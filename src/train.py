#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-config trainer for SigWavNet.

CHANGES vs upstream main.py:
  * Ray Tune + ASHA over a 5-value batch-size grid x 10 samples is ~50 full
    training runs. That is not runnable on a Colab session, so it is replaced by
    one explicit config plus an optional short LR sweep.
  * Mixed precision (torch.amp) + gradient clipping.
  * Cosine LR schedule with warmup, instead of StepLR(20, 0.1) which never fires
    inside a short run.
  * Model selection on validation macro-F1, not accuracy (GTZAN is balanced but
    RAVDESS is not, and F1 is what the paper reports).
  * Checkpoints are plain dicts on disk; no tune.checkpoint_dir.
"""

import os, json, time, math, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from model import SigWavNet
from custom_layers import FocalLoss


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(train_ds, val_ds, batch_size, num_workers=2):
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=num_workers > 0)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw))


@torch.no_grad()
def evaluate_segments(model, loader, device, criterion=None):
    model.eval()
    ys, ps, logps, ridxs, loss_sum, n = [], [], [], [], 0.0, 0
    for x, y, r in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x)
        if criterion is not None:
            loss_sum += criterion(out.float(), y).item() * y.size(0)
        n += y.size(0)
        logps.append(out.float().cpu())
        ps.append(out.argmax(1).cpu())
        ys.append(y.cpu())
        ridxs.append(r)
    y = torch.cat(ys).numpy(); p = torch.cat(ps).numpy()
    return {"acc": float((y == p).mean()),
            "f1": float(f1_score(y, p, average="macro")),
            "loss": loss_sum / max(n, 1),
            "y": y, "pred": p,
            "logp": torch.cat(logps).numpy(),
            "ridx": torch.cat(ridxs).numpy()}


def train_model(model, train_ds, val_ds, *, classes, weights=None, epochs=25,
                batch_size=16, lr_head=1e-3, lr_encoder=3e-4, lr_wavelet=1e-4,
                weight_decay=1e-5, gamma=2.0, label_smoothing=0.05,
                freeze_epochs=0, warmup_frac=0.05, grad_clip=5.0,
                ckpt_path="best.pt", num_workers=2, log_every=50, seed=0):
    """
    freeze_epochs > 0 -> Stage B1: head-only for that many epochs, then unfreeze
    everything (Stage B2). Set to 0 for from-scratch training.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device = get_device()
    model.to(device)

    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size, num_workers)
    criterion = FocalLoss(alpha=weights, gamma=gamma,
                          label_smoothing=label_smoothing).to(device)

    opt = torch.optim.AdamW(model.param_groups(lr_head, lr_encoder, lr_wavelet),
                            weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    warmup = max(1, int(warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if freeze_epochs > 0:
        model.set_trainable_stages(wavelet=False, encoder=False, head=True)
        print(f"[stage B1] head-only for {freeze_epochs} epoch(s)")

    history, best_f1, best_state = [], -1.0, None
    for ep in range(1, epochs + 1):
        if freeze_epochs > 0 and ep == freeze_epochs + 1:
            model.set_trainable_stages(True, True, True)
            print("[stage B2] full network unfrozen")

        model.train()
        t0, run_loss, correct, seen = time.time(), 0.0, 0, 0
        for i, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update(); sched.step()

            run_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += y.size(0)
            if log_every and i % log_every == 0:
                print(f"  ep{ep} [{i}/{len(train_loader)}] "
                      f"loss {run_loss/seen:.4f} acc {correct/seen:.3f}")

        va = evaluate_segments(model, val_loader, device, criterion)
        row = {"epoch": ep, "train_loss": run_loss / seen, "train_acc": correct / seen,
               "val_loss": va["loss"], "val_acc": va["acc"], "val_f1": va["f1"],
               "secs": round(time.time() - t0, 1)}
        history.append(row)
        print(f"epoch {ep:>3} | train {row['train_acc']:.3f} | "
              f"val acc {va['acc']:.3f} f1 {va['f1']:.3f} | {row['secs']}s")

        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model": best_state, "classes": classes,
                        "val_f1": best_f1, "epoch": ep}, ckpt_path)

    if best_state is not None:
        model.load_state_dict(best_state)
    with open(os.path.splitext(ckpt_path)[0] + "_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"best val macro-F1 {best_f1:.4f} -> {ckpt_path}")
    return model, history


def build_model(n_output, level=8, hidden_dim=64, n_layers=3, n_channel=32,
                kernel="db10", mode="PerFilter", initHT=1.0, alpha=10.0, dropout=0.1):
    import pywt
    k = np.array(pywt.Wavelet(kernel).filter_bank[0], dtype=np.float32)
    return SigWavNet(n_input=1, hidden_dim=hidden_dim, n_layers=n_layers,
                     n_output=n_output, n_channel=n_channel, kernelInit=k,
                     kernTrainable=True, level=level, kernelsConstraint=mode,
                     initHT=initHT, trainHT=True, alpha=alpha, dropout=dropout)
