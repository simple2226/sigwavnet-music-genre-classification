#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-config trainer for SigWavNet.

CHANGES vs upstream main.py:
  * Ray Tune + ASHA over a 5-value batch-size grid x 10 samples is ~50 full
    training runs, which cannot finish in a Colab session. Replaced by one
    explicit config.
  * Mixed precision (torch.amp) + gradient clipping.
  * Cosine LR with warmup instead of StepLR(20, 0.1), which never fires in a
    short run.
  * Model selection on validation macro-F1, not accuracy.
  * Checkpoints are plain dicts on disk; no tune.checkpoint_dir.

CHANGES in this revision (the accuracy fix):
  * `gamma` now defaults to 0.0. Focal loss down-weights examples the model
    already classifies correctly, which is counterproductive while the model is
    underfitting — and GTZAN is exactly balanced, so the class-weighting half of
    focal loss buys nothing either. gamma>0 stays available as an ablation.
  * Optional mixup on the raw waveform. Mixing two waveforms is physically
    meaningful (it is just superposition) and is the strongest single
    regulariser for small audio corpora.
  * The scheduler only steps when the optimizer actually stepped. GradScaler
    skips steps on fp16 overflow, which is what produced the
    "lr_scheduler.step() before optimizer.step()" warning.
  * `min_lr_frac` keeps the cosine from decaying fully to zero, which was
    freezing the fine-tuned run flat after ~epoch 18.
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


def make_loaders(train_ds, val_ds, batch_size, num_workers=4):
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=num_workers > 0,
              prefetch_factor=4 if num_workers > 0 else None)
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
        out = out.float()
        if criterion is not None:
            loss_sum += criterion(out, y).item() * y.size(0)
        n += y.size(0)
        logps.append(out.cpu()); ps.append(out.argmax(1).cpu())
        ys.append(y.cpu()); ridxs.append(r)
    y = torch.cat(ys).numpy(); pr = torch.cat(ps).numpy()
    return {"acc": float((y == pr).mean()),
            "f1": float(f1_score(y, pr, average="macro")),
            "loss": loss_sum / max(n, 1),
            "y": y, "pred": pr,
            "logp": torch.cat(logps).numpy(),
            "ridx": torch.cat(ridxs).numpy()}


def train_model(model, train_ds, val_ds, *, classes, weights=None, epochs=40,
                batch_size=32, lr_head=2e-3, lr_encoder=1e-3, lr_wavelet=5e-4,
                weight_decay=1e-4, gamma=0.0, label_smoothing=0.05,
                mixup_alpha=0.3, freeze_epochs=0, warmup_frac=0.03,
                min_lr_frac=0.05, grad_clip=5.0, ckpt_path="best.pt",
                num_workers=4, log_every=100, seed=0):
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

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        prog = (s - warmup) / max(1, total_steps - warmup)
        return min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(f"[setup] {len(train_ds)} train segments | {len(train_loader)} batches/epoch "
          f"| {total_steps} total steps | gamma={gamma} mixup={mixup_alpha}")

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

            lam, y_b = 1.0, y
            if mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(x.size(0), device=x.device)
                x = lam * x + (1.0 - lam) * x[perm]
                y_b = y[perm]

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out = model(x)
            out = out.float()
            loss = (criterion(out, y) if lam == 1.0
                    else lam * criterion(out, y) + (1 - lam) * criterion(out, y_b))

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            prev_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= prev_scale:   # step was not skipped by AMP
                sched.step()

            run_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()   # vs the dominant label
            seen += y.size(0)
            if log_every and i % log_every == 0:
                print(f"  ep{ep} [{i}/{len(train_loader)}] "
                      f"loss {run_loss/seen:.4f} acc {correct/seen:.3f}")

        va = evaluate_segments(model, val_loader, device, criterion)
        row = {"epoch": ep, "train_loss": run_loss / seen, "train_acc": correct / seen,
               "val_loss": va["loss"], "val_acc": va["acc"], "val_f1": va["f1"],
               "lr": opt.param_groups[0]["lr"], "secs": round(time.time() - t0, 1)}
        history.append(row)
        print(f"epoch {ep:>3} | train {row['train_acc']:.3f} | "
              f"val acc {va['acc']:.3f} f1 {va['f1']:.3f} | "
              f"lr {row['lr']:.2e} | {row['secs']}s")

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
