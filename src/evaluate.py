#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation: segment-level metrics, clip-level pooling, confusion matrix,
learned-kernel inspection.

The clip-level number is the one to report. GTZAN is a 1000-clip dataset; a
segment-level score is a score over ~6600 correlated windows and is not
comparable to anything in the literature.
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)

from train import get_device, evaluate_segments


@torch.no_grad()
def evaluate_clips(model, test_ds, classes, batch_size=16, num_workers=2):
    """Average log-probabilities over every window of a clip, then argmax."""
    device = get_device()
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())
    seg = evaluate_segments(model, loader, device)

    n_clips = len(test_ds.df)
    acc = np.zeros((n_clips, len(classes)), dtype=np.float64)
    cnt = np.zeros(n_clips, dtype=np.int64)
    for lp, r in zip(seg["logp"], seg["ridx"]):
        acc[r] += lp
        cnt[r] += 1
    keep = cnt > 0
    clip_pred = acc[keep].argmax(1)
    clip_true = np.array([test_ds.c2i[l] for l in test_ds.df.label.values])[keep]

    return {
        "segment_acc": seg["acc"], "segment_f1": seg["f1"],
        "clip_acc": float(accuracy_score(clip_true, clip_pred)),
        "clip_f1": float(f1_score(clip_true, clip_pred, average="macro")),
        "report": classification_report(clip_true, clip_pred,
                                        target_names=classes, digits=3, zero_division=0),
        "cm": confusion_matrix(clip_true, clip_pred, labels=range(len(classes))),
        "clip_true": clip_true, "clip_pred": clip_pred,
    }


def plot_confusion(cm, classes, out="confusion_matrix.png", normalize=True, title=None):
    m = cm.astype(float)
    if normalize:
        m = m / np.maximum(m.sum(1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(m, cmap="viridis")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title or ("Confusion matrix (%)" if normalize else "Confusion matrix"))
    thr = m.max() / 2
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center",
                    color="white" if m[i, j] < thr else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out


def plot_kernels(model, out="learned_kernels.png", n_levels=3):
    """Low/high-pass filters after training vs their db10 initialisation."""
    fig, axes = plt.subplots(n_levels, 2, figsize=(9, 2.2 * n_levels), squeeze=False)
    for lev in range(min(n_levels, model.level)):
        h = model.kernelsH_[lev]().detach().cpu().numpy()
        g = np.flip(h) * np.array([(-1.0) ** i for i in range(len(h))])
        for col, (v, name) in enumerate([(h, "low-pass  h"), (g, "high-pass g")]):
            ax = axes[lev][col]
            ax.stem(v, basefmt=" ")
            ax.set_title(f"level {lev+1} — {name}", fontsize=10)
            ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out


def plot_history(histories, out="training_curves.png"):
    """histories: {'label': [ {epoch, val_acc, val_f1, train_acc}, ... ]}"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_acc"] for r in h], "--", alpha=.6, label=f"{name} train")
        axes[0].plot(ep, [r["val_acc"] for r in h], label=f"{name} val")
        axes[1].plot(ep, [r["val_f1"] for r in h], label=name)
    axes[0].set_title("Accuracy"); axes[1].set_title("Validation macro-F1")
    for a in axes:
        a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out


def save_results(tag, res, path="results.json"):
    try:
        with open(path) as f:
            all_res = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_res = {}
    all_res[tag] = {k: v for k, v in res.items()
                    if k in ("segment_acc", "segment_f1", "clip_acc", "clip_f1")}
    all_res[tag]["cm"] = res["cm"].tolist()
    with open(path, "w") as f:
        json.dump(all_res, f, indent=2)
    return path


@torch.no_grad()
def ensemble_clips(models, test_ds, classes, batch_size=32, num_workers=2):
    """
    Average clip-level log-probabilities across several trained models (seeds).
    Reliably worth 3-5 points on GTZAN and costs nothing but the extra runs:
    the members disagree most on exactly the confusable classes, so averaging
    cancels a chunk of the variance.
    """
    device = get_device()
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    n_clips, K = len(test_ds.df), len(classes)
    acc = np.zeros((n_clips, K), dtype=np.float64)
    cnt = np.zeros(n_clips, dtype=np.int64)

    for m in models:
        seg = evaluate_segments(m, loader, device)
        for lp, r in zip(seg["logp"], seg["ridx"]):
            acc[r] += lp
            cnt[r] += 1

    keep = cnt > 0
    pred = acc[keep].argmax(1)
    true = np.array([test_ds.c2i[l] for l in test_ds.df.label.values])[keep]
    return {
        "clip_acc": float(accuracy_score(true, pred)),
        "clip_f1": float(f1_score(true, pred, average="macro")),
        "report": classification_report(true, pred, target_names=classes,
                                        digits=3, zero_division=0),
        "cm": confusion_matrix(true, pred, labels=range(K)),
        "segment_acc": float("nan"), "segment_f1": float("nan"),
        "clip_true": true, "clip_pred": pred,
    }


def unweighted_and_weighted_accuracy(y_true, y_pred):
    """
    SER convention: WA = plain accuracy, UA = mean per-class recall (= macro
    recall). Papers on IEMOCAP and EMO-DB report both, and UA is the one that
    matters on imbalanced corpora. Report both so your numbers are comparable
    to the base paper's table.
    """
    from sklearn.metrics import recall_score
    return (float(accuracy_score(y_true, y_pred)),
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)))
