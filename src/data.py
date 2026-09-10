#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data pipeline for Stage A (RAVDESS speech emotion) and Stage B (GTZAN genres).

Design notes worth defending in the presentation:

  * SPLIT BY CLIP, NEVER BY SEGMENT. A 30 s GTZAN track chopped into 3 s windows
    yields 10 near-identical segments. Splitting those at random puts siblings in
    both train and test and inflates accuracy by 10-20 points. Every split here
    is on the clip id, stratified by class.
  * RAVDESS is split BY ACTOR, so Stage A is speaker-independent like the paper.
  * Waveforms are cached once as 16 kHz mono float32 .npy, then windowed on the
    fly. Decoding a 30 s mp3/wav per __getitem__ is the actual bottleneck
    otherwise.
  * Train uses a random crop per epoch (free augmentation + decorrelates the
    segments); eval tiles the clip deterministically and votes at clip level.
"""

import os, glob, hashlib
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

SR = 16000

GTZAN_GENRES = ["blues", "classical", "country", "disco", "hiphop",
                "jazz", "metal", "pop", "reggae", "rock"]

# RAVDESS filename field 3 -> emotion
RAVDESS_EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
                    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}


# --------------------------------------------------------------------- index

def index_gtzan(root):
    """root contains genres_original/<genre>/<genre>.000NN.wav"""
    rows = []
    base = os.path.join(root, "genres_original")
    if not os.path.isdir(base):
        base = root
    for g in GTZAN_GENRES:
        for p in sorted(glob.glob(os.path.join(base, g, "*.wav"))):
            rows.append({"path": p, "label": g, "clip_id": os.path.basename(p),
                         "group": os.path.basename(p)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"no GTZAN wavs under {root}")
    return df


def index_ravdess(root):
    """RAVDESS Audio_Speech_Actors_01-24: Actor_XX/03-01-EE-..-AA.wav"""
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)):
        f = os.path.basename(p).replace(".wav", "").split("-")
        if len(f) != 7:
            continue
        rows.append({"path": p, "label": RAVDESS_EMOTIONS[f[2]],
                     "clip_id": os.path.basename(p), "group": f"actor{f[6]}"})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"no RAVDESS wavs under {root}")
    return df


# --------------------------------------------------------------------- cache

def build_cache(df, cache_dir, sr=SR, verbose=True):
    """Resample every clip to 16 kHz mono float32 and memo it as .npy."""
    os.makedirs(cache_dir, exist_ok=True)
    npy, lengths, bad = [], [], []
    resamplers = {}
    for i, row in enumerate(df.itertuples()):
        key = hashlib.md5(row.path.encode()).hexdigest()[:16]
        out = os.path.join(cache_dir, key + ".npy")
        if not os.path.exists(out):
            try:
                w, in_sr = torchaudio.load(row.path)
            except Exception as e:                       # GTZAN ships one corrupt wav
                bad.append((row.path, repr(e)))
                npy.append(None); lengths.append(0); continue
            w = w.mean(0, keepdim=True)
            if in_sr != sr:
                if in_sr not in resamplers:
                    resamplers[in_sr] = torchaudio.transforms.Resample(in_sr, sr)
                w = resamplers[in_sr](w)
            np.save(out, w.squeeze(0).numpy().astype(np.float32))
        npy.append(out)
        lengths.append(int(np.load(out, mmap_mode="r").shape[0]))
        if verbose and (i + 1) % 200 == 0:
            print(f"  cached {i+1}/{len(df)}")
    df = df.copy()
    df["npy"] = npy
    df["n_samples"] = lengths
    if bad:
        print(f"[cache] skipped {len(bad)} unreadable file(s): "
              f"{[os.path.basename(b[0]) for b in bad]}")
    return df[df.npy.notna() & (df.n_samples > 0)].reset_index(drop=True)


# --------------------------------------------------------------------- split

def split_by_group(df, test_size=0.15, val_size=0.15, seed=42):
    """
    Clip/actor-level stratified split. Returns (train, val, test) DataFrames.
    Groups never straddle a split boundary.
    """
    g = df.groupby("group").agg(label=("label", "first")).reset_index()
    tr, te = train_test_split(g, test_size=test_size, stratify=g.label, random_state=seed)
    tr, va = train_test_split(tr, test_size=val_size / (1 - test_size),
                              stratify=tr.label, random_state=seed)
    pick = lambda s: df[df.group.isin(set(s.group))].reset_index(drop=True)
    return pick(tr), pick(va), pick(te)


# ------------------------------------------------------------------- dataset

class SegmentDataset(Dataset):
    """
    train=True  -> one random `seg_len` crop per clip per epoch.
    train=False -> deterministic tiling with `hop`; item carries its clip index
                   so predictions can be pooled per clip at eval time.
    """

    def __init__(self, df, classes, seg_sec=3.0, hop_sec=1.5, sr=SR,
                 train=True, augment=True):
        self.df = df.reset_index(drop=True)
        self.classes = list(classes)
        self.c2i = {c: i for i, c in enumerate(self.classes)}
        self.seg_len = int(seg_sec * sr)
        self.train = train
        self.augment = augment and train
        self.rng = np.random.default_rng(0)

        if train:
            self.items = [(i, None) for i in range(len(self.df))]
        else:
            hop = int(hop_sec * sr)
            self.items = []
            for i, n in enumerate(self.df.n_samples.values):
                if n <= self.seg_len:
                    self.items.append((i, 0))
                else:
                    for off in range(0, n - self.seg_len + 1, hop):
                        self.items.append((i, off))

    def __len__(self):
        return len(self.items)

    def _load(self, ridx, off):
        arr = np.load(self.df.npy[ridx], mmap_mode="r")
        n = arr.shape[0]
        if off is None:
            off = 0 if n <= self.seg_len else int(self.rng.integers(0, n - self.seg_len + 1))
        seg = np.array(arr[off:off + self.seg_len], dtype=np.float32)
        if seg.shape[0] < self.seg_len:                   # pad short clips by wrapping
            reps = int(np.ceil(self.seg_len / max(seg.shape[0], 1)))
            seg = np.tile(seg, reps)[:self.seg_len]
        return seg

    def __getitem__(self, k):
        ridx, off = self.items[k]
        seg = self._load(ridx, off)

        if self.augment:
            if self.rng.random() < 0.5:                   # random gain, +/- 6 dB
                seg = seg * float(10 ** (self.rng.uniform(-6, 6) / 20))
            if self.rng.random() < 0.5:                   # polarity flip (label-safe)
                seg = -seg
            if self.rng.random() < 0.3:                   # light gaussian noise
                seg = seg + self.rng.normal(0, 0.005, seg.shape).astype(np.float32)

        x = torch.from_numpy(np.ascontiguousarray(seg)).float()
        x = (x - x.mean()) / (x.std() + 1e-5)             # per-segment standardisation
        y = self.c2i[self.df.label[ridx]]
        return x.unsqueeze(0), y, ridx


def class_weights(df, classes):
    counts = df.label.value_counts().reindex(classes).fillna(0).values.astype(np.float32)
    w = counts.sum() / (len(classes) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)
