#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data pipeline for Stage A (RAVDESS speech emotion) and Stage B (GTZAN genres).

Design notes worth defending in the presentation:

  * SPLIT BY CLIP, NEVER BY SEGMENT. A 30 s GTZAN track chopped into 3 s windows
    yields ~19 near-identical segments. Splitting those at random puts siblings
    in both train and test and inflates accuracy by 10-20 points. Every split
    here is on the clip id, stratified by class.
  * RAVDESS is split BY ACTOR, so Stage A is speaker-independent like the paper.
  * Waveforms are cached once as 16 kHz mono float32 .npy, then windowed on the
    fly. Decoding a 30 s wav per __getitem__ is the actual bottleneck otherwise.
  * `repeats` controls how many random crops each clip contributes per epoch.
    With repeats=1 and 700 training clips an "epoch" is 43 batches — nowhere
    near enough optimizer steps to converge. repeats=10 means each epoch sees
    roughly the whole 30 s of every track, at different random offsets.
  * Normalisation uses the CLIP's mean/std, not the segment's. Per-segment
    standardisation flattens within-track dynamics — loud compressed passages
    versus quiet ones — which is a genuine genre cue (metal versus classical).
"""

import os, glob, hashlib, json
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

SR = 16000

GTZAN_GENRES = ["blues", "classical", "country", "disco", "hiphop",
                "jazz", "metal", "pop", "reggae", "rock"]

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
    """
    Resample every clip to 16 kHz mono float32, memo as .npy, and record each
    clip's length plus its global mean/std for normalisation at load time.
    Stats are themselves cached in _stats.json so re-running is instant.
    """
    os.makedirs(cache_dir, exist_ok=True)
    stats_path = os.path.join(cache_dir, "_stats.json")
    try:
        stats = json.load(open(stats_path))
    except (FileNotFoundError, json.JSONDecodeError):
        stats = {}

    npy, lengths, means, stds, bad = [], [], [], [], []
    resamplers = {}
    for i, row in enumerate(df.itertuples()):
        key = hashlib.md5(row.path.encode()).hexdigest()[:16]
        out = os.path.join(cache_dir, key + ".npy")
        if not os.path.exists(out):
            try:
                w, in_sr = torchaudio.load(row.path)
            except Exception as e:                       # GTZAN ships one corrupt wav
                bad.append((row.path, repr(e)))
                npy.append(None); lengths.append(0); means.append(0.0); stds.append(1.0)
                continue
            w = w.mean(0, keepdim=True)
            if in_sr != sr:
                if in_sr not in resamplers:
                    resamplers[in_sr] = torchaudio.transforms.Resample(in_sr, sr)
                w = resamplers[in_sr](w)
            np.save(out, w.squeeze(0).numpy().astype(np.float32))
            stats.pop(key, None)

        if key not in stats:
            arr = np.asarray(np.load(out, mmap_mode="r"), dtype=np.float32)
            stats[key] = [int(arr.shape[0]), float(arr.mean()), float(arr.std() + 1e-8)]
        n, mu, sd = stats[key]
        npy.append(out); lengths.append(n); means.append(mu); stds.append(sd)
        if verbose and (i + 1) % 200 == 0:
            print(f"  cached {i+1}/{len(df)}")

    json.dump(stats, open(stats_path, "w"))
    df = df.copy()
    df["npy"], df["n_samples"] = npy, lengths
    df["clip_mean"], df["clip_std"] = means, stds
    if bad:
        print(f"[cache] skipped {len(bad)} unreadable file(s): "
              f"{[os.path.basename(b[0]) for b in bad]}")
    return df[df.npy.notna() & (df.n_samples > 0)].reset_index(drop=True)


# --------------------------------------------------------------------- split

def split_by_group(df, test_size=0.15, val_size=0.15, seed=42):
    """Clip/actor-level stratified split. Groups never straddle a boundary."""
    g = df.groupby("group").agg(label=("label", "first")).reset_index()
    tr, te = train_test_split(g, test_size=test_size, stratify=g.label, random_state=seed)
    tr, va = train_test_split(tr, test_size=val_size / (1 - test_size),
                              stratify=tr.label, random_state=seed)
    pick = lambda s: df[df.group.isin(set(s.group))].reset_index(drop=True)
    return pick(tr), pick(va), pick(te)


# ------------------------------------------------------------------- dataset

class SegmentDataset(Dataset):
    """
    train=True  -> `repeats` random crops per clip per epoch.
    train=False -> deterministic tiling with `hop_sec`; each item carries its
                   clip index so predictions can be pooled per clip at eval time.
    """

    def __init__(self, df, classes, seg_sec=3.0, hop_sec=1.5, sr=SR,
                 train=True, augment=True, repeats=1):
        self.df = df.reset_index(drop=True)
        self.classes = list(classes)
        self.c2i = {c: i for i, c in enumerate(self.classes)}
        self.seg_len = int(seg_sec * sr)
        self.train = train
        self.augment = augment and train
        self._rng = None                       # created lazily, once per worker

        if train:
            self.items = [(i, None) for i in range(len(self.df))
                          for _ in range(max(1, repeats))]
        else:
            hop = int(hop_sec * sr)
            self.items = []
            for i, n in enumerate(self.df.n_samples.values):
                if n <= self.seg_len:
                    self.items.append((i, 0))
                else:
                    for off in range(0, n - self.seg_len + 1, hop):
                        self.items.append((i, off))

    @property
    def rng(self):
        # Each DataLoader worker forks a copy of this object. One rng seeded in
        # __init__ would hand every worker the identical crop sequence, so a
        # large share of each batch would be duplicates. torch.initial_seed() is
        # unique per worker and per epoch, so derive the seed from it.
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            seed = (torch.initial_seed() + 7919 * (info.id if info else 0)) % (2 ** 32)
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __len__(self):
        return len(self.items)

    def _load(self, ridx, off):
        arr = np.load(self.df.npy[ridx], mmap_mode="r")
        n = arr.shape[0]
        if off is None:
            off = 0 if n <= self.seg_len else int(self.rng.integers(0, n - self.seg_len + 1))
        seg = np.array(arr[off:off + self.seg_len], dtype=np.float32)
        if seg.shape[0] < self.seg_len:                   # wrap-pad short clips
            reps = int(np.ceil(self.seg_len / max(seg.shape[0], 1)))
            seg = np.tile(seg, reps)[:self.seg_len]
        return seg

    def __getitem__(self, k):
        ridx, off = self.items[k]
        seg = self._load(ridx, off)

        # Clip-level standardisation preserves within-track dynamics.
        seg = (seg - self.df.clip_mean[ridx]) / self.df.clip_std[ridx]

        if self.augment:
            if self.rng.random() < 0.5:                   # polarity flip (label-safe)
                seg = -seg
            if self.rng.random() < 0.5:                   # gain, meaningful now
                seg = seg * np.float32(10 ** (self.rng.uniform(-4, 4) / 20))
            if self.rng.random() < 0.25:
                seg = seg + self.rng.normal(0, 0.02, seg.shape).astype(np.float32)

        x = torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32))
        return x.unsqueeze(0), self.c2i[self.df.label[ridx]], ridx


def class_weights(df, classes):
    counts = df.label.value_counts().reindex(classes).fillna(0).values.astype(np.float32)
    w = counts.sum() / (len(classes) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)
