#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data pipeline for Stage A (FMA-small music genres) and Stage B (GTZAN genres).

Both stages are MUSIC GENRE corpora, so the pre-training source and the target
task share a domain. An earlier version pre-trained on RAVDESS (speech emotion);
that is a different domain and it did not transfer.

Design notes worth defending in the presentation:

  * SPLIT BY CLIP, NEVER BY SEGMENT. A 30 s GTZAN track chopped into 5 s windows
    yields ~11 overlapping, near-identical segments. Splitting those at random
    puts siblings in both train and test and inflates accuracy by 10-20 points.
    Every split here is on the clip id, stratified by class.
  * Waveforms are cached once as 16 kHz mono float32 .npy, then windowed on the
    fly. Decoding a 30 s track per __getitem__ is the actual bottleneck otherwise.
  * `repeats` controls how many random crops each clip contributes per epoch.
    With repeats=1 and 700 training clips an "epoch" is 43 batches — nowhere
    near enough optimizer steps to converge. repeats=10 means each epoch sees
    roughly the whole 30 s of every track, at different random offsets.
  * Normalisation uses the CLIP's mean/std, not the segment's. Per-segment
    standardisation flattens within-track dynamics — loud compressed passages
    versus quiet ones — which is a genuine genre cue (metal versus classical).
"""

import os, glob, hashlib, json
from concurrent.futures import ProcessPoolExecutor, as_completed
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

def _load_any(path, sr):
    """
    Decode one audio file to 16 kHz mono float32.

    torchaudio first; librosa/audioread as a fallback. The fallback is not
    optional for FMA: its archive is mp3, and roughly a hundred of its files are
    truncated or have a broken header, which torchaudio raises on but audioread
    usually recovers.
    """
    try:
        w, in_sr = torchaudio.load(path)
        a = w.mean(0).numpy().astype(np.float32)
        if in_sr != sr:
            a = torchaudio.functional.resample(
                torch.from_numpy(a), in_sr, sr).numpy().astype(np.float32)
        if a.size == 0:
            raise RuntimeError("decoded to zero samples")
        return a
    except Exception:
        import librosa
        a, _ = librosa.load(path, sr=sr, mono=True)
        a = np.asarray(a, dtype=np.float32)
        if a.size == 0:
            raise RuntimeError("decoded to zero samples")
        return a


def _cache_one(args):
    """Worker: decode -> optional centre crop -> save .npy -> return stats."""
    path, out, sr, max_seconds = args
    try:
        if not os.path.exists(out):
            a = _load_any(path, sr)
            if max_seconds is not None:               # keep the centre only
                keep = int(max_seconds * sr)
                if a.shape[0] > keep:
                    off = (a.shape[0] - keep) // 2
                    a = a[off:off + keep]
            tmp = out + ".tmp.npy"                    # atomic: no half-written file
            np.save(tmp, a)
            os.replace(tmp, out)
        arr = np.asarray(np.load(out, mmap_mode="r"), dtype=np.float32)
        return out, [int(arr.shape[0]), float(arr.mean()), float(arr.std() + 1e-8)], None
    except Exception as e:
        return out, None, f"{os.path.basename(path)}: {type(e).__name__}"


def build_cache(df, cache_dir, sr=SR, verbose=True, max_seconds=None, workers=None):
    """
    Resample every clip to 16 kHz mono float32, memo as .npy, and record each
    clip's length plus its global mean/std for normalisation at load time.
    Stats are cached in _stats.json so re-running is instant.

    Decoding is the wall-clock bottleneck (8000 FMA mp3s single-threaded is well
    over half an hour), so it runs in a process pool.
    """
    os.makedirs(cache_dir, exist_ok=True)
    stats_path = os.path.join(cache_dir, "_stats.json")
    try:
        stats = json.load(open(stats_path))
    except (FileNotFoundError, json.JSONDecodeError):
        stats = {}

    outs = [os.path.join(cache_dir, hashlib.md5(p.encode()).hexdigest()[:16] + ".npy")
            for p in df.path]

    todo = [(p, o, sr, max_seconds) for p, o in zip(df.path, outs)
            if o not in stats or not os.path.exists(o)]

    bad = []
    if todo:
        workers = workers or min(8, (os.cpu_count() or 2) * 2)
        if verbose:
            print(f"[cache] decoding {len(todo)} file(s) with {workers} worker(s)")
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_cache_one, t) for t in todo]
            for fut in as_completed(futures):
                out, st, err = fut.result()
                if err:
                    bad.append(err)
                    stats.pop(out, None)
                else:
                    stats[out] = st
                done += 1
                if verbose and done % 500 == 0:
                    print(f"  cached {done}/{len(todo)}")
        json.dump(stats, open(stats_path, "w"))

    npy, lengths, means, stds = [], [], [], []
    for out in outs:
        st = stats.get(out)
        if st is None or not os.path.exists(out):
            npy.append(None); lengths.append(0); means.append(0.0); stds.append(1.0)
        else:
            npy.append(out); lengths.append(st[0]); means.append(st[1]); stds.append(st[2])

    df = df.copy()
    df["npy"], df["n_samples"] = npy, lengths
    df["clip_mean"], df["clip_std"] = means, stds
    kept = df[df.npy.notna() & (df.n_samples > 0)].reset_index(drop=True)
    if bad:
        print(f"[cache] skipped {len(bad)} undecodable file(s), e.g. {bad[:3]}")
    if verbose:
        print(f"[cache] {len(kept)}/{len(df)} clips usable in {cache_dir}")
    return kept


# --------------------------------------------------------------------- split

def split_by_group(df, test_size=0.15, val_size=0.15, seed=42, verbose=True):
    """
    Group-level split: a group (clip, speaker, ...) never straddles a boundary.

    Stratification is only meaningful when each group carries ONE label, which
    holds for GTZAN (a group is a clip, a clip is one genre) but NOT for speech
    corpora, where a group is a speaker and every speaker acts every emotion.
    In that case `("label", "first")` would be an arbitrary pick, so we split the
    groups without stratifying rather than stratify on noise.
    """
    g = (df.groupby("group")
           .agg(label=("label", "first"), n_labels=("label", "nunique"))
           .reset_index())
    pure = bool(g.n_labels.max() == 1)
    if verbose and not pure:
        print(f"[split] groups span multiple labels "
              f"(max {int(g.n_labels.max())}); splitting {len(g)} groups unstratified")

    def _tts(frame, size):
        if pure:
            try:
                return train_test_split(frame, test_size=size,
                                        stratify=frame.label, random_state=seed)
            except ValueError:      # too few groups per class to stratify
                pass
        return train_test_split(frame, test_size=size, random_state=seed)

    tr, te = _tts(g, test_size)
    tr, va = _tts(tr, val_size / (1 - test_size))
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
                 train=True, augment=True, repeats=1, speed_perturb=0.0):
        self.df = df.reset_index(drop=True)
        self.classes = list(classes)
        self.c2i = {c: i for i, c in enumerate(self.classes)}
        self.seg_len = int(seg_sec * sr)
        self.train = train
        self.augment = augment and train
        # speed_perturb = p: with probability p, resample the crop by a random
        # factor in [0.9, 1.1]. This shifts tempo AND pitch together, like a
        # record played off-speed. Standard for music, worth 2-4 points on GTZAN;
        # leave it at 0 for speech, where formant shifts change the label space.
        self.speed_perturb = speed_perturb if train else 0.0
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

        want = self.seg_len
        rate = 1.0
        if self.speed_perturb > 0 and self.rng.random() < self.speed_perturb:
            rate = float(self.rng.uniform(0.9, 1.1))
            want = int(round(self.seg_len * rate))

        if off is None:
            off = 0 if n <= want else int(self.rng.integers(0, n - want + 1))
        seg = np.array(arr[off:off + want], dtype=np.float32)

        if rate != 1.0 and seg.shape[0] > 1:      # linear resample back to seg_len
            src = np.linspace(0.0, seg.shape[0] - 1, self.seg_len, dtype=np.float32)
            seg = np.interp(src, np.arange(seg.shape[0], dtype=np.float32),
                            seg).astype(np.float32)
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


# ------------------------------------------------------------------- FMA

FMA_SMALL_GENRES = ["Electronic", "Experimental", "Folk", "Hip-Hop",
                    "Instrumental", "International", "Pop", "Rock"]


def index_fma(audio_root, meta_csv, subset="small"):
    """
    FMA (https://github.com/mdeff/fma) as a MUSIC source domain for pretraining.
    fma_small is 8000 30 s clips over 8 balanced genres -- an order of magnitude
    more music than GTZAN, and far closer to the target domain than RAVDESS.

      audio_root : extracted fma_small/   (000/000002.mp3, 001/..., ...)
      meta_csv   : fma_metadata/tracks.csv  (two-row header, hence header=[0,1])
    """
    tracks = pd.read_csv(meta_csv, index_col=0, header=[0, 1], low_memory=False)
    sel = tracks[tracks[("set", "subset")] == subset]
    rows, missing = [], 0
    for tid, genre in sel[("track", "genre_top")].items():
        if not isinstance(genre, str):
            continue
        stem = os.path.join(audio_root, f"{tid // 1000:03d}", f"{tid:06d}")
        p = next((stem + ext for ext in (".mp3", ".wav") if os.path.exists(stem + ext)), None)
        if p is None:
            missing += 1
            continue
        rows.append({"path": p, "label": genre,
                     "clip_id": f"{tid:06d}", "group": f"{tid:06d}"})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"no FMA audio under {audio_root} — expected {audio_root}/000/000002.mp3")
    if missing:
        print(f"[fma] {missing} track(s) listed in metadata but absent on disk")
    print(f"[fma] {len(df)} tracks | {df.label.nunique()} genres "
          f"| {df.label.value_counts().to_dict()}")
    return df