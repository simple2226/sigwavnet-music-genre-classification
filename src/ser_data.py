#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speech-emotion corpora for the paper's ORIGINAL task: EMO-DB and IEMOCAP.

Both indexers return the same frame schema as data.py (path / label / clip_id /
group), so build_cache, split_by_group and SegmentDataset work unchanged.

  EMO-DB   free, 40 MB, http://emodb.bilderbar.info/download/download.zip
           535 utterances, 10 actors, 7 emotions, already 16 kHz mono.
           `group` = speaker, so splits are speaker-independent by construction.

  IEMOCAP  NOT publicly downloadable. Request a licence from USC SAIL at
           https://sail.usc.edu/iemocap/ and point IEMOCAP_ROOT at the
           extracted IEMOCAP_full_release. The standard benchmark protocol is
           4-class (angry / happy+excited / neutral / sad) with leave-one-
           session-out cross-validation, which is what `session_folds` gives.
"""

import os, re, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ------------------------------------------------------------------- EMO-DB

# filename 03a01Fa.wav -> speaker 03, text a01, emotion F, version a
EMODB_EMOTIONS = {"W": "anger", "L": "boredom", "E": "disgust", "A": "fear",
                  "F": "happiness", "T": "sadness", "N": "neutral"}
EMODB_CLASSES = ["anger", "boredom", "disgust", "fear",
                 "happiness", "neutral", "sadness"]


def index_emodb(root):
    """root is the extracted download.zip (contains a wav/ folder)."""
    base = os.path.join(root, "wav")
    if not os.path.isdir(base):
        base = root
    rows = []
    for p in sorted(glob.glob(os.path.join(base, "*.wav"))):
        stem = os.path.basename(p)[:-4]
        if len(stem) < 6 or stem[5] not in EMODB_EMOTIONS:
            continue
        rows.append({"path": p, "label": EMODB_EMOTIONS[stem[5]],
                     "clip_id": stem, "group": f"spk{stem[:2]}"})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"no EMO-DB wavs under {root}")
    return df


# ------------------------------------------------------------------ IEMOCAP

# The 4-class benchmark: excitement is merged into happiness, which is what
# almost every published IEMOCAP number does. Without that merge happy has only
# ~600 utterances and the task is a different (harder, non-comparable) one.
IEMOCAP_MAP_4 = {"ang": "angry", "hap": "happy", "exc": "happy",
                 "neu": "neutral", "sad": "sad"}
IEMOCAP_CLASSES_4 = ["angry", "happy", "neutral", "sad"]

_EVAL_LINE = re.compile(
    r"^\[(?P<start>[\d.]+)\s*-\s*(?P<end>[\d.]+)\]\s+(?P<utt>\S+)\s+(?P<emo>\w+)")


def index_iemocap(root, label_map=None, sessions=(1, 2, 3, 4, 5)):
    """
    root = .../IEMOCAP_full_release

    Parses Session{n}/dialog/EmoEvaluation/*.txt, whose utterance lines look
    like:  [6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5, 2.5, 2.5]
    and resolves each to Session{n}/sentences/wav/<dialog>/<utt>.wav.

    `group` is the SPEAKER (e.g. Ses01F), so a grouped split is speaker-
    independent. Use session_folds() for the leave-one-session-out protocol.
    """
    label_map = label_map or IEMOCAP_MAP_4
    rows, missing = [], 0
    for s in sessions:
        evald = os.path.join(root, f"Session{s}", "dialog", "EmoEvaluation")
        wavd = os.path.join(root, f"Session{s}", "sentences", "wav")
        for txt in sorted(glob.glob(os.path.join(evald, "*.txt"))):
            for line in open(txt, encoding="utf-8", errors="ignore"):
                m = _EVAL_LINE.match(line)
                if not m:
                    continue
                emo = m.group("emo")
                if emo not in label_map:
                    continue
                utt = m.group("utt")
                wav = os.path.join(wavd, utt.rsplit("_", 1)[0], utt + ".wav")
                if not os.path.exists(wav):
                    missing += 1
                    continue
                # Ses01F_impro01_F000 -> speaker Ses01F (session 1, female)
                speaker = f"Ses0{s}{utt.rsplit('_', 1)[1][0]}"
                rows.append({"path": wav, "label": label_map[emo], "clip_id": utt,
                             "group": speaker, "session": s})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"no IEMOCAP utterances under {root} — expected "
            f"{root}/Session1/dialog/EmoEvaluation/*.txt")
    if missing:
        print(f"[iemocap] {missing} annotated utterances had no wav, skipped")
    print(f"[iemocap] {len(df)} utterances | {df.label.value_counts().to_dict()}")
    return df


def session_folds(df):
    """Leave-one-session-out: yields (fold_name, train_df, test_df) x 5."""
    for s in sorted(df.session.unique()):
        te = df[df.session == s].reset_index(drop=True)
        tr = df[df.session != s].reset_index(drop=True)
        yield f"session{s}", tr, te


# --------------------------------------------------------- speaker-grouped CV

def speaker_folds(df, n_folds=5, seed=42):
    """
    Stratified grouped CV over speakers. EMO-DB has 10 actors, so n_folds=5
    holds out 2 per fold; n_folds=10 is the classic LOSO protocol.
    Yields (fold_name, train_df, test_df).
    """
    spk = df.groupby("group").agg(label=("label", "first")).reset_index()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr_i, te_i) in enumerate(skf.split(spk, spk.label), start=1):
        tr_spk, te_spk = set(spk.group[tr_i]), set(spk.group[te_i])
        yield (f"fold{k}",
               df[df.group.isin(tr_spk)].reset_index(drop=True),
               df[df.group.isin(te_spk)].reset_index(drop=True))


def holdout_from_train(train_df, val_frac=0.2, seed=0):
    """Carve a speaker-disjoint validation set out of a CV training split."""
    rng = np.random.default_rng(seed)
    spk = sorted(train_df.group.unique())
    n_val = max(1, int(round(val_frac * len(spk))))
    val_spk = set(rng.choice(spk, size=n_val, replace=False))
    return (train_df[~train_df.group.isin(val_spk)].reset_index(drop=True),
            train_df[train_df.group.isin(val_spk)].reset_index(drop=True))
