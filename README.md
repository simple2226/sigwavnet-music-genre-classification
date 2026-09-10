# SigWavNet for Music Genre Classification

Adapting **SigWavNet** — a learnable fast-discrete-wavelet-transform network built for
speech emotion recognition — to **music genre classification** on GTZAN, by pretraining on
speech (RAVDESS) and fine-tuning the wavelet front end and encoders on music.

> **Base paper:** A. Nfissi, W. Bouachir, N. Bouguila, B. Mishara.
> *SigWavNet: Learning Multiresolution Signal Wavelet Network for Speech Emotion Recognition.*
> IEEE Transactions on Affective Computing, 2025. [arXiv:2502.00310](https://arxiv.org/abs/2502.00310)
>
> **Original code:** <https://github.com/alaaNfissi/SigWavNet-Learning-Multiresolution-Signal-Wavelet-Network-for-Speech-Emotion-Recognition>
> © 2024 Alaa Nfissi, BSD 3-Clause. Full text in [`third_party/LICENSE-SigWavNet`](third_party/LICENSE-SigWavNet).

---

## Why this is a fine-tuning task and not a re-implementation

The upstream repository publishes the architecture and the training script but **no trained
weights**. There is therefore nothing to fine-tune off the shelf. This repo produces the
checkpoint first and then fine-tunes it:

| Stage | Data | What trains |
|---|---|---|
| **A** | RAVDESS — 1440 clips, 8 emotions, split by *actor* | whole network, from db10-initialised wavelet kernels |
| **B1** | GTZAN — 1000 clips, 10 genres, split by *clip* | new 10-class head only, backbone frozen |
| **B2** | GTZAN | everything, with head > encoder > wavelet learning rates |
| **Control** | GTZAN | identical architecture from random init |

The Stage-B2 vs control gap is the transfer result.

---

## Quick start

Open [`notebooks/SigWavNet_MusicGenre_Colab.ipynb`](notebooks/SigWavNet_MusicGenre_Colab.ipynb)
in Google Colab, set **Runtime → T4 GPU**, and run top to bottom. It downloads both datasets,
builds the 16 kHz cache, runs all three trainings, and writes `results.json` plus the figures.

Locally:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -c "
from data import *; from train import *; from evaluate import *
gtz = build_cache(index_gtzan('data/gtzan'), 'cache/gtzan')
tr, va, te = split_by_group(gtz)
m = build_model(n_output=10, level=8)
m.load_pretrained('stageA_ravdess.pt').replace_head(10)
train_model(m, SegmentDataset(tr, GTZAN_GENRES), SegmentDataset(va, GTZAN_GENRES, train=False),
            classes=GTZAN_GENRES, epochs=30, freeze_epochs=3, ckpt_path='stageB.pt')
"
```

---

## What I changed relative to the upstream code

**Correctness**

1. **GRU hidden state was carried across mini-batches.** `main.py` initialised `h` once per
   epoch and threaded it through every batch. Batches are shuffled independent clips, so one
   clip's recurrent state was seeding another's forward pass. `forward()` now takes only `x`
   and zero-initialises `h` per batch — which also removes the upstream requirement that every
   batch be exactly `batch_size` long.
2. **Focal loss was not focal.** Upstream computed `F.cross_entropy(..., reduction='mean')`,
   a scalar, then applied `(1 - exp(-ce))**gamma` to it. The modulating factor was therefore
   identical for every sample and the loss reduced to weighted CE times a constant. Fixed to
   per-sample CE (`custom_layers.FocalLoss`).
3. **Wavelet kernels were invisible to PyTorch's device/replication machinery.** `self.kernelsG`
   was a plain Python list built in `__init__`, so `.to(device)`, `DataParallel`, and
   `state_dict` round-trips did not see it. Kernels are now materialised inside `forward()`
   from the registered `nn.ModuleList`.
4. **Hard-coded `device = torch.device("cuda:0")` at module scope** in `custom_layers.py` made
   the code unimportable without a GPU. Everything now derives device from the input tensor.
5. **`main.test()` referenced an undefined global `test_ds`.** Rewritten as `evaluate.py`.
6. **LAHT thresholds could go negative** during training, which inverts the thresholding
   semantics. They are now `softplus`-parameterised, and `alpha` is separately controllable
   from `trainBias`.

**Method**

7. **Clip-level splits.** A 30 s GTZAN track windowed into 3 s segments gives ~19 highly
   correlated views. Splitting segments at random leaks siblings across train/test. All splits
   here are on the clip id (GTZAN) or actor id (RAVDESS), stratified by class.
8. **Clip-level evaluation.** Log-probabilities are averaged over every window of a track before
   the argmax, so the reported number is over 150 test tracks, not ~2800 correlated windows.
9. **Cross-domain transfer** — head swap, staged unfreezing, discriminative learning rates.
10. **Waveform augmentation** — random crop per epoch, random gain ±6 dB, polarity flip,
    light Gaussian noise.

**Engineering**

11. **Ray Tune / ASHA removed.** The upstream grid is 5 batch sizes × 10 samples ≈ 50 full runs;
    it cannot finish in a Colab session. Replaced with one explicit config.
12. Mixed precision + gradient clipping, cosine schedule with warmup, model selection on
    validation macro-F1, `.npy` waveform cache so decoding isn't the bottleneck.
13. `n_channel` reduced 128 → 32 to fit a 16 GB T4 at `level=8`.

---

## Licence

The upstream code is **BSD 3-Clause**, which is permissive. It lets you copy, modify, and
redistribute the code (commercially too), on three conditions:

1. keep the copyright notice, the condition list, and the disclaimer in source you redistribute;
2. reproduce them in the documentation of any binary distribution;
3. do not use the authors' names to endorse your derived work.

There is **no copyleft** — this repo is not obliged to be BSD. It stays BSD 3-Clause anyway
because that is the least friction. Compliance here is:

- [`third_party/LICENSE-SigWavNet`](third_party/LICENSE-SigWavNet) — upstream licence, verbatim, unmodified.
- [`LICENSE`](LICENSE) — this repo, listing both copyright holders.
- Every derived file in `src/` carries a header naming the origin and its licence.
- Attribution in this README, plus the paper citation below.

**GTZAN** carries no formal licence; it is distributed for research use. Cite Tzanetakis & Cook
(2002) and note its documented faults — exact duplicates, mislabelled tracks, and distorted
files (Sturm, 2013) — when reporting results.
**RAVDESS** is CC BY-NC-SA 4.0: non-commercial, share-alike, attribution required.

```bibtex
@article{nfissi2025sigwavnet,
  title   = {SigWavNet: Learning Multiresolution Signal Wavelet Network for Speech Emotion Recognition},
  author  = {Nfissi, Alaa and Bouachir, Wassim and Bouguila, Nizar and Mishara, Brian},
  journal = {IEEE Transactions on Affective Computing},
  year    = {2025}
}
@article{tzanetakis2002musical,
  title   = {Musical genre classification of audio signals},
  author  = {Tzanetakis, George and Cook, Perry},
  journal = {IEEE Transactions on Speech and Audio Processing},
  volume  = {10}, number = {5}, pages = {293--302}, year = {2002}
}
```
