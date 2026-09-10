#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SigWavNet architecture, adapted for music genre classification.

Derived from Alaa Nfissi's SigWavNet (BSD 3-Clause, see third_party/LICENSE-SigWavNet).

CHANGES vs upstream:
  * forward(x) instead of forward(x, h). Upstream threaded a GRU hidden state
    across mini-batches; since batches are shuffled *independent* clips, that
    carried one clip's state into another clip's forward pass. Hidden state is
    now zero-initialised per batch (nn.GRU's default when h=None), which also
    removes the requirement that every batch be exactly `batch_size` long.
  * Kernels are materialised inside forward() rather than cached as a plain
    Python list at construction time. The upstream list is built once in
    __init__ and is invisible to .to(), DataParallel replication, and
    state_dict, which silently breaks multi-GPU and device moves.
  * `replace_head(n_output)` + `load_pretrained()` for cross-domain transfer.
  * `set_trainable_stages()` for staged unfreezing.
  * Dropout on the GRU stack when n_layers > 1 (upstream hardcoded dropout=0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_layers import (Kernel, LowPassWave, HighPassWave, HardThresholdAssym,
                           SpatialAttentionBlock, TemporalAttn)


class CNN1DSABiGRUTA(nn.Module):
    """Per-band encoder: 1D dilated CNN -> spatial attention -> Bi-GRU -> temporal attention."""

    def __init__(self, n_input, n_channel, hidden_dim, n_layers,
                 normalize_attn=True, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=7, stride=4, dilation=3)
        self.in1 = nn.InstanceNorm1d(n_channel)
        self.relu1 = nn.LeakyReLU()

        self.conv1_1 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=5, stride=4, dilation=2)
        self.in1_1 = nn.InstanceNorm1d(2 * n_channel)
        self.relu1_1 = nn.LeakyReLU()

        self.dense1 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=7, padding=3, bias=True)
        self.in1_2 = nn.InstanceNorm1d(2 * n_channel)
        self.relu1_2 = nn.LeakyReLU()

        self.spatialAttn = SpatialAttentionBlock(2 * n_channel, normalize_attn)
        self.reluAtt1 = nn.LeakyReLU()

        self.gru1 = nn.GRU(2 * n_channel, hidden_dim, n_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.tempAttn = TemporalAttn(hidden_size=2 * hidden_dim)

    def forward(self, x):
        x = self.relu1(self.in1(self.conv1(x)))
        x = self.relu1_1(self.in1_1(self.conv1_1(x)))

        g1 = self.relu1_2(self.in1_2(self.dense1(x)))
        _, g_attended = self.spatialAttn(x, g1)
        x = self.reluAtt1(x + g_attended.unsqueeze(2))

        x = x.permute(0, 2, 1)
        x, _ = self.gru1(x)                # h defaults to zeros -> no cross-sample leakage
        x, _ = self.tempAttn(x)
        return x.unsqueeze(1)              # (B, 1, 2*hidden_dim)


class ChannelWeighting(nn.Module):
    """Learnable per-frequency-band scaling (eq. 17)."""

    def __init__(self, num_channels):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        return x * self.weights.view(1, -1, 1)


class SigWavNet(nn.Module):
    """
    Learnable FDWT -> per-band CNN/attention/Bi-GRU -> channel weighting -> GAP -> log_softmax.

    kernelsConstraint:
      'CQF'       one kernel shared by every level (h_0 learned, rest via CQF)
      'PerLayer'  one kernel per level, g derived from h by CQF
      'PerFilter' independent h and g kernels per level
    """

    def __init__(self, n_input=1, hidden_dim=64, n_layers=3, n_output=10,
                 n_channel=32, kernelInit=20, kernTrainable=True, level=8,
                 kernelsConstraint="PerFilter", initHT=1.0, trainHT=True,
                 alpha=10.0, dropout=0.1):
        super().__init__()
        self.n_input, self.n_output = n_input, n_output
        self.hidden_dim, self.n_layers = hidden_dim, n_layers
        self.n_channel, self.level = n_channel, level
        self.kernelsConstraint = kernelsConstraint

        if kernelsConstraint == "CQF":
            shared = Kernel(kernelInit, trainKern=kernTrainable)
            self.kernelsG_ = nn.ModuleList([shared] * level)
            self.kernelsH_ = self.kernelsG_
        elif kernelsConstraint == "PerLayer":
            self.kernelsG_ = nn.ModuleList(
                [Kernel(kernelInit, trainKern=kernTrainable) for _ in range(level)])
            self.kernelsH_ = self.kernelsG_
        elif kernelsConstraint == "PerFilter":
            self.kernelsG_ = nn.ModuleList(
                [Kernel(kernelInit, trainKern=kernTrainable) for _ in range(level)])
            self.kernelsH_ = nn.ModuleList(
                [Kernel(kernelInit, trainKern=kernTrainable) for _ in range(level)])
        else:
            raise ValueError(f"unknown kernelsConstraint {kernelsConstraint!r}")

        self.LowPassWave = nn.ModuleList([LowPassWave() for _ in range(level)])
        self.HighPassWave = nn.ModuleList([HighPassWave() for _ in range(level)])
        self.HardThresholdAssymH = nn.ModuleList(
            [HardThresholdAssym(init=initHT, alpha=alpha, trainBias=trainHT)
             for _ in range(level)])

        self.conv1ds = nn.ModuleList([
            CNN1DSABiGRUTA(n_input, n_channel, hidden_dim, n_layers, True, dropout)
            for _ in range(level + 1)])

        self.channel_weighting = ChannelWeighting(level + 1)
        self.conv1_3 = nn.Conv1d(level + 1, n_output, kernel_size=5, stride=1)
        self.in1_3 = nn.InstanceNorm1d(n_output)
        self.relu1_3 = nn.LeakyReLU()

    # ---------------------------------------------------------------- forward

    def forward(self, x):
        """x: (B, 1, T) raw waveform -> (B, n_output) log-probabilities."""
        bands = []
        for lev in range(self.level):
            kh = self.kernelsH_[lev]()
            kg = self.kernelsG_[lev]()
            d = self.HighPassWave[lev]([x, kh])          # detail  (high-pass)
            d = self.HardThresholdAssymH[lev](d)         # LAHT denoising
            x = self.LowPassWave[lev]([x, kg])           # approximation (low-pass)
            bands.append(self.conv1ds[lev](d))
        bands.insert(0, self.conv1ds[self.level](x))     # final low-frequency band

        x = torch.cat(bands, dim=1)                      # (B, level+1, 2*hidden_dim)
        x = self.channel_weighting(x)
        x = self.relu1_3(self.in1_3(self.conv1_3(x)))
        return F.log_softmax(x.mean(2), dim=1)

    # ------------------------------------------------------- transfer helpers

    def replace_head(self, n_output):
        """Swap the class-projection conv for a new task. Wavelet+encoder weights kept."""
        dev = self.conv1_3.weight.device
        self.conv1_3 = nn.Conv1d(self.level + 1, n_output, kernel_size=5, stride=1).to(dev)
        self.in1_3 = nn.InstanceNorm1d(n_output).to(dev)
        self.n_output = n_output
        return self

    def load_pretrained(self, ckpt_path, map_location="cpu", verbose=True):
        """Load a Stage-A checkpoint, tolerating the head shape mismatch."""
        sd = torch.load(ckpt_path, map_location=map_location)
        sd = sd.get("model", sd)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        own = self.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        skipped = sorted(set(sd) - set(keep))
        self.load_state_dict(keep, strict=False)
        if verbose:
            print(f"[transfer] loaded {len(keep)}/{len(own)} tensors; "
                  f"re-initialised: {skipped}")
        return self

    def set_trainable_stages(self, wavelet=True, encoder=True, head=True):
        """Staged unfreezing. Stage B1: (False, False, True). Stage B2: all True."""
        for m in (self.kernelsG_, self.kernelsH_, self.HardThresholdAssymH):
            for p in m.parameters():
                p.requires_grad = wavelet
        for p in self.conv1ds.parameters():
            p.requires_grad = encoder
        for m in (self.conv1_3, self.in1_3, self.channel_weighting):
            for p in m.parameters():
                p.requires_grad = head
        return self

    def param_groups(self, lr_head, lr_encoder, lr_wavelet):
        """Discriminative learning rates: head > encoder > wavelet."""
        wav = list(self.kernelsG_.parameters()) + list(self.HardThresholdAssymH.parameters())
        if self.kernelsH_ is not self.kernelsG_:
            wav += list(self.kernelsH_.parameters())
        head = (list(self.conv1_3.parameters()) + list(self.in1_3.parameters())
                + list(self.channel_weighting.parameters()))
        return [
            {"params": wav, "lr": lr_wavelet},
            {"params": list(self.conv1ds.parameters()), "lr": lr_encoder},
            {"params": head, "lr": lr_head},
        ]

    def min_input_length(self):
        """Smallest T that survives `level` decimations plus the encoder strides."""
        k = self.kernelsG_[0].kernelSize
        t = 64  # need >= ~64 samples entering the deepest encoder
        for _ in range(self.level):
            t = t * 2 + k
        return t
