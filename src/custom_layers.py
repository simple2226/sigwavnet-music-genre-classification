#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom layers for SigWavNet.

Derived from:
  SigWavNet: Learning Multiresolution Signal Wavelet Network for Speech Emotion
  Recognition -- Alaa Nfissi, Wassim Bouachir, Nizar Bouguila, Brian Mishara.
  https://github.com/alaaNfissi/SigWavNet-Learning-Multiresolution-Signal-Wavelet-Network-for-Speech-Emotion-Recognition
  Copyright (c) 2024, Alaa Nfissi. Licensed under BSD 3-Clause (see third_party/LICENSE-SigWavNet).

CHANGES vs upstream:
  * Removed the module-level `device = torch.device("cuda:0")`. Every op now
    derives its device from the incoming tensor, so the code runs on CPU, MPS,
    or any CUDA ordinal without edits.
  * HighPassWave: the QMF sign-flip vector is registered as a buffer keyed by
    kernel length instead of being lazily attached as a stray attribute, so it
    moves with .to()/.cuda() and survives state_dict round-trips.
  * HardThresholdAssym: `alpha` is no longer tied to `trainBias`; sharpness and
    thresholds are separately controllable, and thresholds are kept positive via
    softplus so the LAHT stays a true two-sided threshold during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel(nn.Module):
    """A learnable 1D filter. Initialised randomly (int) or from an array."""

    def __init__(self, kernelInit=20, trainKern=True):
        super().__init__()
        self.trainKern = trainKern
        if isinstance(kernelInit, int):
            self.kernelSize = kernelInit
            self.kernel = nn.Parameter(torch.empty(self.kernelSize), requires_grad=trainKern)
            nn.init.normal_(self.kernel, std=0.1)
        else:
            k = torch.as_tensor(kernelInit, dtype=torch.float32).flatten()
            self.kernelSize = k.numel()
            self.kernel = nn.Parameter(k.clone(), requires_grad=trainKern)

    def forward(self, inputs=None):
        return self.kernel


class LowPassWave(nn.Module):
    """Approximation branch: stride-2 convolution with the scaling filter h."""

    def forward(self, inputs):
        x, k = inputs
        return F.conv1d(x, k.view(1, 1, -1), padding=0, stride=2)


class HighPassWave(nn.Module):
    """Detail branch: stride-2 convolution with g[n] = (-1)^n * h[-n] (CQF)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("qmfFlip", torch.empty(0), persistent=False)

    def forward(self, inputs):
        x, k = inputs
        n = k.numel()
        if self.qmfFlip.numel() != n or self.qmfFlip.device != k.device:
            flip = torch.tensor([(-1.0) ** i for i in range(n)],
                                dtype=k.dtype, device=k.device)
            self.qmfFlip = flip.view(1, 1, -1)
        g = torch.flip(k, [0]).view(1, 1, -1) * self.qmfFlip
        return F.conv1d(x, g, padding=0, stride=2)


class HardThresholdAssym(nn.Module):
    """
    Learnable Asymmetric Hard Threshold (LAHT), eq. 16 of the paper:
        LAHT(x) = x * [ sigmoid(alpha * (x - thrP)) + sigmoid(-alpha * (x + thrN)) ]
    Thresholds are parameterised through softplus so they stay strictly positive.
    """

    def __init__(self, init=1.0, alpha=10.0, trainBias=True, trainAlpha=True):
        super().__init__()
        inv_sp = lambda v: float(torch.log(torch.expm1(torch.tensor(max(v, 1e-3)))))
        self.thrP_raw = nn.Parameter(torch.tensor([inv_sp(init)]), requires_grad=trainBias)
        self.thrN_raw = nn.Parameter(torch.tensor([inv_sp(init)]), requires_grad=trainBias)
        self.alpha = nn.Parameter(torch.tensor([float(alpha)]), requires_grad=trainAlpha)

    def forward(self, x):
        thrP = F.softplus(self.thrP_raw)
        thrN = F.softplus(self.thrN_raw)
        return x * (torch.sigmoid(self.alpha * (x - thrP))
                    + torch.sigmoid(-self.alpha * (x + thrN)))


class SpatialAttentionBlock(nn.Module):
    """1D spatial attention over (N, C, W) features against a global context g."""

    def __init__(self, in_features, normalize_attn=True):
        super().__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv1d(in_features, 1, kernel_size=1, bias=False)

    def forward(self, l, g):
        N, C, W = l.size()
        c = self.op(l + g)
        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W)
            g_out = torch.mul(a.expand_as(l), l).view(N, C, -1).sum(dim=2)
        else:
            a = torch.sigmoid(c)
            g_out = F.adaptive_avg_pool1d(torch.mul(a.expand_as(l), l), 1).view(N, C)
        return c.view(N, 1, W), g_out


class TemporalAttn(nn.Module):
    """Luong-style temporal attention over Bi-GRU hidden states (eqs. 39-42)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, hidden_states):
        score_first_part = self.fc1(hidden_states)              # (B, T, H)
        h_t = hidden_states[:, -1, :]                            # (B, H)
        score = torch.bmm(score_first_part, h_t.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(score, dim=1)              # (B, T)
        context = torch.bmm(hidden_states.permute(0, 2, 1),
                            attention_weights.unsqueeze(2)).squeeze(2)
        v = torch.tanh(self.fc2(torch.cat((context, h_t), dim=1)))
        return v, attention_weights


class FocalLoss(nn.Module):
    """
    Focal loss with class weighting (eq. 19).

    CHANGE vs upstream: the original computed `ce_loss` with reduction='mean'
    and then applied a *scalar* modulating factor to every sample, which makes
    the focal term a no-op up to a constant. Here CE is per-sample, so
    (1 - p_t)^gamma actually down-weights easy examples as intended.

    Expects `inputs` to be log-probabilities (the model ends in log_softmax).
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, log_probs, targets):
        n_classes = log_probs.size(-1)

        # true-class negative log-likelihood; also drives the focal factor
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.exp(-nll).clamp(0.0, 1.0)

        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            ce = (1.0 - eps) * nll - (eps / n_classes) * log_probs.sum(dim=1)
        else:
            ce = nll

        loss = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()