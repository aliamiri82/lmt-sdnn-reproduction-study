"""
Every network in the LMT-SDNN paper, plus the five student baselines of
Table VI and the five non-distilled baselines of Fig. 3.

Common contract
---------------
`forward(x) -> (logits, feat_seq)` where

    x        : (B, 32, 8)  channels = the 32 per-record features,
                           length   = the 8-step flow context
    logits   : (B, C)
    feat_seq : (B, T', D)  the intermediate temporal representation the paper
                           feeds to the TRW loss -- "the residual inception
                           teacher model outputs the feature maps from each
                           Inception-Residual block, while the 1D-CNN+BiLSTM
                           teacher model outputs the hidden state sequences
                           from the BiLSTM layers" (Sec. III-C-2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (BASELINE_WIDTHS, N_FEATURES, SEQ_LEN, STUDENT_CH, STUDENT_HIDDEN,
                    STUDENT_KERNEL, TEACHER1_BLOCKS, TEACHER1_CHANNELS,
                    TEACHER2_CONV_CH, TEACHER2_HIDDEN, TEACHER2_KERNELS,
                    TEACHER2_POOL_STRIDE)


def _same_pad(k: int) -> int:
    return (k - 1) // 2


class _Head(nn.Module):
    """Temporal average pooling + linear classifier, shared by every model."""

    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, seq):                      # seq: (B, T, D)
        return self.fc(seq.mean(dim=1))


# --------------------------------------------------------------------------- #
#  Teacher 1 -- Residual Inception               (Sec. III-C-1, Table II, Eq. 1)
# --------------------------------------------------------------------------- #
class InceptionResidualBranch(nn.Module):
    """
    One of the three parallel paths of Table II.

    Implements Eq. (1) literally: layer 1 sees x0, layer 2 sees (x0, x_w1) and
    layer 3 sees (x0, x_w1, x_w2).  That dense wiring is what produces the
    "six skip connections" the paper counts for the block (two per branch).
    """

    def __init__(self, ch: int, kernels: tuple[int, int, int]):
        super().__init__()
        k1, k2, k3 = kernels
        self.c1 = nn.Conv1d(ch, ch, k1, padding=_same_pad(k1))
        self.c2 = nn.Conv1d(2 * ch, ch, k2, padding=_same_pad(k2))
        self.c3 = nn.Conv1d(3 * ch, ch, k3, padding=_same_pad(k3))
        self.b1, self.b2, self.b3 = (nn.BatchNorm1d(ch) for _ in range(3))

    def forward(self, x0):
        x1 = F.relu(self.b1(self.c1(x0)))
        x2 = F.relu(self.b2(self.c2(torch.cat([x0, x1], 1))))
        x3 = F.relu(self.b3(self.c3(torch.cat([x0, x1, x2], 1))))
        return x3


class InceptionResidualBlock(nn.Module):
    # [PAPER, Table II] branch kernels; 2-D notation in the table, 1-D in Eq. (1)
    KERNELS = ((1, 3, 1), (1, 5, 1), (3, 3, 3))

    def __init__(self, ch: int):
        super().__init__()
        self.branches = nn.ModuleList(
            InceptionResidualBranch(ch, k) for k in self.KERNELS)

    def forward(self, x):
        # Eq. (1) last line: the three multiscale paths are fused by averaging
        fused = sum(b(x) for b in self.branches) / len(self.branches)
        return fused + x                     # residual connection of the block


class ResidualInceptionTeacher(nn.Module):
    def __init__(self, n_classes: int, in_ch: int = N_FEATURES,
                 ch: int = TEACHER1_CHANNELS, n_blocks: int = TEACHER1_BLOCKS):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU())
        self.blocks = nn.ModuleList(InceptionResidualBlock(ch) for _ in range(n_blocks))
        self.head = _Head(ch, n_classes)
        self.feat_dim = ch

    def forward(self, x):
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        seq = h.transpose(1, 2)                       # (B, T, C)
        return self.head(seq), seq


# --------------------------------------------------------------------------- #
#  Teacher 2 -- 1D-CNN + BiLSTM                            (Sec. III-C-2, Eq. 2)
# --------------------------------------------------------------------------- #
class CNNBiLSTMTeacher(nn.Module):
    """
    "Our model employs two convolutional channels, each with the structure
    Conv o BN o ReLU o MaxPool, with a stride of 2.  While both channels share
    the same structure, the convolutional parameters in each channel differ."
    """

    def __init__(self, n_classes: int, in_ch: int = N_FEATURES,
                 conv_ch: int = TEACHER2_CONV_CH,
                 kernels: tuple[int, ...] = TEACHER2_KERNELS,
                 hidden: int = TEACHER2_HIDDEN):
        super().__init__()
        self.channels = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(in_ch, conv_ch, k, padding=_same_pad(k)),
                nn.BatchNorm1d(conv_ch), nn.ReLU(),
                nn.MaxPool1d(kernel_size=TEACHER2_POOL_STRIDE,
                             stride=TEACHER2_POOL_STRIDE),
            ) for k in kernels)
        self.lstm = nn.LSTM(conv_ch * len(kernels), hidden,
                            batch_first=True, bidirectional=True)
        self.feat_dim = hidden * 2
        self.head = _Head(self.feat_dim, n_classes)

    def forward(self, x):
        h = torch.cat([c(x) for c in self.channels], dim=1)   # (B, 2*ch, T/2)
        seq, _ = self.lstm(h.transpose(1, 2))                 # (B, T/2, 2H)
        return self.head(seq), seq


# --------------------------------------------------------------------------- #
#  Student -- 1D-DSC + BiGRU                                      (Sec. III-D)
# --------------------------------------------------------------------------- #
class DSCBlock(nn.Module):
    """
    Depthwise-separable 1-D convolution: "separates the standard convolution
    operation into depthwise convolution and pointwise convolution ... while
    incorporating residual connections to maintain gradient stability".
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = STUDENT_KERNEL):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, k, padding=_same_pad(k),
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.residual = (in_ch == out_ch)

    def forward(self, x):
        h = self.bn(self.pointwise(self.depthwise(x)))
        if self.residual:
            h = h + x
        return F.relu(h)


class LMTSDNNStudent(nn.Module):
    def __init__(self, n_classes: int, in_ch: int = N_FEATURES,
                 ch: int = STUDENT_CH, hidden: int = STUDENT_HIDDEN,
                 n_dsc: int = 2):
        super().__init__()
        blocks = [DSCBlock(in_ch, ch)]
        blocks += [DSCBlock(ch, ch) for _ in range(n_dsc - 1)]
        self.dsc = nn.Sequential(*blocks)
        self.gru = nn.GRU(ch, hidden, batch_first=True, bidirectional=True)
        self.feat_dim = hidden * 2
        self.head = _Head(self.feat_dim, n_classes)

    def forward(self, x):
        h = self.dsc(x)
        seq, _ = self.gru(h.transpose(1, 2))                  # (B, T, 2H)
        return self.head(seq), seq


# --------------------------------------------------------------------------- #
#  Student baselines of Table VI  (KD-1DCNN / KD-SNN / KD-RNN / KD-TCN /
#  KD-1DCNN-BiLSTM) -- also reused un-distilled for Fig. 3 and Fig. 7.
# --------------------------------------------------------------------------- #
class CNN1D(nn.Module):
    def __init__(self, n_classes, in_ch=N_FEATURES, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU())
        self.feat_dim = ch
        self.head = _Head(ch, n_classes)

    def forward(self, x):
        seq = self.net(x).transpose(1, 2)
        return self.head(seq), seq


class SNN(nn.Module):
    """
    The paper names "SNN" as a baseline but never defines it.  In the malicious
    traffic literature the acronym is used for a *shallow* neural network, so we
    implement a two-layer perceptron applied per time step -- the smallest model
    that still exposes a temporal feature sequence for the TRW loss.
    """

    def __init__(self, n_classes, in_ch=N_FEATURES, hidden=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_ch, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.feat_dim = hidden
        self.head = _Head(hidden, n_classes)

    def forward(self, x):
        seq = self.net(x.transpose(1, 2))
        return self.head(seq), seq


class RNNBaseline(nn.Module):
    def __init__(self, n_classes, in_ch=N_FEATURES, hidden=96):
        super().__init__()
        self.rnn = nn.RNN(in_ch, hidden, batch_first=True, bidirectional=True)
        self.feat_dim = hidden * 2
        self.head = _Head(self.feat_dim, n_classes)

    def forward(self, x):
        seq, _ = self.rnn(x.transpose(1, 2))
        return self.head(seq), seq


class TCNBlock(nn.Module):
    def __init__(self, ch_in, ch_out, k, dilation):
        super().__init__()
        pad = (k - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(ch_in, ch_out, k, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(ch_out, ch_out, k, padding=pad, dilation=dilation)
        self.down = nn.Conv1d(ch_in, ch_out, 1) if ch_in != ch_out else None
        self.bn = nn.BatchNorm1d(ch_out)

    def forward(self, x):
        h = self.conv1(x)[..., :x.size(-1)]                # causal crop
        h = F.relu(h)
        h = self.conv2(h)[..., :x.size(-1)]
        h = self.bn(h)
        res = x if self.down is None else self.down(x)
        return F.relu(h + res)


class TCN(nn.Module):
    def __init__(self, n_classes, in_ch=N_FEATURES, ch=64, levels=2, k=3):
        super().__init__()
        layers, c = [], in_ch
        for i in range(levels):
            layers.append(TCNBlock(c, ch, k, dilation=2 ** i))
            c = ch
        self.net = nn.Sequential(*layers)
        self.feat_dim = ch
        self.head = _Head(ch, n_classes)

    def forward(self, x):
        seq = self.net(x).transpose(1, 2)
        return self.head(seq), seq


class CNNBiLSTMSmall(nn.Module):
    """The KD-1DCNN-BiLSTM student: teacher 2's topology at student scale."""

    def __init__(self, n_classes, in_ch=N_FEATURES, ch=48, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU())
        self.lstm = nn.LSTM(ch, hidden, batch_first=True, bidirectional=True)
        self.feat_dim = hidden * 2
        self.head = _Head(self.feat_dim, n_classes)

    def forward(self, x):
        seq, _ = self.lstm(self.conv(x).transpose(1, 2))
        return self.head(seq), seq


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
TEACHERS = {
    "Teacher1-ResidualInception": ResidualInceptionTeacher,
    "Teacher2-1DCNN-BiLSTM": CNNBiLSTMTeacher,
}

STUDENTS = {
    "LMT-SDNN": LMTSDNNStudent,        # 1D-DSC + BiGRU   (ours / the paper's)
    "1DCNN": CNN1D,
    "SNN": SNN,
    "RNN": RNNBaseline,
    "TCN": TCN,
    "1DCNN-BiLSTM": CNNBiLSTMSmall,
}


def build(name: str, n_classes: int, in_ch: int = N_FEATURES, **kw) -> nn.Module:
    if name in TEACHERS:
        return TEACHERS[name](n_classes, in_ch=in_ch, **kw)
    if name in STUDENTS:
        # Baseline widths default to the values that reproduce the parameter
        # budgets the paper reports for them in Table VII.
        defaults = dict(BASELINE_WIDTHS.get(name, {}))
        defaults.update(kw)
        return STUDENTS[name](n_classes, in_ch=in_ch, **defaults)
    raise KeyError(name)


class FeatureProjector(nn.Module):
    """
    Training-time only.  The TRW loss compares student and teacher feature
    increments, but the two live in different spaces; a linear map on the
    *student* side (teachers stay frozen, so the objective cannot be collapsed
    by shrinking the teacher representation) puts them on a common footing.
    Discarded before deployment, hence excluded from the reported student size.
    """

    def __init__(self, student_dim: int, teacher_dims: list[int]):
        super().__init__()
        self.maps = nn.ModuleList(nn.Linear(student_dim, d) for d in teacher_dims)

    def forward(self, seq, i):
        return self.maps[i](seq)


if __name__ == "__main__":
    from complexity import summarise
    x = torch.randn(4, N_FEATURES, SEQ_LEN)
    for nc, ds in ((10, "ToN_IoT"), (5, "BoT_IoT")):
        print(f"\n===== {ds} ({nc} classes) =====")
        for name in list(TEACHERS) + list(STUDENTS):
            m = build(name, nc).eval()
            with torch.no_grad():
                logits, seq = m(x)
            s = summarise(m, nc)
            print(f"  {name:28s} logits {tuple(logits.shape)}  feat {tuple(seq.shape)}"
                  f"  params {s['params']:>9,}  {s['size_kb']:>9.2f} KB")
