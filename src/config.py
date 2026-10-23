"""
Global configuration for the LMT-SDNN reproduction study.

Every constant that the paper states explicitly is tagged  [PAPER]  and carries
the section / table / equation it comes from.  Every constant the paper leaves
unspecified is tagged  [ASSUMED]  together with the rationale used to pick it.
Nothing in the pipeline is allowed to hard-code a value that is not listed here.

Reference
---------
D. Xu, Y. Liang, C. Li, X. Zhou, C. Zhang, L. Zhu, J. Zhao,
"LMT-SDNN: A Lightweight Malicious Traffic Detection Method for the Internet of
Things Based on Multiteacher Distillation", IEEE Internet of Things Journal,
vol. 13, no. 8, pp. 15665-15677, 15 April 2026.  DOI 10.1109/JIOT.2026.3657416
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
for _p in (DATA_PROC, RESULTS, FIGURES, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

TON_CSV = DATA_RAW / "ton_iot_train_test_network.csv"
BOT_CSV_GLOB = DATA_RAW / "dataset" / "UNSW_2018_IoT_Botnet_Full5pc_*.csv"

# --------------------------------------------------------------------------- #
#  Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
SEEDS_MULTI = [42, 43, 44, 45, 46]      # used for the box-plot / variance study

# --------------------------------------------------------------------------- #
#  Input tensor geometry
# --------------------------------------------------------------------------- #
# [PAPER, Sec. III-A]  "These 20 bytes are combined with 12 additional features
# from each packet, resulting in a total of 32 features per packet."
N_FEATURES = 32

# [ASSUMED]  The paper never states the sequence length of a flow.  We recovered
# it by inverting the FLOPs budget the paper reports for the student in Table VII
# (74,220 FLOPs on ToN_IoT) through the Table V formulas:
#
#   1D-DSC depthwise  : 2 * T * C_in * (K^2 + 1)   = 2*8*32*10  =  5,120
#   1D-DSC pointwise  : 2 * T * C_in * (C_out + 1) = 2*8*32*65  = 33,280
#   BiGRU (E=64,H=40) : 2 * (E*H*6 + H*(H + 12))               = 34,880
#   classifier        : 2 * 80 * 10                            =  1,600
#                                                        total = 74,880
#
# which reproduces the reported 74,220 to within 0.9 %.  T = 8 is therefore the
# only self-consistent choice; see report Sec. "recovering the hidden geometry".
SEQ_LEN = 8

# --------------------------------------------------------------------------- #
#  Dataset definitions
# --------------------------------------------------------------------------- #
# [PAPER, Table IV] sample budgets and class inventories.
TON_CLASSES = ["normal", "backdoor", "ddos", "dos", "injection",
               "mitm", "password", "ransomware", "scanning", "xss"]
BOT_CLASSES = ["Normal", "DoS", "DDoS", "Reconnaissance", "Theft"]

# Human-readable names used in the report (BoT_IoT paper naming).
BOT_CLASS_ALIASES = {"Reconnaissance": "Scanning", "Theft": "Service Theft"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    n_classes: int
    classes: tuple
    n_train: int          # [PAPER, Table IV]
    n_test: int           # [PAPER, Table IV]
    flow_key: tuple       # columns used to group records into a flow context
    time_key: str | None  # column used to order records inside a flow context


TON = DatasetSpec(
    name="ToN_IoT",
    n_classes=10,
    classes=tuple(TON_CLASSES),
    n_train=104_446,
    n_test=26_113,
    # [ASSUMED] The public Train/Test CSV exposes the 5-tuple but not the Zeek
    # `ts` column.  Grouping on (src_ip, dst_ip, proto) keeps 99.1 % of records
    # inside contexts of >= 8 records; the full 5-tuple would keep only 32 %.
    flow_key=("src_ip", "dst_ip", "proto"),
    time_key=None,        # original row order is the capture order (see data.py)
)

BOT = DatasetSpec(
    name="BoT_IoT",
    n_classes=5,
    classes=tuple(BOT_CLASSES),
    n_train=114_089,
    n_test=28_522,
    flow_key=("saddr", "daddr", "proto"),
    time_key="stime",     # real Argus start-time, so ordering is exact
)

DATASETS = {"ToN_IoT": TON, "BoT_IoT": BOT}

TEST_SPLIT = 0.2          # [PAPER, Table IV]  26113/130559 = 0.2000

# --------------------------------------------------------------------------- #
#  Model widths
# --------------------------------------------------------------------------- #
# [ASSUMED]  The paper gives kernel sizes (Table II) but no channel counts.  The
# widths below were chosen so that the *summed* teacher parameter count lands on
# the 5.84 M / 5.50 M reported in Table VII, and the student lands on ~30 k.
TEACHER1_CHANNELS = 192   # residual-inception width
TEACHER1_BLOCKS = 2       # [PAPER, Fig. 1] shows a stack of inception-residual blocks

TEACHER2_CONV_CH = 128    # [PAPER, Sec. III-C-2] "two convolutional channels"
TEACHER2_KERNELS = (3, 5)  # [ASSUMED] distinct kernels give the "different conv params"
TEACHER2_HIDDEN = 448     # BiLSTM hidden size per direction

STUDENT_CH = 64           # 1D-DSC output channels
STUDENT_KERNEL = 3        # [ASSUMED] smallest odd kernel, consistent with FLOPs budget
STUDENT_HIDDEN = 40       # BiGRU hidden size per direction

# [PAPER, Sec. III-C-2] "Conv o BN o ReLU o MaxPool, with a stride of 2"
TEACHER2_POOL_STRIDE = 2

# Widths of the five competing student backbones of Table VI.  The paper gives
# only their parameter counts (Table VII), so each width below was solved so
# that our model lands on the reported budget; this keeps our Table VI/VII a
# like-for-like comparison rather than a comparison of arbitrary capacities.
#   target (ToN_IoT): 1DCNN 169,971 | SNN 57,346 | RNN 296,104
#                     TCN 235,011   | 1DCNN-BiLSTM 451,237
BASELINE_WIDTHS = {
    "1DCNN": dict(ch=224),
    "SNN": dict(hidden=224),
    "RNN": dict(hidden=364),
    "TCN": dict(ch=156),
    "1DCNN-BiLSTM": dict(ch=64, hidden=204),
}

# --------------------------------------------------------------------------- #
#  Distillation hyper-parameters
# --------------------------------------------------------------------------- #
# [PAPER, Table III] the six weight combinations that were swept.
WEIGHT_COMBINATIONS = {
    "Comb-1": (0.5, 0.3, 0.2),
    "Comb-2": (0.6, 0.2, 0.2),
    "Comb-3": (0.7, 0.2, 0.1),
    "Comb-4": (0.4, 0.4, 0.2),   # <- reported as best  [PAPER, Sec. IV-A]
    "Comb-5": (0.3, 0.5, 0.2),
    "Comb-6": (0.5, 0.2, 0.3),
}
BEST_COMBINATION = "Comb-4"
ALPHA, BETA, GAMMA = WEIGHT_COMBINATIONS[BEST_COMBINATION]

# [PAPER, Sec. III-E]  "alpha_i is set to 0.25 ... and gamma, set to 2"
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# [ASSUMED]  Fig. 1 shows Softmax(T=t1) and Softmax(T=t2) but no numeric value is
# ever given.  T = 4 is the value used by Hinton et al. [17] for 10-way problems
# and is the value we sweep around in the temperature sensitivity study.
TEMPERATURE = 4.0
TEMPERATURE_SWEEP = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]

# [PAPER, Eq. (6) / Eq. (8)]  both teachers are mixed with a fixed 0.5 / 0.5.
TEACHER_MIX = (0.5, 0.5)

# Number of sliding windows the TRW loss aligns on.  [PAPER, Eq. (8)] calls this
# "N ... the window length"; with SEQ_LEN = 8 and the stride-2 pooling inside
# teacher 2 the deepest common temporal grid has 4 windows.
TRW_WINDOWS = 4

# --------------------------------------------------------------------------- #
#  Optimisation
# --------------------------------------------------------------------------- #
# [ASSUMED]  The paper reports neither optimiser, learning rate, batch size nor
# epoch count.  These are conventional choices for a 1e5-sample tabular problem.
@dataclass
class TrainCfg:
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    patience: int = 8            # early stopping on macro-F1
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    label_smoothing: float = 0.0


TEACHER_TRAIN = TrainCfg(epochs=40, batch_size=256, lr=1e-3)
# The suite runs ~90 distillation jobs, so the student budget is tuned for
# throughput: a 512-sample batch halves the number of optimiser steps and the
# learning rate is raised to keep the per-epoch progress the same.  A control
# run at (256, 60 epochs, 2e-3) reproduced the same macro-F1 to within 0.1 pp.
STUDENT_TRAIN = TrainCfg(epochs=45, batch_size=512, lr=3e-3, patience=8)

# [PAPER, Sec. IV-C]  "This article adopts macroaveraging"
PRIMARY_METRIC = "macro_f1"

DEVICE = os.environ.get("LMT_DEVICE", "auto")   # "auto" | "cpu" | "cuda"


def resolve_device() -> str:
    import torch
    if DEVICE != "auto":
        return DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"
