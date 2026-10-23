"""
Data acquisition -> 32-feature records -> flow-context sequences -> tensors.

The paper builds one sample per *flow*, where the flow is a chronologically
ordered list of packets and each packet is described by 32 numbers (the first
20 header bytes plus 12 derived statistics, Sec. III-A).  The 48.1 GB / 69.3 GB
raw PCAPs that would be needed to rebuild that exact tensor are not part of the
public release, so this module rebuilds the same *tensor geometry* from the
public flow-level CSVs:

    paper                       this reproduction
    -----------------------     ----------------------------------------------
    packet                 ->   one CSV flow record, described by 32 features
    flow = [pkt_1..pkt_i]  ->   flow context = the record plus its 7 chrono-
                                logical predecessors in the same
                                (src_ip, dst_ip, proto) conversation
    label of the flow      ->   label of the *target* (most recent) record

The result is a (N, 8, 32) tensor with exactly one sample per record, so the
sample budgets of Table IV can be matched exactly.  Everything the paper leaves
open is recorded in `PREPROCESSING_NOTES` and reproduced verbatim in the report.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict

import numpy as np
import pandas as pd

from config import (BOT, BOT_CSV_GLOB, DATA_PROC, N_FEATURES, SEED, SEQ_LEN,
                    TEST_SPLIT, TON, TON_CSV, DatasetSpec)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# --------------------------------------------------------------------------- #
#  Feature inventories -- exactly 32 columns each, mirroring the paper's R^32
# --------------------------------------------------------------------------- #

# ToN_IoT.  Raw IPs are deliberately *excluded* from the feature vector (they are
# only used to group records) because in ToN_IoT the attacker host is a constant
# and would make the task trivially separable; see the leakage study in the
# report.  Ports are kept, matching common practice on this dataset.
TON_NUMERIC = [
    "src_port", "dst_port", "duration", "src_bytes", "dst_bytes",
    "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    "dns_qclass", "dns_qtype", "dns_rcode",
    "http_request_body_len", "http_response_body_len", "http_status_code",
]
TON_CATEGORICAL = [
    "proto", "service", "conn_state",
    "ssl_version", "ssl_cipher", "http_method", "http_version",
    "http_user_agent", "weird_name", "http_trans_depth",
]
TON_BINARY = [
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_resumed", "ssl_established",
]
TON_FEATURES = TON_NUMERIC + TON_CATEGORICAL + TON_BINARY   # 16 + 10 + 6 = 32

# BoT_IoT.  `pkSeqID`, `stime`, `ltime` and `seq` are dropped: they are monotone
# capture counters, and because the BoT_IoT attack scenarios were recorded in
# disjoint time slots any of them alone separates the classes almost perfectly.
# `iat` (inter-arrival time inside the conversation) is the paper's
# "timestamp differences" feature and is derived, not copied.
BOT_FEATURES = [
    "flgs_number", "proto_number", "sport", "dport", "pkts", "bytes",
    "state_number", "dur", "mean", "stddev", "sum", "min", "max",
    "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate", "drate",
    "TnBPSrcIP", "TnBPDstIP", "TnP_PSrcIP", "TnP_PDstIP",
    "TnP_PerProto", "TnP_Per_Dport",
    "AR_P_Proto_P_SrcIP", "AR_P_Proto_P_DstIP",
    "N_IN_Conn_P_DstIP", "N_IN_Conn_P_SrcIP",
    "Pkts_P_State_P_Protocol_P_DestIP",
    "iat",
]                                                            # 32

PREPROCESSING_NOTES: dict[str, object] = {}


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _to_number(s: pd.Series) -> pd.Series:
    """Parse Argus/Zeek fields that mix decimal, hex ('0x1f90') and '-' tokens."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")
    t = s.astype(str).str.strip()
    out = pd.to_numeric(t, errors="coerce")
    hex_mask = out.isna() & t.str.lower().str.startswith("0x")
    if hex_mask.any():
        out.loc[hex_mask] = t[hex_mask].apply(
            lambda v: float(int(v, 16)) if v not in ("", "-") else np.nan)
    return out.astype("float64").fillna(-1.0)


def _ordinal(s: pd.Series) -> pd.Series:
    """Frequency-ranked ordinal code: rank 0 = most frequent value."""
    t = s.astype(str).fillna("-")
    order = t.value_counts().index
    mapping = {v: i for i, v in enumerate(order)}
    return t.map(mapping).astype("float64")


def _binary(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip().str.lower()
    return t.isin(["t", "true", "1", "yes"]).astype("float64")


def _signed_log(x: np.ndarray) -> np.ndarray:
    """Tame the heavy tails of byte/packet counters without losing the sign."""
    return np.sign(x) * np.log1p(np.abs(x))


def _contiguous_block_subsample(pos: np.ndarray, quota: int, block: int,
                                rng: np.random.Generator) -> np.ndarray:
    """
    Pick `quota` items out of the ordered array `pos` as a union of randomly
    placed *contiguous* blocks of length `block`.

    Random per-record sampling would shred the burst structure that the temporal
    branch of the model is supposed to learn; sampling contiguous runs keeps
    inter-arrival dynamics intact while still hitting an exact budget.
    """
    n = len(pos)
    if quota >= n:
        return pos
    n_starts = max(1, n - block)
    # Enough candidate starts to cover the quota even if blocks overlap.
    n_draw = min(n_starts, int(np.ceil(quota / block)) * 4 + 8)
    starts = rng.choice(n_starts, size=n_draw, replace=False)
    keep = np.zeros(n, dtype=bool)
    for st in starts:
        keep[st:st + block] = True
        if keep.sum() >= quota:
            break
    idx = np.flatnonzero(keep)
    if len(idx) > quota:                       # trim the tail of the last block
        idx = idx[:quota]
    elif len(idx) < quota:                     # top up with unused records
        spare = np.flatnonzero(~keep)
        rng.shuffle(spare)
        idx = np.sort(np.concatenate([idx, spare[: quota - len(idx)]]))
    return pos[idx]


# --------------------------------------------------------------------------- #
#  Raw loaders
# --------------------------------------------------------------------------- #
def load_ton_raw() -> pd.DataFrame:
    df = pd.read_csv(TON_CSV, low_memory=False)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    # The public CSV has no `ts`; within each class block the row order is the
    # original Zeek conn.log order (verifiable from the runs of retried REJ
    # connections), so the row index is a valid chronological key.
    df["_order"] = np.arange(len(df), dtype=np.int64)
    return df


def load_bot_raw() -> pd.DataFrame:
    files = sorted(BOT_CSV_GLOB.parent.glob(BOT_CSV_GLOB.name))
    if not files:
        raise FileNotFoundError(f"No BoT_IoT CSVs under {BOT_CSV_GLOB}")
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["category"] = df["category"].astype(str).str.strip()
    df["stime"] = _to_number(df["stime"])
    df["_order"] = df["stime"]
    return df


# --------------------------------------------------------------------------- #
#  Feature construction
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame, spec: DatasetSpec) -> np.ndarray:
    if spec.name == "ToN_IoT":
        cols = {}
        for c in TON_NUMERIC:
            cols[c] = _to_number(df[c])
        for c in TON_CATEGORICAL:
            cols[c] = _ordinal(df[c])
        for c in TON_BINARY:
            cols[c] = _binary(df[c])
        feats = pd.DataFrame(cols)[TON_FEATURES]
    else:
        work = df.copy()
        # inter-arrival time inside the conversation == "timestamp differences"
        work["iat"] = (work.groupby(list(spec.flow_key), sort=False)["stime"]
                           .diff().fillna(0.0).clip(lower=0.0))
        cols = {c: _to_number(work[c]) for c in BOT_FEATURES}
        feats = pd.DataFrame(cols)[BOT_FEATURES]

    X = feats.to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    assert X.shape[1] == N_FEATURES, f"{spec.name}: got {X.shape[1]} features, need {N_FEATURES}"
    return X.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Sample budgeting  (Table IV)
# --------------------------------------------------------------------------- #
def plan_quotas_proportional(counts: dict[str, int], total: int) -> dict[str, int]:
    """
    The competing reading of Sec. IV-B: keep the raw class ratios and simply
    scale them down to the Table IV budget.  Used only to measure how much the
    unstated subsampling rule moves the macro-averaged headline numbers.
    """
    if sum(counts.values()) <= total:
        return dict(counts)
    pool = sum(counts.values())
    quotas, acc, names = {}, 0, list(counts)
    for i, c in enumerate(names):
        if i == len(names) - 1:
            quotas[c] = total - acc
        else:
            quotas[c] = max(1, int(round(total * counts[c] / pool)))
            acc += quotas[c]
    assert sum(quotas.values()) == total
    return quotas


def plan_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """
    Reproduce Sec. IV-B -- "Balanced subsets of normal and attack samples were
    selected to meet the training requirements of the lightweight neural
    networks" -- while landing exactly on the Table IV budget.

    Rule (max-min fair / water-filling): every class that cannot supply an even
    share is kept whole, and the freed budget is redistributed evenly over the
    classes that still have samples to give.  This is the allocation that makes
    the retained subset as balanced as the raw data allows, and it is what the
    sentence above describes.  It still leaves BoT_IoT "highly imbalanced"
    (Sec. IV-D) because Normal and Theft only hold 477 and 79 records in total.
    """
    if sum(counts.values()) <= total:
        return dict(counts)

    quotas: dict[str, int] = {}
    remaining = sorted(counts, key=counts.get)        # small classes first
    budget = total
    while remaining:
        even = budget / len(remaining)
        smallest = remaining[0]
        if counts[smallest] <= even:                  # cannot fill its share
            quotas[smallest] = counts[smallest]
            budget -= counts[smallest]
            remaining.pop(0)
        else:                                         # split the rest evenly
            share, extra = divmod(budget, len(remaining))
            for i, c in enumerate(remaining):
                quotas[c] = share + (1 if i < extra else 0)
            break
    assert sum(quotas.values()) == total, (sum(quotas.values()), total)
    return quotas


# --------------------------------------------------------------------------- #
#  Flow-context windows
# --------------------------------------------------------------------------- #
def build_windows(order_key: np.ndarray, group_id: np.ndarray,
                  seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    For every record return the row indices of its `seq_len`-long causal context
    (itself last).  Contexts shorter than `seq_len` are left-padded by repeating
    the earliest record of the conversation, so no future information leaks in.
    """
    n = len(group_id)
    sort_idx = np.lexsort((order_key, group_id))          # group, then time
    g_sorted = group_id[sort_idx]

    # position of each record inside its own group
    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    group_start = np.repeat(starts, np.diff(np.r_[starts, n]))
    pos_in_group = np.arange(n) - group_start

    offsets = np.arange(seq_len - 1, -1, -1)              # [L-1 ... 0]
    back = np.minimum(offsets[None, :], pos_in_group[:, None])
    ctx_sorted = np.arange(n)[:, None] - back             # positions in sorted order
    windows_sorted = sort_idx[ctx_sorted]                 # -> original row ids

    windows = np.empty_like(windows_sorted)
    windows[sort_idx] = windows_sorted
    return windows.astype(np.int64)


# --------------------------------------------------------------------------- #
#  End-to-end preparation
# --------------------------------------------------------------------------- #
def prepare(spec: DatasetSpec, seed: int = SEED, split_mode: str = "record",
            include_ip: bool = False, quota_rule: str = "balanced",
            force: bool = False) -> dict:
    """
    Returns a dict with
        records   (M, 32)  float32   scaled feature table (context pool)
        windows   (N, 8)   int64     row indices into `records`
        y         (N,)     int64     class index of the target record
        train_idx, test_idx          index arrays into `windows` / `y`
        classes, scaler_stats, notes
    """
    tag = (f"{spec.name}_seed{seed}_{split_mode}"
           f"{'_ip' if include_ip else ''}"
           f"{'' if quota_rule == 'balanced' else '_' + quota_rule}")
    cache = DATA_PROC / f"{tag}.npz"
    meta_f = DATA_PROC / f"{tag}.json"
    if cache.exists() and meta_f.exists() and not force:
        z = np.load(cache)
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        return {**{k: z[k] for k in z.files}, **meta}

    rng = np.random.default_rng(seed)
    notes: dict[str, object] = {"dataset": spec.name, "seed": seed,
                                "split_mode": split_mode, "include_ip": include_ip,
                                "quota_rule": quota_rule}

    df = load_ton_raw() if spec.name == "ToN_IoT" else load_bot_raw()
    label_col = "type" if spec.name == "ToN_IoT" else "category"
    notes["raw_rows"] = int(len(df))

    # ---- 1. sample budget ------------------------------------------------- #
    total = spec.n_train + spec.n_test
    counts = df[label_col].value_counts().to_dict()
    counts = {c: counts.get(c, 0) for c in spec.classes}
    quotas = (plan_quotas(counts, total) if quota_rule == "balanced"
              else plan_quotas_proportional(counts, total))
    notes["raw_class_counts"] = counts
    notes["quotas"] = quotas

    # order records by conversation then time, so blocks are contiguous in time
    df = df.sort_values(list(spec.flow_key) + ["_order"], kind="mergesort")
    df = df.reset_index(drop=True)

    keep = []
    block = 200 if spec.name == "ToN_IoT" else 500
    for cls, q in quotas.items():
        pos = np.flatnonzero((df[label_col] == cls).to_numpy())
        if q >= len(pos):
            keep.append(pos)
        else:
            keep.append(_contiguous_block_subsample(pos, q, block, rng))
    keep = np.sort(np.concatenate(keep))
    df = df.iloc[keep].reset_index(drop=True)
    assert len(df) == total, (len(df), total)

    # ---- 2. features ------------------------------------------------------ #
    X = build_features(df, spec)
    feat_names = list(TON_FEATURES if spec.name == "ToN_IoT" else BOT_FEATURES)
    if include_ip:                       # leakage probe: append raw host identity
        ipc = ["src_ip", "dst_ip"] if spec.name == "ToN_IoT" else ["saddr", "daddr"]
        extra = np.stack([_ordinal(df[c]).to_numpy(np.float32) for c in ipc], 1)
        X = np.concatenate([X, extra], axis=1)
        feat_names += ipc

    y = df[label_col].map({c: i for i, c in enumerate(spec.classes)}).to_numpy(np.int64)
    assert not np.isnan(y).any()

    # ---- 3. flow-context windows ------------------------------------------ #
    gkey = (df[list(spec.flow_key)].astype(str).agg("|".join, axis=1)).to_numpy()
    _, group_id = np.unique(gkey, return_inverse=True)
    order_key = df["_order"].to_numpy(np.float64)
    windows = build_windows(order_key, group_id)
    notes["n_flow_groups"] = int(group_id.max() + 1)

    # ---- 4. split --------------------------------------------------------- #
    n = len(y)
    if split_mode == "record":
        # [PAPER] plain stratified random split, 80 / 20 (Table IV)
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            np.arange(n), test_size=spec.n_test, random_state=seed, stratify=y)
    elif split_mode == "block":
        # Robustness probe.  Each conversation is cut into contiguous temporal
        # blocks; whole blocks go to train or test, and every test record whose
        # 8-record context reaches back into a training block is discarded.  No
        # test sample then shares a single record with any training sample.
        train_idx, test_idx = _blocked_split(group_id, order_key, windows, rng)
    else:
        raise ValueError(split_mode)
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    # ---- 5. scaling (fit on training targets only) ------------------------- #
    Xl = _signed_log(X)
    mu = Xl[train_idx].mean(0)
    sd = Xl[train_idx].std(0)
    sd[sd < 1e-6] = 1.0
    Xs = ((Xl - mu) / sd).astype(np.float32)

    notes["n_train"] = int(len(train_idx))
    notes["n_test"] = int(len(test_idx))
    notes["features"] = feat_names
    notes["train_class_counts"] = {spec.classes[i]: int((y[train_idx] == i).sum())
                                   for i in range(spec.n_classes)}
    notes["test_class_counts"] = {spec.classes[i]: int((y[test_idx] == i).sum())
                                  for i in range(spec.n_classes)}
    notes["classes"] = list(spec.classes)
    notes["fingerprint"] = hashlib.md5(Xs.tobytes()).hexdigest()[:12]

    np.savez_compressed(cache, records=Xs, windows=windows, y=y,
                        train_idx=train_idx, test_idx=test_idx,
                        group_id=group_id.astype(np.int64))
    meta_f.write_text(json.dumps(notes, indent=2), encoding="utf-8")
    PREPROCESSING_NOTES[tag] = notes
    return {"records": Xs, "windows": windows, "y": y,
            "train_idx": train_idx, "test_idx": test_idx,
            "group_id": group_id.astype(np.int64), **notes}


def _blocked_split(group_id, order_key, windows, rng, block: int = 512,
                   test_frac: float = TEST_SPLIT):
    """
    Contiguous-block split with a purge zone.

    Records are walked conversation by conversation in time order and cut into
    runs of `block` records.  Each run is assigned to train or test as a whole.
    Afterwards every test sample whose context window touches a training record
    is dropped, which removes the last channel through which information could
    cross the split.  The price is a slightly smaller test set; the benefit is
    that the score is an honest estimate of generalisation to unseen traffic.
    """
    n = len(group_id)
    sort_idx = np.lexsort((order_key, group_id))
    g_sorted = group_id[sort_idx]

    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    group_start = np.repeat(starts, np.diff(np.r_[starts, n]))
    pos_in_group = np.arange(n) - group_start
    block_id = g_sorted.astype(np.int64) * 10_000_000 + (pos_in_group // block)

    uniq = np.unique(block_id)
    is_test_block = rng.random(len(uniq)) < test_frac
    test_of_block = dict(zip(uniq.tolist(), is_test_block.tolist()))

    in_test_sorted = np.array([test_of_block[b] for b in block_id], dtype=bool)
    in_test = np.empty(n, dtype=bool)
    in_test[sort_idx] = in_test_sorted

    # purge test samples whose context reaches into a training block
    ctx_all_test = in_test[windows].all(axis=1)
    test_mask = in_test & ctx_all_test
    train_mask = ~in_test
    return np.flatnonzero(train_mask), np.flatnonzero(test_mask)


# --------------------------------------------------------------------------- #
#  Torch plumbing
# --------------------------------------------------------------------------- #
def make_tensors(bundle: dict, idx: np.ndarray):
    """Materialise (B, 32, 8) float32 inputs for the given sample indices."""
    import torch
    rec = bundle["records"]
    win = bundle["windows"][idx]                 # (B, L)
    x = rec[win]                                 # (B, L, F)
    x = np.transpose(x, (0, 2, 1))               # (B, F, L)  channels = features
    return torch.from_numpy(np.ascontiguousarray(x)), \
        torch.from_numpy(bundle["y"][idx])


class WindowDataset:
    """Tiny in-memory dataset; the whole tensor is ~130 MB so we keep it hot."""

    def __init__(self, bundle: dict, idx: np.ndarray):
        import torch
        self.x, self.y = make_tensors(bundle, idx)
        self.idx = torch.from_numpy(np.asarray(idx))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.idx[i]


if __name__ == "__main__":
    for spec in (TON, BOT):
        b = prepare(spec, force=True)
        print(f"\n=== {spec.name} ===")
        print("  raw rows        :", b["raw_rows"])
        print("  kept            :", len(b["y"]), f"(paper: {spec.n_train + spec.n_test})")
        print("  train / test    :", b["n_train"], "/", b["n_test"],
              f"(paper: {spec.n_train} / {spec.n_test})")
        print("  flow groups     :", b["n_flow_groups"])
        print("  features        :", len(b["features"]))
        print("  train counts    :", b["train_class_counts"])
        print("  test  counts    :", b["test_class_counts"])
