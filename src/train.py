"""
Training and evaluation engine.

Stage 2 of the paper's Algorithm 1 ("Save logits Zt1, Zt2 as teachers logits ...
Save both Teacher 1 and Teacher 2 models") is taken literally: each teacher is
trained once per dataset, frozen, and then its logits *and* its windowed
intermediate features are written to a memory-mapped cache.  Every subsequent
student run reads that cache instead of re-running the teachers, which is what
makes the ~40 distillation runs of this study affordable.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from complexity import summarise
from config import (RESULTS, SEQ_LEN, TRW_WINDOWS, TrainCfg, resolve_device)
from data import make_tensors
from losses import DistillationLoss, to_windows
from models import FeatureProjector, build

CACHE_DIR = Path(os.environ.get("LMT_CACHE", str(RESULTS.parent / ".cache" / "teachers")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = RESULTS / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batches(n: int, bs: int, shuffle: bool, gen: torch.Generator | None = None):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def _stage(x: torch.Tensor, device: str) -> torch.Tensor:
    """
    Keep the whole split on the accelerator when it fits (107 MB for ToN_IoT's
    training half), so batching costs an index_select instead of a host copy.
    """
    if device != "cuda":
        return x
    free, _ = torch.cuda.mem_get_info()
    return x.to(device) if x.numel() * x.element_size() < 0.25 * free else x


def _make_scheduler(opt, cfg: TrainCfg, steps_per_epoch: int):
    total = cfg.epochs * steps_per_epoch
    warm = cfg.warmup_epochs * steps_per_epoch

    def fn(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        if cfg.scheduler == "cosine":
            p = (step - warm) / max(1, total - warm)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
             classes: list[str], device: str, batch: int = 4096,
             light: bool = False) -> dict:
    """`light=True` skips the per-class report and confusion matrix; used for the
    per-epoch model-selection check, where only macro-F1 is consulted."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(y), batch):
            xb = x[i:i + batch].to(device, non_blocking=True)
            preds.append(model(xb)[0].argmax(1))
    yp = torch.cat(preds).cpu().numpy()
    yt = y.numpy()
    # [PAPER, Sec. IV-C] "This article adopts macroaveraging"
    out = {
        "accuracy": accuracy_score(yt, yp) * 100,
        "macro_precision": precision_score(yt, yp, average="macro", zero_division=0) * 100,
        "macro_recall": recall_score(yt, yp, average="macro", zero_division=0) * 100,
        "macro_f1": f1_score(yt, yp, average="macro", zero_division=0) * 100,
    }
    if light:
        return out
    out["weighted_f1"] = f1_score(yt, yp, average="weighted", zero_division=0) * 100
    out["per_class"] = classification_report(
        yt, yp, target_names=classes, output_dict=True, zero_division=0,
        labels=list(range(len(classes))))
    out["confusion_matrix"] = confusion_matrix(
        yt, yp, labels=list(range(len(classes)))).tolist()
    return out


@torch.no_grad()
def measure_latency(model: nn.Module, in_ch: int, seq_len: int = SEQ_LEN,
                    n_warm: int = 20, n_iter: int = 200) -> dict:
    """Single-sample CPU latency -- the proxy for an edge deployment."""
    import copy
    model = copy.deepcopy(model).to("cpu").eval()   # never mutate the caller's model
    x1 = torch.randn(1, in_ch, seq_len)
    for _ in range(n_warm):
        model(x1)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model(x1)
    single = (time.perf_counter() - t0) / n_iter * 1000

    xb = torch.randn(256, in_ch, seq_len)
    for _ in range(5):
        model(xb)
    t0 = time.perf_counter()
    for _ in range(30):
        model(xb)
    batched = (time.perf_counter() - t0) / 30 / 256 * 1000
    return {"latency_ms_single": single, "latency_ms_per_sample_batch256": batched,
            "throughput_flows_per_s": 1000.0 / max(single, 1e-9)}


# --------------------------------------------------------------------------- #
#  Supervised training (teachers and the non-distilled baselines of Fig. 3)
# --------------------------------------------------------------------------- #
def train_supervised(model, bundle, cfg: TrainCfg, device: str, seed: int,
                     class_weights: torch.Tensor | None = None,
                     verbose: bool = False, tag: str = ""):
    set_seed(seed)
    classes = list(bundle["classes"])
    xtr, ytr = make_tensors(bundle, bundle["train_idx"])
    xte, yte = make_tensors(bundle, bundle["test_idx"])
    xtr, xte = _stage(xtr, device), _stage(xte, device)
    ytr = ytr.to(xtr.device)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    steps = max(1, math.ceil(len(ytr) / cfg.batch_size))
    sched = _make_scheduler(opt, cfg, steps)
    crit = nn.CrossEntropyLoss(weight=None if class_weights is None
                               else class_weights.to(device),
                               label_smoothing=cfg.label_smoothing)
    gen = torch.Generator().manual_seed(seed)

    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(cfg.epochs):
        model.train()
        tot = torch.zeros((), device=device)
        for bidx in _batches(len(ytr), cfg.batch_size, True, gen):
            bi = bidx.to(xtr.device)
            xb = xtr[bi].to(device)
            yb = ytr[bi].to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb)[0], yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            tot = tot + loss.detach() * len(bidx)   # stays on device; no sync
        tot = float(tot)
        m = evaluate(model, xte, yte, classes, device, light=True)
        history.append({"epoch": ep, "loss": tot / len(ytr), **{
            k: m[k] for k in ("accuracy", "macro_f1", "macro_recall")}})
        if verbose:
            print(f"    [{tag}] ep{ep:02d} loss {tot/len(ytr):.4f} "
                  f"acc {m['accuracy']:.2f} macroF1 {m['macro_f1']:.2f}")
        if m["macro_f1"] > best:
            best, bad = m["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, evaluate(model, xte, yte, classes, device), history


# --------------------------------------------------------------------------- #
#  Teacher bank -- train once, freeze, cache logits + windowed features
# --------------------------------------------------------------------------- #
class TeacherBank:
    def __init__(self, bundle: dict, dataset: str, device: str, seed: int,
                 cfg: TrainCfg, in_ch: int, force: bool = False,
                 verbose: bool = True, variant: str = "base"):
        self.bundle, self.dataset, self.device = bundle, dataset, device
        self.in_ch, self.seed, self.variant = in_ch, seed, variant
        self.names = ["Teacher1-ResidualInception", "Teacher2-1DCNN-BiLSTM"]
        self.models, self.metrics = {}, {}
        n_classes = len(bundle["classes"])

        for name in self.names:
            ck = CKPT_DIR / f"{dataset}_{name}_seed{seed}_in{in_ch}_{variant}.pt"
            mdl = build(name, n_classes, in_ch=in_ch)
            if ck.exists() and not force:
                blob = torch.load(ck, map_location="cpu", weights_only=False)
                mdl.load_state_dict(blob["state"])
                self.metrics[name] = blob["metrics"]
                if verbose:
                    print(f"    [teacher] {name}: loaded  "
                          f"acc {blob['metrics']['accuracy']:.2f} "
                          f"macroF1 {blob['metrics']['macro_f1']:.2f}")
            else:
                if verbose:
                    print(f"    [teacher] {name}: training ...")
                mdl, met, hist = train_supervised(mdl, bundle, cfg, device, seed,
                                                  tag=name, verbose=verbose)
                self.metrics[name] = met
                torch.save({"state": {k: v.cpu() for k, v in mdl.state_dict().items()},
                            "metrics": met, "history": hist}, ck)
                if verbose:
                    print(f"    [teacher] {name}: acc {met['accuracy']:.2f} "
                          f"macroF1 {met['macro_f1']:.2f}")
            mdl.eval()
            for p in mdl.parameters():
                p.requires_grad_(False)
            self.models[name] = mdl.to(device)

        self.feat_dims = [self.models[n].feat_dim for n in self.names]
        self._build_cache(force=force, verbose=verbose)

    # ------------------------------------------------------------------ #
    def _build_cache(self, force: bool, verbose: bool):
        n = len(self.bundle["y"])
        n_classes = len(self.bundle["classes"])
        stem = f"{self.dataset}_seed{self.seed}_in{self.in_ch}_{self.variant}"
        self.logits = []
        self.feats = []
        for ti, name in enumerate(self.names):
            lf = CACHE_DIR / f"{stem}_{ti}_logits.npy"
            ff = CACHE_DIR / f"{stem}_{ti}_feats.npy"
            d = self.feat_dims[ti]
            if lf.exists() and ff.exists() and not force:
                self.logits.append(np.load(lf, mmap_mode="r"))
                self.feats.append(np.load(ff, mmap_mode="r"))
                continue
            if verbose:
                print(f"    [cache] materialising {name} logits+features ...")
            lg = np.lib.format.open_memmap(lf, mode="w+", dtype=np.float32,
                                           shape=(n, n_classes))
            ft = np.lib.format.open_memmap(ff, mode="w+", dtype=np.float16,
                                           shape=(n, TRW_WINDOWS, d))
            mdl = self.models[name]
            bs = 2048
            with torch.no_grad():
                for i in range(0, n, bs):
                    idx = np.arange(i, min(i + bs, n))
                    xb, _ = make_tensors(self.bundle, idx)
                    z, seq = mdl(xb.to(self.device))
                    lg[idx] = z.float().cpu().numpy()
                    ft[idx] = to_windows(seq, TRW_WINDOWS).half().cpu().numpy()
            lg.flush(); ft.flush()
            self.logits.append(np.load(lf, mmap_mode="r"))
            self.feats.append(np.load(ff, mmap_mode="r"))

        self._pin_to_device(verbose)

    def _pin_to_device(self, verbose: bool):
        """
        Keep the whole teacher cache resident on the accelerator when it fits.
        ~1.1 GB in fp16 for ToN_IoT, which turns the per-batch teacher lookup
        from a disk read into a device-side gather.
        """
        self.resident = False
        if self.device != "cuda":
            return
        need = sum(a.nbytes for a in self.logits) + sum(a.nbytes for a in self.feats)
        free, _ = torch.cuda.mem_get_info()
        if need > 0.45 * free:
            if verbose:
                print(f"    [cache] {need/2**30:.2f} GiB teacher cache stays on disk "
                      f"({free/2**30:.2f} GiB free)")
            return
        self.d_logits = [torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
                         for a in self.logits]
        self.d_feats = [torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
                        for a in self.feats]
        self.resident = True
        if verbose:
            print(f"    [cache] {need/2**30:.2f} GiB teacher cache resident on GPU")

    def fetch(self, idx: np.ndarray, device: str):
        """Teacher logits and windowed features for a batch of sample indices."""
        if getattr(self, "resident", False):
            t = torch.as_tensor(np.asarray(idx), device=self.device, dtype=torch.long)
            return ([l.index_select(0, t) for l in self.d_logits],
                    [f.index_select(0, t).float() for f in self.d_feats])
        i = np.sort(np.asarray(idx))
        order = np.argsort(np.argsort(np.asarray(idx)))     # restore batch order
        lg = [torch.from_numpy(np.asarray(l[i])[order]).to(device) for l in self.logits]
        ft = [torch.from_numpy(np.asarray(f[i])[order]).float().to(device)
              for f in self.feats]
        return lg, ft


# --------------------------------------------------------------------------- #
#  Distillation training (stage 3 of Algorithm 1)
# --------------------------------------------------------------------------- #
def train_student(student_name: str, bundle: dict, bank: TeacherBank,
                  cfg: TrainCfg, device: str, seed: int,
                  alpha: float, beta: float, gamma: float, temperature: float,
                  cs_alpha_mode: str = "constant", cs_target: str = "hard",
                  kl_temperature_scaling: bool = False,
                  adaptive_teachers: bool = False,
                  teacher_subset: list[int] | None = None,
                  in_ch: int | None = None, verbose: bool = False, tag: str = ""):
    set_seed(seed)
    classes = list(bundle["classes"])
    n_classes = len(classes)
    in_ch = in_ch if in_ch is not None else bundle["records"].shape[1]

    tr_idx = bundle["train_idx"]
    te_idx = bundle["test_idx"]
    xtr, ytr = make_tensors(bundle, tr_idx)
    xte, yte = make_tensors(bundle, te_idx)
    xtr, xte = _stage(xtr, device), _stage(xte, device)
    ytr = ytr.to(xtr.device)

    student = build(student_name, n_classes, in_ch=in_ch).to(device)
    sel = teacher_subset if teacher_subset is not None else [0, 1]
    dims = [bank.feat_dims[i] for i in sel]
    projector = FeatureProjector(student.feat_dim, dims).to(device)

    counts = torch.tensor([(ytr == c).sum().item() for c in range(n_classes)])
    crit = DistillationLoss(alpha, beta, gamma, temperature, counts,
                            cs_alpha_mode=cs_alpha_mode, cs_target=cs_target,
                            kl_temperature_scaling=kl_temperature_scaling,
                            adaptive_teachers=adaptive_teachers).to(device)

    params = list(student.parameters()) + list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = max(1, math.ceil(len(ytr) / cfg.batch_size))
    sched = _make_scheduler(opt, cfg, steps)
    gen = torch.Generator().manual_seed(seed)

    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(cfg.epochs):
        student.train()
        agg = {k: torch.zeros((), device=device) for k in ("kl", "cs", "trw", "total")}
        for bidx in _batches(len(ytr), cfg.batch_size, True, gen):
            b = bidx.numpy()
            bi = bidx.to(xtr.device)
            xb = xtr[bi].to(device)
            yb = ytr[bi].to(device)
            t_lg, t_ft = bank.fetch(tr_idx[b], device)
            t_lg = [t_lg[i] for i in sel]
            t_ft = [t_ft[i] for i in sel]

            opt.zero_grad(set_to_none=True)
            z, seq = student(xb)
            loss, parts = crit(z, seq, t_lg, t_ft, yb, projector)
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()
            for k in agg:
                agg[k] = agg[k] + parts[k] * len(b)   # stays on device; no sync

        agg = {k: float(v) for k, v in agg.items()}
        m = evaluate(student, xte, yte, classes, device, light=True)
        history.append({"epoch": ep, **{k: v / len(ytr) for k, v in agg.items()},
                        **{k: m[k] for k in ("accuracy", "macro_f1", "macro_recall")}})
        if verbose:
            print(f"    [{tag}] ep{ep:02d} L {agg['total']/len(ytr):.4f} "
                  f"(kl {agg['kl']/len(ytr):.3f} cs {agg['cs']/len(ytr):.3f} "
                  f"trw {agg['trw']/len(ytr):.3f})  acc {m['accuracy']:.2f} "
                  f"macroF1 {m['macro_f1']:.2f}")
        if m["macro_f1"] > best:
            best, bad = m["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        student.load_state_dict(best_state)
    metrics = evaluate(student, xte, yte, classes, device)
    return student, metrics, history


# --------------------------------------------------------------------------- #
def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
