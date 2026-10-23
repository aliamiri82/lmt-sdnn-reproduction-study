"""
Experiment driver.

Every run is described by a dict, hashed, and appended to results/runs.jsonl.
Re-invoking the script skips runs whose hash is already recorded, so the suite
can be executed in stages and resumed after an interruption.

Stages
------
teachers      stage 2 of Algorithm 1 on both datasets
main          Table VI  -- six distilled student architectures
nodistill     Fig. 3 / Fig. 7 -- the same six trained without a teacher
tablation     Fig. 4  -- teacher 1 only / teacher 2 only / both
weights       Table III + Fig. 2 -- the six (alpha, beta, gamma) settings
lossablation  contribution of L_KL, L_CS, L_TRW
temperature   sensitivity to the distillation temperature the paper never states
csalpha       constant alpha_i (as written) vs a genuinely class-aware alpha_i
cstarget      hard-label vs teacher-target reading of Eq. (7)
klt2          with / without the Hinton T^2 factor missing from Eq. (9)
seeds         repeated runs for the stability box plots
proposal      LMT-SDNN++ : entropy-gated teachers + class-aware cost weights
leakage       what the scores become when host identity is a feature
blocksplit    what the scores become under a contamination-free split
complexity    Table VII + edge latency
all           everything above, in order
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from complexity import audit_paper_table_vii, compression, summarise
from config import (BEST_COMBINATION, DATASETS, RESULTS, SEED, SEEDS_MULTI,
                    STUDENT_TRAIN, TEACHER_TRAIN, TEMPERATURE,
                    TEMPERATURE_SWEEP, WEIGHT_COMBINATIONS, resolve_device)
from data import prepare
from models import STUDENTS, TEACHERS, build
from train import (TeacherBank, evaluate, measure_latency, save_json,
                   set_seed, train_student, train_supervised)

RUNS = RESULTS / "runs.jsonl"
DEFAULT_STUDENTS = ["LMT-SDNN", "1DCNN", "SNN", "RNN", "TCN", "1DCNN-BiLSTM"]


# --------------------------------------------------------------------------- #
# Fields whose default value is dropped before hashing.  This keeps the identity
# of an experiment stable when a new switch is added to the driver: a run that
# predates the switch and a run that leaves it at its default hash identically,
# so the suite never re-executes work it has already recorded.
# Only fields added *after* runs were already recorded belong here; a field that
# was present when a row was written must keep taking part in its hash.
_NEUTRAL_DEFAULTS = {
    "quota_rule": "balanced",
}


def _key(d: dict) -> str:
    canon = {k: v for k, v in d.items() if _NEUTRAL_DEFAULTS.get(k, object()) != v}
    return hashlib.md5(json.dumps(canon, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _done() -> dict[str, dict]:
    if not RUNS.exists():
        return {}
    out = {}
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["key"]] = r
    return out


def _record(cfg: dict, metrics: dict, extra: dict | None = None):
    row = {"key": _key(cfg), "config": cfg, "metrics": metrics, **(extra or {})}
    with RUNS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
    return row


def _fmt(m: dict) -> str:
    return (f"acc {m['accuracy']:6.2f}  P {m['macro_precision']:6.2f}  "
            f"R {m['macro_recall']:6.2f}  F1 {m['macro_f1']:6.2f}")


# --------------------------------------------------------------------------- #
class Context:
    """Caches prepared datasets and teacher banks across stages."""

    def __init__(self, device: str, force: bool = False):
        self.device = device
        self.force = force
        self._bundles: dict[tuple, dict] = {}
        self._banks: dict[tuple, TeacherBank] = {}

    def bundle(self, ds: str, seed: int = SEED, split_mode: str = "record",
               include_ip: bool = False, quota_rule: str = "balanced"):
        k = (ds, seed, split_mode, include_ip, quota_rule)
        if k not in self._bundles:
            self._bundles[k] = prepare(DATASETS[ds], seed=seed,
                                       split_mode=split_mode, include_ip=include_ip,
                                       quota_rule=quota_rule)
        return self._bundles[k]

    def bank(self, ds: str, seed: int = SEED, split_mode: str = "record",
             include_ip: bool = False, quota_rule: str = "balanced") -> TeacherBank:
        k = (ds, seed, split_mode, include_ip, quota_rule)
        if k not in self._banks:
            # Each bank pins ~1.2 GiB of teacher logits/features on the GPU, so
            # only one may be alive at a time on an 8 GB card.  Evict the rest
            # (and their cached bundles) before building the new one.
            self._evict()
            b = self.bundle(ds, seed, split_mode, include_ip, quota_rule)
            variant = (f"{split_mode}{'_ip' if include_ip else ''}"
                       f"{'' if quota_rule == 'balanced' else '_' + quota_rule}")
            self._banks[k] = TeacherBank(b, ds, self.device, seed, TEACHER_TRAIN,
                                         in_ch=b["records"].shape[1],
                                         force=self.force, variant=variant)
        return self._banks[k]

    def _evict(self):
        """Release every pinned teacher bank and free the accelerator cache."""
        if not self._banks:
            return
        for bank in self._banks.values():
            for attr in ("d_logits", "d_feats", "models"):
                if hasattr(bank, attr):
                    setattr(bank, attr, None)
            bank.resident = False
        self._banks.clear()
        self._bundles.clear()
        import gc
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
#  Stages
# --------------------------------------------------------------------------- #
def stage_teachers(ctx: Context, done: dict):
    for ds in DATASETS:
        bank = ctx.bank(ds)
        for name, met in bank.metrics.items():
            cfg = {"stage": "teachers", "dataset": ds, "model": name, "seed": SEED}
            if _key(cfg) in done:
                continue
            mdl = build(name, len(ctx.bundle(ds)["classes"]))
            _record(cfg, met, {"complexity": summarise(mdl, len(ctx.bundle(ds)["classes"]))})
            print(f"  [teachers] {ds:8s} {name:28s} {_fmt(met)}")


def _distil(ctx, done, *, stage, ds, student, seed=SEED, combo=BEST_COMBINATION,
            temperature=TEMPERATURE, cs_alpha_mode="constant", cs_target="hard",
            kl_t2=False, adaptive=False, teacher_subset=None, split_mode="record",
            include_ip=False, quota_rule="balanced", extra_tag=None):
    a, b, g = WEIGHT_COMBINATIONS[combo] if isinstance(combo, str) else combo
    cfg = {"stage": stage, "dataset": ds, "model": student, "seed": seed,
           "alpha": a, "beta": b, "gamma": g, "temperature": temperature,
           "cs_alpha_mode": cs_alpha_mode, "cs_target": cs_target,
           "kl_t2": kl_t2, "adaptive": adaptive,
           "teachers": teacher_subset or [0, 1], "split_mode": split_mode,
           "include_ip": include_ip, "quota_rule": quota_rule, "tag": extra_tag}
    k = _key(cfg)
    if k in done:
        m = done[k]["metrics"]
        print(f"  [{stage}] {ds:8s} {student:14s} {extra_tag or '':18s} (cached) {_fmt(m)}")
        return done[k]
    bundle = ctx.bundle(ds, seed, split_mode, include_ip, quota_rule)
    bank = ctx.bank(ds, seed, split_mode, include_ip, quota_rule)
    # the leakage probe appends the two host-identity columns, so the input width
    # is a property of the bundle, not the global feature count
    in_ch = bundle["records"].shape[1]
    t0 = time.time()
    mdl, met, hist = train_student(student, bundle, bank, STUDENT_TRAIN, ctx.device,
                                   seed, a, b, g, temperature,
                                   cs_alpha_mode=cs_alpha_mode, cs_target=cs_target,
                                   kl_temperature_scaling=kl_t2,
                                   adaptive_teachers=adaptive,
                                   teacher_subset=teacher_subset,
                                   in_ch=in_ch)
    row = _record(cfg, met, {"complexity": summarise(mdl, len(bundle["classes"]), in_ch=in_ch),
                             "seconds": time.time() - t0,
                             "epochs_run": len(hist)})
    print(f"  [{stage}] {ds:8s} {student:14s} {extra_tag or '':18s} {_fmt(met)}"
          f"  ({time.time()-t0:.0f}s)")
    return row


def stage_main(ctx, done):
    for ds in DATASETS:
        for s in DEFAULT_STUDENTS:
            _distil(ctx, done, stage="main", ds=ds, student=s)


def _plain(ctx, done, *, stage, ds, student, quota_rule="balanced", extra_tag=None):
    """Train `student` on hard labels only -- the no-teacher control."""
    cfg = {"stage": stage, "dataset": ds, "model": student, "seed": SEED,
           "quota_rule": quota_rule}
    if extra_tag is not None:          # keeps the key identical to earlier runs
        cfg["tag"] = extra_tag
    k = _key(cfg)
    if k in done:
        print(f"  [{stage}] {ds:8s} {student:14s} {extra_tag or '':16s} (cached) "
              f"{_fmt(done[k]['metrics'])}")
        return done[k]
    bundle = ctx.bundle(ds, SEED, "record", False, quota_rule)
    in_ch = bundle["records"].shape[1]
    mdl = build(student, len(bundle["classes"]), in_ch=in_ch)
    t0 = time.time()
    mdl, met, _ = train_supervised(mdl, bundle, STUDENT_TRAIN, ctx.device, SEED)
    row = _record(cfg, met, {"complexity": summarise(mdl, len(bundle["classes"]), in_ch=in_ch),
                             "seconds": time.time() - t0})
    print(f"  [{stage}] {ds:8s} {student:14s} {extra_tag or '':16s} {_fmt(met)}")
    return row


def stage_nodistill(ctx, done):
    for ds in DATASETS:
        for s in DEFAULT_STUDENTS:
            _plain(ctx, done, stage="nodistill", ds=ds, student=s)


def stage_tablation(ctx, done):
    for ds in DATASETS:
        for sub, tag in (([0], "teacher1-only"), ([1], "teacher2-only"), ([0, 1], "both")):
            _distil(ctx, done, stage="tablation", ds=ds, student="LMT-SDNN",
                    teacher_subset=sub, extra_tag=tag)


def stage_weights(ctx, done):
    for ds in DATASETS:
        for name in WEIGHT_COMBINATIONS:
            _distil(ctx, done, stage="weights", ds=ds, student="LMT-SDNN",
                    combo=name, extra_tag=name)


def stage_lossablation(ctx, done):
    variants = {
        "KL only":            (1.0, 0.0, 0.0),
        "CS only":            (0.0, 1.0, 0.0),
        "KL + CS":            (0.5, 0.5, 0.0),
        "KL + CS + TRW":      WEIGHT_COMBINATIONS[BEST_COMBINATION],
        "CS + TRW (no KD)":   (0.0, 0.8, 0.2),
    }
    for ds in DATASETS:
        for tag, w in variants.items():
            _distil(ctx, done, stage="lossablation", ds=ds, student="LMT-SDNN",
                    combo=w, extra_tag=tag)


def stage_temperature(ctx, done):
    for ds in DATASETS:
        for t in TEMPERATURE_SWEEP:
            _distil(ctx, done, stage="temperature", ds=ds, student="LMT-SDNN",
                    temperature=t, extra_tag=f"T={t}")


def stage_csalpha(ctx, done):
    for ds in DATASETS:
        for mode in ("constant", "inverse_freq", "effective_number"):
            _distil(ctx, done, stage="csalpha", ds=ds, student="LMT-SDNN",
                    cs_alpha_mode=mode, extra_tag=mode)


def stage_cstarget(ctx, done):
    for ds in DATASETS:
        for mode in ("hard", "teacher"):
            _distil(ctx, done, stage="cstarget", ds=ds, student="LMT-SDNN",
                    cs_target=mode, extra_tag=f"p={mode}")


def stage_klt2(ctx, done):
    for ds in DATASETS:
        for flag in (False, True):
            _distil(ctx, done, stage="klt2", ds=ds, student="LMT-SDNN",
                    kl_t2=flag, extra_tag=f"T^2={flag}")


def stage_seeds(ctx, done, n_seeds: int = 3):
    for ds in DATASETS:
        for s in DEFAULT_STUDENTS:
            for seed in SEEDS_MULTI[:n_seeds]:
                _distil(ctx, done, stage="seeds", ds=ds, student=s, seed=seed,
                        extra_tag=f"seed{seed}")


def stage_proposal(ctx, done):
    for ds in DATASETS:
        _distil(ctx, done, stage="proposal", ds=ds, student="LMT-SDNN",
                adaptive=True, cs_alpha_mode="effective_number",
                extra_tag="LMT-SDNN++")
        # each ingredient on its own, so the gain can be attributed
        _distil(ctx, done, stage="proposal", ds=ds, student="LMT-SDNN",
                adaptive=True, extra_tag="+adaptive teachers")
        _distil(ctx, done, stage="proposal", ds=ds, student="LMT-SDNN",
                cs_alpha_mode="effective_number", extra_tag="+class-aware alpha")


def stage_leakage(ctx, done):
    for ds in DATASETS:
        _distil(ctx, done, stage="leakage", ds=ds, student="LMT-SDNN",
                include_ip=True, extra_tag="host IPs as features")


def stage_blocksplit(ctx, done):
    for ds in DATASETS:
        _distil(ctx, done, stage="blocksplit", ds=ds, student="LMT-SDNN",
                split_mode="block", extra_tag="contamination-free split")


def stage_quota(ctx, done):
    """
    Sec. IV-B only says "balanced subsets ... were selected".  This stage runs the
    identical pipeline under the competing reading -- keep the raw class ratios
    and scale them down -- to measure how much of the headline macro number is
    decided by a rule the paper never states.
    """
    for ds in DATASETS:
        _distil(ctx, done, stage="quota", ds=ds, student="LMT-SDNN",
                quota_rule="proportional", extra_tag="proportional quotas")
        # the no-teacher control under the same quotas, so the distillation gain
        # can be measured on a harder, genuinely imbalanced version of the task
        _plain(ctx, done, stage="quota", ds=ds, student="LMT-SDNN",
               quota_rule="proportional", extra_tag="proportional, no KD")


def stage_complexity(ctx, done):
    out = {"models": {}, "paper_table_vii_audit": audit_paper_table_vii()}
    for ds, spec in DATASETS.items():
        nc = spec.n_classes
        tt = {"params": 0, "size_kb": 0.0, "paper_flops": 0, "true_flops": 0,
              "latency_ms_single": 0.0, "latency_ms_per_sample_batch256": 0.0}
        rows = {}
        for name in TEACHERS:
            s = summarise(build(name, nc), nc)
            s.update(measure_latency(build(name, nc), 32))
            rows[name] = s
            for k in tt:
                tt[k] += s[k]
        # the ensemble runs both teachers, so its latency is their sum
        tt["throughput_flows_per_s"] = 1000.0 / max(tt["latency_ms_single"], 1e-9)
        rows["Teacher(Total)"] = tt
        for name in STUDENTS:
            m = build(name, nc)
            s = summarise(m, nc)
            s.update(measure_latency(m, 32))
            s.update(compression(tt, s))
            rows[name] = s
        out["models"][ds] = rows
        print(f"  [complexity] {ds}: teacher total {tt['params']:,} params, "
              f"student {rows['LMT-SDNN']['params']:,} params "
              f"(-{rows['LMT-SDNN']['params_drop_pct']:.2f}%)")
    save_json(out, RESULTS / "complexity.json")


STAGES = {
    "teachers": stage_teachers, "main": stage_main, "nodistill": stage_nodistill,
    "tablation": stage_tablation, "weights": stage_weights,
    "lossablation": stage_lossablation, "temperature": stage_temperature,
    "csalpha": stage_csalpha, "cstarget": stage_cstarget, "klt2": stage_klt2,
    "seeds": stage_seeds, "proposal": stage_proposal, "leakage": stage_leakage,
    "blocksplit": stage_blocksplit, "quota": stage_quota,
    "complexity": stage_complexity,
}
ORDER = ["teachers", "main", "nodistill", "tablation", "weights", "lossablation",
         "csalpha", "cstarget", "klt2", "temperature", "proposal", "seeds",
         "leakage", "blocksplit", "quota", "complexity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", default=["all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = resolve_device()
    print(f"device = {device} | torch {torch.__version__}")
    if device == "cuda":
        print(f"gpu    = {torch.cuda.get_device_name(0)}")

    ctx = Context(device, force=args.force)
    todo = ORDER if args.stages == ["all"] or "all" in args.stages else args.stages
    failures = []
    for s in todo:
        if s not in STAGES:
            print(f"unknown stage: {s}", file=sys.stderr)
            continue
        print(f"\n===== stage: {s} =====", flush=True)
        t0 = time.time()
        try:
            STAGES[s](ctx, _done())
            print(f"----- {s} finished in {time.time()-t0:.0f}s -----", flush=True)
        except Exception as exc:               # one bad stage must not sink the suite
            import traceback
            traceback.print_exc()
            failures.append((s, repr(exc)))
            print(f"----- {s} FAILED after {time.time()-t0:.0f}s -----", flush=True)
    if failures:
        print("\n!! failed stages:")
        for s, e in failures:
            print(f"   {s}: {e}")


if __name__ == "__main__":
    main()
