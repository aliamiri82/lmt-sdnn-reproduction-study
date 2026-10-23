"""
Parameter / model-size / FLOPs accounting for Table VII.

Two independent budgets are produced for every model.

`paper_flops`
    Table V of the paper, applied verbatim, so that our Table VII is directly
    comparable with theirs:

        standard 1-D conv   2 * H * W * C_out * (C_in * K^2 + 1)
        1D-DSC depthwise    2 * H * W * C_in  * (K^2 + 1)
        1D-DSC pointwise    2 * H * W * C_in  * (C_out + 1)
        BiGRU               2 * (E*H*6 + H*(H + 12))
        BiLSTM              2 * (E*H*8 + H*(H + 20))

`true_flops`
    The same quantities computed correctly for 1-D data: the kernel enters
    linearly (K, not K^2) because a 1-D kernel has K taps, and the recurrent
    cost is multiplied by the number of time steps, which Table V omits.  Both
    columns are reported in the study; the gap between them is itself one of the
    findings (see the critique section of the report).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import N_FEATURES, SEQ_LEN

BYTES_PER_PARAM = 4          # float32; reproduces the paper's teacher row exactly


# --------------------------------------------------------------------------- #
def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    ps = model.parameters()
    return int(sum(p.numel() for p in ps if (p.requires_grad or not trainable_only)))


def size_kb(n_params: int) -> float:
    """The paper's own convention: 5,837,316 params -> 22,802.02 KB (Table VII)."""
    return n_params * BYTES_PER_PARAM / 1024.0


# --------------------------------------------------------------------------- #
def _shape_trace(model: nn.Module, in_ch: int, seq_len: int):
    """Record the input/output shape of every leaf module with one forward pass."""
    trace: list[tuple[nn.Module, tuple, tuple]] = []
    handles = []

    def hook(mod, inp, out):
        i = inp[0].shape if isinstance(inp, tuple) and torch.is_tensor(inp[0]) else None
        o = out[0].shape if isinstance(out, tuple) else out.shape
        trace.append((mod, tuple(i) if i is not None else (), tuple(o)))

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear, nn.LSTM, nn.GRU, nn.RNN)):
            handles.append(m.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device       # probe must live where the model does
    try:
        with torch.no_grad():
            model(torch.zeros(2, in_ch, seq_len, device=dev))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return trace


def _rnn_terms(mod) -> tuple[int, int, int]:
    """(input size E, hidden size H, gates) for an RNN-family module."""
    gates = {nn.LSTM: 4, nn.GRU: 3, nn.RNN: 1}[type(mod)]
    return mod.input_size, mod.hidden_size, gates


def flops(model: nn.Module, in_ch: int = N_FEATURES, seq_len: int = SEQ_LEN,
          mode: str = "paper") -> int:
    """`mode` is 'paper' (Table V verbatim) or 'true' (corrected 1-D accounting)."""
    total = 0
    for mod, ish, osh in _shape_trace(model, in_ch, seq_len):
        if isinstance(mod, nn.Conv1d):
            c_in, c_out = mod.in_channels, mod.out_channels
            k = mod.kernel_size[0]
            w = osh[-1]                                   # output length; H = 1
            depthwise = mod.groups == c_in and c_in > 1
            pointwise = k == 1 and mod.groups == 1
            kk = k * k if mode == "paper" else k           # the K^2 / K question
            if depthwise:
                total += 2 * w * c_in * (kk + 1)
            elif pointwise:
                total += 2 * w * c_in * (c_out + 1)
            else:
                total += 2 * w * c_out * (c_in * kk + 1)
        elif isinstance(mod, nn.Linear):
            n_pos = 1
            for d in osh[:-1]:
                n_pos *= d
            n_pos //= osh[0]                               # per sample
            total += 2 * n_pos * mod.in_features * mod.out_features
        elif isinstance(mod, (nn.LSTM, nn.GRU, nn.RNN)):
            e, h, gates = _rnn_terms(mod)
            dirs = 2 if mod.bidirectional else 1
            steps = ish[1] if len(ish) == 3 else seq_len
            if mode == "paper":
                # Table V: BiGRU 2*(E*H*6 + H*(H+12)); BiLSTM 2*(E*H*8 + H*(H+20))
                c = {3: (6, 12), 4: (8, 20), 1: (2, 4)}[gates]
                total += 2 * (e * h * c[0] + h * (h + c[1]))
            else:
                # per step: gates * (E*H + H*H) MACs, x2 FLOPs, x directions
                total += 2 * steps * dirs * gates * (e * h + h * h)
    return int(total)


def summarise(model: nn.Module, n_classes: int, in_ch: int = N_FEATURES,
              seq_len: int = SEQ_LEN) -> dict:
    p = count_params(model)
    return {
        "params": p,
        "size_kb": size_kb(p),
        "paper_flops": flops(model, in_ch, seq_len, "paper"),
        "true_flops": flops(model, in_ch, seq_len, "true"),
        "n_classes": n_classes,
    }


def compression(teacher_total: dict, student: dict) -> dict:
    """Reduction percentages, in the direction Table VII reports them."""
    def drop(a, b):
        return 100.0 * (a - b) / a if a else 0.0
    return {
        "params_drop_pct": drop(teacher_total["params"], student["params"]),
        "size_drop_pct": drop(teacher_total["size_kb"], student["size_kb"]),
        "flops_drop_pct": drop(teacher_total["paper_flops"], student["paper_flops"]),
        "true_flops_drop_pct": drop(teacher_total["true_flops"], student["true_flops"]),
    }


# --------------------------------------------------------------------------- #
#  Consistency audit of the paper's own Table VII
# --------------------------------------------------------------------------- #
PAPER_TABLE_VII = {
    # model: (ToN params, ToN size KB, ToN FLOPs, BoT params, BoT size KB, BoT FLOPs)
    "Teacher(Total)":   (5_837_316, 22_802.02, 12_126_273, 5_503_851, 21_499.42, 11_011_845),
    "KD-1DCNN":         (169_971,      166.95,     392_121,   178_773,     198.33,    330_866),
    "KD-SNN":           (57_346,       132.63,     317_614,    46_915,     118.79,    305_427),
    "KD-RNN":           (296_104,      663.95,   1_016_526,   265_862,   1_016.32,    903_647),
    "KD-TCN":           (235_011,      614.63,     840_768,   234_754,     917.01,    763_023),
    "KD-1DCNN-BiLSTM":  (451_237,      918.01,   1_048_654,   423_725,   1_038.53,    932_194),
    "LMT-SDNN(ours)":   (32_795,        88.99,      74_220,    29_751,      77.13,     71_437),
}


def audit_paper_table_vii() -> list[dict]:
    """
    Check whether the paper's reported (parameters, model size) pairs are
    internally consistent.  A float32 checkpoint must satisfy
    size_KB = params * 4 / 1024; the teacher row obeys it to 5 significant
    figures, every other row does not.
    """
    rows = []
    for name, v in PAPER_TABLE_VII.items():
        for ds, (p, s, f) in (("ToN_IoT", (v[0], v[1], v[2])),
                              ("BoT_IoT", (v[3], v[4], v[5]))):
            implied = size_kb(p)
            rows.append({
                "model": name, "dataset": ds,
                "reported_params": p, "reported_size_kb": s, "reported_flops": f,
                "size_kb_if_fp32": round(implied, 2),
                "bytes_per_param_implied": round(s * 1024 / p, 3),
                "consistent_fp32": abs(implied - s) / s < 0.01,
                # A forward pass costs at least 2 FLOPs per weight per position of
                # the feature map.  A ratio near 2 therefore means the reported
                # figure was evaluated with H = W = 1, i.e. the temporal extent of
                # the input was left out of the budget entirely.
                "flops_per_param": round(f / p, 2),
            })
    return rows


if __name__ == "__main__":
    import json
    from models import STUDENTS, TEACHERS, build

    print("=== our models ===")
    for nc, ds in ((10, "ToN_IoT"), (5, "BoT_IoT")):
        print(f"\n-- {ds} --")
        tt = {"params": 0, "size_kb": 0.0, "paper_flops": 0, "true_flops": 0}
        for name in TEACHERS:
            s = summarise(build(name, nc), nc)
            for k in tt:
                tt[k] += s[k]
            print(f"  {name:28s} {s['params']:>10,}  {s['size_kb']:>10.2f} KB  "
                  f"paperFLOPs {s['paper_flops']:>12,}  trueFLOPs {s['true_flops']:>12,}")
        print(f"  {'Teacher(Total)':28s} {tt['params']:>10,}  {tt['size_kb']:>10.2f} KB  "
              f"paperFLOPs {tt['paper_flops']:>12,}  trueFLOPs {tt['true_flops']:>12,}")
        for name in STUDENTS:
            s = summarise(build(name, nc), nc)
            c = compression(tt, s)
            print(f"  {name:28s} {s['params']:>10,}  {s['size_kb']:>10.2f} KB  "
                  f"paperFLOPs {s['paper_flops']:>12,}  trueFLOPs {s['true_flops']:>12,}"
                  f"   [-{c['params_drop_pct']:.2f}% params]")

    print("\n=== audit of the paper's Table VII ===")
    bad = [r for r in audit_paper_table_vii() if not r["consistent_fp32"]]
    for r in audit_paper_table_vii():
        flag = "OK " if r["consistent_fp32"] else "!! "
        print(f"  {flag}{r['model']:20s} {r['dataset']:8s} "
              f"reported {r['reported_size_kb']:>9.2f} KB vs fp32 "
              f"{r['size_kb_if_fp32']:>9.2f} KB  "
              f"({r['bytes_per_param_implied']:.2f} bytes/param)")
    print(f"  -> {len(bad)}/{len(audit_paper_table_vii())} rows are not fp32-consistent")
