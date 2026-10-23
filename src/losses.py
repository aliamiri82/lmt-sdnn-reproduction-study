"""
The three loss terms of Eq. (5)-(9), implemented literally, plus the switches
needed to test the places where the paper contradicts itself.

    L_KD = alpha * L_KL  +  beta * L_CS  +  gamma * L_TRW          Eq. (9)

Deliberate switches
-------------------
`kl_temperature_scaling`
    Hinton's soft-target loss carries a T^2 factor so that its gradient stays
    comparable with the hard-label term.  Eq. (6) and Eq. (9) of the paper have
    no such factor.  Default follows the paper; the ablation turns it on.

`cs_target`
    Sec. III-E says the cost-sensitive term measures "the fitting error of the
    student model on the hard labels", but Eq. (7) writes p_i(T) -- a
    temperature-softened *teacher* distribution -- and the surrounding text
    explains the focusing factor in terms of "the confidence of the teacher's
    prediction".  Those are different objectives; both are implemented.

`cs_alpha`
    Eq. (7) introduces alpha_i as "the weighting factor for class i.  This
    allows the model to assign different importance to each class", and then
    fixes alpha_i = 0.25 for every class.  A constant cannot re-weight classes,
    so the term as specified is a plain focal loss and does nothing for the
    long-tail problem it is introduced to solve.  'constant' reproduces the
    paper; 'inverse_freq' and 'effective_number' are the intended behaviour.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FOCAL_ALPHA, FOCAL_GAMMA, TEACHER_MIX, TRW_WINDOWS


# --------------------------------------------------------------------------- #
def _normalise_mix(mix: tuple[float, ...], n: int) -> tuple[float, ...]:
    """
    Eq. (6) and Eq. (8) mix the two teachers with a fixed 0.5 / 0.5.  In the
    single-teacher ablation only one term survives, and leaving its weight at
    0.5 would silently halve the whole distillation signal -- so the ablation
    would be comparing "one teacher" against "two teachers at twice the weight"
    rather than against "two teachers".  Renormalising to sum to one keeps the
    comparison about the teachers and not about the loss scale.
    """
    if len(mix) < n:                       # no weight supplied for every teacher
        return tuple(1.0 / n for _ in range(n))
    m = tuple(mix[:n])
    s = sum(m)
    return tuple(v / s for v in m) if s > 0 else tuple(1.0 / n for _ in range(n))


# --------------------------------------------------------------------------- #
#  Eq. (5) / (6) -- divergence distillation
# --------------------------------------------------------------------------- #
def kl_distillation(student_logits: torch.Tensor,
                    teacher_logits: list[torch.Tensor],
                    temperature: float,
                    mix: tuple[float, ...] = TEACHER_MIX,
                    temperature_scaling: bool = False,
                    sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    L_KL = 0.5 * D_KL(p_t1 || q_s) + 0.5 * D_KL(p_t2 || q_s)          Eq. (6)

    `sample_weights` (B, n_teachers) replaces the fixed 0.5 / 0.5 mixture with a
    per-sample, per-teacher weighting; used by the adaptive-fusion proposal.
    """
    T = temperature
    mix = _normalise_mix(mix, len(teacher_logits))
    log_q = F.log_softmax(student_logits / T, dim=1)
    total = student_logits.new_zeros(())
    for i, zt in enumerate(teacher_logits):
        p = F.softmax(zt / T, dim=1)
        kl = (p * (torch.log(p.clamp_min(1e-8)) - log_q)).sum(1)      # (B,)
        w = sample_weights[:, i] if sample_weights is not None else mix[i]
        total = total + (kl * w).mean() if torch.is_tensor(w) else total + w * kl.mean()
    if temperature_scaling:
        total = total * (T ** 2)
    return total


# --------------------------------------------------------------------------- #
#  Eq. (7) -- cost-sensitive / focal loss
# --------------------------------------------------------------------------- #
def class_alpha(counts: torch.Tensor, mode: str = "constant",
                constant: float = FOCAL_ALPHA, beta: float = 0.999) -> torch.Tensor:
    """Per-class weight vector alpha_i for Eq. (7)."""
    if mode == "constant":
        return torch.full_like(counts, constant, dtype=torch.float32)
    if mode == "inverse_freq":
        w = 1.0 / counts.clamp_min(1).float()
    elif mode == "effective_number":
        # Cui et al., "Class-Balanced Loss Based on Effective Number of Samples"
        eff = (1.0 - torch.pow(beta, counts.float())) / (1.0 - beta)
        w = 1.0 / eff.clamp_min(1e-8)
    else:
        raise ValueError(mode)
    return (w / w.sum() * len(counts)) * constant       # keep the paper's scale


def cost_sensitive_loss(student_logits: torch.Tensor,
                        targets: torch.Tensor,
                        alpha: torch.Tensor,
                        gamma: float = FOCAL_GAMMA,
                        teacher_probs: torch.Tensor | None = None,
                        mode: str = "hard",
                        temperature: float = 1.0) -> torch.Tensor:
    """
    L_CS = - sum_i alpha_i (1 - q_i)^gamma p_i log q_i                Eq. (7)

    mode='hard'    : p is the one-hot ground truth  (Sec. III-E prose)
    mode='teacher' : p is the temperature-softened teacher mixture (Eq. 7 symbols)
    """
    T = temperature if mode == "teacher" else 1.0
    log_q = F.log_softmax(student_logits / T, dim=1)
    q = log_q.exp()
    focus = (1.0 - q).clamp_min(0).pow(gamma)

    if mode == "hard":
        p = F.one_hot(targets, student_logits.size(1)).float()
    elif mode == "teacher":
        if teacher_probs is None:
            raise ValueError("mode='teacher' needs teacher_probs")
        p = teacher_probs
    else:
        raise ValueError(mode)

    per_class = alpha.view(1, -1) * focus * p * log_q
    return -per_class.sum(1).mean()


# --------------------------------------------------------------------------- #
#  Eq. (8) -- time-related window (TRW) loss
# --------------------------------------------------------------------------- #
def to_windows(seq: torch.Tensor, n_windows: int = TRW_WINDOWS) -> torch.Tensor:
    """
    Average-pool a (B, T, D) feature sequence onto a common grid of `n_windows`
    sliding windows.  Teacher 2 halves the temporal axis with its stride-2
    pooling, so teacher and student sequences must be re-gridded before their
    increments can be compared.
    """
    if seq.size(1) == n_windows:
        return seq
    return F.adaptive_avg_pool1d(seq.transpose(1, 2), n_windows).transpose(1, 2)


def trw_loss(student_seq: torch.Tensor,
             teacher_seqs: list[torch.Tensor],
             projector: nn.Module,
             n_windows: int = TRW_WINDOWS,
             mix: tuple[float, ...] = TEACHER_MIX,
             normalize: bool = True,
             sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    L_TRW = 0.5 * (1/N) sum_k || d f_T1(k) - d f_S(k) ||^2
          + 0.5 * (1/N) sum_k || d f_T2(k) - d f_S(k) ||^2           Eq. (8)

    with  d f(k) = f(k+1) - f(k)  the increment between adjacent windows.
    Note the paper normalises by N while summing k = 1 .. N-1; we keep the 1/N
    it prints, which is only a constant rescaling of gamma.
    """
    mix = _normalise_mix(mix, len(teacher_seqs))
    s_w = to_windows(student_seq, n_windows)
    total = student_seq.new_zeros(())
    for i, t_seq in enumerate(teacher_seqs):
        t_w = to_windows(t_seq, n_windows)
        s_p = projector(s_w, i)                       # student -> teacher space
        if normalize:
            # teachers are frozen and their activations have arbitrary scale;
            # unit-norming each window keeps the two increment fields comparable
            s_p = F.normalize(s_p, dim=-1)
            t_w = F.normalize(t_w, dim=-1)
        ds = s_p[:, 1:] - s_p[:, :-1]
        dt = t_w[:, 1:] - t_w[:, :-1]
        per_sample = (dt - ds).pow(2).sum(-1).sum(-1) / n_windows
        w = sample_weights[:, i] if sample_weights is not None else mix[i]
        total = total + ((per_sample * w).mean() if torch.is_tensor(w)
                         else w * per_sample.mean())
    return total


# --------------------------------------------------------------------------- #
#  Proposal: entropy-gated adaptive teacher fusion
# --------------------------------------------------------------------------- #
def confidence_weights(teacher_logits: list[torch.Tensor],
                       temperature: float = 1.0,
                       sharpness: float = 1.0) -> torch.Tensor:
    """
    Replace the fixed 0.5 / 0.5 mixture of Eq. (6) and Eq. (8) with per-sample
    weights driven by how certain each teacher is about *that* flow.

    For teacher t we take the normalised predictive entropy
        H_t = -sum_c p_c log p_c / log C     in [0, 1]
    and set w_t = softmax_t(sharpness * (1 - H_t)).  A teacher that hedges on a
    sample stops dictating the student's target for it, which is the cheapest
    possible form of the "dynamic teacher selection" the paper defers to future
    work (Sec. V).  Returns (B, n_teachers) rows that sum to 1.
    """
    conf = []
    for zt in teacher_logits:
        p = F.softmax(zt / temperature, dim=1)
        h = -(p * torch.log(p.clamp_min(1e-8))).sum(1) / torch.log(
            torch.tensor(float(p.size(1)), device=p.device))
        conf.append(1.0 - h)
    conf = torch.stack(conf, dim=1)                    # (B, n_teachers)
    return F.softmax(sharpness * conf, dim=1)


# --------------------------------------------------------------------------- #
#  Eq. (9) -- the composite objective
# --------------------------------------------------------------------------- #
class DistillationLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, gamma: float,
                 temperature: float, class_counts: torch.Tensor,
                 cs_alpha_mode: str = "constant",
                 cs_target: str = "hard",
                 kl_temperature_scaling: bool = False,
                 trw_windows: int = TRW_WINDOWS,
                 adaptive_teachers: bool = False,
                 focal_gamma: float = FOCAL_GAMMA):
        super().__init__()
        self.a, self.b, self.g = alpha, beta, gamma
        self.T = temperature
        self.cs_target = cs_target
        self.kl_t2 = kl_temperature_scaling
        self.trw_windows = trw_windows
        self.adaptive = adaptive_teachers
        self.focal_gamma = focal_gamma
        self.register_buffer("alpha_vec", class_alpha(class_counts, cs_alpha_mode))

    def forward(self, student_logits, student_seq, teacher_logits, teacher_seqs,
                targets, projector):
        w = confidence_weights(teacher_logits) if self.adaptive else None

        l_kl = kl_distillation(student_logits, teacher_logits, self.T,
                               temperature_scaling=self.kl_t2, sample_weights=w)

        t_probs = None
        if self.cs_target == "teacher":
            mix = w if w is not None else None
            stack = torch.stack([F.softmax(z / self.T, 1) for z in teacher_logits], 1)
            t_probs = ((stack * mix.unsqueeze(-1)).sum(1) if mix is not None
                       else stack.mean(1))
        l_cs = cost_sensitive_loss(student_logits, targets, self.alpha_vec,
                                   gamma=self.focal_gamma, teacher_probs=t_probs,
                                   mode=self.cs_target, temperature=self.T)

        l_trw = trw_loss(student_seq, teacher_seqs, projector,
                         n_windows=self.trw_windows, sample_weights=w)

        total = self.a * l_kl + self.b * l_cs + self.g * l_trw
        # returned as detached tensors, not floats: converting here would force a
        # device synchronisation on every optimiser step
        return total, {"kl": l_kl.detach(), "cs": l_cs.detach(),
                       "trw": l_trw.detach(), "total": total.detach()}
