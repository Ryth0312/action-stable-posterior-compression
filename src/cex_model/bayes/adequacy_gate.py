"""Model / operating-domain adequacy gate for decision recommendations.

The decision-determinability criterion ``worst_dec`` (:mod:`cex_model.bayes.decision`)
measures *conditional* posterior precision -- whether the pooled decision ``g`` is pinned
*given the fitted model*.  It does **not** certify that the model's mean prediction is adequate
at the target operating condition.  The decision-level hold-out
(:func:`cex_model.bayes.validation.decision_leave_one_experiment_out`; docs SI S13) shows this
matters: at high loading the SMA model mispredicts the main-peak retention (a documented overload
effect -- the protein binding mode changes), so the pooled decision does *not* extrapolate across
loading.  In one mAb C fold the model was precisely but incorrectly centered
(predicted pooled yield ``0.33`` with a narrow Monte-Carlo band, observed ``0.185``).

This module gates recommendations on an empirically supported operating domain:

    decision-adequate(c)  iff  model adequate at c  AND  worst_dec(c) < tau,

where ``model adequate at c`` is, here, ``c`` lying within the supported loading domain.
Out-of-domain conditions are marked ``abstain`` -- a state *distinct* from ``take_data``:
``take_data`` means the model is adequate but the posterior decision uncertainty is too large
(collect data to sharpen the posterior), whereas ``abstain`` means the model's adequacy is not
established at ``c`` (the fix is model revision or model-discrimination data, not more of the same).

Pure Python (no solver, no numpy): consumes committed operating-window-map rows so the gated
outputs are recomputed -- not merely footnoted -- from the same artifacts the paper cites.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_LOADING_CAP_G_PER_L",
    "SupportedDomain",
    "default_domains",
    "eligibility",
    "gate_operating_window",
    "gated_widest_adequate",
    "gated_action",
]

# Empirical loading boundary from the decision-level LOEO (docs SI S13): the pooled decision
# predicts out-of-sample up to ~35 g/L and fails by 40 g/L (the HLXSYN 45 g/L and HLXSYN 40 g/L
# held-out folds fail; the 25/35 g/L folds pass).  A conservative product-agnostic default;
# the gated deploy verdicts are invariant for any cap in [25, 35] g/L (docs SI S13).
DEFAULT_LOADING_CAP_G_PER_L = 35.0


@dataclass(frozen=True)
class SupportedDomain:
    """Empirically supported operating domain for one ``(product, qoi)`` pair.

    ``loading_max_g_per_l`` is the largest loading at which the decision-level hold-out supports
    the QoI; ``provenance`` records where the boundary came from (never a bare literal in caller
    code -- the cutoff must be traceable to the audit that set it).
    """

    product: str
    qoi: str  # 'pool_purity' | 'pool_yield'
    loading_max_g_per_l: float
    provenance: str

    def contains(self, op) -> bool:
        """Is operating condition ``op = [loading, grad_start, grad_end, elution_cv]`` in domain?
        (Loading is ``op[0]``; the boundary is the empirically supported maximum loading.)"""
        return float(op[0]) <= self.loading_max_g_per_l + 1e-9


def default_domains(product: str, *, cap: float = DEFAULT_LOADING_CAP_G_PER_L,
                    provenance: str = "decision_LOEO_loading_audit (docs SI S13)"):
    """The default per-QoI supported domains for a product (purity and yield share the cap)."""
    return [SupportedDomain(product, q, cap, provenance) for q in ("pool_purity", "pool_yield")]


def eligibility(op, domains) -> dict:
    """Joint (worst-QoI) eligibility of an operating condition ``c``.

    ``A_p(c) = min_q A_{p,q}(c)`` -- the joint purity/yield decision is eligible only if *every*
    QoI is in its supported domain.  Returns ``{'eligibility', 'reason', 'failed_qoi'}`` with
    ``eligibility in {'eligible','abstain'}``; ``reason`` is ``None`` when eligible.
    """
    failed = [d.qoi for d in domains if not d.contains(op)]
    if failed:
        return {"eligibility": "abstain",
                "reason": "outside_empirically_supported_domain",
                "failed_qoi": failed}
    return {"eligibility": "eligible", "reason": None, "failed_qoi": []}


def gate_operating_window(rows, domains):
    """Annotate each operating-window-map row with its eligibility.

    Returns ``(eligible_rows, abstained_rows)``; each returned row is a copy carrying the
    ``eligibility`` / ``reason`` / ``failed_qoi`` fields.  ``rows`` are the committed
    ``operating_window_map['rows']`` (each a dict with an ``'op'`` key).
    """
    elig, abstained = [], []
    for r in rows:
        e = eligibility(r["op"], domains)
        r2 = {**r, **e}
        (elig if e["eligibility"] == "eligible" else abstained).append(r2)
    return elig, abstained


def gated_widest_adequate(rows, domains, *, p_meet_key: str = "p_meet"):
    """The widest decision-adequate window *restricted to eligible candidates*: the eligible row
    with the highest ``P(meet)`` (``None`` if no candidate is eligible).  An out-of-domain
    operating condition can never be selected as the widest adequate window.
    """
    elig, _ = gate_operating_window(rows, domains)
    return max(elig, key=lambda r: r.get(p_meet_key, 0.0)) if elig else None


def gated_action(decision_op, base_action, domains) -> dict:
    """Gate the four-state action on the *deployed* operating condition.

    If the decision OP is out of the supported domain, the action becomes ``'abstain'``
    (explicitly *not* ``'take_data'``); otherwise the base four-state action is kept.  Returns
    ``{'action', 'reason', 'base_action'}``.
    """
    e = eligibility(decision_op, domains)
    if e["eligibility"] == "abstain":
        return {"action": "abstain", "reason": e["reason"], "base_action": base_action}
    return {"action": base_action, "reason": None, "base_action": base_action}
