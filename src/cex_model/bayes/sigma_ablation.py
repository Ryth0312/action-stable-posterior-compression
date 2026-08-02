"""σ freeze / compress ablation — does fixing or compressing σ change the *decision*?

The paper's headline is a σ **decision-null** mechanism (§2.10/§3.7): the steric factor σ enters SMA
only through the shared available-charge term ``Λ̄``, so it carries only 0.0–0.7 % of the
tolerance-whitened decision sensitivity while keq+ν carry 97–99 %. Today that is a *sensitivity*
statement (an exact-autograd σ-share). This module turns it into an **operational** one: it actually
**freezes or compresses σ** in six variants, refits, and asks whether the *fit* and — separately — the
*decision, operating window and uncertainty judgment* are unchanged.

The whole point is to keep the two questions apart:

  * **fit equivalence** — χ²/dof, RMSE, predictive residuals (does the model still describe the curves?);
  * **decision equivalence** — tolerance-whitened decision mean shift, decision covariance / worst_dec /
    interval change, spec-probability P(meet) delta, operating recommendation (does the process call move?).

A variant can change the *fit* yet leave the *decision* untouched — that is exactly the honest bucket
``WARN_FIT_CHANGED`` (e.g. pinning σ at a nominal value discards the *identifiable* common-mode σ, so the
fit degrades, yet the decision is σ-null so it does not move).

**Negative controls (two modes).** Freezing the decision-relevant ν / keq confirms the test has power.
The *power test* is the **@nominal** control (freeze at the prior-mean, refit) — the same operation
variant E applies to σ: σ survives it (decision-null) but ν / keq do not, so ``NC_{nu,keq}_nominal`` are
expected to ``FAIL_DECISION_CHANGED``. The **@MAP** control (freeze at the fitted value, refit) is a
weaker, complementary probe: because keq↔ν is a *tight, identified* ridge (r ≈ −0.999, §2.10), freezing
one endpoint and refitting the other lets the partner absorb it, so ``NC_{nu,keq}_map`` typically *pass* —
the decision uncertainty lives in the ridge *combination*, robust to fixing either endpoint. So the
negative-control verdict keys on the @nominal controls, and the @MAP verdicts are reported as-is.

**Discipline (matches ``docs/decision_null_theorem.md`` / paper App. B):**
  * σ is NOT uniformly unidentifiable — the **common-mode** direction ``σ→σ+δ·(1,…,1)`` (a uniform steric
    shift that moves all peaks) is comparatively *identified*; the sloppy content is the differential /
    minor-component σ directions. Variant D keeps only that identified common mode and freezes the rest.
  * The compression basis is the **fixed physical common mode** ``(1,…,1)`` (:func:`common_mode_direction`),
    NOT a Fisher / covariance eigenvector — an eigenvector basis would (wrongly) certify the stiff
    direction (App. B). This is guarded by an AST test.

**Uniform construction.** Every arm is an affine reparametrization ``u = A·u_free + c`` of the full
``u = [log10 keq, log10 kkin, ν, σ]`` (dim ``4n``, σ block ``u[3n:4n]``). The MAP reuses ``map_fit``'s
loss on the reassembled ``u`` (the ``profile.py`` reduced-vector idiom generalized to a matrix); the
reduced Laplace is ``H_free = J_freeᵀJ_free/σ_obs² + Aᵀ·prior.precision()·A`` with ``J_free = J_full·A``
(``J_full`` reverse-mode only — double-backward is wrong through the detached-IFT solver); the full-space
decision covariance is ``Σ_full = A·cov_free·Aᵀ`` and is wrapped in a :class:`Posterior` so every existing
decision tool consumes it unchanged. Variant C is the one exception (prior-only σ): at the full-model MAP,
zero σ's columns of ``J`` before the GGN so σ collapses to its prior with no refit.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from cex_model.bayes.decision import (
    _report_from_covariance,
    decision_covariance,
    decision_jacobian,
    decision_report,
)
from cex_model.bayes.decision_window import (
    gaussian_meet_prob,
    meet_prob_and_regret,
)
from cex_model.bayes.likelihood import (
    AKTA_NOISE_FLOOR_G_L,
    gaussian_loglik,
    mechanistic_model,
)
from cex_model.bayes.posterior import Posterior, map_fit
from cex_model.bayes.prior import components_to_u, physical_prior
from cex_model.bayes.validation import empirical_sigma_obs
from cex_model.diffsolver.calibrate_diff import targets_from_bundle
from cex_model.diffsolver.torch_solver import DTYPE

__all__ = [
    "ARMS",
    "DEFAULT_THRESHOLDS",
    "common_mode_direction",
    "build_arm",
    "run_ablation",
    "run_synthetic",
    "run_product",
]

# Arm ids. A = baseline; B–E freeze/compress σ; NC_* are the ν/keq negative controls (both freeze modes).
ARMS: tuple[str, ...] = (
    "A_full",
    "B_sigma_map",
    "C_sigma_prior",
    "D_sigma_commonmode",
    "E_sigma_nominal",
    "NC_nu_map",
    "NC_keq_map",
    "NC_nu_nominal",
    "NC_keq_nominal",
)

# Decision-equivalence / fit-equivalence thresholds (heuristic, argparse-overridable).
# The PRIMARY mean-shift gate is ``meanshift_over_std`` (Δg relative to the decision's OWN posterior std):
# a mean shift within the decision posterior is not a decision change, and this is robust to refit optimizer
# slop in the sloppy directions. But a shift can be small vs the posterior yet large vs the TOLERANCE (the
# thing the decision is judged against) when the decision is very determined (std ≪ tol) -- so ``meanshift_tw``
# (tolerance-whitened, the requested metric) is ALSO a graded guard, and any hard decision change (a changed
# operating recommendation, a flipped determinability/spec-met status, or |ΔP(meet)| over ``dp_meet``) forces
# a FAIL regardless of the mean shift.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "rel_wd": 0.20,               # |Δworst_dec| / worst_dec_A                 (FAIL)
    "rel_covfrob": 0.25,          # ‖C_arm − C_A‖_F / ‖C_A‖_F                  (FAIL)
    "dp_meet": 0.05,              # |P(meet)_arm − P(meet)_A|                  (FAIL)
    "meanshift_over_std": 1.0,    # max_q |Δg_q| / decision_std_A[q]           (FAIL)  primary, posterior-relative
    "meanshift_tw_fail": 0.50,    # max_q |Δg_q| / tol_q                       (FAIL)  large operational shift
    "meanshift_tw_warn": 0.25,    # max_q |Δg_q| / tol_q                       (WARN)  moderate operational shift
    "rel_rmse": 0.05,             # (RMSE_arm − RMSE_A) / RMSE_A               (WARN)  fit-equivalence
}

_JITTER = 1e-9


# --------------------------------------------------------------------------- compression basis (guarded)

def common_mode_direction(n: int) -> np.ndarray:
    """The fixed physical common-mode σ direction ``(1,…,1)/‖·‖`` (a uniform steric shift).

    This is the compression basis of variant D. It is a PRE-CHOSEN physical direction — NOT an eigenvector
    of the posterior covariance or the σ-block Fisher. Per paper App. B / ``decision_null_theorem.md`` this
    common mode is comparatively *identified* (it lowers the shared available charge and moves all peaks),
    so keeping it (and freezing the sloppy differential σ) is the physically-motivated compression; a
    Fisher-eigenvector basis would instead certify the stiff direction and is deliberately avoided.
    """
    v = np.ones(int(n), dtype=float)
    return v / np.linalg.norm(v)


def _blocks(n: int) -> dict[str, np.ndarray]:
    """Index arrays of the four u-blocks (keq, kkin, ν, σ), each length ``n``."""
    idx = np.arange(4 * n)
    return {"keq": idx[0:n], "kkin": idx[n:2 * n], "nu": idx[2 * n:3 * n], "sigma": idx[3 * n:4 * n]}


# --------------------------------------------------------------------------- arm specification

class _Arm:
    """One ablation arm as an affine map ``u = A·u_free + c``.

    ``A`` (4n×k), ``c`` (4n,) holds the frozen values, ``free_init`` (k,) warm-starts the refit from the
    full-model MAP.
    """

    def __init__(self, name, A, c, free_init):
        self.name = name
        self.A = np.asarray(A, float)
        self.c = np.asarray(c, float)
        self.free_init = np.asarray(free_init, float)

    @property
    def k(self) -> int:
        return self.A.shape[1]


def build_arm(name: str, n: int, u_map_A: np.ndarray, prior) -> _Arm:
    """Build the affine spec for one arm, warm-started from the full-model MAP ``u_map_A``."""
    u_map_A = np.asarray(u_map_A, float)
    dim = 4 * n
    blk = _blocks(n)
    I = np.eye(dim)
    nominal = np.asarray(prior.mean, float)   # prior mean = nominal σ / ν / keq

    def _select(keep_idx, frozen_block, frozen_value):
        """Freeze one block at ``frozen_value``, keep the other 3 blocks free (selection A)."""
        keep = np.asarray(keep_idx, int)
        A = I[:, keep]
        c = np.zeros(dim)
        c[frozen_block] = frozen_value[frozen_block]
        return A, c, u_map_A[keep]

    if name in ("A_full", "C_sigma_prior"):
        # A: identity. C (prior-only σ): also identity — its σ-Fisher is dropped in `_at_map_cov` (not by a
        # reparametrization), and it reuses arm A's MAP (no refit).
        return _Arm(name, I, np.zeros(dim), u_map_A)

    if name == "D_sigma_commonmode":
        # Free = [keq, kkin, ν, δ]; σ = σ_MAP (differential frozen) + δ·common_mode (identified d.o.f.).
        non_sigma = np.concatenate([blk["keq"], blk["kkin"], blk["nu"]])
        v = common_mode_direction(n)                    # the fixed (1,…,1) basis — NOT an eigenvector
        A = np.zeros((dim, 3 * n + 1))
        A[:, :3 * n] = I[:, non_sigma]
        A[blk["sigma"], 3 * n] = v                       # δ column: common-mode σ shift
        c = np.zeros(dim)
        c[blk["sigma"]] = u_map_A[blk["sigma"]]          # differential σ frozen at MAP
        free_init = np.concatenate([u_map_A[non_sigma], [0.0]])   # δ starts at 0 (σ = σ_MAP)
        return _Arm(name, A, c, free_init)

    non = {"keq": np.concatenate([blk["kkin"], blk["nu"], blk["sigma"]]),
           "kkin": np.concatenate([blk["keq"], blk["nu"], blk["sigma"]]),
           "nu": np.concatenate([blk["keq"], blk["kkin"], blk["sigma"]]),
           "sigma": np.concatenate([blk["keq"], blk["kkin"], blk["nu"]])}

    if name == "B_sigma_map":
        A, c, fi = _select(non["sigma"], blk["sigma"], u_map_A); return _Arm(name, A, c, fi)
    if name == "E_sigma_nominal":
        A, c, fi = _select(non["sigma"], blk["sigma"], nominal); return _Arm(name, A, c, fi)
    if name == "NC_nu_map":
        A, c, fi = _select(non["nu"], blk["nu"], u_map_A); return _Arm(name, A, c, fi)
    if name == "NC_keq_map":
        A, c, fi = _select(non["keq"], blk["keq"], u_map_A); return _Arm(name, A, c, fi)
    if name == "NC_nu_nominal":
        A, c, fi = _select(non["nu"], blk["nu"], nominal); return _Arm(name, A, c, fi)
    if name == "NC_keq_nominal":
        A, c, fi = _select(non["keq"], blk["keq"], nominal); return _Arm(name, A, c, fi)

    raise ValueError(f"unknown arm {name!r}")


# --------------------------------------------------------------------------- fit + laplace (reduced)

def _map_fit_reduced(predict_fn, obs, prior, arm: _Arm, *, sigma_obs, iters, lr) -> np.ndarray:
    """Adam MAP over the free coords; returns the reassembled full ``u`` (``A·u_free+c``).

    Identical loss to :func:`map_fit` (``−loglik − log_prior``) evaluated on the reassembled ``u`` — so
    with ``A=I, c=0`` it reproduces the full MAP; a warm start from ``u_map_A`` keeps the refit cheap.
    """
    A = torch.tensor(arm.A, dtype=DTYPE)
    c = torch.tensor(arm.c, dtype=DTYPE)
    u_free = torch.tensor(arm.free_init, dtype=DTYPE, requires_grad=True)
    opt = torch.optim.Adam([u_free], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        u_full = A @ u_free + c
        loss = -gaussian_loglik(u_full, predict_fn, obs, sigma_obs) - prior.log_prob(u_full)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (A @ u_free.detach() + c).numpy()


def _jacobian_full(predict_fn, u_full) -> np.ndarray:
    """Reverse-mode ``J = ∂predict/∂u`` (n_resid × 4n) at ``u_full`` — forward/double-backward are wrong
    through the detached-IFT solver (``posterior._ggn_hessian``)."""
    u = torch.tensor(np.asarray(u_full, float), dtype=DTYPE)
    return torch.autograd.functional.jacobian(predict_fn, u).detach().numpy()


def _laplace_reduced(arm: _Arm, prior, J_full, u_full, sigma_obs):
    """Reduced Laplace at a refit MAP: ``H_free = J_freeᵀJ_free/σ² + AᵀPA``, ``Σ_full = A·H_free⁻¹·Aᵀ``.

    Used by the @nominal refit arms (frozen at a non-MAP value, so ``J`` is evaluated at the new MAP).
    Returns ``(Sigma_full, H_free)``.
    """
    A = arm.A
    P = prior.precision()
    J_free = np.asarray(J_full, float) @ A
    H_free = (J_free.T @ J_free) / (sigma_obs ** 2) + A.T @ P @ A
    H_free = 0.5 * (H_free + H_free.T) + _JITTER * np.eye(H_free.shape[0])
    cov_free = np.linalg.inv(H_free)
    cov_free = 0.5 * (cov_free + cov_free.T)
    Sigma_full = A @ cov_free @ A.T
    return Sigma_full, H_free


# --------------------------------------------------------------------------- per-arm metrics

def _fit_metrics(predict_fn, obs, u_full, k, sigma_obs) -> dict:
    """χ²/dof + RMSE of the MAP curve (``resid = predict_fn(u_map) − obs``)."""
    with torch.no_grad():
        pred = predict_fn(torch.tensor(np.asarray(u_full, float), dtype=DTYPE))
    resid = (pred - obs).detach().numpy()
    n_obs = int(resid.size)
    chi2 = float(np.sum((resid / sigma_obs) ** 2))
    dof = max(n_obs - int(k), 1)
    return {"chi2": chi2, "dof": dof, "chi2_dof": chi2 / dof, "n_obs": n_obs,
            "rmse_g_l": float(empirical_sigma_obs(resid)), "n_free": int(k)}


def _decision_metrics(post, bundle, op, tol, spec, *, n_steps, precomp=None, mc=False, n_mc=200,
                      seed=0) -> dict:
    """worst_dec / decision covariance / g_map / P(meet) at ``op`` (+ MC cross-check when ``mc``).

    ``precomp = (G, g_map_arr, window)`` supplies a decision Jacobian already computed at ``post.u_map``
    (shared by the freeze-at-MAP arms, whose MAP equals arm A's), avoiding a redundant solver pass; when
    ``None`` the Jacobian is recomputed via :func:`decision_report`.
    """
    if precomp is not None:
        G, g_map_arr, window = precomp
        C0 = decision_covariance(G, post.cov)
        rep = _report_from_covariance(C0, g_map_arr, tol=tuple(tol), window=window, op=op)
    else:
        rep = decision_report(post, bundle, op, tol=tuple(tol), n_steps=n_steps)
    names = rep["names"]
    g_map = np.array([rep["g_map"][nm] for nm in names], float)
    C = np.array(rep["C"], float)
    out = {
        "worst_dec": rep["worst_dec"], "met": rep["met"],
        "g_map": rep["g_map"], "decision_std": rep["decision_std"],
        "p_meet": float(gaussian_meet_prob(g_map, C, tuple(spec))),
        "C": C.tolist(),
    }
    if mc:
        from cex_model.bayes.decision import decision_covariance_mc
        from cex_model.bayes.decision_window import worst_dec_from_cov
        m = decision_covariance_mc(post, bundle, op, n_samples=n_mc, seed=seed,
                                   n_steps=n_steps, return_gs=True)
        gs = m.get("gs", [])
        mr = meet_prob_and_regret(gs, spec=tuple(spec), tol=tuple(tol))
        C_mc = np.array(m["C_mc"], float)
        out["mc"] = {"worst_dec": worst_dec_from_cov(C_mc, tol) if np.all(np.isfinite(C_mc)) else float("nan"),
                     "p_meet": mr["p_meet"], "n_used": mr["n_used"],
                     "rel_frobenius": (float(np.linalg.norm(C - C_mc) / max(np.linalg.norm(C_mc), 1e-30))
                                       if np.all(np.isfinite(C_mc)) else float("nan"))}
    return out


def _compute_benefit(H_free, prior, arm: _Arm, runtime_s) -> dict:
    """Runtime, Hessian condition number, and posterior rank (informed directions) for one arm."""
    eig = np.linalg.eigvalsh(0.5 * (H_free + H_free.T))
    lam_min = float(eig[0])
    cond = float(eig[-1] / lam_min) if lam_min > 0 else float("inf")
    # prior-whitened FREE posterior shrinkage (AᵀPA is diagonal for every arm here) -> informed rank
    P_free = arm.A.T @ prior.precision() @ arm.A
    prior_std_free = 1.0 / np.sqrt(np.clip(np.diag(P_free), 1e-300, None))
    cov_free = np.linalg.inv(0.5 * (H_free + H_free.T))
    d = 1.0 / prior_std_free
    M = cov_free * d[:, None] * d[None, :]
    s = np.sqrt(np.clip(np.linalg.eigvalsh(0.5 * (M + M.T)), 0.0, None))   # per-direction shrinkage
    return {
        "runtime_s": float(runtime_s),
        "nominal_dim": int(arm.k),
        "hessian_cond": cond,
        "hessian_min_eig": lam_min,
        "worst_dir": float(s.max()),                          # prior-whitened worst direction (identifiability)
        "informed_rank": int(np.sum(s < 0.9)),                # directions the data constrains (shrinkage<0.9)
    }


# --------------------------------------------------------------------------- verdict

def _spec_met(dec, floor: float = 0.5) -> bool:
    """Binary spec-meeting status of a decision (more-likely-than-not to meet the joint spec)."""
    return bool(np.isfinite(dec["p_meet"]) and dec["p_meet"] >= floor)


def verdict(arm_dec, base_dec, arm_fit, base_fit, tol, thr, *, spec=None, rec_changed=False) -> dict:
    """Classify one arm: FAIL_DECISION_CHANGED > WARN_FIT_CHANGED > PASS_DECISION_EQUIVALENT.

    Decision dominates fit. **Primary hard FAIL** on any real decision change: a changed operating
    recommendation, a flipped determinability (``worst_dec < τ``) or spec-met (``P(meet) ≥ 0.5``) status,
    ``|ΔP(meet)| > dp_meet``, a covariance/worst_dec move, a mean shift beyond the decision's own posterior
    std, or a large tolerance-whitened mean shift (``> meanshift_tw_fail``). **Secondary WARN** (equivalent
    decision, with caveats): a degraded fit, a moderate tolerance-whitened mean shift (``> meanshift_tw_warn``),
    or -- when ``spec`` is given -- the MAP quantity ``g_MAP`` crossing a purity/yield spec threshold while
    P(meet)/recommendation/determinability all hold (a record-only crossing, deliberately NOT a hard FAIL).
    ``reasons`` records which guards fired.
    """
    tol = np.asarray(tol, float)
    rel_wd = abs(arm_dec["worst_dec"] - base_dec["worst_dec"]) / max(abs(base_dec["worst_dec"]), 1e-30)
    C_a, C_b = np.array(arm_dec["C"], float), np.array(base_dec["C"], float)
    rel_covfrob = float(np.linalg.norm(C_a - C_b) / max(np.linalg.norm(C_b), 1e-30))
    dp_meet = abs(arm_dec["p_meet"] - base_dec["p_meet"])
    names = list(base_dec["g_map"].keys())
    dg = {nm: abs(arm_dec["g_map"][nm] - base_dec["g_map"][nm]) for nm in names}
    meanshift_tw = float(max(dg[nm] / tol[i] for i, nm in enumerate(names)))          # requested metric
    std = base_dec["decision_std"]
    meanshift_over_std = float(max(dg[nm] / max(std[nm], 1e-9) for nm in names))       # primary gate
    rel_rmse = (arm_fit["rmse_g_l"] - base_fit["rmse_g_l"]) / max(base_fit["rmse_g_l"], 1e-30)
    det_met_flip = bool(arm_dec.get("met") != base_dec.get("met"))
    spec_met_flip = bool(_spec_met(arm_dec) != _spec_met(base_dec))
    # record-only: does the MAP quantity itself cross a purity/yield spec threshold?
    g_map_spec_cross = False
    if spec is not None:
        g_map_spec_cross = any((base_dec["g_map"][nm] >= spec[i]) != (arm_dec["g_map"][nm] >= spec[i])
                               for i, nm in enumerate(names))

    fail_reasons = []
    if rec_changed:
        fail_reasons.append("recommendation_changed")
    if det_met_flip:
        fail_reasons.append("determinability_met_flip")
    if spec_met_flip:
        fail_reasons.append("spec_met_flip")
    if dp_meet > thr["dp_meet"]:
        fail_reasons.append("p_meet")
    if meanshift_over_std > thr["meanshift_over_std"]:
        fail_reasons.append("meanshift_over_std")
    if meanshift_tw > thr["meanshift_tw_fail"]:
        fail_reasons.append("meanshift_tw")
    if rel_wd > thr["rel_wd"]:
        fail_reasons.append("worst_dec")
    if rel_covfrob > thr["rel_covfrob"]:
        fail_reasons.append("cov_frobenius")

    warn_reasons = []
    if rel_rmse > thr["rel_rmse"]:
        warn_reasons.append("fit_rmse")
    if meanshift_tw > thr["meanshift_tw_warn"]:
        warn_reasons.append("meanshift_tw_moderate")
    if g_map_spec_cross:
        warn_reasons.append("g_map_spec_cross")

    label = ("FAIL_DECISION_CHANGED" if fail_reasons
             else "WARN_FIT_CHANGED" if warn_reasons
             else "PASS_DECISION_EQUIVALENT")
    # The ACTIONABLE decision (what a process engineer does) = the operating recommendation, the spec-met
    # status, and the determinability verdict. It is unchanged iff none of those three DISCRETE outcomes
    # flipped -- separate from the CONTINUOUS precision magnitude (worst_dec/cov Frobenius), which can cross
    # its threshold (hence FAIL) while the outcome holds, especially on a tight-decision product where a
    # small absolute cov change is a large relative one (the §3.7 small-denominator effect).
    outcome_equivalent = not (rec_changed or det_met_flip or spec_met_flip)
    return {"verdict": label,
            "decision_outcome_equivalent": bool(outcome_equivalent),
            "reasons": {"fail": fail_reasons, "warn": warn_reasons},
            "decision_deltas": {"rel_worst_dec": float(rel_wd), "rel_cov_frobenius": rel_covfrob,
                                "dp_meet": float(dp_meet), "meanshift_tw": meanshift_tw,
                                "meanshift_over_std": meanshift_over_std,
                                "determinability_met_flip": det_met_flip, "spec_met_flip": spec_met_flip,
                                "g_map_spec_cross": bool(g_map_spec_cross),
                                "recommendation_changed": bool(rec_changed)},
            "fit_deltas": {"rel_rmse": float(rel_rmse),
                           "d_chi2": float(arm_fit["chi2"] - base_fit["chi2"])}}


# --------------------------------------------------------------------------- recommendation (optional)

def _psd_regularize(cov: np.ndarray) -> np.ndarray:
    """Floor the eigenvalues of ``cov`` to a tiny fraction of its top eigenvalue so it is invertible.

    A freeze/compress arm's full-dim ``Σ = A·cov_free·Aᵀ`` is rank-deficient (frozen dims have zero
    variance; variant D's σ-block is rank-1), which makes ``decision_voi``'s ``inv(Σ)`` (the preposterior
    precision) fail. Flooring the null eigenvalues gives the frozen directions a near-infinite precision --
    exactly right: a frozen parameter is *known*, so no candidate experiment can inform it -- while leaving
    the data-informed (free) directions untouched. Decision quantities ``C = G Σ Gᵀ`` are unchanged to the
    floor (1e-12 of the top eigenvalue).
    """
    cov = np.asarray(cov, float)
    w, V = np.linalg.eigh(0.5 * (cov + cov.T))
    floor = max(float(w.max()), 1e-300) * 1e-12
    return (V * np.clip(w, floor, None)) @ V.T


def _recommendation(post, bundle, op, tol, spec, *, n_steps, n_candidates, seed) -> str:
    """Four-state operating recommendation (operate_as_is / move_operating_point / redesign_pool /
    take_data[_low_value]) via the §2.12 deliverable — gated (grid of ops per arm is Colab-heavy).

    The freeze/compress arms yield a rank-deficient posterior; :func:`_psd_regularize` makes it invertible
    for ``decision_voi`` (which needs ``inv(Σ)``) without changing the decision quantities.
    """
    from dataclasses import replace

    from cex_model.bayes.decision_window import (
        candidate_ops_for,
        decision_voi,
        operating_window_map,
        synthesize_recommendation,
    )
    post = replace(post, cov=_psd_regularize(post.cov))
    ops = [list(op)] + list(candidate_ops_for(bundle, n_candidates=n_candidates, seed=seed))
    owm = operating_window_map(post, bundle, ops, spec=tuple(spec), tol=tuple(tol),
                               method="gauss", n_steps=n_steps)
    voi = decision_voi(post, bundle, list(op), ops[1:], spec=tuple(spec), tol=tuple(tol), n_steps=n_steps)
    return synthesize_recommendation(owm, voi)["action"]


# --------------------------------------------------------------------------- driver

# Freeze-at-MAP arms: their conditional MAP equals arm A's MAP (σ/ν/keq already at the joint optimum), so
# they need NO solver -- the reduced posterior is a block of the committed Hessian H_A = Σ_A⁻¹. Only the
# @nominal arms (E, NC_*_nominal), frozen at a NON-MAP value, force a genuine refit.
_AT_MAP_ARMS: tuple[str, ...] = ("B_sigma_map", "C_sigma_prior", "D_sigma_commonmode",
                                 "NC_nu_map", "NC_keq_map")


def _at_map_cov(name, arm, H_A, F_A, prior, n):
    """``(Σ_full, H_free)`` for a freeze-at-MAP arm from the full Hessian ``H_A`` -- pure linear algebra.

    B/D/NC_*_map = conditioning: the reduced Hessian is the projection ``Aᵀ H_A A`` (a block of ``H_A``),
    ``Σ_full = A·(Aᵀ H_A A)⁻¹·Aᵀ``. C (prior-only σ) = drop σ's data information: zero the σ rows/cols of the
    data Fisher ``F_A = H_A − P`` and re-add the prior precision, so σ collapses to its (independent) prior.
    """
    if name == "C_sigma_prior":
        s = _blocks(n)["sigma"]
        Hc = np.array(F_A, float)
        Hc[s, :] = 0.0
        Hc[:, s] = 0.0
        Hc = Hc + prior.precision()
        H_free = 0.5 * (Hc + Hc.T) + _JITTER * np.eye(Hc.shape[0])
        Sigma_full = np.linalg.inv(H_free)
        return 0.5 * (Sigma_full + Sigma_full.T), H_free
    A = arm.A
    H_free = A.T @ H_A @ A
    H_free = 0.5 * (H_free + H_free.T) + _JITTER * np.eye(H_free.shape[0])
    cov_free = np.linalg.inv(H_free)
    Sigma_full = A @ (0.5 * (cov_free + cov_free.T)) @ A.T
    return 0.5 * (Sigma_full + Sigma_full.T), H_free


def run_ablation(bundle, prior, n, *, op, tol=(0.02, 0.05), spec=(0.70, 0.50), arms=ARMS,
                 n_steps=120, map_iters=150, lr=0.05, sigma_obs=AKTA_NOISE_FLOOR_G_L,
                 thresholds=None, recommend=False, mc=False, n_mc=200, n_candidates=12,
                 seed=0, product="?", targets=None, post_A=None, verbose=False) -> dict:
    """Run the σ freeze/compress ablation on one product (or synthetic bundle).

    Establishes arm A (full model) -- either **reused** from a committed ``post_A`` (a
    ``{product}_posterior.npz`` Laplace posterior; no re-fit, and arm A byte-matches the paper) or fit
    fresh -- then each arm. The freeze-at-MAP arms (:data:`_AT_MAP_ARMS`) are pure linear algebra on the
    committed Hessian and share arm A's decision Jacobian (no solver); only the @nominal arms (E, NC_*_nominal)
    re-fit. Reports fit-equivalence, decision-equivalence and computational-benefit metrics + a per-arm
    verdict, all against arm A. Returns a JSON-ready dict.

    ``targets`` (optional) supplies a pre-built :class:`ExperimentTarget` list (the synthetic path, which
    has no measured ``.curve``); when ``None`` the targets are read from the bundle's real curves.
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if targets is None:
        targets = targets_from_bundle(bundle, n_steps=n_steps)
    predict_fn, obs = mechanistic_model(targets, n)
    tmpl = bundle.components
    P = prior.precision()

    # ----- arm A (baseline): reuse a committed posterior, or fit fresh -----
    t0 = time.time()
    if post_A is not None:
        u_map_A = np.asarray(post_A.u_map, float)
        cov_A = np.asarray(post_A.cov, float)
        H_A = 0.5 * (np.linalg.inv(cov_A) + np.linalg.inv(cov_A).T)
        reused = True
    else:
        u_map_A, _ = map_fit(components_to_u(tmpl), predict_fn, obs, prior,
                             sigma_obs=sigma_obs, iters=map_iters, lr=lr, progress=verbose)
        J_A = _jacobian_full(predict_fn, u_map_A)
        H_A = (J_A.T @ J_A) / sigma_obs**2 + P
        H_A = 0.5 * (H_A + H_A.T) + _JITTER * np.eye(H_A.shape[0])
        cov_A = 0.5 * (np.linalg.inv(H_A) + np.linalg.inv(H_A).T)
        reused = False
    F_A = H_A - P                                          # data GGN/Fisher (for variant C)
    rt_A = time.time() - t0

    # shared decision Jacobian at u_map_A -- reused by every freeze-at-MAP arm (their MAP == u_map_A)
    G_A, g_map_A, sel_A = decision_jacobian(bundle, op, u_map_A, n_steps=n_steps, return_extra=True)
    precomp_A = (G_A, g_map_A, sel_A)

    armA = build_arm("A_full", n, u_map_A, prior)
    post_A_obj = Posterior.from_template(mean=u_map_A, cov=cov_A, u_map=u_map_A, prior=prior,
                                         template=tmpl, sigma_obs=sigma_obs, engine="ablation:A_full")
    fit_A = _fit_metrics(predict_fn, obs, u_map_A, armA.k, sigma_obs)
    dec_A = _decision_metrics(post_A_obj, bundle, op, tol, spec, n_steps=n_steps, precomp=precomp_A,
                              mc=mc, n_mc=n_mc, seed=seed)
    ben_A = _compute_benefit(H_A, prior, armA, rt_A)
    rec_A = (_recommendation(post_A_obj, bundle, op, tol, spec, n_steps=n_steps,
                             n_candidates=n_candidates, seed=seed) if recommend else None)
    results = {"A_full": {"fit": fit_A, "decision": dec_A, "benefit": ben_A, "recommendation": rec_A,
                          "verdict": "BASELINE", "reused_posterior": reused}}
    if verbose:
        print(f"  A_full        worst_dec={dec_A['worst_dec']:.3f}  P(meet)={dec_A['p_meet']:.3f}  "
              f"RMSE={fit_A['rmse_g_l']:.4f}  cond={ben_A['hessian_cond']:.2e}  rec={rec_A}"
              f"{'  (reused posterior)' if reused else ''}")

    # ----- the remaining arms -----
    for name in arms:
        if name == "A_full":
            continue
        arm = build_arm(name, n, u_map_A, prior)
        t0 = time.time()
        if name in _AT_MAP_ARMS:                          # no solver: block algebra on the committed Hessian
            Sigma_full, H_free = _at_map_cov(name, arm, H_A, F_A, prior, n)
            u_full = u_map_A
            precomp = precomp_A                            # same MAP -> same decision Jacobian
        else:                                              # E / NC_*_nominal: genuine refit (frozen at nominal)
            u_full = _map_fit_reduced(predict_fn, obs, prior, arm, sigma_obs=sigma_obs, iters=map_iters, lr=lr)
            J = _jacobian_full(predict_fn, u_full)
            Sigma_full, H_free = _laplace_reduced(arm, prior, J, u_full, sigma_obs)
            precomp = None
        rt = time.time() - t0
        post = Posterior.from_template(mean=u_full, cov=Sigma_full, u_map=u_full, prior=prior,
                                       template=tmpl, sigma_obs=sigma_obs, engine=f"ablation:{name}")
        fit = _fit_metrics(predict_fn, obs, u_full, arm.k, sigma_obs)
        dec = _decision_metrics(post, bundle, op, tol, spec, n_steps=n_steps, precomp=precomp,
                                mc=mc, n_mc=n_mc, seed=seed)
        ben = _compute_benefit(H_free, prior, arm, rt)
        rec = (_recommendation(post, bundle, op, tol, spec, n_steps=n_steps,
                               n_candidates=n_candidates, seed=seed) if recommend else None)
        rec_changed = bool(recommend and rec_A is not None and rec != rec_A)
        vd = verdict(dec, dec_A, fit, fit_A, tol, thr, spec=spec, rec_changed=rec_changed)
        results[name] = {"fit": fit, "decision": dec, "benefit": ben, "recommendation": rec, **vd}
        if verbose:
            print(f"  {name:14s} worst_dec={dec['worst_dec']:.3f}  P(meet)={dec['p_meet']:.3f}  "
                  f"RMSE={fit['rmse_g_l']:.4f}  cond={ben['hessian_cond']:.2e}  rec={rec}  -> {vd['verdict']}")

    # ----- negative-control verdict -----
    # Power test = the @nominal controls (the wrong-value operation E also applies to σ): they MUST fail.
    # The @MAP controls are reported as-is -- they typically pass via keq<->nu ridge compensation (a
    # corroboration of the tight identified ridge, NOT a weak test), so they do not gate the verdict.
    nominal = [name for name in ("NC_nu_nominal", "NC_keq_nominal") if name in results]
    map_ctrls = {name: results[name]["verdict"] for name in ("NC_nu_map", "NC_keq_map") if name in results}
    nominal_fail = [name for name in nominal if results[name]["verdict"] == "FAIL_DECISION_CHANGED"]
    if nominal:
        nc_verdict = "CONTROLS_DISCRIMINATE" if len(nominal_fail) == len(nominal) else "WEAK_TEST"
    elif map_ctrls:
        nc_verdict = "MAP_CONTROLS_ONLY"
    else:
        nc_verdict = "NO_CONTROLS"

    return {
        "product": product, "n_protein": n, "n_steps": n_steps, "map_iters": map_iters,
        "op": [float(x) for x in op], "tol": [float(x) for x in tol], "spec": [float(x) for x in spec],
        "thresholds": thr, "arms": results, "reused_posterior": bool(post_A is not None),
        "negative_control_verdict": nc_verdict,
        "nominal_controls_failed": nominal_fail,
        "map_controls": map_ctrls,
        "sigma_arms_pass": [name for name in results
                            if name.startswith(("B_", "C_", "D_", "E_"))
                            and results[name]["verdict"] == "PASS_DECISION_EQUIVALENT"],
        "sigma_arms_outcome_equivalent": [name for name in results
                                          if name.startswith(("B_", "C_", "D_", "E_"))
                                          and results[name].get("decision_outcome_equivalent")],
    }


# --------------------------------------------------------------------------- product / synthetic loaders

def run_product(product, *, in_dir="results/bayes", n_steps=120, map_iters=150, arms=ARMS,
                recommend=False, mc=False, n_candidates=12, seed=0, reuse_posterior=True,
                verbose=False) -> dict:
    """Ablation for a real product at its committed decision OP + tol (``{product}_decision.json``).

    ``reuse_posterior=True`` (default) loads the committed ``{product}_posterior.npz`` for arm A -- so arm
    A byte-matches the paper and the five freeze-at-MAP arms need no solver (block algebra on the committed
    Hessian). Use ``--n-steps`` matching the committed fit (the production posteriors are n_steps=300) so
    arm A and the @nominal refits share the same forward-model discretization. Falls back to a fresh fit if
    the ``.npz`` is absent.
    """
    import json
    from pathlib import Path

    import cex_model.app_support as A
    from cex_model.bayes.compare import filter_experiments
    from cex_model.bayes.groenwall import _PRODUCT_MAP

    in_dir = Path(in_dir)
    dec = json.loads((in_dir / f"{product}_decision.json").read_text())
    op = [float(x) for x in dec["decision_op"]]
    tol = tuple(dec.get("tol", (0.02, 0.05)))
    prod, drop = _PRODUCT_MAP.get(product, (product, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
    n = bundle.components.n_protein
    prior = physical_prior(n)
    post_A = None
    if reuse_posterior:
        npz = in_dir / f"{product}_posterior.npz"
        if npz.exists():
            post_A = Posterior.load(npz)
        elif verbose:
            print(f"  [warn] {npz} absent -> fitting arm A fresh")
    return run_ablation(bundle, prior, n, op=op, tol=tol, arms=arms, n_steps=n_steps, map_iters=map_iters,
                        recommend=recommend, mc=mc, n_candidates=n_candidates, seed=seed,
                        product=product, post_A=post_A, verbose=verbose)


def run_synthetic(*, n_comp=3, fractions=(18.0, 70.0, 12.0), keq_ladder=True, nu=None, sigma=None,
                  loading=10.0, n_steps=40, n_points=60, n_exp=3, map_iters=120, arms=ARMS,
                  recommend=False, mc=False, n_candidates=8, seed=0, verbose=False) -> dict:
    """Ablation on a synthetic SMA bundle (CI-friendly).

    Simulates ``n_exp`` noisy experiments at spread-out operating conditions under the ground-truth ``u``
    (via :func:`simulate_target`), so σ is genuinely partly-informed as on a real product; the decision OP
    is the first (highest-loading) condition. Default ``n_comp=3, fractions=[18,70,12]`` gives a proper
    acid/main/basic split for the pooled-purity decision (matching the decision unit tests).
    """
    from cex_model.bayes.active import simulate_target
    from cex_model.bayes.synthetic import synthetic_sma_bundle

    bundle, comps, u_true = synthetic_sma_bundle(n_comp, fractions=list(fractions), keq_ladder=keq_ladder,
                                                 nu=nu, sigma=sigma, loading=loading)
    n = comps.n_protein
    e0 = bundle.experiments[0]
    base = [e0.loading_g_l, e0.gradient_start_pct, e0.gradient_end_pct, e0.elution_cv]
    # a small spread of operating conditions (loading + gradient), decision OP = the first
    deltas = [(0.0, 0.0, 0.0), (4.0, 5.0, 5.0), (-3.0, -5.0, -5.0), (6.0, 10.0, 0.0)][:max(1, n_exp)]
    ops = [[base[0] + dl, base[1] + dg0, base[2] + dg1, base[3]] for (dl, dg0, dg1) in deltas]
    targets = [simulate_target(bundle, op, u_true, n_steps=n_steps, n_points=n_points,
                               sigma_obs=AKTA_NOISE_FLOOR_G_L, seed=seed + i) for i, op in enumerate(ops)]
    prior = physical_prior(n)
    return run_ablation(bundle, prior, n, op=ops[0], tol=(0.02, 0.05), arms=arms, n_steps=n_steps,
                        map_iters=map_iters, recommend=recommend, mc=mc, n_candidates=n_candidates,
                        seed=seed, product=f"SYN{n_comp}", targets=targets, verbose=verbose)
