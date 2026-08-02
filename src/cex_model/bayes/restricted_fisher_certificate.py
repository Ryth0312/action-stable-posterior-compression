"""σ-forcing-restricted finite-horizon OBSERVATION gain — a NON-circular a-priori upper bound on the
σσ **Fisher** block, closing the Fisher-null (observation) side of Theorem 1's dual
(docs/decision_null_theorem.md §7-1).  The observation twin of ``restricted_sigma_gain.py`` (state side).

Theorem 1's headline is a dual: the steric factor σ is *simultaneously* (i) **Fisher-null** ``O(ε²)`` (so σ
is unidentified, ``worst_dir_σ → 1``) and (ii) **decision-null** ``O(ε)``.  ``restricted_sigma_gain`` bounds
the **full-state** sensitivity ``‖S_σ(τ)‖`` at the final time (Prop 1M magnitude); the ratio law
(``decision_null_ratio``) witnesses (ii).  The Fisher-null leg (i) — the one that makes σ *unidentified* —
had **no** dedicated non-circular a-priori bound (only the unconditional constant-free scaling + the
measured ``κ_eff`` + the ratio law reproducing the σ-share).  This module supplies it.

**Object.** The data Fisher information of σ is ``F_σσ = J_σᵀ J_σ / σ_obs²`` with ``J_σ = ∂y_obs/∂σ`` the
σ-sensitivity of the OBSERVED elution curve, i.e. ``C_obs S_σ(t_k)`` summed over the observation times
(``C_obs`` = the outlet observation operator; ``S_σ`` = the full-state variational sensitivity).  This is a
different output of the *same* fixed propagator Φ than ``restricted_sigma_gain`` bounds: observation-over-time
instead of full-state@τ.

**Bound (non-circular, no full-state Φ norm).** Since ``λmax(Σ_k S_kᵀ S_k) = ‖J_σ‖₂² ≤ ‖J_σ‖_F² =
Σ_k ‖S_k‖_F²`` and each ``‖S_k‖_F = ‖C_obs S_σ(t_k)‖ = ‖C_obs ∫₀^{t_k} Φ F_σ ds‖ ≤ ∫₀^{t_k}‖C_obs Φ F_σ‖ds
=: b_k`` (triangle inequality, norm INSIDE the Duhamel integral — an inequality, not the exact integral, so
non-circular, and using **no** full-state Φ norm, only the low-dim σ-forcing propagated then observed):

    λmax(F_σσ)  ≤  (1/σ_obs²) Σ_k b_k²  =  (ε · κ_obs,σ^reach)² / σ_obs²  =: F_σσ_bound ,
    κ_obs,σ^reach := sqrt(Σ_k b_k²) / ε   (the normalized σ-restricted observation gain).

**Shrinkage (honest).**  With a uniform σ prior (std ``σ_prior``) and Schur ``(H⁻¹)_σσ ⪰ (H_σσ)⁻¹``,
``H_σσ = F_σσ + I/σ_prior²`` gives the a-priori LOWER bound on the σ-block worst-direction shrinkage

    worst_dir_σ  ≥  1 / sqrt(1 + σ_prior² · F_σσ_bound)  =: posterior_shrinkage_bound  →  1  as ε → 0 .

Because ``F_σσ_bound ∝ ε²``, this lower bound **rises to 1 at the Fisher-null O(ε²) rate** as ε → 0 — the
scaling that makes σ unidentified.  At the *finite* operating ε it is **loose**: the σ block is not uniformly
null (its most-identified direction has ``λmax(F_σσ) ≫ λmin(F_σσ)``, so a λmax-based bound cannot pin the
sloppiest direction).  We therefore report it honestly and gate the verdict on the **tight Fisher-magnitude
bound** ``bound/exact`` (the Fisher-null scaling), not on the shrinkage clearing a threshold.  The committed
``posterior_worst_dir_exact`` (the actual sloppiness) is reported alongside for context.

**Observation model & rigor.**  We observe the grouped protein outlet on the propagator's own solver grid;
the bound and the "exact" (propagator ``F_σσ_exact = λmax((1/σ_obs²)Σ_k S_kᵀS_k)``) use that *same* model, so
``bound/exact ≥ 1`` is a clean triangle slack (cross-checked against an independent finite-difference J_σ
Fisher — the analogue of ``restricted_sigma_gain``'s semi/FD check).  The solver grid is *denser* than the
real data sample times ⇒ more Fisher information ⇒ its ``worst_dir_σ`` ≤ the committed data-time
``worst_dir_σ``; hence ``posterior_shrinkage_bound`` is a conservative, a-fortiori lower bound on the
committed ``posterior_worst_dir_exact`` (denser-data monotonicity).

**Honest framing (same as the state gain).**  A REFRAME, not a closure: it does not close the open full-state
a-priori ``κ_S``; it certifies the σ-restricted, observation-restricted subspace that carries the Fisher-null.
Verdict: PASS_RESTRICTED_FISHER / WARN_USEFUL_DIAGNOSTIC / FAIL_VACUOUS.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sla
import torch

from cex_model.bayes.groenwall import _PRODUCT_MAP, _epsilon_from_traj, _forward_full_state
from cex_model.bayes.likelihood import AKTA_NOISE_FLOOR_G_L, unpack_u
from cex_model.bayes.restricted_sigma_gain import sigma_forcing
from cex_model.sma import MOL_TO_G_L

__all__ = [
    "observed_columns",
    "restricted_fisher_sweep",
    "restricted_fisher_diagnostic",
    "certify",
    "certify_synthetic",
]


# --------------------------------------------------------------------------- observation operator C_obs

def _obs_indices(sim) -> list[int]:
    """Flat state indices of the protein outlet nodes (last spatial node of each protein C-block),
    in protein order j=0..npr-1.  Block c (c=1..npr) occupies flat ``[c*gs:(c+1)*gs]`` ⇒ outlet ``(c+1)*gs-1``
    (torch_solver.integrate).  Salt (block 0) is produced by the solver but dropped by the likelihood."""
    gs = int(sim.gs)
    return [(c + 1) * gs - 1 for c in range(1, int(sim.npr) + 1)]


def observed_columns(S, obs_idx, groups, npr) -> np.ndarray:
    """Apply the (exactly linear) observation operator ``C_obs`` to state columns ``S`` (ndim, k):
    select the protein outlet rows, scale by ``MOL_TO_G_L``, then sum proteins per ``observation_groups``.

    Returns ``(n_obs, k)``.  ``C_obs`` commutes with the Duhamel time integral, so it can be applied to the
    propagated per-source blocks ``Φ F_σ`` directly."""
    S = np.asarray(S)
    prot = S[obs_idx, :] * MOL_TO_G_L                       # (npr, k) protein outlet sensitivities (g/L)
    grp = groups if groups is not None else [[j] for j in range(npr)]
    rows = [prot[list(g), :].sum(axis=0) for g in grp]      # sum protein columns per observation group
    return np.stack(rows, axis=0)                           # (n_obs, k)


def _trapezoid_weights(t_src: np.ndarray) -> np.ndarray:
    """Trapezoid Duhamel weights over the (possibly non-uniform) source-time grid (mirrors
    restricted_sigma_gain.restricted_gain_sweep)."""
    w = np.zeros_like(t_src)
    if t_src.size >= 2:
        w[1:-1] = 0.5 * (t_src[2:] - t_src[:-2])
        w[0] = 0.5 * (t_src[1] - t_src[0])
        w[-1] = 0.5 * (t_src[-1] - t_src[-2])
    elif t_src.size == 1:
        w[0] = 1.0
    return w


# --------------------------------------------------------------------------- the propagator sweep

def restricted_fisher_sweep(sim, keq, kkin, nu, sig, n, groups, *, stride: int = 1) -> dict:
    """One forward BDF-matched variational sweep that, at every step ``t_k``, snapshots the OBSERVED
    σ-sensitivity ``S_k = C_obs S_σ(t_k)`` and the norm-inside bound ``b_k = Σ_{m≤k}‖C_obs Φ(t_k,t_m)F_m‖ w_m``.

    The BDF variational propagation is byte-identical to ``restricted_sigma_gain.restricted_gain_sweep``
    (implicit-Euler BDF1 first step then BDF2 ``a0=1.5, hist=2V−0.5V_prev``, per-step LU over the stacked
    active source blocks, ``A_k = a0 I − h J_k``, ``J_k = sim.analytic_jac``); the only additions are the
    per-step observation snapshots.  ``S_σ(t_k) = Σ_{m≤k} G_m(t_k) w_m`` and ``b_k`` use the SAME per-source
    blocks/weights, so ``b_k ≥ ‖S_k‖_F`` holds by the triangle inequality at every step.

    Returns the accumulated observed Fisher matrix ``Σ_k S_kᵀ S_k`` (unscaled by σ_obs²), ``Σ_k b_k²``, and the
    full-state ``‖S_σ(τ)‖`` (impulse-sum, for the propagator FD cross-check)."""
    with torch.no_grad():
        Y = _forward_full_state(sim, keq, kkin, nu, sig)          # (nt, ndim)
    t = sim.t.detach().numpy()
    nt = len(t)
    ndim = int(Y.shape[1])
    npr = int(sim.npr)
    eye = np.eye(ndim)
    obs_idx = _obs_indices(sim)

    source_set = sorted(set(range(1, nt, max(1, stride))) | {nt - 1})   # strided source steps (t>0)
    t_src_full = np.array([t[m] for m in source_set])
    w_full = _trapezoid_weights(t_src_full)
    w_of_step = {m: float(w_full[i]) for i, m in enumerate(source_set)}

    Vs = np.zeros((ndim, 0)); Vs_prev = np.zeros((ndim, 0))    # stacked active blocks + 1-step-back value
    src_order: list[int] = []                                  # source step per n-column group (inject order)

    FTF = np.zeros((n, n))          # Σ_k (C_obs S_σ(t_k))ᵀ (C_obs S_σ(t_k))  = observed Fisher × σ_obs²
    bound_sq = 0.0                  # Σ_k b_k²
    S_final = np.zeros((ndim, n))   # ∫ Φ F_σ ds ≈ S_σ(τ) (impulse-sum, full state) for the FD cross-check

    for k in range(1, nt):
        h = float(t[k] - t[k - 1])
        a0 = 1.0 if k == 1 else 1.5                            # BDF1 first step, BDF2 thereafter
        J = sim.analytic_jac(Y[k], keq, kkin, nu, sig).detach().numpy()
        lu, piv = sla.lu_factor(a0 * eye - h * J)              # A_k = a0 I − h J_k
        if Vs.shape[1]:
            hist = Vs if k == 1 else (2.0 * Vs - 0.5 * Vs_prev)
            Vs_new = sla.lu_solve((lu, piv), hist)
            Vs_prev = Vs; Vs = Vs_new
        if k in w_of_step:
            F_k = sigma_forcing(sim, Y[k], sim.inlet_c[k], keq, kkin, nu, sig)   # (ndim, n) = ∂f/∂σ at t_k
            new = sla.lu_solve((lu, piv), F_k)                # unit source injection at m=k
            Vs = np.hstack([Vs, new]); Vs_prev = np.hstack([Vs_prev, np.zeros((ndim, n))])
            src_order.append(k)

        # ---- observation snapshot at step k (all active source blocks propagated to t_k) ----
        if src_order:
            w_now = np.array([w_of_step[m] for m in src_order])        # (n_active,)
            CVs = observed_columns(Vs, obs_idx, groups, npr)          # (n_obs, n_active*n)
            n_obs = CVs.shape[0]
            blocks = CVs.reshape(n_obs, len(src_order), n)            # (n_obs, n_active, n)
            b_k = float(np.sum(np.linalg.norm(blocks, axis=(0, 2)) * w_now))   # Σ_m ‖C_obs G_m‖_F w_m
            S_obs = np.tensordot(blocks, w_now, axes=([1], [0]))     # (n_obs, n) = C_obs S_σ(t_k)
            FTF += S_obs.T @ S_obs
            bound_sq += b_k * b_k
            if k == nt - 1:
                S_final = (Vs.reshape(ndim, len(src_order), n) * w_now[None, :, None]).sum(axis=1)

    return {
        "nt": nt, "ndim": ndim, "n_sources": len(src_order), "stride": int(stride),
        "tau": float(t[-1]),
        "FTF": FTF, "bound_sq_sum": float(bound_sq),
        "S_sigma_tau_norm": float(np.linalg.norm(S_final)),
    }


def _fd_observed_fisher(sim, keq, kkin, nu, sig, n, groups, *, fd: float = 1e-4) -> np.ndarray:
    """Independent finite-difference observed Fisher ``J_σᵀ J_σ`` (unscaled), ``J_σ = ∂(grouped outlet
    trajectory)/∂σ`` stacked over all solver steps — the cross-check for the propagator's ``FTF``."""
    obs_idx = _obs_indices(sim)
    npr = int(sim.npr)
    cols = []
    for j in range(n):
        sp = sig.clone(); sp[j] += fd
        sm = sig.clone(); sm[j] -= fd
        with torch.no_grad():
            Yp = _forward_full_state(sim, keq, kkin, nu, sp).detach().numpy()   # (nt, ndim)
            Ym = _forward_full_state(sim, keq, kkin, nu, sm).detach().numpy()
        dS = ((Yp - Ym) / (2.0 * fd)).T                                        # (ndim, nt) = ∂y(t)/∂σ_j
        cols.append(observed_columns(dS, obs_idx, groups, npr).reshape(-1))    # (n_obs*nt,)
    Jobs = np.stack(cols, axis=1)                                              # (n_obs*nt, n)
    return Jobs.T @ Jobs


# --------------------------------------------------------------------------- diagnostic + verdict

def restricted_fisher_diagnostic(sim, keq, kkin, nu, sig, n, groups, *, stride: int = 1,
                                 crosscheck: bool = True) -> dict:
    """Single-experiment σ-restricted observation-gain diagnostic (the full construction on one ``sim``)."""
    sweep = restricted_fisher_sweep(sim, keq, kkin, nu, sig, n, groups, stride=stride)
    with torch.no_grad():
        Y = _forward_full_state(sim, keq, kkin, nu, sig)
    eps = _epsilon_from_traj(sim, Y, nu + sig)
    out = {**sweep, "epsilon": eps}
    if crosscheck:
        out["FTF_fd"] = _fd_observed_fisher(sim, keq, kkin, nu, sig, n, groups)
    return out


def _lam_max(M) -> float:
    M = np.asarray(M, float)
    return float(np.linalg.eigvalsh(0.5 * (M + M.T))[-1])


def _worst_dir_sigma(cov, n, sigma_prior) -> float:
    """Committed posterior σ-block worst-direction shrinkage ``sqrt(λmax(Σ_σσ))/σ_prior`` (§2.4 restricted
    to the σ block; uniform σ prior)."""
    Sig = np.asarray(cov, float)[3 * n:4 * n, 3 * n:4 * n]
    return math.sqrt(max(_lam_max(Sig), 0.0)) / sigma_prior


def _shrinkage_bound(F_bound, sigma_prior) -> float:
    """A-priori lower bound on ``worst_dir_σ`` from the Fisher upper bound: ``1/sqrt(1+σ_prior²·F_bound)``."""
    return 1.0 / math.sqrt(1.0 + sigma_prior ** 2 * F_bound)


def _decide(ratio: float, xcheck_ok: bool) -> str:
    """PASS if the non-circular σ-Fisher bound is tight (``bound/exact ≤ 1e3`` — O(1)–moderate slack) and the
    hand-written propagator is validated against the finite difference.  The verdict is on the Fisher-block
    MAGNITUDE bound (the Fisher-null O(ε²) scaling), NOT on the worst_dir shrinkage: because the σ block has a
    wide eigenvalue spread (some σ directions are identified, λmax ≫ λmin), the λmax-based shrinkage lower
    bound is loose at the finite operating ε (it → 1 only as ε → 0).  WARN if non-vacuous but looser; FAIL if
    ``>1e6`` or non-finite."""
    if (not math.isfinite(ratio)) or ratio > 1e6:
        return "FAIL_VACUOUS"
    if xcheck_ok and 1.0 - 1e-6 <= ratio <= 1e3:
        return "PASS_RESTRICTED_FISHER"
    return "WARN_USEFUL_DIAGNOSTIC"


def _package(product, n_steps, stride, agg, sigma_obs, sigma_prior, worst_dir_exact, kappa_state) -> dict:
    FTF, bound_sq, eps = agg["FTF"], agg["bound_sq_sum"], agg["epsilon"]
    F_exact = _lam_max(FTF) / sigma_obs ** 2                       # λmax((1/σ²)Σ S_kᵀ S_k)
    F_bound = bound_sq / sigma_obs ** 2                            # (1/σ²)Σ b_k² ≥ F_exact
    kappa_obs = math.sqrt(bound_sq) / eps if eps > 0 else math.inf   # normalized σ-restricted obs gain
    ratio = F_bound / F_exact if F_exact > 0 and math.isfinite(F_bound) else math.inf
    shrink_bound = _shrinkage_bound(F_bound, sigma_prior)          # a-priori worst_dir_σ lower bound (→1 as ε→0)
    F_exact_fd = agg.get("F_fd", math.nan)
    xcheck = F_exact / F_exact_fd if math.isfinite(F_exact_fd) and F_exact_fd > 0 else math.inf
    xcheck_ok = bool(math.isfinite(xcheck) and 0.5 <= xcheck <= 2.0)
    bound_over_fd = F_bound / F_exact_fd if math.isfinite(F_exact_fd) and F_exact_fd > 0 else math.inf
    verdict = _decide(ratio, xcheck_ok)
    return {
        "product": product, "n_steps": n_steps, "stride": int(stride),
        "epsilon": eps, "sigma_obs": sigma_obs, "sigma_prior": sigma_prior,
        "restricted_fisher": {
            "kappa_state_sigma_restricted": kappa_state,          # committed state gain (‖∫Φ F_σ‖/ε), for context
            "kappa_obs_sigma_restricted": kappa_obs,              # σ-restricted observation gain sqrt(Σ b_k²)/ε
            "F_sigma_sigma_bound": F_bound,                       # (ε·κ_obs,σ)²/σ_obs² ≥ λmax(F_σσ)
            "F_sigma_sigma_exact": F_exact,                       # λmax((1/σ²)Σ S_kᵀS_k) (propagator observed Fisher)
            "bound_over_exact": ratio,                            # triangle slack (≥1); the certificate tightness
            "posterior_shrinkage_bound": shrink_bound,            # a-priori worst_dir_σ lower bound (loose, →1 as ε→0)
            "posterior_worst_dir_exact": worst_dir_exact,         # committed σ-block worst_dir (the actual sloppiness)
            "n_sources": agg["n_sources"],
        },
        "cross_check": {
            "F_exact_over_fd": xcheck,              # propagator observed Fisher / FD Fisher (fidelity, ~1 ideal)
            "bound_over_fd": bound_over_fd,         # the a-priori bound also exceeds the independent FD Fisher (rigor)
            "propagator_ok": xcheck_ok,
            "S_sigma_tau_norm": agg.get("S_sigma_tau_norm"),
        },
        "decision": verdict,
    }


# --------------------------------------------------------------------------------- aggregation over experiments

def _fisher_over_experiments(bundle, keq, kkin, nu, sig, n, groups, *, n_steps, stride,
                             crosscheck: bool = True) -> dict:
    """Sum the σ-restricted observation gain over the product's real experiments (the calibration Fisher is
    ``Σ_exp J_exp(θ_MAP)ᵀ J_exp(θ_MAP)/σ²``)."""
    from cex_model.bayes.active import _sim_for_op

    FTF = np.zeros((n, n)); FTF_fd = np.zeros((n, n)); bound_sq = 0.0
    eps = 0.0; s_tau = 0.0; n_sources = 0
    for e in bundle.experiments:
        op = [float(e.loading_g_l), float(e.gradient_start_pct), float(e.gradient_end_pct),
              float(e.elution_cv)]
        sim = _sim_for_op(bundle, op, n_steps)
        diag = restricted_fisher_diagnostic(sim, keq, kkin, nu, sig, n, groups, stride=stride,
                                            crosscheck=crosscheck)
        FTF += diag["FTF"]; bound_sq += diag["bound_sq_sum"]
        eps = max(eps, diag["epsilon"]); s_tau = max(s_tau, diag["S_sigma_tau_norm"])
        n_sources = max(n_sources, diag["n_sources"])
        if crosscheck:
            FTF_fd += diag["FTF_fd"]
    agg = {"FTF": FTF, "bound_sq_sum": bound_sq, "epsilon": eps, "S_sigma_tau_norm": s_tau,
           "n_sources": n_sources}
    if crosscheck:
        agg["F_fd"] = _lam_max(FTF_fd) / AKTA_NOISE_FLOOR_G_L ** 2
    return agg


def _read_state_gain(product, in_dir) -> float | None:
    """Reuse the committed state-side restricted gain (``restricted_sigma_gain_{p}.json``): the normalized
    ``κ_state,σ = ‖∫Φ F_σ ds‖_bound / ε``, for side-by-side reporting with the observation gain."""
    from pathlib import Path
    import json
    p = Path(in_dir) / f"restricted_sigma_gain_{product}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    eps = d.get("epsilon")
    b = d.get("restricted_gain", {}).get("bound_norm_inside")
    return float(b) / float(eps) if b and eps else None


# --------------------------------------------------------------------------------------------- drivers

def _sigma_prior(n) -> float:
    from cex_model.bayes.prior import physical_prior
    return float(physical_prior(n).std[3 * n])


def certify(product, *, in_dir="results/bayes", n_steps: int = 120, stride: int = 1) -> dict:
    """σ-restricted observation-gain Fisher certificate for a real product at its committed MAP over its
    real experiments."""
    import cex_model.app_support as A
    from cex_model.bayes.compare import filter_experiments
    from cex_model.bayes.posterior import Posterior
    from cex_model.diffsolver.torch_solver import DTYPE

    post = Posterior.load(f"{in_dir}/{product}_posterior.npz")
    n = post.n_protein
    prod, drop = _PRODUCT_MAP.get(product, (product, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
    keq, kkin, nu, sig = unpack_u(torch.tensor(post.u_map, dtype=DTYPE), n)
    groups = bundle.observation_groups
    s_p = _sigma_prior(n)

    agg = _fisher_over_experiments(bundle, keq, kkin, nu, sig, n, groups, n_steps=n_steps, stride=stride)
    worst_dir_exact = _worst_dir_sigma(post.cov, n, s_p)
    kappa_state = _read_state_gain(product, in_dir)
    return _package(product, n_steps, stride, agg, AKTA_NOISE_FLOOR_G_L, s_p, worst_dir_exact, kappa_state)


def certify_synthetic(*, n_comp: int = 2, nu=4.0, sigma=30.0, loading: float = 8.0,
                      fractions=(70.0, 30.0), keq_ladder=True, n_steps: int = 40, stride: int = 1) -> dict:
    """σ-restricted observation-gain Fisher certificate on a synthetic bundle (CI-friendly).  The exact
    posterior worst_dir is taken from a Laplace posterior at the true params (a small consistency object,
    not committed)."""
    from cex_model.bayes.synthetic import synthetic_sma_bundle
    from cex_model.diffsolver.torch_solver import DTYPE

    bundle, comps, u_true = synthetic_sma_bundle(n_comp, fractions=list(fractions), keq_ladder=keq_ladder,
                                                 nu=nu, sigma=sigma, loading=loading)
    n = comps.n_protein
    keq, kkin, nu_t, sig = unpack_u(torch.tensor(np.asarray(u_true, float), dtype=DTYPE), n)
    groups = bundle.observation_groups
    s_p = _sigma_prior(n)

    agg = _fisher_over_experiments(bundle, keq, kkin, nu_t, sig, n, groups, n_steps=n_steps, stride=stride)
    # synthetic exact worst_dir: build the σ-block posterior cov from the same summed Fisher + prior
    from cex_model.bayes.prior import physical_prior
    prior = physical_prior(n)
    F_full = np.zeros((4 * n, 4 * n))
    F_full[3 * n:4 * n, 3 * n:4 * n] = agg["FTF"] / AKTA_NOISE_FLOOR_G_L ** 2
    cov = np.linalg.inv(F_full + prior.precision())
    worst_dir_exact = _worst_dir_sigma(cov, n, s_p)
    return _package(f"SYN{n_comp}", n_steps, stride, agg, AKTA_NOISE_FLOOR_G_L, s_p, worst_dir_exact, None)
