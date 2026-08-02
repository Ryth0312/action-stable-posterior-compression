"""Thermodynamic entropy metric for the σ decision-null a-priori constant κ_S
(``ENTROPY_METRIC_KAPPA_S_PLAN.md``, Tasks 1-4).

The Euclidean logarithmic-norm Grönwall constant (``bayes/groenwall.py``) and the
constant/per-species-diagonal contraction-metric family (``bayes/contraction.py``) are both vacuous
for SMA: the discretized operator ``J=∂f/∂y`` is strongly non-normal from the stiff **adsorption
reaction**, and neither metric family expresses the off-diagonal capacity coupling that reaction
induces through the shared ``Λ̄``. This module tries a metric that is *induced by the reaction
itself* — the Hessian of the natural SMA adsorption/desorption entropy

    H(C,Q) = Σ_i θ_i [C_i(log C_i - 1) + Q_i(log Q_i - 1)] + ρ·Λ(log Λ - 1)
    θ_i = ρ (ν_i+σ_i) γ_i / ν_i ,   a_i = (ν_i+σ_i) γ_i ,   Λ = Λ_0 - Σ_i a_i Q_i

which contains exactly the missing off-diagonal Q-block term ``ρ a_i a_j / Λ`` (rank-one in the
shared capacity direction) and makes the reversible exchange part of the reaction dissipative
(``(a-b)log(a/b) ≥ 0``). It does **not** by itself close a-priori contraction (the ``Λ^{ν_i}``
capacity nonlinearity leaves an unsigned residual — see the plan), so this module is a
**diagnostic**: it reports whether the entropy-metric log-norm ``μ_ent`` is materially better than
the Euclidean one and whether the resulting candidate constant ``κ_S_ent`` is usable, not a proof.

State layout follows ``diffsolver/torch_solver.py::rhs`` exactly (Fortran/species-major:
``[C_salt, C_1..C_npr, Q_1..Q_npr]``, each species block ``gs``-contiguous). The entropy has no
term in the shared salt channel ``C_0`` (θ is only defined for the protein species), so ``M_ent``
is only positive-definite on the protein sub-block ``(C_1..C_npr, Q_1..Q_npr)`` — see
``protein_indices``; the log-norm diagnostic (Task 2) and the candidate ``κ_S_ent`` are computed
on that sub-block, with the full Euclidean ``μ₂`` (over the whole state, from ``groenwall.py``)
reported alongside for context.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import torch

from cex_model.bayes.groenwall import _EXP_OVERFLOW, _ffill, _epsilon_from_traj, _forward_full_state, log_norm2
from cex_model.bayes.likelihood import unpack_u

__all__ = [
    "protein_indices",
    "local_entropy_metric",
    "assemble_entropy_metric",
    "entropy_log_norm",
    "residual_diagnostics",
    "entropy_diagnostic",
    "certify",
    "certify_synthetic",
]


# --------------------------------------------------------------------------- pure helpers

def _floors(floors) -> tuple[float, float, float]:
    floors = floors or {}
    return (float(floors.get("C", 1e-8)), float(floors.get("Q", 1e-8)), float(floors.get("Lambda", 1e-12)))


def protein_indices(gs: int, nc: int) -> np.ndarray:
    """Full-state indices of the protein-only (non-salt) sub-block ``[C_1..C_{nc-1}, Q_1..Q_{nc-1}]``
    (each ``gs``-contiguous), matching ``torch_solver``'s Fortran/species-major layout."""
    d = np.arange(gs)
    npr = nc - 1
    idx = [(i + 1) * gs + d for i in range(npr)] + [(nc + j) * gs + d for j in range(npr)]
    return np.concatenate(idx)


def _protein_metric_dense(C, Q, nu, sigma, gamma, Lambda0, *, rho=1.0, floors=None) -> np.ndarray:
    """Dense ``(2*npr*gs, 2*npr*gs)`` entropy metric restricted to the protein sub-block, ordered
    ``[C_1(all nodes), ..., C_npr(all nodes), Q_1(all nodes), ..., Q_npr(all nodes)]`` (a local
    species-major layout used by ``local_entropy_metric``/``assemble_entropy_metric``/the diagnostic
    loop; ``C``, ``Q`` are ``(gs, npr)``)."""
    C = np.atleast_2d(np.asarray(C, float))
    Q = np.atleast_2d(np.asarray(Q, float))
    nu = np.asarray(nu, float); sigma = np.asarray(sigma, float); gamma = np.asarray(gamma, float)
    gs, npr = C.shape
    fC, fQ, fL = _floors(floors)
    a = (nu + sigma) * gamma
    theta = rho * a / nu
    Lambda = np.maximum(Lambda0 - Q @ a, fL)   # (gs,)

    ndim = 2 * npr * gs
    M = np.zeros((ndim, ndim))
    for i in range(npr):
        iC = slice(i * gs, (i + 1) * gs)
        M[iC, iC] = np.diag(theta[i] / (C[:, i] + fC))
    for i in range(npr):
        iQi = slice((npr + i) * gs, (npr + i + 1) * gs)
        M[iQi, iQi] = np.diag(theta[i] / (Q[:, i] + fQ) + rho * a[i] * a[i] / Lambda)
        for j in range(i + 1, npr):
            iQj = slice((npr + j) * gs, (npr + j + 1) * gs)
            off = np.diag(rho * a[i] * a[j] / Lambda)
            M[iQi, iQj] = off
            M[iQj, iQi] = off
    return M


def local_entropy_metric(C, Q, nu, sigma, gamma, Lambda0, rho=1.0, floors=None) -> np.ndarray:
    """``M_ent`` ``(2*npr, 2*npr)`` quadratic-form matrix at ONE spatial node (Task 1), ordered
    ``[C_1..C_npr, Q_1..Q_npr]``. ``C``, ``Q``, ``nu``, ``sigma``, ``gamma`` are length-``npr``."""
    return _protein_metric_dense(np.atleast_2d(C), np.atleast_2d(Q), nu, sigma, gamma, Lambda0,
                                 rho=rho, floors=floors)


def assemble_entropy_metric(Y, nu, sigma, gamma, Lambda0, *, rho=1.0, floors=None) -> sp.csr_matrix:
    """Sparse ``(ndim, ndim)`` entropy metric for the full discretized SMA state (Task 1), following
    the exact state layout of ``torch_solver.rhs``. ``Y`` is ``(gs, 2nc-1)`` (post smooth-clip),
    ``nu``/``sigma``/``gamma`` are ``(npr,)``. The shared salt channel ``C_0`` carries no entropy
    weight (H has no term in it) so its rows/cols are structurally zero — ``M_ent`` is only
    positive-definite on the protein sub-block (see ``protein_indices``)."""
    Y = np.asarray(Y, float)
    gs = Y.shape[0]
    npr = int(np.asarray(nu).shape[0]); nc = npr + 1
    ndim = gs * (2 * nc - 1)
    C, Q = Y[:, 1:nc], Y[:, nc:]
    M_dense = _protein_metric_dense(C, Q, nu, sigma, gamma, Lambda0, rho=rho, floors=floors)
    idx = protein_indices(gs, nc)
    rows, cols = np.meshgrid(idx, idx, indexing="ij")
    return sp.coo_matrix((M_dense.ravel(), (rows.ravel(), cols.ravel())), shape=(ndim, ndim)).tocsr()


def entropy_log_norm(J, M, Mdot=None) -> float:
    """``μ_M(J) = 1/2 λ_max`` of the generalized eigenproblem ``(M J + J^T M + Ṁ, M)`` (Task 2) —
    the entropy-metric log-norm, with an optional metric-derivative term. Reduces to the plain
    weighted log-norm (``contraction.full_log_norm``) when ``Mdot=None``."""
    J = np.asarray(J, float); M = np.asarray(M, float)
    A = M @ J
    S = 0.5 * (A + A.T)
    if Mdot is not None:
        S = S + 0.5 * np.asarray(Mdot, float)
    return float(sla.eigh(S, M, eigvals_only=True)[-1])


# ----------------------------------------------------------------------------- Task 3: residuals

def residual_diagnostics(sim, keq, kkin, nu, sig) -> dict:
    """ζ_i (reaction disequilibrium), ω_i (total capacity-occupancy contribution) and ε_i
    (per-molecule loading fraction) along the nominal trajectory (Task 3), at every grid step
    (cheap: no autograd, plain elementwise algebra on the already-simulated state)."""
    nc, gs = sim.nc, sim.gs
    nusig = nu + sig
    gamma = sim.gamma_p
    with torch.no_grad():
        Y = _forward_full_state(sim, keq, kkin, nu, sig)
        Yr = sim._smooth(Y.reshape(Y.shape[0], 2 * nc - 1, gs))       # (nt, 2nc-1, gs)
        Cp = Yr[:, 1:nc, :]            # (nt, npr, gs)
        Qp = Yr[:, nc:, :]             # (nt, npr, gs)
        csalt = Yr[:, 0, :]            # (nt, gs)
        bound = torch.einsum("j,njg->ng", gamma * nusig, Qp)
        Lam = (sim.inocap - bound).clamp(min=1e-12)                   # (nt, gs)
        ads = keq[None, :, None] * Lam[:, None, :] ** nu[None, :, None] * Cp
        des = gamma[None, :, None] * Qp * csalt[:, None, :] ** nu[None, :, None]
        zeta = (ads - des).abs() / (ads + des + 1e-30)
        omega = (nusig[None, :, None] * gamma[None, :, None] * Qp) / Lam[:, None, :]
        epsilon = (gamma[None, :, None] * Qp) / Lam[:, None, :]

    def _stats(x):
        v = x.detach().numpy().ravel()
        return {"max": float(v.max()), "p95": float(np.percentile(v, 95.0))}

    return {"zeta": _stats(zeta), "omega": _stats(omega), "epsilon": _stats(epsilon)}


# ----------------------------------------------------------------------------- Task 2: mu_ent(t)

def _finite_diff_Mdot(M_at, k: int, nt: int, t: np.ndarray) -> np.ndarray:
    """Central finite difference of ``M_ent(t)`` in time (forward/backward at the boundary)."""
    if 0 < k < nt - 1:
        return (M_at(k + 1) - M_at(k - 1)) / (t[k + 1] - t[k - 1])
    if k == 0:
        return (M_at(1) - M_at(0)) / (t[1] - t[0])
    return (M_at(nt - 1) - M_at(nt - 2)) / (t[nt - 1] - t[nt - 2])


def entropy_diagnostic(sim, keq, kkin, nu, sig, n, *, rho=1.0, floors=None, stride=4,
                       dot_M="fd") -> dict:
    """Entropy-metric log-norm diagnostic along the nominal trajectory (Task 2), plus the candidate
    a-priori entropy Grönwall constant κ_S_ent (plan §"Candidate a-priori constant").

    ``dot_M``: ``"fd"`` (default; central finite difference of ``M_ent(t)`` in time) or ``"zero"``
    (metric held frozen — explicitly incomplete, the plan's phase-3 fallback if FD is unusable).
    Heavy per-step linear algebra (the generalized eigenproblem, the σ-forcing autograd Jacobian) is
    subsampled by ``stride`` (mirrors ``contraction.py``'s ``t_stride`` for the same class of
    computation, with forward-fill onto the full grid, mirroring ``groenwall.py``'s ``normF``); the
    cheap Euclidean log-norm is not.
    """
    if dot_M not in ("fd", "zero"):
        raise ValueError(f"dot_M must be 'fd' or 'zero', got {dot_M!r}")
    gs, nc = sim.gs, sim.nc
    idx = protein_indices(gs, nc)
    t = sim.t.detach().numpy()
    nt = len(t)
    tidx = sorted(set(range(0, nt, max(1, stride))) | {0, nt - 1})

    with torch.no_grad():
        Y = _forward_full_state(sim, keq, kkin, nu, sig)
    nu_np, sig_np, gamma_np = nu.detach().numpy(), sig.detach().numpy(), sim.gamma_p.detach().numpy()
    Lambda0 = sim.inocap
    # ε = max_{t,node,j} γ_j Q_j/Λ̄ over the FULL (unstrided) trajectory -- cheap (no autograd), so
    # unlike the per-step generalized-eigenproblem/σ-Jacobian below it is not stride-subsampled
    # (an understated eps would inflate kappa_S_ent = integ/eps, biasing the route to look more
    # vacuous than it is).
    eps = _epsilon_from_traj(sim, Y, nu + sig)

    def node_state(k):
        Ynode = sim._smooth(Y[k].reshape(2 * nc - 1, gs).t()).detach().numpy()   # (gs, 2nc-1)
        return Ynode[:, 1:nc], Ynode[:, nc:]

    def M_at(k):
        C, Q = node_state(k)
        return _protein_metric_dense(C, Q, nu_np, sig_np, gamma_np, Lambda0, rho=rho, floors=floors)

    mu_ent_zero = np.full(nt, np.nan)
    mu_ent_fd = np.full(nt, np.nan)
    mu2_sub = np.full(nt, np.nan)
    mu2_full = np.full(nt, np.nan)
    normF_M = np.full(nt, np.nan)

    for k in tidx:
        y_k = Y[k]
        J_full = sim.analytic_jac(y_k, keq, kkin, nu, sig).detach().numpy()
        J_sub = J_full[np.ix_(idx, idx)]
        mu2_full[k] = log_norm2(J_full)
        mu2_sub[k] = log_norm2(J_sub)
        M_k = M_at(k)
        # both dot(M) variants are cheap once M_at(k±1) is available (Task 2's phase-3 "zero" mode
        # and the finite-difference refinement), so both are always computed and reported
        # side-by-side (dot_M selects only which becomes the headline "entropy" summary below) --
        # a large zero-vs-fd divergence is itself a diagnostic (FD instability near floor-scale states).
        Mdot = _finite_diff_Mdot(M_at, k, nt, t)
        mu_ent_zero[k] = entropy_log_norm(J_sub, M_k, None)
        mu_ent_fd[k] = entropy_log_norm(J_sub, M_k, Mdot)

        def f_of_sig(s):
            return sim.rhs(y_k, sim.inlet_c[k], keq, kkin, nu, nu + s)
        F = torch.autograd.functional.jacobian(f_of_sig, sig).detach().numpy()   # (ndim, n)
        F_sub = F[idx, :]
        normF_M[k] = float(np.sqrt(np.einsum("ik,ij,jk->", F_sub, M_k, F_sub)))

    mu_ent_zero = _ffill(mu_ent_zero); mu_ent_fd = _ffill(mu_ent_fd)
    mu2_sub = _ffill(mu2_sub); mu2_full = _ffill(mu2_full)
    normF_M = _ffill(normF_M)

    def _summ(mu):
        pos = np.clip(mu, 0.0, None)
        int_pos = float(np.trapezoid(pos, t))
        return {"sup_mu": float(np.max(mu)), "int_mu": float(np.trapezoid(mu, t)),
                "int_positive_mu": int_pos,
                "exp_int_positive_mu": float(math.exp(min(int_pos, _EXP_OVERFLOW)))}

    def _kappa_S_ent(mu):
        # candidate a-priori κ_S_ent: cumulative entropy log-norm exp(∫_s^τ μ_ent dr) — same
        # Coppel-style construction as groenwall._certify_on_sim's Euclidean κ_S, in metric M(t).
        seg = 0.5 * (mu[:-1] + mu[1:]) * np.diff(t)
        Mcum = np.append(np.cumsum(seg[::-1])[::-1], 0.0)
        Aexp = np.exp(np.clip(Mcum, None, _EXP_OVERFLOW))
        integ = float(np.trapezoid(Aexp * normF_M, t))
        return (integ / eps) * math.sqrt(1.0 / lam_min_tau) if eps > 0 and lam_min_tau > 0 else float("nan")

    zero_summ = _summ(mu_ent_zero)
    fd_summ = _summ(mu_ent_fd)
    euc_sub_summ = _summ(mu2_sub)
    euc_full_summ = _summ(mu2_full)
    M_tau = M_at(nt - 1)
    lam_min_tau = float(np.linalg.eigvalsh(M_tau)[0])
    zero_summ["kappa_S_ent"] = _kappa_S_ent(mu_ent_zero)
    fd_summ["kappa_S_ent"] = _kappa_S_ent(mu_ent_fd)
    max_abs_divergence = float(np.max(np.abs(mu_ent_fd - mu_ent_zero)))

    headline = fd_summ if dot_M == "fd" else zero_summ
    return {
        "dot_M_mode": dot_M, "stride": int(stride), "epsilon": eps,
        "entropy": headline, "entropy_zero": zero_summ, "entropy_fd": fd_summ,
        "max_zero_fd_divergence": max_abs_divergence,
        "euclidean_protein_subspace": euc_sub_summ, "euclidean_full_state": euc_full_summ,
        "kappa_S_ent": headline["kappa_S_ent"], "lambda_min_M_tau": lam_min_tau,
    }


# ----------------------------------------------------------------------------- Task 4: decision

def _decide(entropy: dict, euclidean: dict, residual: dict) -> str:
    """Task 4 decision criterion (plan §"Success criteria"/"abandon"): a threshold-based translation
    of the plan's qualitative gates. ``continue_entropy`` needs the entropy log-norm integral to be
    O(1) or orders of magnitude below the Euclidean one, OR the bound-to-measured ratio to be a
    non-catastrophic first pass (<1e3), OR the residual small in most (p95) of the state tube;
    ``fallback_adjoint_gain`` fires when the ratio is catastrophic (>=1e6) or the entropy integral
    stays huge with no material improvement over Euclidean; anything else is ``inconclusive``
    rather than guessed."""
    mu_ent_pos = entropy["int_positive_mu_ent"]
    mu2_pos = euclidean["int_positive_mu2"]
    ratio = entropy["bound_to_measured_ratio"]
    reduction = (mu2_pos / mu_ent_pos) if mu_ent_pos > 1e-12 else float("inf")

    entropy_small = mu_ent_pos < 10.0
    entropy_much_improved = np.isfinite(reduction) and reduction >= 100.0
    ratio_ok = np.isfinite(ratio) and 0.0 < ratio < 1e3
    ratio_catastrophic = (not np.isfinite(ratio)) or ratio >= 1e6
    residual_localized = residual["p95_zeta"] < 0.5 and residual["p95_omega"] < 0.5

    succeeds = entropy_small or entropy_much_improved or ratio_ok or residual_localized
    fails = ratio_catastrophic or (not entropy_small and not entropy_much_improved and mu_ent_pos > 1e3)
    if fails and not succeeds:
        return "fallback_adjoint_gain"
    if succeeds and not fails:
        return "continue_entropy"
    return "inconclusive"


def _package(product: str, n_steps: int, floors, residual: dict, diag: dict, euclid: dict) -> dict:
    entropy = {
        "sup_mu_ent": diag["entropy"]["sup_mu"],
        "int_mu_ent": diag["entropy"]["int_mu"],
        "int_positive_mu_ent": diag["entropy"]["int_positive_mu"],
        "exp_int_positive_mu_ent": diag["entropy"]["exp_int_positive_mu"],
        "kappa_S_ent": diag["kappa_S_ent"],
        "kappa_eff": euclid["kappa_eff"],
        "bound_to_measured_ratio": (diag["kappa_S_ent"] / euclid["kappa_eff"]
                                    if euclid["kappa_eff"] > 0 and np.isfinite(diag["kappa_S_ent"])
                                    else float("inf")),
    }
    euclidean = {
        "sup_mu2": euclid["mu_bar"],
        "int_positive_mu2": diag["euclidean_full_state"]["int_positive_mu"],
    }
    residual_block = {
        "max_zeta": residual["zeta"]["max"], "p95_zeta": residual["zeta"]["p95"],
        "max_omega": residual["omega"]["max"], "p95_omega": residual["omega"]["p95"],
        "max_epsilon": residual["epsilon"]["max"], "p95_epsilon": residual["epsilon"]["p95"],
    }
    return {
        "product": product, "n_steps": n_steps,
        "floors": floors or {"C": 1e-8, "Q": 1e-8, "Lambda": 1e-12},
        "euclidean": euclidean, "entropy": entropy, "residual": residual_block,
        "diagnostics": {"dot_M_mode": diag["dot_M_mode"], "stride": diag["stride"],
                       "euclidean_protein_subspace": diag["euclidean_protein_subspace"],
                       "entropy_zero": diag["entropy_zero"], "entropy_fd": diag["entropy_fd"],
                       "max_zero_fd_divergence": diag["max_zero_fd_divergence"]},
        "decision": _decide(entropy, euclidean, residual_block),
    }


# --------------------------------------------------------------------------------- drivers

def certify(product, *, in_dir="results/bayes", n_steps: int = 120, stride: int = 4,
           fd_delta: float = 1e-4, rho: float = 1.0, floors=None, dot_M: str = "fd") -> dict:
    """Entropy-metric certificate for a real product at its committed decision OP + MAP (Tasks 1-4)."""
    from pathlib import Path
    import json

    import cex_model.app_support as A
    from cex_model.bayes.active import _sim_for_op
    from cex_model.bayes.compare import filter_experiments
    from cex_model.bayes.groenwall import _PRODUCT_MAP, certify as _certify_euclid
    from cex_model.bayes.posterior import Posterior
    from cex_model.diffsolver.torch_solver import DTYPE

    in_dir = Path(in_dir)
    post = Posterior.load(in_dir / f"{product}_posterior.npz")
    op = json.loads((in_dir / f"{product}_decision.json").read_text())["decision_op"]
    n = post.n_protein
    prod, drop = _PRODUCT_MAP.get(product, (product, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
    sim = _sim_for_op(bundle, op, n_steps)
    keq, kkin, nu, sig = unpack_u(torch.tensor(post.u_map, dtype=DTYPE), n)

    residual = residual_diagnostics(sim, keq, kkin, nu, sig)
    diag = entropy_diagnostic(sim, keq, kkin, nu, sig, n, rho=rho, floors=floors, stride=stride, dot_M=dot_M)
    euclid = _certify_euclid(product, in_dir=str(in_dir), n_steps=n_steps, stride=stride, fd_delta=fd_delta)
    return _package(product, n_steps, floors, residual, diag, euclid)


def certify_synthetic(*, n_comp: int = 2, nu=4.0, sigma=30.0, loading: float = 8.0,
                      fractions=(70.0, 30.0), keq_ladder=True, n_steps: int = 60, stride: int = 4,
                      fd_delta: float = 1e-4, rho: float = 1.0, floors=None, dot_M: str = "fd") -> dict:
    """Entropy-metric certificate on a synthetic bundle (CI-friendly; mirrors ``groenwall.certify_synthetic``)."""
    from cex_model.bayes.active import _sim_for_op
    from cex_model.bayes.groenwall import certify_synthetic as _certify_euclid
    from cex_model.bayes.synthetic import synthetic_sma_bundle
    from cex_model.diffsolver.torch_solver import DTYPE

    bundle, comps, u_true = synthetic_sma_bundle(n_comp, fractions=list(fractions), keq_ladder=keq_ladder,
                                                 nu=nu, sigma=sigma, loading=loading)
    n = comps.n_protein
    e0 = bundle.experiments[0]
    op = [e0.loading_g_l, e0.gradient_start_pct, e0.gradient_end_pct, e0.elution_cv]
    sim = _sim_for_op(bundle, op, n_steps)
    keq_t, kkin_t, nu_t, sig_t = unpack_u(torch.tensor(np.asarray(u_true, float), dtype=DTYPE), n)

    residual = residual_diagnostics(sim, keq_t, kkin_t, nu_t, sig_t)
    diag = entropy_diagnostic(sim, keq_t, kkin_t, nu_t, sig_t, n, rho=rho, floors=floors,
                              stride=stride, dot_M=dot_M)
    euclid = _certify_euclid(n_comp=n_comp, nu=nu, sigma=sigma, loading=loading, fractions=fractions,
                             keq_ladder=keq_ladder, n_steps=n_steps, stride=stride, fd_delta=fd_delta)
    return _package(f"SYN{n_comp}", n_steps, floors, residual, diag, euclid)
