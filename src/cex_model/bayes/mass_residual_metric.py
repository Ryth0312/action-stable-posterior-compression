"""Structured mass--residual metric route to the open a-priori κ_S constant
(``docs/decision_null_theorem.md`` §7-1: the *general off-diagonal / time-varying* metric family that the
constant/per-species-diagonal family (``bayes/contraction.py``) and the entropy-Hessian metric
(``bayes/entropy_metric.py``) both leave unclosed).

Idea: change coordinates ``(C,Q) ↦ (M,R)`` so the strictly-stable conserved **mass** mode
``M_i = C_i + c1·Q_i`` (pure transport — the reaction cancels in ``rhs``) is split from the stiff reaction
**residual** ``R_i = Q_i − Q_i*`` (departure from the local SMA adsorption equilibrium
``Q_i* = keq_i Λ̄^{ν_i} C_i /(γ_i c_salt^{ν_i})``).  The transform is time-varying (``Q_i*`` moves with the
state), so ``H(t) = ∂(M,R)/∂(C,Q)`` carries an ``Ḣ`` term.  A block metric ``diag(W_M, ω²W_R)`` is pulled
back to ``P_x(t) = H^T diag(W_M, ω²W_R) H`` and a 2×2 block-comparison contraction rate
``ρ(t) = (a+d − √((a−d)²+4bc))/2`` gives a candidate ``κ_S^MR = B₀·χ_P·(1−e^{−ρτ})/ρ``.

**Load-bearing sign correction.** The rank-one SMA capacity coupling in ``analytic_jac`` (block ``E5``,
``∂(dQ_i/dt)/∂Q_j`` for ``i≠j``) is ``p_i v_j`` with ``p_i = −keq_i C_i ν_i Λ̄^{ν_i−1}/kkin_i < 0`` and
``v_j = γ_j(ν_j+σ_j) > 0``.  The metric that symmetrizes it must be SPD, so ``W_R = diag(v_i/|p_i|) ≻ 0``
(NOT ``diag(v_i/p_i)``, which is negative-definite):  ``W_R (p vᵀ) = −v vᵀ`` is symmetric
negative-semidefinite, i.e. in the ``W_R`` metric the stiff capacity coupling becomes **dissipative**.
``1/|p_i| ∝ 1/(keq_i C_i Λ̄^{ν_i−1})`` and ``Q_i* ∝ c_salt^{−ν_i}`` both blow up where protein ``C_i→0`` /
salt is low (front/tail) — the same ``Λ̄^{ν}`` nonlinearity that left the entropy route with an unsigned
residual does not vanish, it relocates into the metric conditioning ``χ_P``.  So this module is a
**diagnostic** (does the mass--residual split give a non-vacuous a-priori constant, and if not, *where* does
it break — the residual transport log-norm accumulated over τ, or the front/tail ``χ_P`` blow-up), NOT a
proof.  It reports a per-product verdict PASS_NONVACUOUS / WARN_LOOSE / FAIL_VACUOUS.

**Construction discipline (criterion 10).** ``P_x`` is built *only* algebraically from ``H`` and the metric
weights ``(W_M, W_R=diag(v/|p|), ω)`` read off the state and the analytic-Jacobian quantities — it is
**never** formed from the state-transition matrix Φ, a backward Lyapunov Gramian
(``scipy.linalg.solve_continuous_lyapunov`` / ``solve_continuous_are``), the empirical sensitivity ``S``, or
a fitted fundamental matrix.  The empirical ``S`` enters only on the *measured* side
(``groenwall.certify``'s ``actual_state_sensitivity``), for the bound-vs-measured ratio.

State layout follows ``diffsolver/torch_solver.py::rhs`` exactly (Fortran/species-major:
``[C_salt, C_1..C_npr, Q_1..Q_npr]``).  Everything lives on the protein sub-block (salt ``C_0`` carries no
mass--residual coordinate) via ``entropy_metric.protein_indices``, in the local species-major ordering
``[C_1(all nodes),..,C_npr,Q_1,..,Q_npr]`` (same ordering as ``J_full[ix_(idx,idx)]``).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from cex_model.bayes.entropy_metric import entropy_log_norm, protein_indices
from cex_model.bayes.groenwall import (
    _EXP_OVERFLOW,
    _epsilon_from_traj,
    _ffill,
    _forward_full_state,
    groenwall_constant,
    log_norm2,
    spectral_norm,
)
from cex_model.bayes.likelihood import unpack_u

__all__ = [
    "equilibrium_Q",
    "residual_weights",
    "local_H",
    "assemble_H",
    "pullback_metric",
    "transform_jacobian",
    "block_rates",
    "rho_from_blocks",
    "groenwall_mr_constant",
    "mass_residual_diagnostic",
    "certify",
    "certify_synthetic",
]

_RESIDUAL_MODES = ("equilibrium", "Q")
_HDOT_MODES = ("fd", "zero")


# --------------------------------------------------------------------------- pure helpers

def _floors(floors) -> tuple[float, float, float, float]:
    floors = floors or {}
    return (float(floors.get("C", 1e-8)), float(floors.get("Q", 1e-8)),
            float(floors.get("salt", 1e-8)), float(floors.get("Lambda", 1e-12)))


def equilibrium_Q(C, csalt, Lam, keq, gamma, nu, *, floors=None) -> np.ndarray:
    """Local SMA adsorption equilibrium ``Q_i* = keq_i Λ̄^{ν_i} C_i /(γ_i c_salt^{ν_i})`` (from ``ads=des``),
    per node.  ``C``/``Q`` are ``(gs, npr)``, ``csalt``/``Lam`` are ``(gs,)`` (Λ̄ already clamped)."""
    fC, _fQ, fS, _fL = _floors(floors)
    C = np.maximum(np.atleast_2d(np.asarray(C, float)), fC)
    csalt = np.maximum(np.asarray(csalt, float), fS)
    Lam = np.asarray(Lam, float)
    keq = np.asarray(keq, float); gamma = np.asarray(gamma, float); nu = np.asarray(nu, float)
    cs_pow = csalt[:, None] ** nu[None, :]
    lam_pow = Lam[:, None] ** nu[None, :]
    return keq[None, :] * lam_pow * C / (gamma[None, :] * cs_pow)


def residual_weights(C, Q, csalt, Lam, keq, kkin, nu, sigma, gamma, *, floors=None):
    """``(W_M, W_R, p, v)`` per (species, node).  ``W_M = 1`` (identity metric on the mass block);
    ``W_R_i = v_i/|p_i|`` with ``p_i = −keq_i C_i ν_i Λ̄^{ν_i−1}/kkin_i`` (analytic-Jacobian ``ads_dsma/kkin``)
    and ``v_i = γ_i(ν_i+σ_i)`` — the SPD metric that makes the rank-one capacity coupling ``p vᵀ``
    dissipative (``W_R p vᵀ = −v vᵀ ⪯ 0``).  All arrays ``(gs, npr)`` (``v`` broadcast per node)."""
    fC, _fQ, _fS, _fL = _floors(floors)
    C = np.maximum(np.atleast_2d(np.asarray(C, float)), fC)
    Lam = np.asarray(Lam, float)
    keq = np.asarray(keq, float); kkin = np.asarray(kkin, float)
    nu = np.asarray(nu, float); sigma = np.asarray(sigma, float); gamma = np.asarray(gamma, float)
    lam_m1 = Lam[:, None] ** (nu[None, :] - 1.0)
    p = -keq[None, :] * C * nu[None, :] * lam_m1 / kkin[None, :]        # (gs, npr) < 0
    v = (gamma * (nu + sigma))[None, :] * np.ones_like(C)               # (gs, npr) > 0
    wM = np.ones_like(C)
    wR = v / np.abs(p)                                                  # (gs, npr) > 0
    return wM, wR, p, v


def local_H(C, csalt, Lam, keq, gamma, nu, sigma, c1, *, residual="equilibrium", floors=None):
    """Per-node mass--residual transform ``H = ∂(M,R)/∂(C,Q)`` and its inverse, ordered
    ``[C_1..C_npr, Q_1..Q_npr] → [M_1..M_npr, R_1..R_npr]`` (``(2npr, 2npr)``).  ``C`` is the length-``npr``
    protein state at ONE node (``csalt``/``Lam`` scalars; the ``Q`` *value* is not needed — the transform
    depends on the state only through ``C, csalt, Λ̄``).  Returns ``(H, Hinv, Qstar)``.

    ``M_i = C_i + c1 Q_i`` ⇒ ``∂M/∂C=I, ∂M/∂Q=c1 I``.  ``residual='equilibrium'``:
    ``R_i = Q_i − Q_i*`` ⇒ ``∂R/∂C = −diag(Q*/C)`` (Q* linear in C_i), ``∂R/∂Q = I + u vᵀ`` with
    ``u_i = Q_i* ν_i/Λ̄``, ``v_j = γ_j(ν_j+σ_j)`` (via ``∂Λ̄/∂Q_j = −(ν_j+σ_j)γ_j``).  ``residual='Q'``:
    ``R_i = Q_i`` ⇒ ``H = [[I, c1 I],[0, I]]`` (constant, ``Ḣ=0``; robust baseline)."""
    if residual not in _RESIDUAL_MODES:
        raise ValueError(f"residual must be one of {_RESIDUAL_MODES}, got {residual!r}")
    fC, _fQ, fS, _fL = _floors(floors)
    npr = int(np.asarray(nu).shape[0])
    Cn = np.maximum(np.asarray(C, float), fC)
    nu = np.asarray(nu, float); sigma = np.asarray(sigma, float); gamma = np.asarray(gamma, float)
    I = np.eye(npr)
    H = np.zeros((2 * npr, 2 * npr))
    H[:npr, :npr] = I
    H[:npr, npr:] = c1 * I
    if residual == "Q":
        H[npr:, npr:] = I
        Qstar = np.zeros(npr)
    else:
        Qstar = equilibrium_Q(Cn[None, :], np.asarray([csalt]), np.asarray([Lam]),
                              keq, gamma, nu, floors=floors)[0]       # (npr,)
        u = Qstar * nu / float(Lam)
        v = gamma * (nu + sigma)
        H[npr:, :npr] = -np.diag(Qstar / Cn)
        H[npr:, npr:] = I + np.outer(u, v)
    return H, np.linalg.inv(H), Qstar


def assemble_H(Y, keq, kkin, nu, sigma, gamma, inocap, c1, *, residual="equilibrium", floors=None):
    """Full mass--residual transform ``(H, Hinv)`` on the protein sub-block, ``(2npr·gs, 2npr·gs)`` dense in
    the local species-major ordering ``[C_1(gs),..,C_npr(gs),Q_1(gs),..,Q_npr(gs)]``.  ``H`` couples C↔Q
    only within a spatial node, so it is block-diagonal over nodes (``Hinv`` inverted per node).  ``Y`` is
    the post-smooth-clip node state ``(gs, 2nc-1)``."""
    Y = np.asarray(Y, float)
    gs = Y.shape[0]
    npr = int(np.asarray(nu).shape[0]); nc = npr + 1
    _fC, _fQ, _fS, fL = _floors(floors)
    C = Y[:, 1:nc]; Q = Y[:, nc:]; csalt = Y[:, 0]
    Lam = np.maximum(inocap - Q @ (gamma * (nu + sigma)), fL)          # (gs,)
    ndim = 2 * npr * gs
    H = np.zeros((ndim, ndim)); Hinv = np.zeros((ndim, ndim))
    # local sub-block index of (species s, node g): C_s -> s*gs+g, Q_s -> (npr+s)*gs+g
    for g in range(gs):
        Hg, Hginv, _ = local_H(C[g], csalt[g], Lam[g], keq, gamma, nu, sigma, c1,
                               residual=residual, floors=floors)
        loc = np.array([s * gs + g for s in range(npr)] + [(npr + s) * gs + g for s in range(npr)])
        H[np.ix_(loc, loc)] = Hg
        Hinv[np.ix_(loc, loc)] = Hginv
    return H, Hinv


def pullback_metric(H, wM, wR, omega) -> np.ndarray:
    """``P_x = Hᵀ diag(W_M, ω²W_R) H`` — the pulled-back block metric (criterion-10 anchor: algebraic in
    ``H`` and the weights only).  ``wM``/``wR`` are ``(gs, npr)``; the diag weight is in the ``(M,R)`` row
    space of ``H`` (species-major)."""
    wM_flat = np.asarray(wM, float).T.reshape(-1)                      # [M_s(gs)] species-major
    wR_flat = np.asarray(wR, float).T.reshape(-1)                      # [R_s(gs)]
    Wdiag = np.concatenate([wM_flat, (omega ** 2) * wR_flat])
    H = np.asarray(H, float)
    return H.T @ (Wdiag[:, None] * H)


def transform_jacobian(J_sub, H, Hinv, Hdot=None) -> np.ndarray:
    """``J̃ = H J H⁻¹ + Ḣ H⁻¹`` (criterion 4) — the generator in the ``(M,R)`` coordinates."""
    J_sub = np.asarray(J_sub, float)
    Jt = H @ J_sub @ Hinv
    if Hdot is not None:
        Jt = Jt + np.asarray(Hdot, float) @ Hinv
    return Jt


def block_rates(Jt, wR, wR_dot, omega, gs, npr):
    """2×2 block-comparison constants of the transformed generator (criterion 5).  ``a,d`` are CONTRACTION
    rates (positive = stable) = negated metric log-norms of the M/R diagonal blocks; ``b,c ≥ 0`` are the
    metric-normalized cross-block coupling gains (``bc`` is ω-independent, so ω tunes only ``χ_P``, not ρ)::

        a = −μ_{W_M}(J̃_MM) = −log_norm2(J̃_MM)          (W_M = I)
        d = −μ_{W_R}(J̃_RR)  via entropy_log_norm(·, W_R, Ẇ_R)   (time-varying metric)
        b = (1/ω)·σmax(J̃_MR W_R^{−1/2}) ,  c = ω·σmax(W_R^{1/2} J̃_RM)
    """
    Jt = np.asarray(Jt, float)
    nM = npr * gs
    Jmm, Jmr = Jt[:nM, :nM], Jt[:nM, nM:]
    Jrm, Jrr = Jt[nM:, :nM], Jt[nM:, nM:]
    wr = np.asarray(wR, float).T.reshape(-1)                          # (nM,) species-major
    WR = np.diag(wr)
    WRdot = np.diag(np.asarray(wR_dot, float).T.reshape(-1)) if wR_dot is not None else None
    a = -log_norm2(Jmm)
    d = -entropy_log_norm(Jrr, WR, WRdot)
    inv_sqrt = 1.0 / np.sqrt(wr)
    sqrt = np.sqrt(wr)
    b = (1.0 / omega) * spectral_norm(Jmr * inv_sqrt[None, :])        # J̃_MR W_R^{-1/2}
    c = omega * spectral_norm(sqrt[:, None] * Jrm)                    # W_R^{1/2} J̃_RM
    return float(a), float(d), float(b), float(c)


def rho_from_blocks(a, d, b, c) -> float:
    """``ρ = (a+d − √((a−d)²+4bc))/2`` — the guaranteed contraction rate of the 2×2 comparison system
    (= ``−λmax`` of the Metzler matrix ``[[−a,b],[c,−d]]``).  ``ρ>0 ⇔ ad>bc ⇔`` coupled contraction."""
    disc = (a - d) ** 2 + 4.0 * b * c
    return float((a + d - math.sqrt(max(disc, 0.0))) / 2.0)


def groenwall_mr_constant(B0, chi_P, rho, tau) -> float:
    """Sup-bound ``κ_S^MR = χ_P·B₀·(1−e^{−ρτ})/ρ`` (contracting ρ>0), reusing ``groenwall_constant`` with
    log-norm ``μ = −ρ`` (so ρ≤0 gives the vacuous ``e^{|ρ|τ}`` growth / ``+inf``)."""
    return chi_P * groenwall_constant(B0, -rho, tau)


# --------------------------------------------------------------------------------- diagnostic

def _finite_diff(arr_at, k: int, nt: int, t: np.ndarray) -> np.ndarray:
    """Central finite difference of a time-indexed array (forward/backward at the boundary); mirrors
    ``entropy_metric._finite_diff_Mdot``."""
    if 0 < k < nt - 1:
        return (arr_at(k + 1) - arr_at(k - 1)) / (t[k + 1] - t[k - 1])
    if k == 0:
        return (arr_at(1) - arr_at(0)) / (t[1] - t[0])
    return (arr_at(nt - 1) - arr_at(nt - 2)) / (t[nt - 1] - t[nt - 2])


def _cumulative_forward_gain(rho: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``A(s) = exp(clip(−∫_s^τ ρ dr))`` on the full grid (Coppel forward gain in the block metric); same
    reversed-cumsum construction as ``groenwall._certify_on_sim``'s ``M(s)``, with the contraction sign."""
    seg = 0.5 * (rho[:-1] + rho[1:]) * np.diff(t)
    Rcum = np.append(np.cumsum(seg[::-1])[::-1], 0.0)                 # ∫_{t_k}^{τ} ρ dr
    return np.exp(np.clip(-Rcum, None, _EXP_OVERFLOW))


def mass_residual_diagnostic(sim, keq, kkin, nu, sig, n, *, omega=1.0, residual="equilibrium",
                             H_dot="fd", stride=4, floors=None) -> dict:
    """Mass--residual metric diagnostic along the nominal trajectory (criteria 1-9).

    Per strided step builds ``H(t)``/``Ḣ(t)`` (``residual`` transform), ``W_R(t)``/``Ẇ_R(t)``,
    ``J̃ = H J H⁻¹ + Ḣ H⁻¹``, the block constants ``(a,d,b,c)`` and rate ``ρ(t)``, the pullback metric
    ``P_x(t)`` (for ``χ_P`` and the σ-forcing norm ``normF_P``).  Both ``H_dot ∈ {fd, zero}`` ρ paths are
    computed for the frozen-vs-FD divergence diagnostic; ``H_dot`` selects the headline.  Heavy per-step
    linear algebra (assembly, eigenproblems, the σ-forcing autograd Jacobian) is subsampled by ``stride``
    (mirrors ``entropy_diagnostic``); ε is over the full grid (an understated ε would bias ``κ_S^MR``)."""
    if residual not in _RESIDUAL_MODES:
        raise ValueError(f"residual must be one of {_RESIDUAL_MODES}, got {residual!r}")
    if H_dot not in _HDOT_MODES:
        raise ValueError(f"H_dot must be one of {_HDOT_MODES}, got {H_dot!r}")
    gs, nc = sim.gs, sim.nc
    npr = n
    idx = protein_indices(gs, nc)
    tau = float(sim.t[-1])
    t = sim.t.detach().numpy()
    nt = len(t)
    tidx = sorted(set(range(0, nt, max(1, stride))) | {0, nt - 1})

    with torch.no_grad():
        Y = _forward_full_state(sim, keq, kkin, nu, sig)
    keq_np, kkin_np = keq.detach().numpy(), kkin.detach().numpy()
    nu_np, sig_np, gamma_np = nu.detach().numpy(), sig.detach().numpy(), sim.gamma_p.detach().numpy()
    inocap, c1 = float(sim.inocap), float(sim.c1)
    eps = _epsilon_from_traj(sim, Y, nu + sig)

    def node_state(k):
        return sim._smooth(Y[k].reshape(2 * nc - 1, gs).t()).detach().numpy()      # (gs, 2nc-1)

    def H_at(k):
        return assemble_H(node_state(k), keq_np, kkin_np, nu_np, sig_np, gamma_np, inocap, c1,
                          residual=residual, floors=floors)[0]

    def wR_at(k):
        Yk = node_state(k)
        C, Q, csalt = Yk[:, 1:nc], Yk[:, nc:], Yk[:, 0]
        _fL = _floors(floors)[3]
        Lam = np.maximum(inocap - Q @ (gamma_np * (nu_np + sig_np)), _fL)
        return residual_weights(C, Q, csalt, Lam, keq_np, kkin_np, nu_np, sig_np, gamma_np,
                                floors=floors)[1]

    rho_fd = np.full(nt, np.nan); rho_zero = np.full(nt, np.nan)
    normF_P = np.full(nt, np.nan)
    a_arr = np.full(nt, np.nan); d_arr = np.full(nt, np.nan)
    b_arr = np.full(nt, np.nan); cc_arr = np.full(nt, np.nan)
    lam_min = math.inf; lam_max = -math.inf
    H_cond_max = 0.0; wR_max = 0.0

    for k in tidx:
        y_k = Y[k]
        Yk = node_state(k)
        C, Q, csalt = Yk[:, 1:nc], Yk[:, nc:], Yk[:, 0]
        Lam = np.maximum(inocap - Q @ (gamma_np * (nu_np + sig_np)), _floors(floors)[3])
        wM, wR, _p, _v = residual_weights(C, Q, csalt, Lam, keq_np, kkin_np, nu_np, sig_np, gamma_np,
                                          floors=floors)
        H, Hinv = assemble_H(Yk, keq_np, kkin_np, nu_np, sig_np, gamma_np, inocap, c1,
                             residual=residual, floors=floors)
        with np.errstate(over="ignore", invalid="ignore"):
            H_cond_max = max(H_cond_max, float(np.linalg.cond(H)))
        wR_max = max(wR_max, float(np.max(wR)))
        J_sub = sim.analytic_jac(y_k, keq, kkin, nu, sig).detach().numpy()[np.ix_(idx, idx)]

        Hdot = _finite_diff(H_at, k, nt, t) if residual == "equilibrium" else np.zeros_like(H)
        wRdot = _finite_diff(wR_at, k, nt, t)

        # frozen-metric (zero) and time-varying (fd) block-comparison rates, both cheap once H/J are built
        Jt_zero = transform_jacobian(J_sub, H, Hinv, None)
        a0, d0, b0, c0 = block_rates(Jt_zero, wR, None, omega, gs, npr)
        rho_zero[k] = rho_from_blocks(a0, d0, b0, c0)
        Jt_fd = transform_jacobian(J_sub, H, Hinv, Hdot)
        a1, d1, b1, c1b = block_rates(Jt_fd, wR, wRdot, omega, gs, npr)
        rho_fd[k] = rho_from_blocks(a1, d1, b1, c1b)
        a_arr[k], d_arr[k], b_arr[k], cc_arr[k] = (a1, d1, b1, c1b) if H_dot == "fd" else (a0, d0, b0, c0)

        P_x = pullback_metric(H, wM, wR, omega)
        ev = np.linalg.eigvalsh(0.5 * (P_x + P_x.T))
        lam_min = min(lam_min, float(ev[0])); lam_max = max(lam_max, float(ev[-1]))

        def f_of_sig(s):
            return sim.rhs(y_k, sim.inlet_c[k], keq, kkin, nu, nu + s)
        F = torch.autograd.functional.jacobian(f_of_sig, sig).detach().numpy()[idx, :]      # (2npr*gs, n)
        normF_P[k] = float(np.sqrt(max(np.einsum("ik,ij,jk->", F, P_x, F), 0.0)))

    rho_head = _ffill(rho_fd if H_dot == "fd" else rho_zero)
    rho_zero = _ffill(rho_zero); rho_fd = _ffill(rho_fd)
    normF_P = _ffill(normF_P)
    for arr in (a_arr, d_arr, b_arr, cc_arr):
        arr[:] = _ffill(arr)

    chi_P = float(math.sqrt(lam_max / lam_min)) if lam_min > 0 and math.isfinite(lam_max) else math.inf
    ill_conditioned = not (math.isfinite(chi_P) and chi_P < 1e12)

    def _kappa_time_resolved(rho):
        A = _cumulative_forward_gain(rho, t)
        with np.errstate(over="ignore", invalid="ignore"):
            integ = float(np.trapezoid(A * normF_P, t))       # A can hit the exp-overflow clip => vacuous
        if not math.isfinite(integ) or eps <= 0:
            return math.inf
        return chi_P * integ / eps

    B0 = float(np.nanmax(normF_P) / eps) if eps > 0 else math.inf
    rho_worst_head = float(np.min(rho_head))
    rho_worst_zero = float(np.min(rho_zero))
    kappa_time = _kappa_time_resolved(rho_head)
    kappa_sup = groenwall_mr_constant(B0, chi_P, rho_worst_head, tau)
    net_int_rho = float(np.trapezoid(rho_head, t))

    return {
        "residual": residual, "H_dot": H_dot, "omega": float(omega), "stride": int(stride),
        "tau": tau, "ndim_subblock": int(2 * npr * gs), "epsilon": eps,
        "chi_P": chi_P, "lambda_min_Px": lam_min, "lambda_max_Px": lam_max,
        "H_cond_max": H_cond_max, "wR_max": wR_max,
        "ill_conditioned": bool(ill_conditioned),
        "B0": B0, "rho_worst": rho_worst_head, "net_integral_rho": net_int_rho,
        "kappa_S_MR": kappa_time, "kappa_S_MR_sup": kappa_sup,
        "block_constants": {"a_sup": float(np.nanmax(a_arr)), "a_inf": float(np.nanmin(a_arr)),
                            "d_inf": float(np.nanmin(d_arr)), "d_sup": float(np.nanmax(d_arr)),
                            "b_sup": float(np.nanmax(b_arr)), "c_sup": float(np.nanmax(cc_arr))},
        "rho_zero_worst": rho_worst_zero,
        "kappa_S_MR_zero": _kappa_time_resolved(rho_zero),
        "max_zero_fd_rho_divergence": float(np.max(np.abs(rho_fd - rho_zero))),
    }


# ----------------------------------------------------------------------------- Task 5: verdict

def _decide(ratio: float, chi_P: float, lam_min: float, rho_worst: float, ill_conditioned: bool) -> str:
    """Verdict (criterion 9): PASS_NONVACUOUS if bound/measured ≤ 1e3; WARN_LOOSE if 1e3 < ratio ≤ 1e6;
    FAIL_VACUOUS if ratio > 1e6 OR ``P_x`` ill-conditioned (χ_P non-finite/huge or λmin ≤ 0)."""
    p_ill = ill_conditioned or (not math.isfinite(chi_P)) or lam_min <= 0.0
    if p_ill or (not math.isfinite(ratio)) or ratio > 1e6:
        return "FAIL_VACUOUS"
    if ratio <= 1e3:
        return "PASS_NONVACUOUS"
    return "WARN_LOOSE"


def _package(product: str, n_steps: int, floors, diag: dict, euclid: dict) -> dict:
    kappa_eff = euclid["kappa_eff"]
    measured = euclid["actual_state_sensitivity"]                     # ‖S_σ‖ = ε·κ_eff
    bound = diag["epsilon"] * diag["kappa_S_MR"]                      # ε·κ_S_MR
    ratio = (diag["kappa_S_MR"] / kappa_eff
             if kappa_eff > 0 and math.isfinite(diag["kappa_S_MR"]) else math.inf)
    verdict = _decide(ratio, diag["chi_P"], diag["lambda_min_Px"], diag["rho_worst"],
                      diag["ill_conditioned"])
    return {
        "product": product, "n_steps": n_steps,
        "floors": floors or {"C": 1e-8, "Q": 1e-8, "salt": 1e-8, "Lambda": 1e-12},
        "residual": diag["residual"], "H_dot": diag["H_dot"], "omega": diag["omega"],
        "epsilon": diag["epsilon"], "tau": diag["tau"],
        "mass_residual": {
            "chi_P": diag["chi_P"], "rho_worst": diag["rho_worst"],
            "net_integral_rho": diag["net_integral_rho"], "B0": diag["B0"],
            "kappa_S_MR": diag["kappa_S_MR"], "kappa_S_MR_sup": diag["kappa_S_MR_sup"],
            "measured_S_sigma": measured, "bound_S_sigma": bound,
            "kappa_eff": kappa_eff, "bound_to_measured_ratio": ratio,
        },
        "euclidean": {"kappa_eff": kappa_eff, "actual_state_sensitivity": measured,
                      "kappa_S_euclid": euclid.get("kappa_S"), "mu_bar": euclid.get("mu_bar")},
        "diagnostics": {
            "lambda_min_Px": diag["lambda_min_Px"], "lambda_max_Px": diag["lambda_max_Px"],
            "H_cond_max": diag["H_cond_max"], "wR_max": diag["wR_max"],
            "ill_conditioned": diag["ill_conditioned"], "block_constants": diag["block_constants"],
            "rho_zero_worst": diag["rho_zero_worst"], "kappa_S_MR_zero": diag["kappa_S_MR_zero"],
            "max_zero_fd_rho_divergence": diag["max_zero_fd_rho_divergence"], "stride": diag["stride"],
        },
        "decision": verdict,
    }


# --------------------------------------------------------------------------------- drivers

def certify(product, *, in_dir="results/bayes", n_steps: int = 120, stride: int = 4, omega: float = 1.0,
            residual: str = "equilibrium", H_dot: str = "fd", floors=None) -> dict:
    """Mass--residual metric certificate for a real product at its committed decision OP + MAP."""
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

    diag = mass_residual_diagnostic(sim, keq, kkin, nu, sig, n, omega=omega, residual=residual,
                                    H_dot=H_dot, stride=stride, floors=floors)
    euclid = _certify_euclid(product, in_dir=str(in_dir), n_steps=n_steps, stride=stride)
    return _package(product, n_steps, floors, diag, euclid)


def certify_synthetic(*, n_comp: int = 2, nu=4.0, sigma=30.0, loading: float = 8.0,
                      fractions=(70.0, 30.0), keq_ladder=True, n_steps: int = 60, stride: int = 4,
                      omega: float = 1.0, residual: str = "equilibrium", H_dot: str = "fd",
                      floors=None) -> dict:
    """Mass--residual metric certificate on a synthetic bundle (CI-friendly; mirrors ``groenwall``)."""
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

    diag = mass_residual_diagnostic(sim, keq_t, kkin_t, nu_t, sig_t, n, omega=omega, residual=residual,
                                    H_dot=H_dot, stride=stride, floors=floors)
    euclid = _certify_euclid(n_comp=n_comp, nu=nu, sigma=sigma, loading=loading, fractions=fractions,
                             keq_ladder=keq_ladder, n_steps=n_steps, stride=stride)
    return _package(f"SYN{n_comp}", n_steps, floors, diag, euclid)
