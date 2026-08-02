"""Theorem-ladder for decision-null posterior compression (docs/decision_compression_theorem_memo.md).

Pure numpy (NO torch / NO ODE) so it is unit-testable and shared by both drivers:
  * ``scripts/step2a_decision_compression_calibration.py`` -- linear-Gaussian calibration leg;
  * ``scripts/step2b_decision_compression_sweep.py``       -- physical CMC refit sweep (Colab).

Objects (memo Definitions/Props):
  * ``rotate_sigma_block``  -- fixed orthogonal change of coords on the sigma block so that
    u = {keq, kkin, nu, sigma-COMMON-mode} (active) and v = sigma-DIFFERENTIAL null modes.
  * ``schur_cov``  -- D_Schur: v <- E[v|u] (conditional-mean compression, THEOREM object).
  * ``freeze_cov`` -- D_freeze: v <- mu_v (blunt fix-at-mean, the engineering reduction).
  * ``prior_only_cov`` -- C: restore a block to its prior variance (negative control).
  * ``bures_w2`` -- EXACT 2-Wasserstein between two equal-mean Gaussians (memo Prop 2 fix).
  * ``shortfall_risk`` -- Gaussian closed-form tolerance-whitened shortfall Bayes risk.
  * ``theorem_ladder`` -- the checked inequality chain (identity / cov bound / Bures / risk).
  * ``action_gap`` -- finite-candidate action-preservation check (memo Prop 3).

Everything is in the TOLERANCE-WHITENED decision space unless noted: ``Cbar = T^{-1} C T^{-1}``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "common_mode_direction",
    "rotate_sigma_block",
    "schur_cov",
    "freeze_cov",
    "prior_only_cov",
    "decision_cov",
    "variant_d_cov",
    "ridge_derivative",
    "compression_certificates",
    "whiten",
    "bures_w2",
    "sqrtm_psd",
    "shortfall_risk",
    "theorem_ladder",
    "action_gap",
]


# --------------------------------------------------------------------------- linear algebra

def sqrtm_psd(C: np.ndarray) -> np.ndarray:
    """Symmetric PSD matrix square root via eigendecomposition (clip tiny negatives)."""
    C = 0.5 * (np.asarray(C, float) + np.asarray(C, float).T)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def bures_w2(C1: np.ndarray, C2: np.ndarray) -> float:
    """EXACT 2-Wasserstein between N(m, C1) and N(m, C2) (equal means): the Bures metric
    ``W2^2 = tr[C1 + C2 - 2 (C2^{1/2} C1 C2^{1/2})^{1/2}]``.  (NOT ||C1^{1/2}-C2^{1/2}||_F,
    which holds only when C1, C2 commute -- the memo Prop-2 correction.)"""
    C1 = np.asarray(C1, float)
    C2 = np.asarray(C2, float)
    s2 = sqrtm_psd(C2)
    mid = sqrtm_psd(s2 @ C1 @ s2)
    val = float(np.trace(C1) + np.trace(C2) - 2.0 * np.trace(mid))
    return float(np.sqrt(max(val, 0.0)))


def sqrt_diff_fro(C1: np.ndarray, C2: np.ndarray) -> float:
    """``||C1^{1/2} - C2^{1/2}||_F`` -- the Procrustes UPPER bound on ``bures_w2`` (W2 <= this)."""
    return float(np.linalg.norm(sqrtm_psd(C1) - sqrtm_psd(C2), "fro"))


# --------------------------------------------------------------------------- active/null split

def common_mode_direction(n: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Unit vector of the sigma COMMON mode within the length-``n`` sigma sub-block.

    Fixed in parameter space (memo scope condition 1).  Default = all-ones/sqrt(n) (the
    variant-D common mode); pass ``weights`` (e.g. gamma_j * Qbar_j at the nominal trajectory)
    for the capacity-weighted direction -- but it must be FIXED, never action/draw-dependent.
    """
    if weights is None:
        e = np.ones(n, float)
    else:
        e = np.asarray(weights, float).copy()
    nrm = np.linalg.norm(e)
    if nrm == 0.0:
        raise ValueError("common-mode weights are all zero")
    return e / nrm


def rotate_sigma_block(Sigma: np.ndarray, G: np.ndarray, n: int,
                       e_c: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed orthogonal rotation on the sigma sub-block (indices ``[3n:4n]`` of ``u``) so the
    common mode is the FIRST sigma coordinate and the ``(n-1)`` differential modes follow.

    Returns ``(Sigma_rot, G_rot, u_idx, v_idx)`` where ``v_idx`` are the differential-null
    coordinates (the compressible block) and ``u_idx`` everything else (incl. sigma common mode).
    Identity on the non-sigma coordinates; a genuine change of basis on the sigma block.
    """
    Sigma = np.asarray(Sigma, float)
    G = np.atleast_2d(np.asarray(G, float))
    dim = Sigma.shape[0]
    if e_c is None:
        e_c = common_mode_direction(n)
    # Householder-style orthonormal basis with e_c first, differential complement after.
    Q, _ = np.linalg.qr(np.column_stack([e_c, np.eye(n)]))  # (n, n), first col ~ ±e_c
    if np.dot(Q[:, 0], e_c) < 0:
        Q[:, 0] *= -1.0
    R = np.eye(dim)
    sig = slice(3 * n, 4 * n)
    R[sig, sig] = Q
    Sigma_rot = R.T @ Sigma @ R
    G_rot = G @ R
    v_idx = np.arange(3 * n + 1, 4 * n)          # differential sigma modes (drop the common mode col)
    u_idx = np.array([i for i in range(dim) if i not in set(v_idx.tolist())])
    return Sigma_rot, G_rot, u_idx, v_idx


def _blocks(Sigma, u_idx, v_idx):
    Suu = Sigma[np.ix_(u_idx, u_idx)]
    Suv = Sigma[np.ix_(u_idx, v_idx)]
    Svv = Sigma[np.ix_(v_idx, v_idx)]
    return Suu, Suv, Svv


def schur_cov(Sigma: np.ndarray, u_idx: np.ndarray, v_idx: np.ndarray) -> np.ndarray:
    """Cov of the conditional-mean compression ``theta~ = (u, E[v|u])`` (memo Def 3).
    v-block becomes ``Svu Suu^{-1} Suv``; u-block and u-v cross-blocks unchanged."""
    Sigma = np.asarray(Sigma, float)
    Suu, Suv, _ = _blocks(Sigma, u_idx, v_idx)
    out = Sigma.copy()
    out[np.ix_(v_idx, v_idx)] = Suv.T @ np.linalg.solve(Suu, Suv)
    return out


def freeze_cov(Sigma: np.ndarray, u_idx: np.ndarray, v_idx: np.ndarray) -> np.ndarray:
    """Cov of the blunt freeze ``theta = (u, mu_v)`` (v fixed at its mean): the v rows/cols are
    zeroed (variance AND the u-v cross-covariance destroyed) -- the engineering variant-D map."""
    Sigma = np.asarray(Sigma, float)
    out = Sigma.copy()
    out[v_idx, :] = 0.0
    out[:, v_idx] = 0.0
    return out


def prior_only_cov(Sigma: np.ndarray, blk_idx: np.ndarray, prior_var: np.ndarray) -> np.ndarray:
    """Restore ``blk_idx`` to its PRIOR variance and drop its cross-covariance (negative control:
    variant-C when ``blk_idx`` is the WHOLE sigma block incl. common mode)."""
    Sigma = np.asarray(Sigma, float)
    out = Sigma.copy()
    out[blk_idx, :] = 0.0
    out[:, blk_idx] = 0.0
    out[np.ix_(blk_idx, blk_idx)] = np.diag(np.asarray(prior_var, float))
    return out


def decision_cov(G: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """``C = G cov G^T`` (raw decision covariance)."""
    G = np.atleast_2d(np.asarray(G, float))
    return G @ np.asarray(cov, float) @ G.T


def variant_d_cov(Sigma: np.ndarray, G: np.ndarray, u_idx: np.ndarray, v_idx: np.ndarray) -> np.ndarray:
    """variant-D decision covariance ``C_D = G_u H_uu^{-1} G_u^T`` (``H = Sigma^{-1}``): freeze v at its
    mean and condition u on v (the paper's restricted local-Laplace posterior)."""
    G = np.atleast_2d(np.asarray(G, float))
    H = np.linalg.inv(np.asarray(Sigma, float))
    Gu = G[:, u_idx]
    return Gu @ np.linalg.inv(H[np.ix_(u_idx, u_idx)]) @ Gu.T


def ridge_derivative(Sigma: np.ndarray, G: np.ndarray, u_idx: np.ndarray, v_idx: np.ndarray) -> np.ndarray:
    """Effective decision derivative along the posterior ridge ``Gamma_v = G_v - G_u H_uu^{-1} H_uv``
    (the total dg/dv when u tracks its conditional mean E[u|v]).  variant-D's exact residual is
    ``C - C_D = Gamma_v Sigma_vv Gamma_v^T`` (law of total variance conditioning on v)."""
    G = np.atleast_2d(np.asarray(G, float))
    H = np.linalg.inv(np.asarray(Sigma, float))
    Gu, Gv = G[:, u_idx], G[:, v_idx]
    return Gv - Gu @ np.linalg.solve(H[np.ix_(u_idx, u_idx)], H[np.ix_(u_idx, v_idx)])


def compression_certificates(Sigma, G, tol, u_idx, v_idx) -> dict:
    """The two exact conditional-variance residuals and their tolerance-whitened certificates (memo §4b R1).

    Schur:     C - C_S = G_v Sigma_{v|u} G_v^T ,  delta_S = ‖T^{-1} G_v Sigma_{v|u}^{1/2}‖_2 .
    variant-D: C - C_D = Gamma_v Sigma_vv Gamma_v^T ,  delta_D = ‖T^{-1} Gamma_v Sigma_vv^{1/2}‖_2 .
    Also returns the identity residuals (should be ~0), ‖Gamma_v‖, the marginal width ‖Sigma_vv^{1/2}‖,
    and c_min = min eig of the tolerance-whitened Schur-compressed decision covariance (non-degeneracy).
    """
    Sigma = np.asarray(Sigma, float)
    G = np.atleast_2d(np.asarray(G, float))
    tol = np.asarray(tol, float)
    Ti = np.diag(1.0 / tol)
    H = np.linalg.inv(Sigma)
    Gv = G[:, v_idx]
    Sv_u = np.linalg.inv(H[np.ix_(v_idx, v_idx)])              # Sigma_{v|u} = (H_vv)^{-1}
    Svv = Sigma[np.ix_(v_idx, v_idx)]
    Gamma = ridge_derivative(Sigma, G, u_idx, v_idx)
    C_full = decision_cov(G, Sigma)
    C_S = decision_cov(G, schur_cov(Sigma, u_idx, v_idx))
    C_D = variant_d_cov(Sigma, G, u_idx, v_idx)
    delta_S = float(np.linalg.norm(Ti @ Gv @ sqrtm_psd(Sv_u), 2))
    delta_D = float(np.linalg.norm(Ti @ Gamma @ sqrtm_psd(Svv), 2))
    c_min = float(np.linalg.eigvalsh(whiten(C_S, tol)).min())
    return {
        "id_S": float(np.linalg.norm((C_full - C_S) - Gv @ Sv_u @ Gv.T)),
        "id_D": float(np.linalg.norm((C_full - C_D) - Gamma @ Svv @ Gamma.T)),
        "delta_S": delta_S, "delta_D": delta_D,
        "gamma_v_norm": float(np.linalg.norm(Gamma, 2)),
        "Svv_sqrt_norm": float(np.linalg.norm(sqrtm_psd(Svv), 2)),
        "c_min": c_min,
    }


def whiten(C: np.ndarray, tol: np.ndarray) -> np.ndarray:
    """Tolerance-whitening ``Cbar = T^{-1} C T^{-1}``, ``T = diag(tol)``."""
    d = 1.0 / np.asarray(tol, float)
    return np.asarray(C, float) * d[:, None] * d[None, :]


# --------------------------------------------------------------------------- risk / ladder

def shortfall_risk(mean: np.ndarray, cov: np.ndarray, spec: np.ndarray, w: np.ndarray) -> float:
    """Gaussian closed-form Bayes risk of the separable shortfall loss
    ``ell = sum_q w_q [spec_q - g_q]_+`` under ``g ~ N(mean, cov)`` (uses marginal variances only):
    ``E[(s-X)_+] = (s-mu) Phi(z) + sigma phi(z)``, ``z = (s-mu)/sigma``."""
    from math import erf, pi, sqrt as _sqrt
    mean = np.asarray(mean, float)
    var = np.clip(np.diag(np.asarray(cov, float)), 0.0, None)
    w = np.asarray(w, float)
    spec = np.asarray(spec, float)
    total = 0.0
    for q in range(mean.size):
        s = float(spec[q]); m = float(mean[q]); sd = float(np.sqrt(var[q]))
        if sd <= 0.0:
            total += float(w[q]) * max(s - m, 0.0)
            continue
        z = (s - m) / sd
        Phi = 0.5 * (1.0 + erf(z / _sqrt(2.0)))
        phi = np.exp(-0.5 * z * z) / _sqrt(2.0 * pi)
        total += float(w[q]) * ((s - m) * Phi + sd * phi)
    return float(total)


def theorem_ladder(Sigma, G, tol, u_idx, v_idx) -> dict:
    """Compute the memo's checked inequality chain for the Schur compression at one posterior.

    Returns identity residual ``e_id`` (memo Prop 1: gap == G_v Sigma_{v|u} G_v^T exactly),
    the tolerance-whitened covariance-gap bound ratio ``q_C``, the exact Bures ``W2``, the
    sqrt-difference upper-bound ratio ``q_W``, and ``c_min`` (active-decision non-degeneracy).
    The per-action risk-gap is computed by the caller via :func:`shortfall_risk`.
    All ratios should be <= 1 up to numerical tolerance.
    """
    Sigma = np.asarray(Sigma, float)
    G = np.atleast_2d(np.asarray(G, float))
    tol = np.asarray(tol, float)
    Suu, Suv, Svv = _blocks(Sigma, u_idx, v_idx)
    Sv_u = Svv - Suv.T @ np.linalg.solve(Suu, Suv)      # Cov(v|u)

    C_full = decision_cov(G, Sigma)
    C_schur = decision_cov(G, schur_cov(Sigma, u_idx, v_idx))
    Cbar_full = whiten(C_full, tol)
    Cbar_schur = whiten(C_schur, tol)
    gap = Cbar_full - Cbar_schur

    Bv = whiten_rows(G[:, v_idx], tol)                  # T^{-1} G_v
    ident = Bv @ Sv_u @ Bv.T                            # memo Prop 1: gap == this exactly
    e_id = float(np.linalg.norm(gap - ident, "fro"))

    bound_cov = (np.linalg.norm(Bv, 2) ** 2) * float(np.linalg.norm(Sv_u, 2))
    q_C = float(np.linalg.norm(gap, 2) / bound_cov) if bound_cov > 0 else float("nan")

    W2 = bures_w2(Cbar_full, Cbar_schur)
    c_min = float(np.linalg.eigvalsh(Cbar_schur).min())
    sqrt_ub = sqrt_diff_fro(Cbar_full, Cbar_schur)
    q_W = float(W2 / sqrt_ub) if sqrt_ub > 0 else float("nan")

    return {"e_id": e_id, "gap_op": float(np.linalg.norm(gap, 2)), "q_C": q_C,
            "W2": W2, "sqrt_ub": sqrt_ub, "q_W": q_W, "c_min": c_min,
            "Sv_u_op": float(np.linalg.norm(Sv_u, 2)), "Bv_op": float(np.linalg.norm(Bv, 2))}


def whiten_rows(Grows: np.ndarray, tol: np.ndarray) -> np.ndarray:
    """``T^{-1} G_rows`` (whiten the QoI rows of a Jacobian block)."""
    d = 1.0 / np.asarray(tol, float)
    return d[:, None] * np.atleast_2d(np.asarray(Grows, float))


def action_gap(risks_full: np.ndarray, risks_comp: np.ndarray) -> dict:
    """Finite-candidate action-preservation check (memo Prop 3).

    ``risks_full``/``risks_comp`` = Bayes risk of each candidate action under the full and the
    compressed posterior.  Reports the full-posterior action gap ``Delta_act``, the sup risk
    perturbation ``eta_R``, the ratio ``2 eta_R / Delta_act``, and whether argmin is preserved.
    """
    rf = np.asarray(risks_full, float)
    rc = np.asarray(risks_comp, float)
    a_full = int(np.argmin(rf))
    a_comp = int(np.argmin(rc))
    order = np.argsort(rf)
    delta_act = float(rf[order[1]] - rf[order[0]]) if rf.size >= 2 else float("inf")
    eta_r = float(np.max(np.abs(rf - rc)))
    return {"a_full": a_full, "a_comp": a_comp, "preserved": bool(a_full == a_comp),
            "delta_act": delta_act, "eta_R": eta_r,
            "ratio": float(2.0 * eta_r / delta_act) if delta_act > 0 else float("inf")}
