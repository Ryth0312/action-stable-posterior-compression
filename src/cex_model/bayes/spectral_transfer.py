"""Spectral-alignment transfer: WHICH posterior subspace the process decision lives in.

This is the proof-grade complement to the empirical decision-vs-parameter decoupling
(:mod:`cex_model.bayes.decision`).  ``identifiability_report`` answers "is theta
identified?" via the worst PRIOR-whitened direction ``worst_dir = sqrt(lambda_max(M))``,
``M = diag(1/sigma_prior) Sigma diag(1/sigma_prior)``; ``decision_report`` answers "is
the decision g(theta) determined?" via the worst TOLERANCE-whitened direction
``worst_dec = sqrt(lambda_max(T^-1 G Sigma Gᵀ T^-1))``.  This module proves WHY the two
can disagree, by resolving ``worst_dec`` over the eigen-subspaces of ``M``.

Exact identity (Proposition 1).  With ``D = diag(1/sigma_prior)`` so ``Sigma = D^-1 M D^-1``,
and ``B = T^-1 G D^-1 = T^-1 G diag(sigma_prior)`` (the tolerance-and-prior-whitened
decision map), the tolerance-whitened decision covariance factorises through ``M``'s
eigenpairs ``M = sum_k lambda_k v_k v_kᵀ``::

    C~ = T^-1 (G Sigma Gᵀ) T^-1 = B M Bᵀ = sum_k lambda_k b_k b_kᵀ ,   b_k = B v_k

so ``worst_dec = sqrt(lambda_max(sum_k lambda_k b_k b_kᵀ))`` splits cleanly into the
SLOPPY block (eigen-directions with ``sqrt(lambda_k) > tau`` -- the ones theta-space
cannot identify) and the STIFF block.  The decision decouples from parameter
non-identifiability iff the decision map has negligible energy ``b_k`` on the sloppy
block.  We report that energy as ``alpha = ||B P_sloppy||_F / ||B P_stiff||_F`` and the
residual ``||b_{k*}||`` of the single widest (k*, sloppiest) direction -- empirically the
steric-factor (sigma) direction -- the witness that the unidentified direction is
decision-null for separation-type QoIs.

Pure numpy -- no solver, no torch.  ``worst_dir`` reproduces
``active.identifiability_report`` and ``worst_dec`` reproduces
``decision._report_from_covariance`` to machine precision (verified in tests), so this is
a re-expression of the committed criteria, not a new estimand.
"""

from __future__ import annotations

import numpy as np

__all__ = ["decompose_decision"]


def decompose_decision(cov, G, prior_std, tol, *, tau: float = 0.5,
                       names: list[str] | None = None, return_matrices: bool = False) -> dict:
    """Subspace-resolved decision identifiability (Proposition 1).

    Parameters
    ----------
    cov : (dim, dim)
        Posterior covariance ``Sigma`` in u-space (e.g. ``Posterior.cov``).
    G : (k_dec, dim)
        Decision Jacobian ``dg/du`` (e.g. ``decision.decision_jacobian``).
    prior_std : (dim,)
        Prior std in u-space (e.g. ``physical_prior(n).std``).
    tol : (k_dec,)
        Decision tolerances (same order as ``G``'s rows / ``DECISION_NAMES``).
    tau : float, default 0.5
        Identifiability threshold: a prior-whitened direction is SLOPPY iff its
        posterior/prior std ``sqrt(lambda_k) > tau`` (matches ``identifiability_report``).
    names : list[str], optional
        u-space parameter names; if given, each reported direction is annotated with the
        parameters carrying most of its weight (top-3 by |eigenvector entry|).
    return_matrices : bool, default False
        Also return the (tolerance-whitened) decision covariance ``C_decision`` and its
        ``C_sloppy`` / ``C_stiff`` blocks -- used by the exact-split test and the figure.

    Returns
    -------
    dict with::

        worst_dir              sqrt(lambda_max(M))            == identifiability_report
        worst_dec              sqrt(lambda_max(C~))           == decision_report
        worst_dec_sloppy       sloppy-block contribution to worst_dec
        worst_dec_stiff        stiff-block contribution
        alpha                  ||B P_sloppy||_F / ||B P_stiff||_F  (decision energy ratio)
        kappa_g_submult        ||B||_2  (the vacuous submultiplicative constant, by design)
        residual_widest        ||b_{k*}|| of the single widest/sloppiest direction (decision-null witness)
        n_sloppy, tau          bookkeeping
        directions             per-direction {s, b_norm, sloppy, [top_params]} widest-first
        [C_decision, C_sloppy, C_stiff]   if return_matrices
    """
    cov = np.asarray(cov, float)
    G = np.atleast_2d(np.asarray(G, float))
    prior_std = np.asarray(prior_std, float)
    tol = np.asarray(tol, float)
    dim = cov.shape[0]
    if cov.shape != (dim, dim):
        raise ValueError(f"cov must be square, got {cov.shape}")
    if G.shape[1] != dim:
        raise ValueError(f"G has {G.shape[1]} columns, expected dim={dim}")
    if prior_std.shape != (dim,):
        raise ValueError(f"prior_std has shape {prior_std.shape}, expected ({dim},)")
    if tol.shape != (G.shape[0],):
        raise ValueError(f"tol has shape {tol.shape}, expected ({G.shape[0]},)")

    # prior-whitened posterior: M = D Sigma D, D = diag(1/sigma_prior). Same construction
    # as identifiability_report, so eig(M) and worst_dir match it exactly.
    d = 1.0 / prior_std
    M = cov * d[:, None] * d[None, :]
    lam, V = np.linalg.eigh(M)                 # ascending; V columns orthonormal v_k
    lam = np.clip(lam, 0.0, None)
    s = np.sqrt(lam)                           # per-direction posterior/prior std ratio
    worst_dir = float(s.max()) if s.size else 0.0

    # tolerance-and-prior-whitened decision map B = T^-1 G diag(sigma_prior); b_k = B v_k.
    dinv = 1.0 / tol
    B = dinv[:, None] * G * prior_std[None, :]  # (k_dec, dim)
    b = B @ V                                   # (k_dec, dim); column k is b_k

    def _block(mask):
        if not np.any(mask):
            k = G.shape[0]
            return np.zeros((k, k))
        bm = b[:, mask]
        return (bm * lam[mask][None, :]) @ bm.T  # sum_k lambda_k b_k b_kᵀ over the block

    sloppy = s > tau
    C_full = _block(np.ones_like(sloppy, dtype=bool))   # == T^-1 G Sigma Gᵀ T^-1
    C_sl = _block(sloppy)
    C_st = _block(~sloppy)

    def _worst(C):
        return float(np.sqrt(max(float(np.linalg.eigvalsh(C).max()), 0.0))) if C.size else 0.0

    worst_dec = _worst(C_full)
    worst_dec_sloppy = _worst(C_sl) if np.any(sloppy) else 0.0
    worst_dec_stiff = _worst(C_st) if np.any(~sloppy) else 0.0

    # decision-energy alignment: how much of B sits on the sloppy vs stiff eigenbasis.
    b_norm = np.linalg.norm(b, axis=0)          # ||b_k||
    energy_sl = float(np.sqrt(np.sum(b_norm[sloppy] ** 2))) if np.any(sloppy) else 0.0
    energy_st = float(np.sqrt(np.sum(b_norm[~sloppy] ** 2))) if np.any(~sloppy) else 0.0
    alpha = energy_sl / energy_st if energy_st > 0.0 else float("inf")
    kappa_g = float(np.linalg.svd(B, compute_uv=False)[0]) if B.size else 0.0

    kstar = int(np.argmax(lam)) if lam.size else 0   # the single widest (sloppiest) direction
    residual_widest = float(b_norm[kstar]) if b_norm.size else 0.0

    order = np.argsort(lam)[::-1]                # widest (most uncertain) first
    directions = []
    for k in order:
        entry = {"s": float(s[k]), "b_norm": float(b_norm[k]), "sloppy": bool(sloppy[k])}
        if names is not None:
            w = np.abs(V[:, k])
            top = np.argsort(w)[::-1][:3]
            entry["top_params"] = [names[i] for i in top]
        directions.append(entry)

    out = {
        "worst_dir": worst_dir,
        "worst_dec": worst_dec,
        "worst_dec_sloppy": worst_dec_sloppy,
        "worst_dec_stiff": worst_dec_stiff,
        "alpha": alpha,
        "energy_sloppy": energy_sl,
        "energy_stiff": energy_st,
        "kappa_g_submult": kappa_g,
        "residual_widest": residual_widest,
        "n_sloppy": int(np.sum(sloppy)),
        "tau": float(tau),
        "directions": directions,
    }
    if return_matrices:
        out["C_decision"] = C_full
        out["C_sloppy"] = C_sl
        out["C_stiff"] = C_st
    return out
