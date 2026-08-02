"""QoI curvature κ_q for the decision-interval coverage bound (docs/decision_null_theorem.md §4, T2).

T2 bounds the error of the linearized (delta-method) decision interval by the QoI curvature

    κ_q = ‖G2_q Σ‖_F / (G_qᵀ Σ G_q),     G_q = ∇_u q ,  G2_q = ∇²_u q ,

equivalently the second-order/first-order delta-method variance-contribution ratio (= a Σ-metric
Bates–Watts (1980) parameter-effects curvature).  For a scalar QoI ``q(θ)`` with ``θ ~ N(u_MAP, Σ)``,
the exact second-order variance is ``Var = G_qᵀΣG_q + ½Tr[(G2_q Σ)²]``, so the linearized variance
``G_qᵀΣG_q`` has relative error ``δ_q = ½Tr[(G2_q Σ)²]/(G_qᵀΣG_q)`` -- the quantity the §2.9/§3.7
rel-Frobenius flag estimates (Corollary 2a), and the predictor of the n=2 delta-method breakdown.

Solver subtlety (load-bearing): the differentiable BDF solver makes the FORWARD pass differentiable
by an implicit-function-theorem correction with a DETACHED Jacobian, so the reverse-mode GRADIENT is
exact but a naive reverse-over-reverse HESSIAN can miss second-order IFT terms.  We therefore take the
**finite difference of the (trusted) reverse-mode gradient** as the reference Hessian and report the
pure-autograd Hessian only as a *validity check* of double-backward through the solver.
"""

from __future__ import annotations

import numpy as np
import torch

from cex_model.bayes.decision import decision_forward
from cex_model.diffsolver.torch_solver import DTYPE

__all__ = [
    "curvature_kappa",
    "qoi_gradient",
    "qoi_hessian_fd",
    "qoi_hessian_autograd",
    "hessian_check",
]


# --------------------------------------------------------------------------- pure helper

def curvature_kappa(G2, Sigma, G_q) -> dict:
    """κ_q and the exact second-order delta-method variance decomposition (pure linear algebra).

    Returns ``kappa_q = ‖G2Σ‖_F/(GᵀΣG)``, ``first_order_var = GᵀΣG`` (linearized), the second-order
    correction ``½Tr[(G2Σ)²]``, and ``delta_rel_var_error = correction/first`` (the relative error of
    the linearized variance; the rel-Frobenius predictor, Corollary 2a)."""
    G2 = np.asarray(G2, float)
    Sigma = np.asarray(Sigma, float)
    G_q = np.asarray(G_q, float).ravel()
    HS = G2 @ Sigma
    first = float(G_q @ Sigma @ G_q)
    second = 0.5 * float(np.trace(HS @ HS))           # ½Tr[(G2Σ)²]  -- exact 2nd-order variance term
    kappa = float(np.linalg.norm(HS, "fro")) / first if first > 0 else float("inf")
    return {
        "kappa_q": kappa,
        "first_order_var": first,
        "second_order_var": second,
        "delta_rel_var_error": (second / first) if first > 0 else float("inf"),
        "q_std_linear": float(np.sqrt(first)) if first > 0 else float("nan"),
    }


# ----------------------------------------------------------------------------- solver path

def _q_fn(bundle, op, u, n_steps, qoi_index, spec):
    """Scalar QoI ``q(u) = g(u)[qoi_index]`` (0=pool purity, 1=pool yield) and the base u tensor."""
    g_fn, _, _ = decision_forward(bundle, op, u, n_steps=n_steps, **spec)
    u_t = torch.tensor(np.asarray(u, float), dtype=DTYPE)

    def q(uu):
        return g_fn(uu)[qoi_index]
    return q, u_t


def qoi_gradient(bundle, op, u, *, n_steps: int = 120, qoi_index: int = 0, **spec) -> np.ndarray:
    """``G_q = ∇_u q`` via reverse-mode (the trusted gradient through the implicit-diff solver)."""
    q, u_t = _q_fn(bundle, op, u, n_steps, qoi_index, spec)
    return torch.autograd.functional.jacobian(q, u_t).detach().numpy()


def qoi_hessian_fd(bundle, op, u, *, n_steps: int = 120, qoi_index: int = 0,
                   delta: float = 1e-4, **spec) -> np.ndarray:
    """Reference Hessian = central finite difference of the trusted reverse-mode gradient (symmetrized)."""
    q, u_t = _q_fn(bundle, op, u, n_steps, qoi_index, spec)

    def grad(uu):
        return torch.autograd.functional.jacobian(q, uu).detach().numpy()
    dim = u_t.numel()
    H = np.zeros((dim, dim))
    for j in range(dim):
        up = u_t.clone(); up[j] += delta
        um = u_t.clone(); um[j] -= delta
        H[:, j] = (grad(up) - grad(um)) / (2.0 * delta)
    return 0.5 * (H + H.T)


def qoi_hessian_autograd(bundle, op, u, *, n_steps: int = 120, qoi_index: int = 0, **spec) -> np.ndarray:
    """Pure reverse-over-reverse Hessian (double-backward through the BDF solver) -- the object whose
    correctness through the detached-IFT solver is checked against ``qoi_hessian_fd``."""
    q, u_t = _q_fn(bundle, op, u, n_steps, qoi_index, spec)
    return torch.autograd.functional.hessian(q, u_t).detach().numpy()


def hessian_check(bundle, op, u, *, n_steps: int = 120, qoi_index: int = 0, delta: float = 1e-4,
                  autograd: bool = True, **spec) -> dict:
    """FD reference Hessian + (optional) autograd-vs-FD validity check of double-backward.

    Returns ``H_fd`` (trusted), symmetry residual, and -- if ``autograd`` -- ``rel_err`` between the
    reverse-over-reverse Hessian and ``H_fd`` (large ⇒ the detached-IFT solver breaks double-backward,
    so κ_q must use the FD Hessian)."""
    H_fd = qoi_hessian_fd(bundle, op, u, n_steps=n_steps, qoi_index=qoi_index, delta=delta, **spec)
    out = {"H_fd": H_fd, "fro": float(np.linalg.norm(H_fd))}
    if autograd:
        H_ag = qoi_hessian_autograd(bundle, op, u, n_steps=n_steps, qoi_index=qoi_index, **spec)
        denom = float(np.linalg.norm(H_fd)) + 1e-30
        out["autograd_vs_fd_rel_err"] = float(np.linalg.norm(H_ag - H_fd) / denom)
        out["H_autograd"] = H_ag
    return out


def curvature_report(Sigma, bundle, op, u, *, n_steps: int = 120, delta: float = 1e-4,
                     qoi_names=("pool_purity", "pool_yield"), check_autograd: bool = True,
                     **spec) -> dict:
    """Per-QoI κ_q (from the FD-verified Hessian) + the delta-method relative-variance error.

    ``Sigma`` is the posterior covariance in u-space (``Posterior.cov``); ``u`` the MAP.  For each
    scalar decision QoI it returns κ_q, the linearized vs second-order variance, ``delta_rel_var_error``
    (the rel-Frobenius predictor) and the autograd-Hessian validity residual."""
    Sigma = np.asarray(Sigma, float)
    out = {"n_steps": n_steps, "qoi": {}}
    for i, name in enumerate(qoi_names):
        G_q = qoi_gradient(bundle, op, u, n_steps=n_steps, qoi_index=i, **spec)
        chk = hessian_check(bundle, op, u, n_steps=n_steps, qoi_index=i, delta=delta,
                            autograd=check_autograd, **spec)
        kap = curvature_kappa(chk["H_fd"], Sigma, G_q)
        rec = dict(kap)
        rec["hessian_fro"] = chk["fro"]
        if check_autograd:
            rec["autograd_vs_fd_rel_err"] = chk["autograd_vs_fd_rel_err"]
        out["qoi"][name] = rec
    return out
