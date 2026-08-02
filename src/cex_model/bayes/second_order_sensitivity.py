"""Path A: analytic BDF-matched second-order sensitivity for a trustworthy kappa_q Hessian-vector
product through the IFT-BDF SMA solver.

Motivation (results/bayes/kappa_value_fd_check.json, docs/decision_null_theorem.md §4.2): reverse-over-
reverse autograd through the detached-IFT solver is wrong, and value-FD blows up on the wide whitened
sloppy axes (d*L*e_j steps into forward breakdown / window-edge discontinuities). Path A instead
evaluates the Hessian LOCALLY at the operating point by propagating the (first and mixed second)
variational equations along the well-behaved MAP trajectory -- it never finite-differences into the
ill-behaved region, so it is immune.

Key idea: the SOLVER's Newton solve detaches its Jacobian (breaking double-backward), but ``sim.rhs``
is a plain differentiable torch function, so all RHS derivatives are SAFE via autograd. We do the TIME
propagation analytically, matched to the solver's BDF1->BDF2 scheme (verified in restricted_sigma_gain).

Discretization (from ``TorchSimulator._newton``: ``a0 y - h f(y) = hist``, ``a0=1`` at k=1 then 1.5,
``hist = 2 y_{k-1} - 0.5 y_{k-2}``):
  first  variational   ``(a0 I - h J_k) S_k   = (2 S_{k-1}-0.5 S_{k-2})   + h F_{u,k}``
  mixed second (dir v) ``(a0 I - h J_k) W_k   = (2 W_{k-1}-0.5 W_{k-2})   + h Q_k``,
     ``Q_k = f_yy(s_v, S_k) + f_yu(S_k, v) + f_yu(s_v, e_j) + f_uu(e_j, v)`` (columns j; s_v=S_k v),
     which is exactly the directional derivative along (s_v, v) of ``J(y,u) S_k + F_u(y,u)`` (S held fixed).
Terminal QoI ``q = g(y(tau))`` -> HVP ``(H v)_j = g_y . W_j(tau) + S_j(tau) . g_yy . (S(tau) v)``.

Window-integral (ratio) decision QoIs need state augmentation (running window integrals as terminal
states); that wiring + the full validation ladder + real-product runs are the next milestone. This
module covers the propagator core + a terminal QoI; tests validate S vs FD, the second forcing vs FD,
and the terminal HVP vs FD-of-gradient.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import torch

from cex_model.bayes.groenwall import _forward_full_state
from cex_model.bayes.likelihood import unpack_u
from cex_model.diffsolver.torch_solver import DTYPE

__all__ = [
    "StepCache",
    "build_step_cache",
    "build_step_cache_generic",
    "forward_first_variational",
    "second_forcing",
    "forward_second_variational",
    "second_variational_hvp",
    "terminal_gradient",
    "decision_trajectory_functional",
    "trajectory_functional_hvp",
]


def _rhs_of(sim, y_t, inlet_c, u_t, n):
    """``f(y; u)`` with the u->model-unit map (``unpack_u``) composed, so autograd w.r.t. ``u`` gives
    ``df/du`` directly and w.r.t. ``y`` gives ``J``. ``u = [log10 keq, log10 kkin, nu, sigma]``."""
    keq, kkin, nu, sig = unpack_u(u_t, n)
    return sim.rhs(y_t, inlet_c, keq, kkin, nu, nu + sig)


class StepCache:
    """Per-step BDF data shared by the first- and second-order variational passes: a generic RHS
    ``rhs_yu(k, y_t, u_t) -> f`` (differentiable in y and u), the forward state ``Y``, step
    sizes/coefficients, the LU of ``A_k = a0 I - h J_k`` (reused for S and W), and ``F_u,k = df/du``.
    Built once per (system, u). The generic RHS lets the SAME propagator run on the SMA solver AND on
    a differentiable-BDF ground-truth model (CMC) where a full-AD Hessian exists."""

    def __init__(self, rhs_yu, Y, t, a0, lu, jac, Fu, u):
        self.rhs_yu = rhs_yu; self.Y = np.asarray(Y, float); self.t = np.asarray(t, float)
        self.a0 = a0; self.lu = lu; self.jac = jac; self.Fu = Fu
        self.u = np.asarray(u, float)
        self.nt = len(self.t); self.ndim = self.Y.shape[1]; self.dim_u = len(self.u)


def build_step_cache_generic(rhs_yu, Y, t, u) -> StepCache:
    """Build the cache from a generic RHS ``rhs_yu(k, y_t, u_t) -> f`` and a precomputed forward state
    ``Y`` on grid ``t``. Uses the EXACT autograd ``df/dy`` (not a clipped Newton-accelerator)."""
    u = np.asarray(u, float); u_t = torch.tensor(u, dtype=DTYPE)
    Y = np.asarray(Y, float); t = np.asarray(t, float)
    nt = len(t); ndim = Y.shape[1]; eye = np.eye(ndim)
    a0_arr = [0.0] + [1.0 if k == 1 else 1.5 for k in range(1, nt)]
    lu = [None] * nt; jac = [None] * nt; Fu = [None] * nt
    for k in range(1, nt):
        h = float(t[k] - t[k - 1])
        y_t = torch.tensor(Y[k], dtype=DTYPE)
        J = torch.autograd.functional.jacobian(lambda yy: rhs_yu(k, yy, u_t), y_t).detach().numpy()
        jac[k] = J
        lu[k] = sla.lu_factor(a0_arr[k] * eye - h * J)
        Fu[k] = torch.autograd.functional.jacobian(lambda uu: rhs_yu(k, y_t, uu), u_t).detach().numpy()
    return StepCache(rhs_yu, Y, t, a0_arr, lu, jac, Fu, u)


def build_step_cache(sim, u, n) -> StepCache:
    """SMA adapter: forward state from the solver's BDF ``_forward_full_state``, RHS via ``sim.rhs``
    with the ``u -> model`` map (``unpack_u``) composed."""
    def rhs_yu(k, y_t, u_t):
        keq, kkin, nu, sig = unpack_u(u_t, n)
        return sim.rhs(y_t, sim.inlet_c[k], keq, kkin, nu, nu + sig)
    u_t = torch.tensor(np.asarray(u, float), dtype=DTYPE)
    with torch.no_grad():
        Y = _forward_full_state(sim, *unpack_u(u_t, n)).detach().numpy()
    return build_step_cache_generic(rhs_yu, Y, sim.t.detach().numpy(), u)


def forward_first_variational(cache: StepCache):
    """S_k = dy_k/du at every step by the direct BDF variational recurrence. Returns a list ``S`` of
    ``(ndim, dim_u)`` arrays (``S[0]=0``)."""
    nt, ndim, dim_u = cache.nt, cache.ndim, cache.dim_u
    S = [np.zeros((ndim, dim_u))]
    for k in range(1, nt):
        h = float(cache.t[k] - cache.t[k - 1])
        hist = S[k - 1] if k == 1 else (2.0 * S[k - 1] - 0.5 * S[k - 2])
        S.append(sla.lu_solve(cache.lu[k], hist + h * cache.Fu[k]))
    return S


def second_forcing(cache: StepCache, k: int, S_k, s_v, v) -> np.ndarray:
    """Mixed second-variational forcing ``Q_k`` (``ndim, dim_u``) at step ``k``: the directional
    derivative along ``(s_v, v)`` of ``J(y,u) S_k + F_u(y,u)`` (with ``S_k`` held fixed), which equals
    ``f_yy(s_v,S_k)+f_yu(S_k,v)+f_yu(s_v,e_j)+f_uu(e_j,v)`` column-wise. Computed as a forward-over-
    forward JVP on the (safe, differentiable) RHS: column ``j`` is
    ``Q[:,j] = D_(s_v,v) [ D_(S_k[:,j], e_j) f ](y,u)`` (an exact identity for the dense form above), so
    it needs only ``dim_u`` directional derivatives -- never the dense ``ndim x ndim`` Jacobian that the
    autograd-jacobian form materializes (ndim reverse passes/step). Matches the dense form to ~1e-15;
    ``vmap`` batches the ``dim_u`` columns."""
    y0 = torch.tensor(cache.Y[k], dtype=DTYPE)
    u0 = torch.tensor(cache.u, dtype=DTYPE)
    S_k_t = torch.tensor(np.asarray(S_k, float), dtype=DTYPE)
    s_v_t = torch.tensor(np.asarray(s_v, float), dtype=DTYPE)
    v_t = torch.tensor(np.asarray(v, float), dtype=DTYPE)
    eye_u = torch.eye(cache.dim_u, dtype=DTYPE)

    def rhs(y, u):
        return cache.rhs_yu(k, y, u)

    def col(y_tan, u_tan):                                       # Q[:,j], inner tangent (S_k[:,j], e_j)
        inner = lambda y, u: torch.func.jvp(rhs, (y, u), (y_tan, u_tan))[1]
        return torch.func.jvp(inner, (y0, u0), (s_v_t, v_t))[1]

    Q = torch.func.vmap(col, in_dims=(1, 0))(S_k_t, eye_u)       # (dim_u, ndim)
    return Q.T.detach().numpy()


def forward_second_variational(cache: StepCache, S, v):
    """Mixed second variational ``W_k = d^2 y_k / du d[v]`` at every step, in direction ``v``, by the
    same BDF recurrence as ``S`` (reusing the ``A_k`` LU) with the 4-term forcing ``Q_k``. Returns a
    list of ``(ndim, dim_u)`` arrays (``W[0]=0``)."""
    nt, ndim, dim_u = cache.nt, cache.ndim, cache.dim_u
    v = np.asarray(v, float)
    W = [np.zeros((ndim, dim_u))]
    W_prev = np.zeros((ndim, dim_u))
    for k in range(1, nt):
        h = float(cache.t[k] - cache.t[k - 1])
        s_v = S[k] @ v
        Q = second_forcing(cache, k, S[k], s_v, v)
        hist = W[k - 1] if k == 1 else (2.0 * W[k - 1] - 0.5 * W_prev)
        Wk = sla.lu_solve(cache.lu[k], hist + h * Q)
        W_prev = W[k - 1]; W.append(Wk)
    return W


def second_variational_hvp(cache: StepCache, S, v, g_y, g_yy=None) -> np.ndarray:
    """Terminal-QoI Hessian-vector product ``H v`` (``dim_u,``) for ``q = g(y(tau))``, given the
    first-order trajectory ``S``, probe ``v``, ``g_y = dg/dy`` (``ndim,``) and optional
    ``g_yy = d2g/dy2`` (``ndim, ndim``; None => linear QoI). ``(H v)_j = g_y . W_j(tau) + S_j(tau) .
    g_yy . (S(tau) v)``."""
    W = forward_second_variational(cache, S, v)
    g_y = np.asarray(g_y, float)
    Hv = W[-1].T @ g_y
    if g_yy is not None:
        s_v_tau = S[-1] @ np.asarray(v, float)
        Hv = Hv + S[-1].T @ (np.asarray(g_yy, float) @ s_v_tau)
    return Hv


def terminal_gradient(cache: StepCache, S, g_y) -> np.ndarray:
    """First-order gradient ``dq/du = S(tau)^T g_y`` for ``q = g(y(tau))`` (``dim_u,``)."""
    return S[-1].T @ np.asarray(g_y, float)


# ------------------------------------------------------ window-integral decision QoI (Phi(Y_traj))

def decision_trajectory_functional(bundle, op, u_map, *, n_steps: int = 120, **spec):
    """Express the pooled decision QoI ``g = [purity, recovery]`` as a torch functional of the FULL
    STATE TRAJECTORY ``Phi(Y_flat) -> (2,)`` (window-integral ratios of the exit-concentration readout),
    with NO solver re-run -- so ``Phi``'s Y-derivatives are autograd-safe. Mirrors
    ``decision.decision_forward``: the collection window is selected ONCE on the MAP curve and held
    fixed by physical time. Returns ``(Phi, sel, n_protein)``.

    ``window=`` (a ``GradientWindowSelection``, in ``spec``): reuse a FIXED physical-time window
    (``start_time_s``/``end_time_s``/``min_total``) instead of re-selecting on THIS mesh's MAP curve.
    Required for a clean mesh-convergence study of the second-order sensitivity -- otherwise the grid
    search snaps the window to different edges per mesh, so the sweep conflates window placement with
    discretization convergence.

    Readout (``TorchSimulator.integrate``): ``salt = Y[:, gs-1]``, ``prot_c = Y[:, (c+1)gs-1]*MOL_TO_G_L``;
    ``elution_curve`` slices from the feed-end index and re-zeros time."""
    from cex_model.bayes.active import _sim_for_op
    from cex_model.bayes.decision import group_indices, _ACID_MAX, _MAIN_MIN, _BASIC_MAX
    from cex_model.diffsolver.collection_objective import (
        differentiable_fixed_window_objective, select_window_for_gradient)
    from cex_model.diffsolver.torch_solver import MOL_TO_G_L

    acid_max = spec.get("acid_max", _ACID_MAX); main_min = spec.get("main_min", _MAIN_MIN)
    basic_max = spec.get("basic_max", _BASIC_MAX); grid = spec.get("grid", 40)
    window = spec.get("window", None)
    n = bundle.components.n_protein
    grp = group_indices(bundle.components)
    sim = _sim_for_op(bundle, op, n_steps)
    gs, npr = sim.gs, sim.npr
    u_t = torch.tensor(np.asarray(u_map, float), dtype=DTYPE)
    if window is not None:
        sel = window
    else:
        with torch.no_grad():
            curve_map = sim.elution_curve(*unpack_u(u_t, n), differentiable=False)
        sel = select_window_for_gradient(curve_map.numpy(), grid=grid, acid_idx=grp["acid"],
                                         main_idx=grp["main"], basic_idx=grp["basic"], acid_max=acid_max,
                                         main_min=main_min, basic_max=basic_max)
    i0 = int(torch.searchsorted(sim.t, torch.as_tensor(sim.feed_end_t, dtype=DTYPE)).item())
    i0 = max(0, min(i0, sim.t.shape[0] - 1))
    t_col = sim.t[:, None]

    def curve_from_Y(Y):
        salt = Y[:, gs - 1]
        prot = torch.stack([Y[:, (c + 1) * gs - 1] for c in range(1, npr + 1)], dim=1) * MOL_TO_G_L
        full = torch.cat([t_col, salt[:, None], prot], dim=1)
        out = full[i0:].clone()
        out[:, 0] = out[:, 0] - out[0, 0]
        return out

    def Phi(Y_flat):
        Y = Y_flat.reshape(sim.t.shape[0], -1)
        r = differentiable_fixed_window_objective(
            curve_from_Y(Y), start_time_s=sel.start_time_s, end_time_s=sel.end_time_s,
            acid_idx=grp["acid"], main_idx=grp["main"], basic_idx=grp["basic"],
            acid_max=acid_max, main_min=main_min, basic_max=basic_max, min_total=sel.min_total)
        return torch.stack([r.main_fraction, r.recovery])

    return Phi, sel, n


def _flat_traj(S, nt, ndim, dim_u):
    return np.stack([S[k] for k in range(nt)], axis=0).reshape(nt * ndim, dim_u)


def trajectory_functional_hvp(cache: StepCache, S, v, Phi, *, dim_g: int = 2):
    """Hessian-vector product ``H_g v`` (``dim_g, dim_u``) for ``g = Phi(Y_flat)`` (a torch trajectory
    functional), assembled from the analytic first/second variational trajectories ``S``, ``W(v)`` and
    autograd derivatives of ``Phi`` w.r.t. ``Y`` (safe -- ``Phi`` never touches the solver):
        ``(H_g^q v) = W(v)^T (dPhi^q/dY) + S^T (d2Phi^q/dY2 . (S v))``.
    Also returns the first-order gradient ``dg/du = (dPhi/dY) S`` (validate vs ``decision_jacobian``)."""
    nt, ndim, dim_u = cache.nt, cache.ndim, cache.dim_u
    S_flat = _flat_traj(S, nt, ndim, dim_u)                      # (nt*ndim, dim_u)
    W = forward_second_variational(cache, S, v)
    W_flat = _flat_traj(W, nt, ndim, dim_u)
    s_v_flat = S_flat @ np.asarray(v, float)                     # (nt*ndim,)

    Y_flat = torch.tensor(cache.Y.reshape(-1), dtype=DTYPE)
    Phi_Y = torch.autograd.functional.jacobian(Phi, Y_flat).detach().numpy()   # (dim_g, nt*ndim)
    grad_g = Phi_Y @ S_flat                                       # (dim_g, dim_u)

    s_v_t = torch.tensor(s_v_flat, dtype=DTYPE)
    Hv = np.zeros((dim_g, dim_u))
    for q in range(dim_g):
        _, hvp_q = torch.autograd.functional.hvp(lambda y: Phi(y)[q], Y_flat, s_v_t)
        Hv[q] = W_flat.T @ Phi_Y[q] + S_flat.T @ hvp_q.detach().numpy()
    return Hv, grad_g
