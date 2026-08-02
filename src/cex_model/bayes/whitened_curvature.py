"""Posterior-whitened QoI curvature diagnostics (kappa closure brief, Deliverable B).

``bayes/curvature.py`` (kept, unmodified, cross-referenced as ``legacy_raw_fd``) computes ``kappa_q``
from a raw-coordinate finite-difference Hessian; per ``docs/decision_null_theorem.md`` §4.2 that path
is numerically unreliable on real products -- the wide sloppy directions of ``Sigma`` amplify
gradient-mesh noise in the Hessian, so ``delta_q`` and ``relFrob`` end up *anti-correlated* on real
data. This module instead differentiates in POSTERIOR-WHITENED coordinates

    u = u_map + L z,   L L^T = Sigma (approx),   z ~ N(0, I)

so a unit perturbation of ``z`` is a unit-scale step along an actually-informative posterior
direction, not an arbitrarily-scaled raw parameter. ``L`` is the symmetric matrix square root
``V diag(sqrt(clip(lambda, eig_clip)))`` from the eigendecomposition of ``Sigma`` (never a plain
Cholesky, which can fail outright on a near-singular covariance); eigenvalue clipping is always
reported, never silent (brief §"Phase 2").

Second-order info is obtained ONLY by Hessian-vector products built from finite differences of the
TRUSTED first-order reverse-mode gradient (``hvp_fd_grad_whitened``) -- never by reverse-over-reverse
autograd through the current implicit-function-theorem BDF solver, which detaches its Jacobian and is
known to miss second-order terms (``bayes/curvature.py``'s own ``qoi_hessian_autograd`` is kept there
purely as the negative-control diagnostic that established this; it is never called from here).

Everything here is a DIAGNOSTIC, not a certificate, unless the 4-level validation ladder in
``bayes/kappa_closure_report.py`` marks a product/QoI ``PASS_CERTIFIED_CURVATURE``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from cex_model.bayes.decision import decision_forward
from cex_model.bayes.loading_sweep import _load_product_setup
from cex_model.diffsolver.torch_solver import DTYPE

__all__ = [
    "WhitenedCurvatureConfig",
    "load_posterior_whitening",
    "whiten", "unwhiten",
    "qoi_value_and_grad_z",
    "hvp_fd_grad_whitened",
    "discrete_bdf_hvp",
    "value_fd_hvp",
    "calibrate_value_fd_step",
    "mixed_symmetry_error",
    "estimate_hessian_frobenius",
    "compute_delta_q",
    "analytic_bdf_hvp_context",
    "analytic_bdf_delta_q",
    "run_whitened_curvature_report",
    "run_level3_synthetic",
    "analytic_toy_qoi",
    "small_ode_qoi",
]

_QOI_INDEX = {"purity": 0, "yield": 1}
_HVP_STEP_GRID = (8e-2, 4e-2, 2e-2, 1e-2, 5e-3)   # halving sequence: h/2 of step i == step i+1
# Value-FD step grid: the small-eps machine-resolved plateau found by bayes_kappa_pathb_spike.py
# (the forward q(u) is smooth to ~1e-13; the clean curvature plateau sits at eps ~ 3e-6..1e-4, BELOW
# the QoI-saturation region that traps the autograd-gradient FD).
_VALUE_FD_STEP_GRID = (3.0e-4, 1.0e-4, 5.0e-5, 3.0e-5, 1.5e-5, 8.0e-6)


@dataclass
class WhitenedCurvatureConfig:
    products: list[str]
    qois: tuple[str, ...] = ("purity", "yield")
    n_steps: list[int] = field(default_factory=lambda: [100, 120])
    hvp_mode: str = "fd_grad_whitened"
    hutchinson_draws: int = 64
    hutchpp_rank: int = 16
    seed: int = 20260702
    in_dir: str = "results/bayes"
    out_path: str = "results/bayes/whitened_curvature.json"
    fail_open_to_mc: bool = True
    hvp_step_rel_tol: float = 0.15
    n_symmetry_pairs: int = 16
    eig_clip: float = 1e-12
    run_level3: bool = True                 # Level-3 synthetic SMA/CMC validation (see run_level3_synthetic)
    level3_cmc_draws: int = 512             # Hutchinson probes for the CMC ground-truth Frobenius estimate
    level3_sma_draws: int = 256             # Hutchinson probes for the SMA self-consistency leg
    level3_cond_target: float = 1.0e6       # cond(Sigma) of the sloppy whitening (>= 1e4; real posteriors ~1e7)
    level3_n_steps: int = 60                # solver mesh for the synthetic-SMA leg (small: it is synthetic)


# --------------------------------------------------------------------------- Phase 2: whitening

def load_posterior_whitening(product: str, *, in_dir: str = "results/bayes", eig_clip: float = 1e-12) -> dict:
    """``u_map``, covariance ``Sigma``, the stable square-root factor ``L``, dimension ``names``,
    and the bundle/op/tol needed to build ``g_fn`` -- with EXPLICIT (never silent) eigenvalue
    clipping if ``Sigma`` is near-singular."""
    post, base_op, tol, n, prior, bundle = _load_product_setup(product, in_dir)
    Sigma = np.asarray(post.cov, float)
    lam, V = np.linalg.eigh(Sigma)
    min_eig_before = float(lam.min())
    num_clipped = int(np.sum(lam < eig_clip))
    lam_c = np.clip(lam, eig_clip, None)
    L = V @ np.diag(np.sqrt(lam_c))
    return {
        "product": product, "u_map": np.asarray(post.u_map, float), "cov": Sigma, "L": L,
        "names": post.names, "n_protein": n, "bundle": bundle, "op": base_op, "tol": tol,
        "whitening": {"method": "eigh_clip", "min_eig_before": min_eig_before,
                     "eig_clip": float(eig_clip), "num_clipped": num_clipped},
    }


def whiten(u: np.ndarray, u_map: np.ndarray, L: np.ndarray) -> np.ndarray:
    """``z = L^{-1}(u - u_map)``. ``L`` is SPD (eigh_clip), so this is a genuine inverse."""
    return np.linalg.solve(L, np.asarray(u, float) - np.asarray(u_map, float))


def unwhiten(z: np.ndarray, u_map: np.ndarray, L: np.ndarray) -> np.ndarray:
    """``u = u_map + L z``."""
    return np.asarray(u_map, float) + L @ np.asarray(z, float)


# ------------------------------------------------------------------- Phase 3: gradient + HVP in z

def qoi_value_and_grad_z(product: str, qoi: str, z, n_steps: int, *,
                         in_dir: str = "results/bayes", _cache: dict | None = None) -> tuple[float, np.ndarray]:
    """``q(u_map + Lz)`` and ``grad_z q`` via the existing first-order differentiable solver
    (``decision.decision_forward``), composed with the whitening map. ``_cache`` (optional) lets
    callers reuse a pre-built ``g_fn``/``L`` across many probes (see ``hvp_fd_grad_whitened``) --
    each rebuild re-solves the MAP window selection, which is wasteful when it never changes with ``z``."""
    ctx = _cache if _cache is not None else _build_qoi_context(product, qoi, n_steps, in_dir=in_dir)
    z_t = torch.tensor(np.asarray(z, float), dtype=DTYPE, requires_grad=True)

    def q_of_z(zz):
        u = ctx["u_map_t"] + ctx["L_t"] @ zz
        return ctx["g_fn"](u)[ctx["idx"]]

    val = q_of_z(z_t)
    grad_z = torch.autograd.functional.jacobian(q_of_z, z_t).detach().numpy()
    return float(val.detach()), grad_z


def _build_qoi_context(product: str, qoi: str, n_steps: int, *, in_dir: str = "results/bayes",
                       w: dict | None = None) -> dict:
    if qoi not in _QOI_INDEX:
        raise ValueError(f"qoi must be one of {list(_QOI_INDEX)}, got {qoi!r}")
    # ``w`` is normally loaded from disk (real product); a caller may inject an in-memory whitening
    # dict (bundle/op/u_map/L/whitening/names) instead -- used by the Level-3 synthetic-SMA leg, whose
    # duck-typed bundle has no committed posterior to load through ``_load_product_setup``.
    if w is None:
        w = load_posterior_whitening(product, in_dir=in_dir)
    g_fn, sel, _ = decision_forward(w["bundle"], w["op"], w["u_map"], n_steps=n_steps)
    return {
        "g_fn": g_fn, "idx": _QOI_INDEX[qoi],
        "u_map_t": torch.tensor(w["u_map"], dtype=DTYPE),
        "L_t": torch.tensor(w["L"], dtype=DTYPE),
        "dim": w["L"].shape[0], "whitening": w["whitening"], "names": w["names"],
    }


def hvp_fd_grad_whitened(product: str, qoi: str, v, n_steps: int, h: float | None = None, *,
                         in_dir: str = "results/bayes", hvp_step_rel_tol: float = 0.15,
                         _cache: dict | None = None) -> dict:
    """``H_z v`` by central finite difference of the trusted reverse-mode gradient in ``z``
    (brief §4.3 Mode 1). ``v`` is normalized to unit norm. If ``h`` is None, adaptively picks the
    largest step in ``_HVP_STEP_GRID`` passing a Richardson stability check between ``h`` and
    ``h/2``; else uses the given ``h`` directly (single evaluation, no adaptive search)."""
    ctx = _cache if _cache is not None else _build_qoi_context(product, qoi, n_steps, in_dir=in_dir)
    grad_fn = lambda z: qoi_value_and_grad_z(product, qoi, z, n_steps, in_dir=in_dir, _cache=ctx)[1]
    return _hvp_fd_core(grad_fn, v, h, hvp_step_rel_tol=hvp_step_rel_tol)


def _hvp_fd_core(grad_fn, v, h: float | None = None, *, hvp_step_rel_tol: float = 0.15) -> dict:
    """Central finite difference of a TRUSTED whitened gradient ``grad_fn(z) -> R^dim`` along ``v``
    (unit-normalized), evaluated at ``z0 = 0`` (``u = u_map``). This is the production HVP arithmetic
    shared by the real-product path (``hvp_fd_grad_whitened``) and the Level-3 synthetic ground-truth
    check, so both exercise the SAME central-difference + adaptive-Richardson step logic -- the CMC
    leg is not a re-implementation but a call into this exact code (brief §4.3 Mode 1)."""
    v = np.asarray(v, float)
    nv = np.linalg.norm(v)
    v = v / nv if nv > 0 else v
    z0 = np.zeros(v.shape[0])   # z=0 <=> u=u_map: the operating point the posterior is centered on

    def hv_at(step):
        return (np.asarray(grad_fn(z0 + step * v), float) - np.asarray(grad_fn(z0 - step * v), float)) / (2.0 * step)

    if h is not None:
        return {"Hv": hv_at(h), "h": float(h), "hvp_step_status": "FIXED_STEP", "rel_diff": float("nan")}

    # _HVP_STEP_GRID is a halving sequence, so each grid point's "h/2" comparison IS the next grid
    # point's "h" -- cache it forward instead of recomputing (5 grid points cost ~6 gradient-pair
    # evals total this way, not ~20).
    tried = []
    Hv_prev = hv_at(_HVP_STEP_GRID[0])
    for i, step in enumerate(_HVP_STEP_GRID[1:], start=1):
        Hv_h, Hv_h2 = Hv_prev, hv_at(step)
        rel = float(np.linalg.norm(Hv_h - Hv_h2) / max(np.linalg.norm(Hv_h2), 1e-12))
        tried.append({"h": _HVP_STEP_GRID[i - 1], "rel_diff": rel})
        if rel < hvp_step_rel_tol:
            return {"Hv": Hv_h2, "h": step, "hvp_step_status": "PASS", "rel_diff": rel, "tried": tried}
        Hv_prev = Hv_h2
    # nothing passed: return the smallest-step estimate, flagged
    return {"Hv": Hv_prev, "h": _HVP_STEP_GRID[-1],
           "hvp_step_status": "WARN_NO_STABLE_STEP", "rel_diff": tried[-1]["rel_diff"] if tried else float("nan"),
           "tried": tried}


def discrete_bdf_hvp(product: str, qoi: str, v, n_steps: int, **kwargs) -> dict:
    """Mode 2 (brief §4.3): HVP by differentiating the fixed BDF time-stepping residual directly.

    NOT implemented -- the current solver's Newton/implicit-differentiation internals
    (``TorchSimulator._newton``/``_diff_step``) are not exposed as a per-step residual API that can
    be second-differentiated without unrolling the adaptive Newton loop (which the brief explicitly
    prohibits: "do not differentiate through adaptive step/order branching"). Implementing this
    would require adding a frozen-schedule residual entry point to ``torch_solver.py`` -- a solver
    change out of scope for a diagnostics module. Per brief §4.3: "If this mode is too invasive,
    implement the interface and return NOT_AVAILABLE with a clear explanation."
    """
    return {"status": "NOT_AVAILABLE",
           "reason": "torch_solver.py has no frozen-schedule per-step residual API to "
                     "second-differentiate without unrolling adaptive Newton; would require a "
                     "solver-side change, out of scope here. fd_grad_whitened is the production MVP."}


def calibrate_value_fd_step(qfn, z0, v, *, grid=_VALUE_FD_STEP_GRID, lin_dq: float = 1e-2,
                            mad_k: float = 5.0) -> dict:
    """Pick the value-FD step in the machine-resolved curvature plateau for unit direction ``v``:
    sweep ``grid``, keep steps where the QoI stays in linear response (``|q(eps v)-q0| < lin_dq``, i.e.
    below saturation), MAD-reject isolated fixed-window/peak-crossing glitches, and return the step
    whose directional curvature is closest to the robust plateau median. Mirrors
    ``scripts/bayes_kappa_pathb_spike.py`` so the module and the feasibility spike agree."""
    v = np.asarray(v, float); v = v / max(np.linalg.norm(v), 1e-30)
    q0 = float(qfn(z0))
    steps, curv, dq = [], [], []
    for d in grid:
        qp, qm = float(qfn(z0 + d * v)), float(qfn(z0 - d * v))
        steps.append(d); dq.append(qp - q0); curv.append((qp - 2 * q0 + qm) / (d * d))
    steps, curv, dq = np.array(steps, float), np.array(curv, float), np.array(dq, float)
    win = np.abs(dq) < lin_dq
    if win.sum() < 3:
        win = np.ones_like(steps, bool)
    cw, sw = curv[win], steps[win]
    med = np.median(cw); mad = np.median(np.abs(cw - med)) + 1e-30
    keep = np.abs(cw - med) < mad_k * mad
    H0 = float(np.median(cw[keep])) if keep.any() else float(med)
    flat = float(np.median(np.abs(cw[keep] - H0)) / (abs(H0) + 1e-30)) if keep.any() else float("nan")
    delta = float(sw[keep][np.argmin(np.abs(cw[keep] - H0))]) if keep.any() else float(np.median(sw))
    return {"delta": delta, "curv_plateau": H0, "robust_flatness": flat,
            "n_window": int(win.sum()), "n_kept": int(keep.sum()), "q0": q0}


def value_fd_hvp(qfn, z0, v, *, delta: float, robust_grid=None, glitch_guard: bool = True,
                 glitch_rel_tol: float = 0.2, mad_k: float = 5.0) -> dict:
    """Hessian-vector product ``H_z v`` at ``z0`` by VALUE finite differences ONLY -- never the
    autograd gradient (whose grid error, ``docs/decision_null_theorem.md`` §4.2, forces the FD step
    into the QoI-saturation regime and breaks mixed symmetry). Each component uses the mixed 4-point
    stencil ``(H v)_j = [q(z0+d(e_j+v)) - q(z0+d(e_j-v)) - q(z0-d(e_j-v)) + q(z0-d(e_j+v))]/(4 d^2)``.
    ``v`` is unit-normalized internally and the result rescaled (``H`` is linear), so Hutchinson's
    non-unit probes keep their perturbations in the plateau.

    Two robustness modes for the SMA non-smoothness (window-edge / peak-crossing kinks, worst along
    the wide whitened SLOPPY axes ``e_j`` whose ``L`` column is a large u-step):
      * ``robust_grid`` (a step grid): per component, sweep the grid, MAD-reject glitch-hit steps and
        take the robust median -- the strong mode for sloppy axes.
      * else ``glitch_guard``: recompute at ``delta/2`` and keep the finer value, counting disagreements.
    Symmetry is NOT imposed: ``H v`` and ``H w`` use independent evaluations, so ``v^T H w`` vs
    ``w^T H v`` stays a live diagnostic."""
    z0 = np.asarray(z0, float); v = np.asarray(v, float); dim = z0.shape[0]
    nv = float(np.linalg.norm(v))
    if nv == 0:
        return {"Hv": np.zeros(dim), "delta": float(delta), "n_glitch": 0, "mode": "zero"}
    vh = v / nv
    Hv = np.zeros(dim); n_glitch = 0

    def _mixed(j, d):
        e = np.zeros(dim); e[j] = 1.0
        return (float(qfn(z0 + d * (e + vh))) - float(qfn(z0 + d * (e - vh)))
                - float(qfn(z0 - d * (e - vh))) + float(qfn(z0 - d * (e + vh)))) / (4.0 * d * d)

    for j in range(dim):
        if robust_grid is not None:
            vals = np.array([_mixed(j, d) for d in robust_grid], float)
            med = np.median(vals); mad = np.median(np.abs(vals - med)) + 1e-30
            keep = np.abs(vals - med) < mad_k * mad
            Hv[j] = float(np.median(vals[keep])) if keep.any() else float(med)
            if (~keep).any():
                n_glitch += 1
        else:
            h1 = _mixed(j, delta)
            if glitch_guard:
                h2 = _mixed(j, delta / 2.0)
                if abs(h1 - h2) > glitch_rel_tol * (abs(h2) + 1e-12):
                    n_glitch += 1
                Hv[j] = h2
            else:
                Hv[j] = h1
    return {"Hv": Hv * nv, "delta": float(delta), "n_glitch": int(n_glitch),
            "mode": "robust_grid" if robust_grid is not None else ("guard" if glitch_guard else "single")}


def mixed_symmetry_error(product: str, qoi: str, v, w, n_steps: int, *, h: float | None = None,
                         in_dir: str = "results/bayes", _cache: dict | None = None) -> float:
    """``|v^T H w - w^T H v| / (1 + |v^T H w| + |w^T H v|)`` (brief §4.6 Level 2). ``h`` fixed (from
    a prior adaptive search) avoids re-running step-size discovery per probe pair."""
    ctx = _cache if _cache is not None else _build_qoi_context(product, qoi, n_steps, in_dir=in_dir)
    Hv = hvp_fd_grad_whitened(product, qoi, v, n_steps, h=h, in_dir=in_dir, _cache=ctx)["Hv"]
    Hw = hvp_fd_grad_whitened(product, qoi, w, n_steps, h=h, in_dir=in_dir, _cache=ctx)["Hv"]
    v = np.asarray(v, float) / max(np.linalg.norm(v), 1e-30)
    w = np.asarray(w, float) / max(np.linalg.norm(w), 1e-30)
    vHw, wHv = float(v @ Hw), float(w @ Hv)
    return float(abs(vHw - wHv) / (1.0 + abs(vHw) + abs(wHv)))


def estimate_hessian_frobenius(product: str, qoi: str, n_steps: int, n_draws: int, method: str = "hutchinson", *,
                               h: float | None = None, in_dir: str = "results/bayes", seed: int = 0,
                               _cache: dict | None = None) -> dict:
    """``||H_z||_F^2 = E||H_z v||^2`` via Hutchinson (Rademacher probes; brief §4.5). ``h`` fixed
    (from a prior adaptive search) avoids re-running step-size discovery per draw.

    ``method="hutchpp"`` is not implemented: a correct variance-reduced Hutch++ estimator of
    ``tr(H^2)`` needs matvecs of the IMPLICIT operator ``H^2`` (``H(Hv)``, two HVP evaluations per
    probe, plus a QR-deflation step), not the single ``||Hv||^2`` per probe that makes plain
    Hutchinson cheap here; shipping a half-implementation would silently return a wrong number, so
    this raises instead (brief §4.5 lists Hutch++ as optional; Hutchinson is the required baseline).
    """
    if method not in ("hutchinson", "hutchpp"):
        raise ValueError(f"method must be 'hutchinson' or 'hutchpp', got {method!r}")
    if method == "hutchpp":
        raise NotImplementedError("hutchpp is not implemented (see docstring); use method='hutchinson'.")

    ctx = _cache if _cache is not None else _build_qoi_context(product, qoi, n_steps, in_dir=in_dir)

    def hv(vec):
        # hvp_fd_grad_whitened normalizes v internally; rescale back so hv(vec) == H @ vec for a
        # non-unit-norm vec (Rademacher probes are unit-scale per-entry, not unit-norm).
        return (hvp_fd_grad_whitened(product, qoi, vec, n_steps, h=h, in_dir=in_dir, _cache=ctx)["Hv"]
               * np.linalg.norm(vec))

    return _hutchinson_core(hv, ctx["dim"], n_draws, seed=seed)


def _hutchinson_core(hv_fn, dim: int, n_draws: int, *, seed: int = 0) -> dict:
    """Hutchinson ``||H||_F^2 = E||H v||^2`` (Rademacher probes; brief §4.5), where ``hv_fn(vec) = H @ vec``
    for a NON-unit-normalized ``vec``. Shared by the real-product path and the Level-3 CMC/SMA checks so
    the Frobenius estimator is the same code everywhere."""
    rng = np.random.default_rng(seed)
    V = rng.choice([-1.0, 1.0], size=(dim, n_draws))
    frob2_vals = np.array([float(np.linalg.norm(np.asarray(hv_fn(V[:, i]), float)) ** 2) for i in range(n_draws)])
    frob2_hat = float(np.mean(frob2_vals))
    se_hat = float(np.std(frob2_vals, ddof=1) / np.sqrt(n_draws)) if n_draws > 1 else float("nan")
    rel_se = se_hat / max(frob2_hat, 1e-30)
    return {"method": "hutchinson", "n_draws": n_draws, "frob2_hat": frob2_hat, "se_hat": se_hat, "rel_se": rel_se}


def compute_delta_q(product: str, qoi: str, n_steps: int, *, in_dir: str = "results/bayes",
                    hutchinson_draws: int = 64, seed: int = 0, hvp_step_rel_tol: float = 0.15,
                    n_symmetry_pairs: int = 16, light: bool = False) -> dict:
    """``C_lin``, ``||H_z||_F^2`` estimate, ``delta_q``, and validation flags for one product/QoI.

    The stable HVP step ``h`` is discovered ONCE (adaptive Richardson search on the gradient
    direction) and then reused FIXED for the symmetry-pair and Hutchinson probes -- re-running the
    full adaptive search per probe (as a literal reading of "adaptive step selection" per-HVP-call
    would imply) costs 5-20x more solver evaluations for no material gain, since the step that is
    numerically stable for one direction is, in practice, stable for nearby-scale directions too;
    ``hvp_step_status`` reports the ONE canonical search plus a cheap same-``h`` re-check on 2 more
    random directions (not a full independent search each).

    ``light=True`` skips the step re-check and symmetry-pair loop entirely (only ``C_lin`` +
    ``delta_q`` are computed) -- used for the extra mesh-stability ``n_steps`` points, where only
    ``delta_q``'s spread is needed and re-verifying symmetry at every mesh point is pure overhead.
    """
    ctx = _build_qoi_context(product, qoi, n_steps, in_dir=in_dir)
    _, grad_z = qoi_value_and_grad_z(product, qoi, np.zeros(ctx["dim"]), n_steps, in_dir=in_dir, _cache=ctx)
    C_lin = float(np.dot(grad_z, grad_z))

    rng = np.random.default_rng(seed)
    v0 = grad_z if np.linalg.norm(grad_z) > 1e-12 else rng.standard_normal(ctx["dim"])
    discovery = hvp_fd_grad_whitened(product, qoi, v0, n_steps, in_dir=in_dir, hvp_step_rel_tol=hvp_step_rel_tol, _cache=ctx)
    h_star = discovery["h"]

    if light:
        hvp_step_status = discovery["hvp_step_status"]
        mixed_symmetry_median = mixed_symmetry_max = float("nan")
    else:
        step_status = [discovery["hvp_step_status"]]
        for _ in range(min(2, ctx["dim"])):
            v = rng.standard_normal(ctx["dim"])
            Hv_h = hvp_fd_grad_whitened(product, qoi, v, n_steps, h=h_star, in_dir=in_dir, _cache=ctx)["Hv"]
            Hv_h2 = hvp_fd_grad_whitened(product, qoi, v, n_steps, h=h_star / 2.0, in_dir=in_dir, _cache=ctx)["Hv"]
            rel = float(np.linalg.norm(Hv_h - Hv_h2) / max(np.linalg.norm(Hv_h2), 1e-12))
            step_status.append("PASS" if rel < hvp_step_rel_tol else "WARN_STEP_UNSTABLE")
        hvp_step_status = "PASS" if all(s == "PASS" for s in step_status) else "WARN_STEP_UNSTABLE"

        sym_errs = []
        for _ in range(n_symmetry_pairs):
            v, w = rng.standard_normal(ctx["dim"]), rng.standard_normal(ctx["dim"])
            sym_errs.append(mixed_symmetry_error(product, qoi, v, w, n_steps, h=h_star, in_dir=in_dir, _cache=ctx))
        sym_errs = np.array(sym_errs)
        mixed_symmetry_median = float(np.median(sym_errs)) if sym_errs.size else float("nan")
        mixed_symmetry_max = float(np.max(sym_errs)) if sym_errs.size else float("nan")

    frob = estimate_hessian_frobenius(product, qoi, n_steps, hutchinson_draws, "hutchinson", h=h_star,
                                      in_dir=in_dir, seed=seed, _cache=ctx)
    delta_q = 0.5 * frob["frob2_hat"] / C_lin if C_lin > 0 else float("nan")

    return {
        "product": product, "qoi": qoi, "n_steps": n_steps,
        "C_lin": C_lin, "grad_z_norm": float(np.sqrt(max(C_lin, 0.0))),
        "frob_H2_hat": frob["frob2_hat"], "frob_H2_rel_se": frob["rel_se"],
        "delta_q": delta_q, "hvp_step_status": hvp_step_status,
        "mixed_symmetry_rel": mixed_symmetry_median, "mixed_symmetry_max": mixed_symmetry_max,
        "whitening": ctx["whitening"],
    }


# ---------------------------------------------------------------- Levels 1-2: solver-free validation

def analytic_bdf_hvp_context(product: str, qoi: str, n_steps: int, *, in_dir: str = "results/bayes",
                             window=None):
    """Build the Path-A analytic BDF Hessian-vector-product context for a real product: the SMA cache
    + first-order sensitivity S + the decision trajectory functional Phi + the whitening L. Returns
    ``(hv_fn, grad_z, dim, sel)`` where ``hv_fn(v_z) = L^T H_u (L v_z)`` is the WHITENED HVP for QoI
    ``qoi`` (the analytic BDF second-order sensitivity, validated to machine precision on the CMC full-AD
    gate, ``scripts/bayes_kappa_pathA_cmc_gate.py``), ``grad_z = L^T (dPhi/dY . S)_qoi`` the whitened
    gradient, and ``sel`` the (resolved) collection window. This is what makes the analytic HVP enter the
    kappa_q pipeline (vs the FD-of-autograd-gradient path). ``window`` (optional) fixes the collection
    window across meshes (see :func:`decision_window_for`)."""
    from cex_model.bayes.active import _sim_for_op
    from cex_model.bayes.second_order_sensitivity import (
        build_step_cache, forward_first_variational, decision_trajectory_functional,
        trajectory_functional_hvp, _flat_traj)
    w = load_posterior_whitening(product, in_dir=in_dir)
    n = w["n_protein"]; L = np.asarray(w["L"], float); qi = _QOI_INDEX[qoi]
    sim = _sim_for_op(w["bundle"], w["op"], n_steps)
    cache = build_step_cache(sim, w["u_map"], n)
    S = forward_first_variational(cache)
    Phi, sel, _ = decision_trajectory_functional(w["bundle"], w["op"], w["u_map"], n_steps=n_steps,
                                                  window=window)
    Phi_Y = torch.autograd.functional.jacobian(
        Phi, torch.tensor(cache.Y.reshape(-1), dtype=DTYPE)).detach().numpy()
    grad_u = (Phi_Y @ _flat_traj(S, cache.nt, cache.ndim, cache.dim_u))[qi]
    grad_z = L.T @ grad_u

    def hv_fn(v_z):
        v_u = L @ np.asarray(v_z, float)
        Hu = trajectory_functional_hvp(cache, S, v_u, Phi)[0][qi]     # (dim_u,) analytic H_u v_u
        return L.T @ Hu                                              # H_z v_z
    return hv_fn, grad_z, int(L.shape[0]), sel


def decision_window_for(product: str, n_steps: int, *, in_dir: str = "results/bayes"):
    """The collection-window selection (``GradientWindowSelection``) at a reference mesh, to REUSE as a
    FIXED physical-time window across a mesh sweep (pass as ``window=`` to :func:`analytic_bdf_delta_q`).
    Select it at the decision-framework mesh (``decision.decision_forward`` default ``n_steps=120``) so
    the swept second-order sensitivity is of the SAME physical decision QoI at every mesh."""
    from cex_model.bayes.second_order_sensitivity import decision_trajectory_functional
    w = load_posterior_whitening(product, in_dir=in_dir)
    _, sel, _ = decision_trajectory_functional(w["bundle"], w["op"], w["u_map"], n_steps=n_steps)
    return sel


def analytic_bdf_delta_q(product: str, qoi: str, n_steps: int, *, n_draws: int = 64,
                         in_dir: str = "results/bayes", seed: int = 0, n_sym: int = 6,
                         exact: bool = True, window=None) -> dict:
    """kappa_q / delta_q for a real product from the ANALYTIC BDF HVP (not FD-of-autograd-gradient):
    ``delta_q = 1/2 ||H_z||_F^2 / (grad_z . grad_z)``, plus a mixed-symmetry diagnostic. Ships the
    analytic HVP into the kappa_q pipeline.

    ``exact=True`` (default): the whitened dim is small (``dim_z=20`` for the real products), so the full
    ``H_z`` is assembled EXACTLY from ``dim`` basis HVPs (``H[:,i] = hv_fn(e_i)``). This gives the exact
    Frobenius norm (no Hutchinson variance -> ``frob_rel_se=0``) and an exact FULL-matrix symmetry check
    ``max|H-H^T|`` (stronger than random-pair probes) in ``dim`` HVPs instead of ``n_draws + 2*n_sym`` --
    both faster and noise-free. ``exact=False`` restores the stochastic Hutchinson + random-pair path
    (kept for large-dim contexts).

    ``window`` (a ``GradientWindowSelection`` from :func:`decision_window_for`): hold the collection
    window FIXED across a mesh sweep instead of re-selecting per mesh -- required for the sweep to test
    discretization convergence rather than window-placement drift."""
    hv_fn, grad_z, dim, sel = analytic_bdf_hvp_context(product, qoi, n_steps, in_dir=in_dir, window=window)
    win = [float(sel.start_time_s), float(sel.end_time_s)]
    C_lin = float(grad_z @ grad_z)
    if exact:
        H = np.column_stack([np.asarray(hv_fn(e), float) for e in np.eye(dim)])   # H[:,i] = H_z e_i
        frob2 = float(np.sum(H * H))
        sym = float(np.max(np.abs(H - H.T)) / (1.0 + np.max(np.abs(H))))
        delta_q = 0.5 * frob2 / C_lin if C_lin > 0 else float("nan")
        return {"product": product, "qoi": qoi, "n_steps": int(n_steps), "hvp": "analytic_bdf",
                "delta_q": delta_q, "C_lin": C_lin, "frob2_hat": frob2, "frob_rel_se": 0.0,
                "mixed_symmetry_max": sym, "n_draws": int(dim), "dim": dim, "frob_method": "exact_basis",
                "window_s": win, "fixed_window": window is not None}
    frob = _hutchinson_core(hv_fn, dim, n_draws, seed=seed)
    delta_q = 0.5 * frob["frob2_hat"] / C_lin if C_lin > 0 else float("nan")
    rng = np.random.default_rng(seed + 1); syms = []
    for _ in range(n_sym):
        v = rng.standard_normal(dim); w2 = rng.standard_normal(dim)
        vn, wn = v / np.linalg.norm(v), w2 / np.linalg.norm(w2)
        Hv, Hw = hv_fn(vn), hv_fn(wn)
        vHw, wHv = float(vn @ Hw), float(wn @ Hv)
        syms.append(abs(vHw - wHv) / (1.0 + abs(vHw) + abs(wHv)))
    return {"product": product, "qoi": qoi, "n_steps": int(n_steps), "hvp": "analytic_bdf",
            "delta_q": delta_q, "C_lin": C_lin, "frob2_hat": frob["frob2_hat"],
            "frob_rel_se": frob["rel_se"], "mixed_symmetry_max": float(np.max(syms)),
            "n_draws": int(n_draws), "dim": dim, "frob_method": "hutchinson",
            "window_s": win, "fixed_window": window is not None}


def analytic_toy_qoi(a, H, z) -> tuple[float, np.ndarray]:
    """``q(z) = a^T z + 1/2 z^T H z + 0.1 sum z_i^3`` and its exact gradient (brief §4.6 Level 1)."""
    a, H, z = np.asarray(a, float), np.asarray(H, float), np.asarray(z, float)
    q = float(a @ z + 0.5 * z @ H @ z + 0.1 * np.sum(z ** 3))
    grad = a + H @ z + 0.3 * z ** 2
    return q, grad


def _analytic_toy_hvp(H, v, *, z=None) -> np.ndarray:
    """Exact HVP of ``analytic_toy_qoi`` (Hessian = H + diag(0.6 z))."""
    H = np.asarray(H, float); v = np.asarray(v, float)
    diag = 0.6 * (np.asarray(z, float) if z is not None else np.zeros(H.shape[0]))
    return H @ v + diag * v


def small_ode_qoi(u, *, T: float = 1.0, n_rk4: int = 20) -> torch.Tensor:
    """A small (dim-3) non-stiff linear-ish ODE integrated by explicit RK4, fully torch-autograd
    differentiable with NO detached/implicit-diff tricks -- reverse-over-reverse Hessians through
    this ARE reliable, unlike the project's IFT-BDF solver, so it is a trustworthy Level-2 reference
    for validating ``hvp_fd_grad_whitened`` (brief §4.6 Level 2)."""
    u = u if torch.is_tensor(u) else torch.tensor(np.asarray(u, float), dtype=DTYPE)
    y = torch.zeros(2, dtype=DTYPE)
    y = y + torch.stack([torch.tensor(1.0, dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE)])
    h = T / n_rk4

    def f(yy):
        a, b, c = u[0], u[1], u[2]
        dy0 = -a * yy[0] + b * yy[1] * yy[1]
        dy1 = c * yy[0] - 0.5 * yy[1]
        return torch.stack([dy0, dy1])

    for _ in range(n_rk4):
        k1 = f(y)
        k2 = f(y + 0.5 * h * k1)
        k3 = f(y + 0.5 * h * k2)
        k4 = f(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y[0] + 0.5 * y[1]   # scalar QoI


def _small_ode_grad(u) -> np.ndarray:
    u_t = torch.tensor(np.asarray(u, float), dtype=DTYPE, requires_grad=True)
    return torch.autograd.functional.jacobian(small_ode_qoi, u_t).detach().numpy()


def _small_ode_hvp_fd(u0, v, h: float = 1e-2) -> np.ndarray:
    v = np.asarray(v, float); v = v / max(np.linalg.norm(v), 1e-30)
    return (_small_ode_grad(u0 + h * v) - _small_ode_grad(u0 - h * v)) / (2.0 * h)


def _small_ode_hessian_ad(u0) -> np.ndarray:
    u_t = torch.tensor(np.asarray(u0, float), dtype=DTYPE, requires_grad=True)
    return torch.autograd.functional.hessian(small_ode_qoi, u_t).detach().numpy()


def run_validation_ladder(config: WhitenedCurvatureConfig) -> dict:
    """Levels 1-2 (solver-free, always run) + a Level-3 summary flag (real per-product levels 3-4
    are reported per product/QoI in ``run_whitened_curvature_report``, not duplicated here)."""
    rng = np.random.default_rng(config.seed)

    # Level 1: analytic toy, z=0.
    dim1 = 5
    H1 = rng.standard_normal((dim1, dim1)); H1 = 0.5 * (H1 + H1.T)
    a1 = rng.standard_normal(dim1)
    z0 = np.zeros(dim1)
    errs = []
    for _ in range(8):
        v = rng.standard_normal(dim1)
        # emulate hvp_fd_grad_whitened's central-difference-of-gradient construction directly:
        # it normalizes the probe to unit norm before differencing, so compare against the exact
        # HVP at the SAME unit-normalized v_n (not the raw v).
        v_n = v / np.linalg.norm(v)
        Hv_exact = _analytic_toy_hvp(H1, v_n, z=z0)
        h = 1e-3
        _, gp = analytic_toy_qoi(a1, H1, z0 + h * v_n)
        _, gm = analytic_toy_qoi(a1, H1, z0 - h * v_n)
        Hv_fd = (gp - gm) / (2 * h)
        errs.append(float(np.linalg.norm(Hv_fd - Hv_exact) / max(np.linalg.norm(Hv_exact), 1e-30)))
    level1_rel_err = float(np.max(errs))
    level1 = "PASS" if level1_rel_err < 1e-6 else "FAIL"

    # Level 2: small ODE, reverse-over-reverse AD Hessian (trustworthy here) vs fd_grad_whitened-style HVP.
    u0 = np.array([1.0, 0.5, 0.8])
    H_ad = _small_ode_hessian_ad(u0)
    rel_errs, sym_errs = [], []
    for _ in range(6):
        v = rng.standard_normal(3)
        Hv_fd = _small_ode_hvp_fd(u0, v)
        Hv_ad = H_ad @ (v / max(np.linalg.norm(v), 1e-30))
        rel_errs.append(float(np.linalg.norm(Hv_fd - Hv_ad) / max(np.linalg.norm(Hv_ad), 1e-30)))
    for _ in range(6):
        v, w = rng.standard_normal(3), rng.standard_normal(3)
        vn, wn = v / np.linalg.norm(v), w / np.linalg.norm(w)
        Hv, Hw = _small_ode_hvp_fd(u0, vn, h=1e-2), _small_ode_hvp_fd(u0, wn, h=1e-2)
        vHw, wHv = float(vn @ Hw), float(wn @ Hv)
        sym_errs.append(float(abs(vHw - wHv) / (1.0 + abs(vHw) + abs(wHv))))
    level2_hvp_rel_err = float(np.max(rel_errs))
    level2_symmetry_err = float(np.max(sym_errs))
    level2 = "PASS" if (level2_hvp_rel_err < 5e-3 and level2_symmetry_err < 5e-3) else "FAIL"

    return {
        "level1_analytic": level1, "level1_rel_err": level1_rel_err,
        "level2_small_ode": level2, "level2_hvp_rel_err": level2_hvp_rel_err,
        "level2_symmetry_err": level2_symmetry_err,
    }


# ---------------------------------------------------------------- Level 3: synthetic SMA/CMC validation

def _sma_sloppy_whitening(n_comp: int, cond: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Diagonal sloppy whitening for the synthetic-SMA leg that MIRRORS the real SMA posterior structure
    (real HLXSYN cov cond ~1e7): the sloppy (wide) block is the steric factor sigma (and minor kkin) -- the
    QoI-irrelevant decision-null direction -- and the stiff block is keq/nu (the selectivity the decision
    needs, identified). Absolute scales come from the physical prior std so the whitened steps are PHYSICAL
    (a unit z-step along sigma stays in-range); keq/nu are shrunk by sqrt(cond) so ``cond(Sigma) == cond``.
    Because the wide direction is genuinely QoI-insensitive, the FD-HVP is not pathologically amplified
    (unlike a random rotation onto a QoI-sensitive axis, which destabilizes the real IFT-BDF solver)."""
    from cex_model.bayes.prior import physical_prior
    n = int(n_comp)
    pstd = np.asarray(physical_prior(n).std, float)             # u = [log-keq(n), log-kkin(n), nu(n), sigma(n)]
    base = float(np.mean(pstd[3 * n:4 * n]))                    # sigma prior std -> physical scale for sloppy dirs
    std_u = np.full(4 * n, base)                               # kkin (minor) + sigma stay prior-wide (sloppy)
    std_u[0:n] = base / np.sqrt(cond)                         # keq: identified (stiff)
    std_u[2 * n:3 * n] = base / np.sqrt(cond)                 # nu:  identified (stiff) -> cond(Sigma) == cond
    L = np.diag(std_u)
    Sigma = np.diag(std_u ** 2)
    return Sigma, L, float((std_u.max() / std_u.min()) ** 2)


def _sloppy_sigma_from_fisher(J_obs: np.ndarray, sigma_obs: float, cond_target: float,
                              prior_floor: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
    """A REALISTIC sloppy ``Sigma = F⁻¹`` from the synthetic obs Fisher ``F = J_obsᵀJ_obs/σ² +
    prior_floor·I`` -- variance ∝ 1/information, so the wide directions are the genuinely low-info ones,
    and (as on real posteriors) their QoI sensitivity is correspondingly modest, unlike an arbitrary
    rotation. ``prior_floor`` is shrunk until ``cond(Sigma) >= cond_target`` so the whitening is sloppy
    enough (>=1e4) to exercise the FD x wide-Sigma amplification WITHOUT the pathological over-stretch of a
    prescribed spectrum on a QoI-sensitive axis (Plan review; validated empirically)."""
    J = np.asarray(J_obs, float)
    FtF = (J.T @ J) / (sigma_obs ** 2)
    dim = J.shape[1]
    floor = float(prior_floor)
    for _ in range(60):                                          # shrink the prior floor until sloppy enough
        lam, V = np.linalg.eigh(FtF + floor * np.eye(dim))
        lam = np.clip(lam, 1e-30, None)
        if lam.max() / lam.min() >= cond_target or floor < 1e-10:
            break
        floor *= 0.5
    inv = 1.0 / lam                                             # Sigma eigenvalues (variances)
    Sigma = (V * inv) @ V.T
    L = V * np.sqrt(inv)
    return Sigma, L, float(inv.max() / inv.min())


def _cmc_ground_truth_check(*, n_consumers: int = 3, cond_target: float = 1.0e6, n_draws: int = 512,
                            seed: int = 0, hvp_rel_tol: float = 0.05, sym_tol: float = 0.10,
                            dq_rel_tol: float = 0.15, hvp_step_rel_tol: float = 0.15) -> dict:
    """Level-3 REAL validation: run the production whitened-HVP arithmetic (``_hvp_fd_core`` +
    ``_hutchinson_core``) on the pure-RK4 CMC ratio QoI ``purity(p)`` and check it against the
    TRUSTWORTHY full-AD Hessian (reliable because ``cmc_toy`` is a plain RK4 integrator with no
    detached-IFT solver). The whitening ``Sigma`` is a Fisher-derived SLOPPY covariance (cond >= 1e4) so
    the test exercises the wide-``Sigma`` regime the diagnostic must survive, not a benign identity.
    Non-circular: FD-of-gradient is an INDEPENDENT numerical route to the same Hessian, validated here
    against the exact AD Hessian action (Plan review)."""
    from cex_model.bayes.cmc_toy import CMCConfig, integrate, _window_idx

    n = int(n_consumers)
    c = CMCConfig(g=tuple(1.0 - 0.12 * i for i in range(n)), nu=tuple(3.0 - 0.4 * i for i in range(n)),
                  delta=tuple(0.5 for _ in range(n)), phi=tuple(5.0 for _ in range(n)),
                  beta=tuple(0.1 for _ in range(n)), R0=1.0, T=10.0, n_steps=100, window=(0.4, 0.8), target=0)
    a, b = _window_idx(c)
    p0 = torch.cat([c.g, c.nu, c.delta, c.phi]).clone().to(DTYPE)     # full parameter vector, dim = 4n
    dim = p0.shape[0]

    def _unpack(p):
        return p[0:n], p[n:2 * n], p[2 * n:3 * n], p[3 * n:4 * n]

    def purity_of_p(p):
        gg, nn, dd, pp = _unpack(p)
        t, Y = integrate(c, g=gg, nu=nn, delta=dd, phi_eff=pp)
        coll = torch.trapezoid(Y[a:b + 1], t[a:b + 1], dim=0)
        return coll[c.target] / coll.sum().clamp(min=1e-12)

    def obs_of_p(p):
        gg, nn, dd, pp = _unpack(p)
        _, Y = integrate(c, g=gg, nu=nn, delta=dd, phi_eff=pp)
        return Y.reshape(-1)

    # sloppy whitening from the synthetic obs Fisher
    J_obs = torch.autograd.functional.jacobian(obs_of_p, p0, strategy="forward-mode", vectorize=True).detach().numpy()
    Sigma, L, cond = _sloppy_sigma_from_fisher(J_obs, sigma_obs=0.05, cond_target=cond_target)
    L_t = torch.tensor(L, dtype=DTYPE)

    # exact ground truth (reliable full-AD Hessian on the pure-RK4 forward)
    grad_p = torch.autograd.functional.jacobian(purity_of_p, p0).detach().numpy()
    H_p = torch.autograd.functional.hessian(purity_of_p, p0).detach().numpy()
    H_z_true = L.T @ H_p @ L
    grad_z_true = L.T @ grad_p
    C_lin_true = float(grad_z_true @ grad_z_true)
    frob2_true = float(np.sum(H_z_true ** 2))
    delta_q_true = 0.5 * frob2_true / C_lin_true if C_lin_true > 0 else float("nan")

    # production arithmetic on the whitened QoI q(z) = purity(p0 + L z)
    p0_t = p0.detach()

    def grad_z_fn(z):
        zt = torch.tensor(np.asarray(z, float), dtype=DTYPE, requires_grad=True)
        return torch.autograd.functional.jacobian(lambda zz: purity_of_p(p0_t + L_t @ zz), zt).detach().numpy()

    grad_z0 = grad_z_fn(np.zeros(dim))
    C_lin_hat = float(grad_z0 @ grad_z0)
    v0 = grad_z0 if np.linalg.norm(grad_z0) > 1e-12 else np.random.default_rng(seed).standard_normal(dim)
    disc = _hvp_fd_core(grad_z_fn, v0, None, hvp_step_rel_tol=hvp_step_rel_tol)
    h_star = disc["h"]

    # PRIMARY (noise-free) gate: FD-HVP vs exact Hessian action + mixed symmetry
    rng = np.random.default_rng(seed + 1)
    hv_errs, sym_errs = [], []
    for _ in range(6):
        vn = rng.standard_normal(dim); vn = vn / np.linalg.norm(vn)
        Hv_fd = _hvp_fd_core(grad_z_fn, vn, h_star)["Hv"]
        Hv_true = H_z_true @ vn
        hv_errs.append(float(np.linalg.norm(Hv_fd - Hv_true) / max(np.linalg.norm(Hv_true), 1e-30)))
    for _ in range(8):
        v, w = rng.standard_normal(dim), rng.standard_normal(dim)
        vn, wn = v / np.linalg.norm(v), w / np.linalg.norm(w)
        Hv = _hvp_fd_core(grad_z_fn, v, h_star)["Hv"]; Hw = _hvp_fd_core(grad_z_fn, w, h_star)["Hv"]
        vHw, wHv = float(vn @ Hw), float(wn @ Hv)
        sym_errs.append(abs(vHw - wHv) / (1.0 + abs(vHw) + abs(wHv)))
    hvp_rel_err = float(np.max(hv_errs))
    sym_max = float(np.max(sym_errs))

    # SECONDARY: Hutchinson Frobenius / delta_q vs exact (fair: within Hutchinson noise + tol)
    hv_fn = lambda vec: _hvp_fd_core(grad_z_fn, vec, h_star)["Hv"] * np.linalg.norm(vec)
    frob = _hutchinson_core(hv_fn, dim, n_draws, seed=seed)
    delta_q_hat = 0.5 * frob["frob2_hat"] / C_lin_hat if C_lin_hat > 0 else float("nan")
    frob2_rel_err = abs(frob["frob2_hat"] - frob2_true) / max(frob2_true, 1e-30)
    dq_rel_err = abs(delta_q_hat - delta_q_true) / max(abs(delta_q_true), 1e-30)

    ok = (disc["hvp_step_status"] == "PASS" and hvp_rel_err < hvp_rel_tol and sym_max < sym_tol
          and frob2_rel_err < max(dq_rel_tol, 3.0 * frob["rel_se"]))
    return {
        "status": "PASS_CMC_GROUND_TRUTH" if ok else "FAIL",
        "model": "shared-resource-competition (non-SMA CMC), full-AD Hessian ground truth",
        "dim": int(dim), "cond_sigma": cond, "hvp_step_status": disc["hvp_step_status"], "h_star": h_star,
        "hvp_rel_err": hvp_rel_err, "mixed_symmetry_max": sym_max,
        "delta_q_true": delta_q_true, "delta_q_hat": delta_q_hat, "delta_q_rel_err": dq_rel_err,
        "frob2_true": frob2_true, "frob2_hat": frob["frob2_hat"], "frob2_rel_err": frob2_rel_err,
        "frob_rel_se": frob["rel_se"], "n_draws": int(n_draws),
    }


def _sma_self_consistency_check(*, cond_target: float = 1.0e4, n_draws: int = 256, seed: int = 0,
                                n_steps: int = 60, hvp_step_rel_tol: float = 0.15,
                                sym_tol: float = 0.10, mesh_tol: float = 0.30) -> dict:
    """Level-3 WEAKER leg: build a synthetic SMA product (the real IFT-BDF ``decision_forward`` path) with
    a known ``u_true`` and a prescribed sloppy whitening, run the production HVP arithmetic end-to-end, and
    check SELF-CONSISTENCY (step stability, mixed symmetry, mesh stability). There is NO independent
    trustworthy Hessian here (reverse-over-reverse through the detached-IFT solver is wrong), and
    self-consistency detects INSTABILITY but not a bias shared by the +/-h gradients -- so this corroborates
    the pipeline integration but can NEVER be a Level-3 PASS on its own (Plan review)."""
    from cex_model.bayes.synthetic import synthetic_sma_bundle

    n_comp = 3
    bundle, comps, u_true = synthetic_sma_bundle(n_comp, fractions=[18, 70, 12], keq_ladder=True)
    e = bundle.experiments[0]
    op = [e.loading_g_l, e.gradient_start_pct, e.gradient_end_pct, e.elution_cv]
    dim = int(np.asarray(u_true, float).shape[0])
    Sigma, L, cond = _sma_sloppy_whitening(n_comp, cond_target)
    names = [c.name for c in comps.components]

    def _ctx(ns):
        w = {"bundle": bundle, "op": op, "u_map": np.asarray(u_true, float), "L": L,
             "whitening": {"method": "sigma_sloppy_diagonal", "cond": cond, "num_clipped": 0}, "names": names}
        return _build_qoi_context(None, "purity", ns, w=w)

    def _delta_q_at(ns, draws):
        ctx = _ctx(ns)
        gfn = lambda z: qoi_value_and_grad_z(None, "purity", z, ns, _cache=ctx)[1]
        g0 = gfn(np.zeros(dim)); cl = float(g0 @ g0)
        v0 = g0 if np.linalg.norm(g0) > 1e-12 else np.random.default_rng(seed).standard_normal(dim)
        disc = _hvp_fd_core(gfn, v0, None, hvp_step_rel_tol=hvp_step_rel_tol)
        hv_fn = lambda vec: _hvp_fd_core(gfn, vec, disc["h"])["Hv"] * np.linalg.norm(vec)
        fr = _hutchinson_core(hv_fn, dim, draws, seed=seed)
        return gfn, disc, cl, (0.5 * fr["frob2_hat"] / cl if cl > 0 else float("nan"))

    grad_fn, disc, C_lin, delta_q = _delta_q_at(n_steps, n_draws)
    h_star = disc["h"]
    rng = np.random.default_rng(seed)

    step_ok = disc["hvp_step_status"] == "PASS"
    for _ in range(2):
        v = rng.standard_normal(dim)
        Hv_h = _hvp_fd_core(grad_fn, v, h_star)["Hv"]; Hv_h2 = _hvp_fd_core(grad_fn, v, h_star / 2.0)["Hv"]
        step_ok = step_ok and float(np.linalg.norm(Hv_h - Hv_h2) / max(np.linalg.norm(Hv_h2), 1e-12)) < hvp_step_rel_tol

    sym_errs = []
    for _ in range(8):
        v, w = rng.standard_normal(dim), rng.standard_normal(dim)
        vn, wn = v / np.linalg.norm(v), w / np.linalg.norm(w)
        Hv = _hvp_fd_core(grad_fn, v, h_star)["Hv"]; Hw = _hvp_fd_core(grad_fn, w, h_star)["Hv"]
        sym_errs.append(abs(float(vn @ Hw) - float(wn @ Hv)) / (1.0 + abs(float(vn @ Hw)) + abs(float(wn @ Hv))))
    sym_max = float(np.max(sym_errs))

    _, _, _, dq_fine = _delta_q_at(2 * n_steps, max(64, n_draws // 2))     # mesh stability
    dqv = np.array([x for x in (delta_q, dq_fine) if np.isfinite(x)])
    mesh_spread = (float((dqv.max() - dqv.min()) / np.mean(np.abs(dqv)))
                   if dqv.size >= 2 and np.mean(np.abs(dqv)) != 0 else float("nan"))

    ok = (step_ok and sym_max < sym_tol and np.isfinite(delta_q)
          and np.isfinite(mesh_spread) and mesh_spread < mesh_tol)
    return {
        "status": "PASS_SELF_CONSISTENT" if ok else "WARN_SELF_INCONSISTENT",
        "note": "self-consistency only; no independent Hessian (detached-IFT reverse-over-reverse is wrong)",
        "dim": dim, "cond_sigma": cond, "n_steps": int(n_steps), "hvp_step_status": disc["hvp_step_status"],
        "step_ok": bool(step_ok), "mixed_symmetry_max": sym_max, "delta_q": delta_q,
        "mesh_delta_q_rel_spread": mesh_spread, "n_draws": int(n_draws),
    }


def run_level3_synthetic(config: WhitenedCurvatureConfig) -> dict:
    """Level-3 synthetic validation: a CMC INDEPENDENT-ground-truth-Hessian check (the real gate) plus a
    synthetic-SMA SELF-CONSISTENCY leg (weaker). Combined ``level3_synthetic_sma == 'PASS'`` requires the
    CMC leg to pass AND the SMA leg to be self-consistent -- certification rests on the CMC leg, never on
    the SMA self-consistency alone."""
    try:
        cmc = _cmc_ground_truth_check(cond_target=config.level3_cond_target,
                                      n_draws=config.level3_cmc_draws, seed=config.seed)
    except Exception as exc:   # a broken leg must not take down the whole report
        cmc = {"status": "FAIL", "error": str(exc)}
    try:
        sma = _sma_self_consistency_check(cond_target=config.level3_cond_target,
                                          n_draws=config.level3_sma_draws, seed=config.seed,
                                          n_steps=config.level3_n_steps)
    except Exception as exc:
        sma = {"status": "WARN_SELF_INCONSISTENT", "error": str(exc)}
    combined = ("PASS" if cmc.get("status") == "PASS_CMC_GROUND_TRUTH" and sma.get("status") == "PASS_SELF_CONSISTENT"
                else f"PARTIAL_cmc={cmc.get('status')}_sma={sma.get('status')}")
    return {"level3_synthetic_sma": combined, "level3_cmc_status": cmc.get("status"),
            "level3_sma_status": sma.get("status"), "level3_detail": {"cmc": cmc, "sma": sma}}


# --------------------------------------------------------------------------------- report driver

def _apply_certificate_gate(products_out: dict, level3_status: str) -> None:
    """Certificate gate (acceptance rule): kappa_q is a DIAGNOSTIC, not a certificate, unless the Level-3
    synthetic ground-truth validation PASSES. If Level-3 is not PASS, downgrade every per-(product,QoI)
    ``PASS_CERTIFIED_CURVATURE`` to ``WARN_DIAGNOSTIC_ONLY`` in place (keeps ``_status_for`` pure; the
    downgrade propagates to the closure report through the written statuses). Certification requires the
    per-QoI gates AND global Level-3 PASS -- not ``real_product_overall``, so one unrelated product's
    failure cannot strip a clean QoI's certification."""
    if level3_status == "PASS":
        return
    for by_qoi in products_out.values():
        for rec in by_qoi.values():
            if rec.get("status") == "PASS_CERTIFIED_CURVATURE":
                rec["status"] = "WARN_DIAGNOSTIC_ONLY"
                rec["downgrade_reason"] = (f"level3_synthetic_sma={level3_status}; "
                                           "certification requires Level-3 PASS")


def _status_for(rec: dict, cfg: WhitenedCurvatureConfig) -> str:
    """Acceptance criteria §4.8: PASS_CERTIFIED_CURVATURE only if all conditions hold."""
    ok = (rec["hvp_step_status"] == "PASS"
         and rec["mixed_symmetry_rel"] < 0.10 and rec["mixed_symmetry_max"] < 0.25
         and np.isfinite(rec["frob_H2_rel_se"]) and rec["frob_H2_rel_se"] < 0.20
         and np.isfinite(rec.get("mesh_delta_q_rel_spread", 0.0)) and rec.get("mesh_delta_q_rel_spread", 0.0) < 0.25)
    if ok:
        return "PASS_CERTIFIED_CURVATURE"
    if rec["hvp_step_status"] == "WARN_NO_STABLE_STEP" or rec["mixed_symmetry_max"] >= 0.5:
        return "FAIL_DIAGNOSTIC_ONLY"
    return "WARN_DIAGNOSTIC_ONLY"


def run_whitened_curvature_report(config: WhitenedCurvatureConfig) -> dict:
    """Run products x QoIs, classify each PASS_CERTIFIED_CURVATURE / WARN_DIAGNOSTIC_ONLY /
    FAIL_DIAGNOSTIC_ONLY (never failing the whole script for one bad product/QoI), and write
    JSON/CSV reports. Attaches ``relFrob_existing`` from the committed ``*_decision.json`` if present."""
    import json

    validation = run_validation_ladder(config)
    level3 = (run_level3_synthetic(config) if config.run_level3
              else {"level3_synthetic_sma": "SKIPPED_LEVEL3_DISABLED", "level3_detail": {}})
    products_out: dict = {}
    for p in config.products:
        products_out[p] = {}
        relfrob = _load_committed_relfrob(p, config.in_dir)
        for qoi in config.qois:
            try:
                headline_ns = max(config.n_steps)
                rec = compute_delta_q(p, qoi, headline_ns, in_dir=config.in_dir,
                                      hutchinson_draws=config.hutchinson_draws, seed=config.seed,
                                      hvp_step_rel_tol=config.hvp_step_rel_tol,
                                      n_symmetry_pairs=config.n_symmetry_pairs)
                if len(config.n_steps) > 1:
                    dq_vals = [rec["delta_q"]]
                    for ns in config.n_steps:
                        if ns == headline_ns:
                            continue
                        dq_vals.append(compute_delta_q(p, qoi, ns, in_dir=config.in_dir,
                                                       hutchinson_draws=max(8, config.hutchinson_draws // 4),
                                                       seed=config.seed, light=True)["delta_q"])
                    dq_vals = np.array([x for x in dq_vals if np.isfinite(x)])
                    rec["mesh_delta_q_rel_spread"] = (float((dq_vals.max() - dq_vals.min()) / np.mean(np.abs(dq_vals)))
                                                      if dq_vals.size >= 2 and np.mean(np.abs(dq_vals)) != 0 else float("nan"))
                else:
                    rec["mesh_delta_q_rel_spread"] = float("nan")
                rec["relFrob_existing"] = relfrob.get(qoi, None)
                rec["status"] = _status_for(rec, config)
            except Exception as exc:   # a single product/QoI must not take down the whole report
                rec = {"product": p, "qoi": qoi, "status": "FAIL_DIAGNOSTIC_ONLY", "error": str(exc)}
            products_out[p][qoi] = rec

    _apply_certificate_gate(products_out, level3["level3_synthetic_sma"])

    any_fail = any(r.get("status") == "FAIL_DIAGNOSTIC_ONLY" for pr in products_out.values() for r in pr.values())
    any_warn = any(r.get("status") == "WARN_DIAGNOSTIC_ONLY" for pr in products_out.values() for r in pr.values())
    real_product_overall = "FAIL_SOME" if any_fail else ("PASS_WITH_WARNINGS" if any_warn else "PASS")

    report = {
        "metadata": {
            "hvp_mode": config.hvp_mode, "coordinate_system": "posterior_whitened_z",
            "raw_fd_hessian_used": False, "reverse_over_reverse_used": False,
            "hutchinson_draws": config.hutchinson_draws, "seed": config.seed,
        },
        "validation_summary": {
            "level1_analytic": validation["level1_analytic"],
            "level2_small_ode": validation["level2_small_ode"],
            "level3_synthetic_sma": level3["level3_synthetic_sma"],
            "level3_cmc_status": level3.get("level3_cmc_status"),
            "level3_sma_status": level3.get("level3_sma_status"),
            "real_product_overall": real_product_overall,
        },
        "validation_detail": {**validation, "level3": level3.get("level3_detail", {})},
        "products": products_out,
    }
    out_path = Path(config.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=_json_default))

    validation_path = out_path.with_name(out_path.stem + "_validation.json")
    validation_path.write_text(json.dumps(validation, indent=2))

    _write_csv(products_out, out_path.with_suffix(".csv"))
    _plot_delta_q(products_out, out_path.with_name(out_path.stem + "_deltaq.png"))
    return report


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _load_committed_relfrob(product: str, in_dir: str) -> dict:
    """The committed rel-Frobenius diagnostic (linearized-vs-MC decision covariance, Corollary 2a)
    from ``{product}_decision.json``'s ``crosscheck.rel_frobenius`` -- NOT ``{product}_curvature.json``'s
    ``delta_rel_var_error`` (that field is the legacy raw-FD-Hessian-predicted delta_q, the object
    ``docs/decision_null_theorem.md`` §4.2 found numerically unreliable on real products; using it
    here would silently compare the new whitened diagnostic against the very quantity it replaces).
    ``rel_frobenius`` is a single joint value over the (purity, yield) covariance, so it applies to
    both QoI rows identically -- see ``scripts/bayes_curvature.py``'s own ``rel_frobenius_committed``
    sourcing, the precedent this mirrors."""
    import json
    path = Path(in_dir) / f"{product}_decision.json"
    if not path.exists():
        return {}
    try:
        d = json.loads(path.read_text())
        rf = d.get("crosscheck", {}).get("rel_frobenius")
        return {"purity": rf, "yield": rf} if rf is not None else {}
    except Exception:
        return {}


def _write_csv(products_out: dict, path: Path) -> None:
    cols = ["product", "qoi", "C_lin", "grad_z_norm", "frob_H2_hat", "frob_H2_rel_se", "delta_q",
           "hvp_step_status", "mixed_symmetry_rel", "mesh_delta_q_rel_spread", "relFrob_existing", "status"]
    lines = [",".join(cols)]
    for p, by_qoi in products_out.items():
        for qoi, rec in by_qoi.items():
            lines.append(",".join(str(rec.get(c, "")) for c in cols))
    path.write_text("\n".join(lines) + "\n")


def _plot_delta_q(products_out: dict, png_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels, deltas, colors = [], [], []
    color_map = {"PASS_CERTIFIED_CURVATURE": "tab:green", "WARN_DIAGNOSTIC_ONLY": "tab:orange",
                "FAIL_DIAGNOSTIC_ONLY": "tab:red"}
    for p, by_qoi in products_out.items():
        for qoi, rec in by_qoi.items():
            if not np.isfinite(rec.get("delta_q", float("nan"))):
                continue
            labels.append(f"{p}\n{qoi}"); deltas.append(rec["delta_q"])
            colors.append(color_map.get(rec.get("status"), "tab:gray"))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(labels)), 4))
    ax.bar(range(len(labels)), deltas, color=colors)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"$\delta_q$ (whitened, posterior-whitened HVP)")
    ax.set_title("Posterior-whitened curvature diagnostic")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
