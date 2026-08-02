"""Posterior-driven optimal next experiment (theta-space Bayesian D-optimal / BOED).

Given a posterior over ``theta`` and a product, score each candidate operating
condition (OP) by how much a hypothetical experiment there would shrink the
posterior -- the local Bayesian D-optimal expected information gain

    EIG(c) = 1/2 [ logdet(H + J_cᵀ J_c / sigma²) - logdet(H) ]

where ``H = inv(cov)`` is the current precision and ``J_c = d(curve)/d(theta)``
at the MAP for the candidate's differentiable forward.  This naturally targets
the sloppy directions (small eigenvalues of ``H``) -- e.g. the structural
keq<->nu degeneracy and the unconstrained sigma that gradient-only data leaves
open -- and reports, per recommendation, which parameters it most pins down.

Candidates come from the same safe-bounds / Sobol pool as the empirical
``diffpeak`` D-optimal design (reused), so the two are directly comparable.
The Jacobian is reverse-mode (forward-mode is wrong through the IFT solver), so
keep ``n_steps`` / ``n_points`` / ``n_candidates`` modest -- this is the cost.
"""

from __future__ import annotations

import numpy as np
import torch

from cex_model.bayes.likelihood import AKTA_NOISE_FLOOR_G_L, unpack_u
from cex_model.diffpeak.design import OP_FEATURES, candidate_pool, safe_bounds
from cex_model.diffsolver.calibrate_diff import _grouped_curve
from cex_model.diffsolver.torch_solver import DTYPE, TorchSimulator
from cex_model.gradients import build_fitting_inlet

__all__ = ["candidate_predict_fn", "expected_info_gain", "recommend_experiments_bayes",
           "candidate_pool_for"]


class _OpData:
    """Lightweight stand-in for diffpeak PeakData (safe_bounds/candidate_pool read op_matrix)."""

    def __init__(self, op_matrix: np.ndarray):
        self.op_matrix = np.asarray(op_matrix, float)
        self.experiments = list(range(len(self.op_matrix)))


def _bundle_op_matrix(bundle) -> np.ndarray:
    return np.array([[e.loading_g_l, e.gradient_start_pct, e.gradient_end_pct, e.elution_cv]
                     for e in bundle.experiments], dtype=float)


def candidate_predict_fn(bundle, op, *, n_steps: int, n_points: int, u_map):
    """Differentiable forward for a candidate OP, evaluated at ``n_points`` elution times.

    ``op = (loading, gradient_start_pct, gradient_end_pct, elution_cv)``.  Reuses
    ``build_fitting_inlet`` + ``TorchSimulator`` exactly like ``targets_from_bundle``
    (buffers fixed per product); returns ``predict_fn(u) -> flat (n_points*n_obs,)``.
    """
    n_protein = bundle.components.n_protein
    groups = bundle.observation_groups
    fr = bundle.components.fraction_array()
    e0 = bundle.experiments[0]
    loading, gstart, gend, ecv = (float(x) for x in op)
    inlet = build_fitting_inlet(
        buffer_a=e0.buffer_a, buffer_b=e0.buffer_b, gradient_start_pct=gstart, gradient_end_pct=gend,
        elution_cv=ecv, rt_min=bundle.column.rt, load_amount_g_l=loading, component_fractions_pct=fr)
    sim = TorchSimulator(bundle.column, bundle.components, inlet, loading, bundle.correction, n_steps=n_steps)

    u0 = torch.tensor(np.asarray(u_map, float), dtype=DTYPE)
    with torch.no_grad():
        curve0 = sim.elution_curve(*unpack_u(u0, n_protein), differentiable=False)
    times = torch.linspace(float(curve0[0, 0]), float(curve0[-1, 0]), n_points, dtype=DTYPE)

    def predict_fn(u: torch.Tensor) -> torch.Tensor:
        curve = sim.elution_curve(*unpack_u(u, n_protein), differentiable=True)
        return _grouped_curve(curve, times, groups, n_protein).reshape(-1)

    return predict_fn


def candidate_pool_for(bundle, *, n_candidates: int = 24, seed: int = 0) -> np.ndarray:
    """The safe-bounds Sobol OP pool for a bundle (shared by BOED, baselines, ablations).

    Every design strategy (theta-D-optimal, random, space-filling) and the Fisher-no-prior
    ablation draws from this SAME deterministic pool so the comparison is apples-to-apples.
    """
    data = _OpData(_bundle_op_matrix(bundle))
    return np.asarray(candidate_pool(data, safe_bounds(data), n=n_candidates, seed=seed), float)


def expected_info_gain(H: np.ndarray, J: np.ndarray, sigma_obs: float) -> float:
    """Local Bayesian D-optimal gain ``0.5*(logdet(H + JᵀJ/sigma²) - logdet(H))`` (nats)."""
    M = (J.T @ J) / (sigma_obs**2)
    _, ld_new = np.linalg.slogdet(H + M)
    _, ld_old = np.linalg.slogdet(H)
    return 0.5 * float(ld_new - ld_old)


def _param_reductions(H: np.ndarray, M: np.ndarray, names: list[str]):
    """Fractional posterior-std reduction per parameter from adding a candidate (sorted)."""
    std_old = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0.0, None))
    std_new = np.sqrt(np.clip(np.diag(np.linalg.inv(H + M)), 0.0, None))
    red = 1.0 - std_new / np.clip(std_old, 1e-30, None)
    return [(names[i], float(red[i])) for i in np.argsort(red)[::-1]]


def recommend_experiments_bayes(posterior, bundle, *, k: int = 5, n_candidates: int = 24,
                                n_steps: int = 120, n_points: int = 12, sigma_obs: float | None = None,
                                seed: int = 0, progress: bool = False) -> dict:
    """Top-``k`` next experiments by theta-space expected information gain."""
    sigma_obs = float(sigma_obs if sigma_obs is not None else posterior.sigma_obs)
    data = _OpData(_bundle_op_matrix(bundle))
    bounds = safe_bounds(data)
    pool = candidate_pool(data, bounds, n=n_candidates, seed=seed)
    H = np.linalg.inv(posterior.cov)
    u_map, names = posterior.u_map, posterior.names

    scored = []
    for idx, op in enumerate(pool):
        J = torch.autograd.functional.jacobian(
            candidate_predict_fn(bundle, op, n_steps=n_steps, n_points=n_points, u_map=u_map),
            torch.tensor(np.asarray(u_map, float), dtype=DTYPE)).detach().numpy()
        M = (J.T @ J) / (sigma_obs**2)
        scored.append({"op": np.asarray(op, float), "eig": expected_info_gain(H, J, sigma_obs),
                       "reductions": _param_reductions(H, M, names)})
        if progress:
            print(f"  candidate {idx + 1}/{len(pool)}: EIG={scored[-1]['eig']:.3f}")
    scored.sort(key=lambda r: r["eig"], reverse=True)

    recs = []
    for r in scored[:k]:
        targets = r["reductions"][:3]
        rec = {OP_FEATURES[j]: float(r["op"][j]) for j in range(len(OP_FEATURES))}
        rec.update({
            "info_gain": round(r["eig"], 3),
            "targets": [n for n, _ in targets],
            "why": "most shrinks " + ", ".join(f"{n}({v:.0%})" for n, v in targets),
            "in_safe_bounds": True,
        })
        recs.append(rec)
    return {"product": getattr(bundle, "product_id", "?"), "safe_bounds": bounds,
            "n_candidates": int(len(pool)), "recommendations": recs}
