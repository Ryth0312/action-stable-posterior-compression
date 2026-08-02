"""D4.2 -- active learning / optimal experiment design for the D4.1 differentiable peak model.

NO SMA, NO RK23: this only uses the real experiments' OPERATING CONDITIONS and the fact that D4.1's
OP->peak-parameter map is LINEAR (mu = mu0 + beta.z, sigma via softplus(sig0 + delta.z)). A linear slope is
identifiable exactly when the OP design matrix has information along its direction, so "which experiment to
run next" is classic **D-optimal experiment design**: pick OP points that most raise the information of the
design (equivalently, that most reduce the slope variance). The acquisition is the leverage z'(A'A)^-1 z -- a
transparent predictive-variance score, not a black box -- and a greedy batch keeps the picks COMPLEMENTARY.

Pipeline: safe_bounds (clamp to product-reasonable process limits) -> candidate_pool (valid, de-duplicated)
-> recommend_experiments (greedy D-optimal + per-pick explanation + a stability flag when the data is too
sparse for a stable design).
"""

from __future__ import annotations

import numpy as np

from cex_model.diffpeak.data import OP_FEATURES

# Product-reasonable process limits (encompass every product's measured experiments + headroom). These CAP the
# heuristic so a recommendation never runs off to an unphysical setpoint (e.g. gradient_end pushed to 126%).
# Overridable per product via ``recommend_experiments(..., bounds=...)`` when the real process limits are known.
OP_LIMITS: dict[str, tuple[float, float]] = {
    "loading_g_l": (5.0, 50.0), "gradient_start_pct": (0.0, 50.0),
    "gradient_end_pct": (40.0, 115.0), "elution_cv": (8.0, 40.0),
}
_GRAD_SPAN = 5.0   # require gradient_end > gradient_start + this (a valid gradient)


def safe_bounds(data, override: dict | None = None) -> dict:
    """Per-product SAFE operating bounds: the observed range expanded (so under-probed variables still get room
    to vary), CLAMPED to ``OP_LIMITS``. ``override`` sets exact bounds for any feature."""
    M, out = data.op_matrix, {}
    for k, name in enumerate(OP_FEATURES):
        if override and name in override:
            out[name] = (float(override[name][0]), float(override[name][1])); continue
        lo_g, hi_g = OP_LIMITS[name]
        obs_lo, obs_hi = float(M[:, k].min()), float(M[:, k].max())
        half = max(0.5 * (obs_hi - obs_lo), 0.25 * (hi_g - lo_g))   # floor lets a CONSTANT feature still vary
        out[name] = (max(lo_g, obs_lo - half), min(hi_g, obs_hi + half))
    return out


def _scale(op: np.ndarray, bounds: dict) -> np.ndarray:
    lo = np.array([bounds[n][0] for n in OP_FEATURES]); hi = np.array([bounds[n][1] for n in OP_FEATURES])
    return (np.asarray(op, dtype=float) - lo) / np.maximum(hi - lo, 1e-9)


def candidate_pool(data, bounds: dict, n: int = 4000, seed: int = 0, min_dist: float = 0.05) -> np.ndarray:
    """Sobol-sample OP points in ``bounds``; keep valid gradients (end > start + span) and drop points within
    ``min_dist`` (scaled) of an EXISTING experiment (no re-running what we already have)."""
    from scipy.stats import qmc
    lo = np.array([bounds[n_][0] for n_ in OP_FEATURES]); hi = np.array([bounds[n_][1] for n_ in OP_FEATURES])
    m = int(np.ceil(np.log2(max(2, n * 2))))
    pts = qmc.scale(qmc.Sobol(d=len(OP_FEATURES), scramble=True, seed=seed).random_base2(m), lo, hi)
    gs, ge = OP_FEATURES.index("gradient_start_pct"), OP_FEATURES.index("gradient_end_pct")
    pts = pts[pts[:, ge] > pts[:, gs] + _GRAD_SPAN]
    Xe = _scale(data.op_matrix, bounds)
    Xc = _scale(pts, bounds)
    far = np.array([float(np.min(np.linalg.norm(Xe - x, axis=1))) for x in Xc]) > min_dist
    return pts[far][:n]


def _augment(X: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)  # intercept + features


def _coverage(data, bounds):
    """Per-feature distinct values + whether the slope is identifiable (needs >=3 distinct OP values)."""
    M = data.op_matrix
    return [{"feature": n, "n_distinct": int(np.unique(np.round(M[:, k], 6)).size),
             "observed": [float(M[:, k].min()), float(M[:, k].max())], "safe_bounds": list(bounds[n]),
             "identifiable": int(np.unique(np.round(M[:, k], 6)).size) >= 3}
            for k, n in enumerate(OP_FEATURES)]


def recommend_experiments(data, bounds: dict | None = None, k: int = 5, n_candidates: int = 4000,
                          ridge: float = 0.05, seed: int = 0) -> dict:
    """Greedy D-optimal next experiments for ``data`` within safe ``bounds``. Returns the bounds, OP coverage,
    a candidate-pool sample, the top-``k`` recommendations (each EXPLAINED), and a design-stability flag."""
    bounds = bounds or safe_bounds(data)
    cov = _coverage(data, bounds)
    cand = candidate_pool(data, bounds, n_candidates, seed)
    Xe, Xc = _scale(data.op_matrix, bounds), _scale(cand, bounds)
    p = len(OP_FEATURES) + 1
    M = _augment(Xe).T @ _augment(Xe) + ridge * np.eye(p)
    rank0 = int(np.linalg.matrix_rank(_augment(Xe), tol=1e-6))

    recs, pool, poolX = [], cand.copy(), Xc.copy()
    for _ in range(k):
        if len(pool) == 0:
            break
        Minv = np.linalg.inv(M)
        A = _augment(poolX)
        lev = np.einsum("ni,ij,nj->n", A, Minv, A)           # D-optimal leverage = predictive variance
        b = int(np.argmax(lev))
        x_best, xs_best = pool[b], poolX[b]
        recs.append({**{n: round(float(x_best[i]), 2) for i, n in enumerate(OP_FEATURES)},
                     "leverage": float(lev[b]), "in_safe_bounds": True, **_explain(xs_best, Xe, cov)})
        M = M + np.outer(A[b], A[b])                          # greedy: fold the pick into the information matrix
        keep = np.linalg.norm(poolX - xs_best, axis=1) > 0.08  # diversify (drop near-duplicates of the pick)
        pool, poolX = pool[keep], poolX[keep]
        Xe = np.vstack([Xe, xs_best])

    deficiency = p - rank0                                    # # of unidentifiable design directions now
    stable = deficiency <= k and len(data.experiments) >= 3
    stability = {"design_rank": rank0, "full_rank": p, "unidentified_directions": deficiency,
                 "n_experiments": len(data.experiments), "stable_after_recommendations": bool(stable),
                 "note": ("" if stable else
                          f"only {len(data.experiments)} experiments and {deficiency} unidentified design "
                          f"directions -> the design is sparse; treat the recommendations as a first batch, "
                          f"re-run the design after collecting them")}
    return {"product": data.product, "safe_bounds": {n: list(bounds[n]) for n in OP_FEATURES},
            "op_coverage": cov, "recommendations": recs, "stability": stability,
            "candidate_pool_sample": cand[:: max(1, len(cand) // 400)].tolist()}


def _explain(xs: np.ndarray, Xe: np.ndarray, cov) -> dict:
    """Which OP direction(s) this pick fills: the feature(s) whose scaled value is most NOVEL vs the existing
    experiments (prioritising the under-probed ones) -> a plain-language reason, not just a score."""
    novelty = np.array([float(np.min(np.abs(Xe[:, k] - xs[k]))) for k in range(len(OP_FEATURES))])
    under = np.array([not c["identifiable"] for c in cov])
    score = novelty + 0.5 * under                              # bias toward under-probed features
    order = np.argsort(score)[::-1]
    fills = [OP_FEATURES[k] for k in order[:2] if novelty[k] > 0.1 or under[k]]
    primary = fills[0] if fills else OP_FEATURES[int(np.argmax(novelty))]
    return {"fills_directions": fills,
            "improves_slopes": [f"{primary} retention/width slope (all peaks)"],
            "why": f"varies {', '.join(fills) or primary} where existing experiments are "
                   f"{'constant/under-probed' if under[order[0]] else 'clustered'} -> highest slope-variance "
                   f"reduction (leverage)"}
