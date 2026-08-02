"""Posterior-predictive checks on the RK23 truth path (the slice-2 centerpiece).

This is where the value lands *without any new experiments*: draw parameters
from the posterior, predict each experiment's chromatogram on the RK23 reference
solver (the project's lossless contract), and report (a) the predictive band's
COVERAGE of the real curve, (b) per-peak RMSE, and (c) main-peak retention /
height / purity errors.  Coverage well below the nominal level flags **model
inadequacy** (SMA+EDM structurally insufficient), not just a bad fit.

Inference runs on the fast differentiable solver; this validation runs on RK23 --
honouring the documented torch<->RK23 gap.  Reuses ``app_support.simulate_elution``
(RK23) and ``metrics.aggregate_observation_curve``.
"""

from __future__ import annotations

import numpy as np

from cex_model.app_support import simulate_elution
from cex_model.bayes.prior import physical_u_bounds
from cex_model.metrics import aggregate_observation_curve

__all__ = ["predict_experiment", "posterior_predictive_rk23"]

# numpy>=2.0 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def predict_experiment(column, correction, components, exp, groups, *, method: str = "RK23") -> np.ndarray:
    """RK23 predicted observed-curve for one experiment, aligned to its sample times.

    Returns ``(n_exp_points, n_obs)`` g/L (proteins grouped into observed peaks).
    """
    curve = simulate_elution(
        column, components, buffer_a=exp.buffer_a, buffer_b=exp.buffer_b,
        gradient_start_pct=exp.gradient_start_pct, gradient_end_pct=exp.gradient_end_pct,
        elution_cv=exp.elution_cv, loading_g_l=exp.loading_g_l, correction=correction, method=method)
    agg = aggregate_observation_curve(curve, groups)  # [time, salt, obs1..]
    t_sim, obs_cols = agg[:, 0], agg[:, 2:]
    t_exp = exp.curve[:, 0]
    return np.stack([np.interp(t_exp, t_sim, obs_cols[:, k]) for k in range(obs_cols.shape[1])], axis=1)


def _peak_stats(curve_2d: np.ndarray, t: np.ndarray):
    """Per-column retention time (s), height (g/L), and purity (area fraction)."""
    rt = t[np.argmax(curve_2d, axis=0)]
    height = curve_2d.max(axis=0)
    area = _trapz(np.clip(curve_2d, 0.0, None), t, axis=0)
    purity = area / max(area.sum(), 1e-12)
    return rt, height, purity


def posterior_predictive_rk23(posterior, bundle, *, n_samples: int = 100, seed: int = 0,
                              level: float = 0.9, method: str = "RK23") -> dict:
    """Posterior-predictive coverage + peak errors for every experiment in ``bundle``."""
    us = posterior.samples(n_samples, seed=seed)
    # Clip draws to the model's physical support: the unbounded Gaussian (Laplace/SVI)
    # can sample e.g. nu < 0 for a very sloppy 2-experiment product, which makes the SMA
    # term csalt**nu blow up (0**negative) and the reference solver fail to integrate.
    lo, hi = physical_u_bounds(posterior.n_protein)
    us = np.clip(us, lo, hi)
    comps = [posterior.to_components(u) for u in us]
    groups = bundle.observation_groups
    rng = np.random.default_rng(seed)
    sigma_obs = float(getattr(posterior, "sigma_obs", 0.0) or 0.0)
    per_exp = []
    for exp in bundle.experiments:
        # Per-draw safety net: a residual pathological draw that still fails to integrate
        # is dropped (counted) rather than killing the whole posterior-predictive run.
        pred_list, n_dropped = [], 0
        for c in comps:
            try:
                pred_list.append(predict_experiment(
                    bundle.column, bundle.correction, c, exp, groups, method=method))
            except (RuntimeError, FloatingPointError, ValueError):
                n_dropped += 1
        if len(pred_list) < 2:
            raise RuntimeError(
                f"posterior-predictive failed for experiment {exp.name!r}: only "
                f"{len(pred_list)}/{len(comps)} posterior draws integrated on {method}")
        preds = np.stack(pred_list)  # (n_used, n_pts, n_obs)
        mean = preds.mean(axis=0)
        # posterior-predictive band = parameter spread (epistemic) (+) observation noise (aleatoric);
        # coverage is checked against NOISY data, so the noise term is required for a fair test.
        ypred = preds + sigma_obs * rng.standard_normal(preds.shape)
        lo = np.quantile(ypred, 0.5 * (1.0 - level), axis=0)
        hi = np.quantile(ypred, 0.5 * (1.0 + level), axis=0)
        real = exp.curve[:, 1:]
        t = exp.curve[:, 0]

        rt_p, h_p, pur_p = _peak_stats(mean, t)
        rt_r, h_r, pur_r = _peak_stats(real, t)
        per_exp.append({
            "name": exp.name,
            "coverage": float(np.mean((real >= lo) & (real <= hi))),
            "rmse_total": float(np.sqrt(np.mean((mean - real) ** 2))),
            "rmse_per_obs": np.sqrt(np.mean((mean - real) ** 2, axis=0)).tolist(),
            "rt_err_s": (rt_p - rt_r).tolist(),
            "height_err_g_l": (h_p - h_r).tolist(),
            "purity_abs_err": np.abs(pur_p - pur_r).tolist(),
            "n_used": int(preds.shape[0]), "n_dropped": int(n_dropped),
            "band": {"t": t, "lo": lo, "hi": hi, "mean": mean, "real": real},  # for plotting
        })
    cov = float(np.mean([e["coverage"] for e in per_exp])) if per_exp else float("nan")
    rmse = float(np.mean([e["rmse_total"] for e in per_exp])) if per_exp else float("nan")
    return {
        "product": getattr(bundle, "product_id", "?"), "level": level, "n_samples": n_samples,
        "method": method, "per_experiment": per_exp,
        "aggregate": {"coverage": cov, "rmse_total": rmse,
                      "model_adequate": cov >= level - 0.15},  # coverage << nominal => misspecified
    }
