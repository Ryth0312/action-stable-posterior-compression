"""Leave-one-experiment-out evaluation + identifiability diagnostics for the differentiable peak model (D4.1).

The point of D4.1 is HONEST identifiability, not a pretty in-sample fit: with only 3-6 experiments per product
the OP->peak-parameter SLOPES are the hard part. This module reports:
  * leave_one_out  -- fit on all-but-one experiment, predict the held-out one (does the OP->param model
                      generalise, or only interpolate the points it saw?);
  * op_coverage    -- how many DISTINCT values each operating variable takes (a slope+curvature needs >=3);
                      under-probed variables have unidentifiable slopes, and we recommend which experiment to
                      add (criterion #5);
  * bootstrap_uncertainty -- residual bootstrap -> per-parameter spread; slopes whose CI spans 0 are
                      unidentifiable (optional, costs n_boot refits);
  * perturbation_trends   -- vary one OP variable a little and check the predicted area/RT/width move smoothly.
"""

from __future__ import annotations

import numpy as np
import torch

from cex_model.diffpeak.data import OP_FEATURES
from cex_model.diffpeak.fit import FitConfig, fit_diffpeak, peak_metrics
from cex_model.diffpeak.model import DTYPE

_LOO_KEYS = ("curve_rmse", "area_rel_err", "rt_err_s", "purity_abs_err", "recovery_abs_err", "yield_rel_err")


def leave_one_out(data, config: FitConfig | None = None) -> dict:
    """Fit on every experiment-but-one and score the held-out experiment. Returns per-fold + aggregate
    held-out metrics (over the PREDICTABLE folds) plus ``n_diverged``: a held-out experiment that VARIES an
    operating variable the training fold held CONSTANT is fundamentally unpredictable (that slope is
    unidentifiable from the rest) -- we report it as diverged with the offending variable, which IS the
    identifiability finding, not a bug to paper over."""
    n = len(data.experiments)
    folds = []
    for held in range(n):
        train_idx = np.array([i for i in range(n) if i != held])
        model, _ = fit_diffpeak(data, config, idx=train_idx)
        m = peak_metrics(model, data, idx=np.array([held]))
        e = data.experiments[held]
        with torch.no_grad():
            pred_total = model.predict_total(torch.as_tensor(e.op, dtype=DTYPE),
                                             torch.as_tensor(e.times, dtype=DTYPE)).cpu().numpy()
        tr_op = data.op_matrix[train_idx]
        unpred = [OP_FEATURES[k] for k in range(e.op.shape[0])
                  if tr_op[:, k].std() < 1e-6 and abs(e.op[k] - tr_op[0, k]) > 1e-6]  # varies a train-constant OP
        folds.append({"held_out": e.name, "predictable": bool(np.isfinite(m["aggregate"]["curve_rmse"])),
                      "unpredicted_op": unpred, **{k: m["aggregate"][k] for k in _LOO_KEYS},
                      "times": e.times.tolist(), "obs_total": e.curves.sum(0).tolist(),
                      "pred_total": np.nan_to_num(pred_total, posinf=0.0, neginf=0.0).tolist()})
    fin = [f for f in folds if f["predictable"]]
    agg = {k: float(np.mean([f[k] for f in fin])) if fin else float("nan") for k in _LOO_KEYS}
    return {"folds": folds, "aggregate": agg, "n_folds": n, "n_predictable": len(fin),
            "n_diverged": n - len(fin)}


def op_coverage(data) -> dict:
    """Per operating variable: range + number of DISTINCT sampled values. A linear slope needs >=2 distinct
    values and a curvature/identifiable slope-with-confidence really needs >=3; variables with fewer are
    UNDER-PROBED and their OP->parameter slopes are unidentifiable. Recommends experiments that vary them."""
    M = data.op_matrix
    cov, recommend = [], []
    for k, name in enumerate(OP_FEATURES):
        vals = np.unique(np.round(M[:, k], 6))
        nd = int(vals.size)
        cov.append({"feature": name, "min": float(M[:, k].min()), "max": float(M[:, k].max()),
                    "n_distinct": nd, "values": vals.tolist(), "identifiable": nd >= 3})
        if nd < 3:
            v = vals.tolist()
            if nd == 1:                                   # constant -> suggest a +-30% bracket
                suggest = [round(v[0] * 0.7, 3), round(v[0] * 1.3, 3)]
            else:                                          # 2 distinct -> fill the midpoint + extend one end
                suggest = [round(float(np.mean(v)), 3), round(max(v) * 1.2, 3)]
            recommend.append({"feature": name, "current_values": v, "suggest_values": suggest,
                              "why": f"only {nd} distinct value(s) -> its retention/width OP-slope is unidentifiable"})
    return {"coverage": cov, "under_probed": [c["feature"] for c in cov if not c["identifiable"]],
            "recommendation": recommend}


@torch.no_grad()
def perturbation_trends(model, data, feature: str, n: int = 11, span_frac: float = 0.25) -> dict:
    """Vary one OP ``feature`` +-span_frac*range around the mean experiment; return the predicted per-peak
    area / retention / width along the sweep (for a smoothness/monotonicity sanity check)."""
    k = OP_FEATURES.index(feature)
    base = data.op_matrix.mean(0)
    rng = float(data.op_matrix[:, k].max() - data.op_matrix[:, k].min()) or abs(base[k]) or 1.0
    xs = np.linspace(base[k] - span_frac * rng, base[k] + span_frac * rng, n)
    area, mu, sigma = [], [], []
    for x in xs:
        op = base.copy(); op[k] = x
        a, m, s, _ = (t.cpu().numpy() for t in model.peak_params_real(torch.as_tensor(op, dtype=DTYPE)))
        area.append(a); mu.append(m); sigma.append(s)
    return {"feature": feature, "x": xs.tolist(), "area": np.array(area), "mu_s": np.array(mu),
            "sigma_s": np.array(sigma)}


def bootstrap_uncertainty(data, config: FitConfig | None = None, n_boot: int = 20, seed: int = 0) -> dict:
    """Residual bootstrap: fit once, then refit on (fitted curve + resampled residuals) ``n_boot`` times.
    Per OP->parameter SLOPE, report mean/std across refits; a slope with |mean| < 2*std is UNIDENTIFIABLE
    (its bootstrap CI spans 0). Costs ``n_boot`` refits -- optional."""
    rng = np.random.default_rng(seed)
    model, _ = fit_diffpeak(data, config)
    resid = []
    with torch.no_grad():
        fitted = [model.predict_peaks(torch.as_tensor(e.op, dtype=DTYPE),
                                      torch.as_tensor(e.times, dtype=DTYPE)).cpu().numpy() for e in data.experiments]
    for e, f in zip(data.experiments, fitted):
        resid.append((e.curves - f).ravel())
    pool = np.concatenate(resid)
    betas, deltas = [], []
    for _ in range(n_boot):
        boot = _resampled(data, fitted, pool, rng)
        bm, _ = fit_diffpeak(boot, config)
        betas.append(bm.beta.detach().cpu().numpy()); deltas.append(bm.delta.detach().cpu().numpy())
    betas, deltas = np.array(betas), np.array(deltas)
    return {"beta": _slope_report(betas, "mu", model), "delta": _slope_report(deltas, "sigma", model),
            "n_boot": n_boot}


def _resampled(data, fitted, pool, rng):
    from cex_model.diffpeak.data import Experiment, PeakData
    exps = []
    for e, f in zip(data.experiments, fitted):
        noisy = np.clip(f + rng.choice(pool, size=f.shape, replace=True), 0.0, None)
        exps.append(Experiment(name=e.name, times=e.times, curves=noisy, op=e.op, loading=e.loading))
    return PeakData(product=data.product, n_peaks=data.n_peaks, peak_types=data.peak_types,
                    frac=data.frac, experiments=exps)


def _slope_report(arr, param, model) -> list:
    """arr: (n_boot, n_peaks, n_op) -> per (peak, op) mean/std + identifiable flag."""
    mean, std = arr.mean(0), arr.std(0)
    out = []
    for j in range(arr.shape[1]):
        for k in range(arr.shape[2]):
            out.append({"param": param, "peak": j, "op": OP_FEATURES[k], "mean": float(mean[j, k]),
                        "std": float(std[j, k]), "identifiable": bool(abs(mean[j, k]) > 2 * std[j, k])})
    return out
