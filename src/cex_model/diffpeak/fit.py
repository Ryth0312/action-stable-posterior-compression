"""Fit the differentiable peak model to REAL outlet curves and read out integration-based metrics (D4.1).

Loss = curve MSE (the only data term) + physical penalties that the parameterisation cannot already enforce
structurally:
  * structural (no penalty): concentration/area >= 0 (softplus + EMG), width > 0 (softplus), area proportional
    to loaded mass x fraction, recovery in [0,1] (it is a ratio of integrals);
  * penalised: retention ORDER (mu increasing along the elution columns), SKEW regularised toward 0, and an
    L2 prior on the OP->parameter SLOPES so directions the few experiments cannot constrain shrink to zero.
Process metrics (purity / recovery / yield) are obtained by INTEGRATING the predicted curve -- there is no
black-box regression head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from cex_model.diffpeak.model import DTYPE, DiffPeakModel

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy>=2.0 renamed trapz -> trapezoid


@dataclass
class FitConfig:
    epochs: int = 4000
    lr: float = 0.02
    weight_order: float = 1.0        # retention-order hinge (per-experiment mu monotonic in elution order)
    order_margin: float = 0.05       # min peak separation in STANDARDIZED time the order penalty asks for
    weight_skew: float = 1e-3        # softplus(tau0)^2 -> small skew unless the curve tails demand it
    weight_slope: float = 1e-2       # L2 on beta/delta (standardized-OP slopes) -> identifiability prior
    seed: int = 0


def _curve_mse(model: DiffPeakModel, data, idx) -> torch.Tensor:
    se, n = model.frac.new_zeros(()), 0
    for i in idx:
        e = data.experiments[i]
        t = torch.as_tensor(e.times, dtype=DTYPE)
        pred = model.predict_peaks(torch.as_tensor(e.op, dtype=DTYPE), t)
        obs = torch.as_tensor(e.curves, dtype=DTYPE)
        se = se + ((pred - obs) ** 2).sum()
        n += obs.numel()
    return se / max(n, 1)


def _order_penalty(model: DiffPeakModel, data, idx, margin) -> torch.Tensor:
    pen = model.frac.new_zeros(())
    for i in idx:
        _, mu, _, _ = model.peak_params(torch.as_tensor(data.experiments[i].op, dtype=DTYPE))  # standardized
        if mu.numel() > 1:
            pen = pen + torch.relu(mu[:-1] - mu[1:] + margin).pow(2).sum()
    return pen


def fit_diffpeak(data, config: FitConfig | None = None, idx: np.ndarray | None = None):
    """Fit on experiments ``idx`` (default all); returns (model, history). Full-batch (the data is tiny)."""
    cfg = config or FitConfig()
    torch.manual_seed(cfg.seed)
    idx = np.arange(len(data.experiments)) if idx is None else np.asarray(idx)
    op_mean, op_std = data.op_standardizer(idx)
    all_t = np.concatenate([data.experiments[i].times for i in idx])      # standardize time -> O(1) params
    model = DiffPeakModel(data.n_peaks, data.frac, op_mean, op_std, float(all_t.mean()),
                          float(all_t.std() + 1e-8)).to(DTYPE)
    model.init_baselines_(_subset(data, idx))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: list[dict] = []
    best_loss, best_state = float("inf"), None
    for epoch in range(cfg.epochs):
        opt.zero_grad()
        mse = _curve_mse(model, data, idx)
        loss = (mse
                + cfg.weight_order * _order_penalty(model, data, idx, cfg.order_margin)
                + cfg.weight_skew * nn.functional.softplus(model.tau0).pow(2).sum()
                + cfg.weight_slope * (model.beta.pow(2).sum() + model.delta.pow(2).sum()))
        if not torch.isfinite(loss):
            break                       # a too-small fold can destabilise a peak -> keep the best finite state
        lv = float(loss.detach())
        if lv < best_loss:              # track the best (also returns the best, not the last, epoch)
            best_loss = lv
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)  # safety net vs transient peak-collapse spikes
        opt.step()
        if epoch % 200 == 0 or epoch == cfg.epochs - 1:
            history.append({"epoch": epoch, "loss": lv, "curve_mse": float(mse.detach())})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _subset(data, idx):
    from cex_model.diffpeak.data import PeakData
    return PeakData(product=data.product, n_peaks=data.n_peaks, peak_types=data.peak_types, frac=data.frac,
                    experiments=[data.experiments[i] for i in idx])


def _fine_grid(times: np.ndarray, n: int = 600) -> np.ndarray:
    return np.linspace(float(times.min()), float(times.max()), n)


@torch.no_grad()
def peak_metrics(model: DiffPeakModel, data, idx: np.ndarray | None = None) -> dict:
    """Per-experiment, integration-based errors of the prediction vs the measured curve: curve RMSE, per-peak
    AREA and RETENTION-TIME error, and purity/recovery/yield (integrated over the main peak +-2 sigma window).
    ``main`` peak = the observed peak typed 'main'. Returns aggregate means + the per-experiment rows."""
    idx = np.arange(len(data.experiments)) if idx is None else np.asarray(idx)
    main_j = data.peak_types.index("main") if "main" in data.peak_types else int(np.argmax(data.frac))
    rows = []
    for i in idx:
        e = data.experiments[i]
        op = torch.as_tensor(e.op, dtype=DTYPE)
        area, mu, sigma, tau = (x.cpu().numpy() for x in model.peak_params_real(op))
        pred = model.predict_peaks(op, torch.as_tensor(e.times, dtype=DTYPE)).cpu().numpy()
        obs = e.curves
        curve_rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
        obs_area = _trapz(obs, e.times, axis=1)                       # (n_peaks,) measured peak integrals
        area_rel = float(np.mean(np.abs(area - obs_area) / np.clip(obs_area, 1e-9, None)))
        obs_ctr = np.array([float((e.times * obs[j]).sum() / max(obs[j].sum(), 1e-9)) for j in range(len(obs))])
        rt_err = float(np.mean(np.abs(mu - obs_ctr)))
        # purity/recovery/yield by integrating on a fine grid over the main +-2 sigma window
        g = _fine_grid(e.times)
        pg = model.predict_peaks(op, torch.as_tensor(g, dtype=DTYPE)).cpu().numpy()          # (n_peaks, G) pred
        og = np.stack([np.interp(g, e.times, obs[j]) for j in range(len(obs))])               # obs interpolated
        w = (g >= mu[main_j] - 2 * sigma[main_j]) & (g <= mu[main_j] + 2 * sigma[main_j])
        pp = _window(pg, g, w, main_j); oo = _window(og, g, w, main_j)
        rows.append({"name": e.name, "curve_rmse": curve_rmse, "area_rel_err": area_rel, "rt_err_s": rt_err,
                     "purity_pred": pp["purity"], "purity_obs": oo["purity"],
                     "recovery_pred": pp["recovery"], "recovery_obs": oo["recovery"],
                     "yield_pred": pp["yield"], "yield_obs": oo["yield"]})
    agg = {k: float(np.mean([r[k] for r in rows])) for k in ("curve_rmse", "area_rel_err", "rt_err_s")}
    agg["purity_abs_err"] = float(np.mean([abs(r["purity_pred"] - r["purity_obs"]) for r in rows]))
    agg["recovery_abs_err"] = float(np.mean([abs(r["recovery_pred"] - r["recovery_obs"]) for r in rows]))
    agg["yield_rel_err"] = float(np.mean([abs(r["yield_pred"] - r["yield_obs"]) / max(r["yield_obs"], 1e-9)
                                          for r in rows]))
    return {"aggregate": agg, "per_experiment": rows, "main_peak": main_j}


def _window(curves: np.ndarray, grid: np.ndarray, w: np.ndarray, main_j: int) -> dict:
    """purity / recovery / yield of the main peak over the boolean window ``w`` (all by integration)."""
    total_main = _trapz(curves[main_j], grid)
    coll_main = _trapz(curves[main_j][w], grid[w]) if w.sum() > 1 else 0.0
    coll_total = _trapz(curves[:, w].sum(0), grid[w]) if w.sum() > 1 else 0.0
    return {"purity": float(coll_main / coll_total) if coll_total > 1e-12 else 0.0,
            "recovery": float(coll_main / total_main) if total_main > 1e-12 else 0.0,
            "yield": float(coll_main)}
