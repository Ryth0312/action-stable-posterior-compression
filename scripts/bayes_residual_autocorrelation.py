"""How serially correlated the within-run residuals are, before and after the deployed noise model.

The independent-residual likelihood treats the fractions of a run as exchangeable draws with a common
scale. That is the assumption the correlated refit replaces, and this measures the gap it is replacing:
the lag-1 autocorrelation of the residuals of the reference fit, pooled over the blocks of a product,
against the same statistic after whitening those residuals by the fitted
Ornstein-Uhlenbeck-plus-nugget covariance

    Sigma_e = sigma_nug^2 I + sigma_corr^2 R_ell,   (R_ell)_ij = exp(-|t_i - t_j| / ell),

at the hyperparameters each product's committed refit selected. Whitening is by the Cholesky factor, so
a correctly specified noise model leaves residuals with no lag-1 structure left to find.

Run:
  python scripts/bayes_residual_autocorrelation.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PRODUCTS = ['HLXSYN']
NAME = {'HLXSYN': 'mAb A'}
_MAP = {"HLXSYN": ("HLXSYN", ["DT"])}


def lag1(x):
    """Lag-1 autocorrelation of one centred block, or None if the block is too short or degenerate."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return None
    x = x - x.mean()
    d = float(x @ x)
    return float(x[:-1] @ x[1:] / d) if d > 0 else None


def blocks_for(pid, in_dir):
    """Residuals of the reference fit, one block per (experiment, observation group), with their times."""
    import torch

    import cex_model.app_support as A
    from cex_model.bayes.compare import filter_experiments
    from cex_model.bayes.likelihood import unpack_u
    from cex_model.bayes.posterior import Posterior
    from cex_model.diffsolver.calibrate_diff import targets_from_bundle
    from cex_model.diffsolver.torch_solver import DTYPE

    prod, drop = _MAP.get(pid, (pid, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
    post = Posterior.load(Path(in_dir) / f"{pid}_posterior.npz")   # the fit the hyperparameters were read off
    u = torch.tensor(np.asarray(post.u_map, float), dtype=DTYPE)
    n = post.n_protein

    out = []
    for tg in targets_from_bundle(bundle, n_steps=300):
        with torch.no_grad():
            curve = np.asarray(tg.sim.elution_curve(*unpack_u(u, n), differentiable=False), float)
        groups = tg.groups if tg.groups is not None else [[j] for j in range(n)]
        t_obs = np.asarray(tg.times_s, float)
        y_obs = np.asarray(tg.values, float)
        for k, g in enumerate(groups):
            fit = np.interp(t_obs, curve[:, 0], curve[:, 2 + np.asarray(g)].sum(1))
            out.append((t_obs, y_obs[:, k] - fit))
    return out


def whiten(t, r, hyper):
    """Sigma^{-1/2} r under the fitted OU-plus-nugget covariance on this block's own sample times."""
    d = np.abs(t[:, None] - t[None, :])
    S = hyper["sigma_nug"] ** 2 * np.eye(len(t)) + hyper["sigma_corr"] ** 2 * np.exp(-d / hyper["ell"])
    return np.linalg.solve(np.linalg.cholesky(S), r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=PRODUCTS)
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--out", default="results/bayes/residual_autocorrelation.json")
    args = ap.parse_args()

    rows = []
    for pid in args.products:
        hyper = json.loads((Path(args.in_dir) / f"{pid}_correlated.json").read_text())["hyper"]
        raw, whi = [], []
        for t, r in blocks_for(pid, args.in_dir):
            a = lag1(r)
            b = lag1(whiten(t, r, hyper))
            if a is not None:
                raw.append(a)
            if b is not None:
                whi.append(b)
        r = {"product": pid, "name": NAME.get(pid, pid), "n_blocks": len(raw),
             "kernel": hyper["kernel"], "ell_s": hyper["ell"], "rho": hyper["rho"],
             "sigma_nug": hyper["sigma_nug"], "sigma_corr": hyper["sigma_corr"],
             "lag1_independent_mean": float(np.mean(raw)), "lag1_independent_median": float(np.median(raw)),
             "lag1_whitened_mean": float(np.mean(whi)), "lag1_whitened_median": float(np.median(whi)),
             "lag1_independent_blocks": [float(x) for x in raw],
             "lag1_whitened_blocks": [float(x) for x in whi]}
        rows.append(r)
        print(f"{r['name']:6s} blocks={r['n_blocks']:3d}  lag-1 residual ACF "
              f"{r['lag1_independent_mean']:+.3f} -> {r['lag1_whitened_mean']:+.3f} (mean), "
              f"{r['lag1_independent_median']:+.3f} -> {r['lag1_whitened_median']:+.3f} (median); "
              f"ell={r['ell_s']:.0f}s rho={r['rho']}")

    Path(args.out).write_text(json.dumps({"rows": rows}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
