"""Wall-clock / peak-memory benchmark for the gradient-Bayesian calibration path.

Times the production MAP + Laplace pipeline (`targets_from_bundle` ->
`mechanistic_model` -> `map_fit` -> `laplace_posterior`) at one or more BDF grid
resolutions on a real product, and records the cheap solver-free PbP-applicability
check for the same product. Reports, per grid size: target-build time, MAP-fit
wall-clock, Laplace-posterior wall-clock (the one-Jacobian cost on top of the MAP),
total, and peak resident set size (`resource.getrusage`). Numbers are machine- and
`--map-iters`-dependent; report the reference machine (cores) alongside them. The
transferable observations are the ratios (Laplace ~= one reverse-mode Jacobian
relative to the MAP fit) and the memory scaling in n_steps x n_resid.

    OMP_NUM_THREADS=4 python scripts/bayes_timing.py --product HLXSYN --n-steps 150 300 --map-iters 120
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

import cex_model.app_support as A
from cex_model.bayes import (AKTA_NOISE_FLOOR_G_L, components_to_u,
                             laplace_posterior, map_fit, mechanistic_model,
                             physical_prior)
from cex_model.bayes.pbp import pbp_applicability
from cex_model.diffsolver.calibrate_diff import targets_from_bundle


def _peak_mb() -> float:
    # ru_maxrss is KB on Linux, bytes on macOS; assume Linux for the reference machine.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--product", default="HLXSYN")
    ap.add_argument("--n-steps", type=int, nargs="+", default=[150, 300])
    ap.add_argument("--map-iters", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default="results/bayes/timing.json")
    args = ap.parse_args()

    b = A.load_product(args.product)
    n = len(b.components.fraction_array())
    opm = [[e.loading_g_l, e.gradient_start_pct, e.gradient_end_pct, e.elution_cv]
           for e in b.experiments]
    pbp = pbp_applicability(opm)
    u_init = components_to_u(b.components)
    prior = physical_prior(n)

    rows = []
    for ns in args.n_steps:
        t0 = time.perf_counter()
        tgs = targets_from_bundle(b, n_steps=ns)
        predict_fn, obs = mechanistic_model(tgs, n)
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        u_map, _ = map_fit(u_init, predict_fn, obs, prior,
                           sigma_obs=AKTA_NOISE_FLOOR_G_L, iters=args.map_iters, lr=args.lr)
        t_map = time.perf_counter() - t0

        t0 = time.perf_counter()
        laplace_posterior(u_map, predict_fn, obs, prior, template=b.components,
                          sigma_obs=AKTA_NOISE_FLOOR_G_L)
        t_lap = time.perf_counter() - t0

        row = dict(n_steps=ns, n_resid=int(np.asarray(obs).size),
                   t_build_s=round(t_build, 2), t_map_s=round(t_map, 2),
                   t_laplace_s=round(t_lap, 2), t_total_s=round(t_build + t_map + t_lap, 2),
                   peak_rss_mb=round(_peak_mb()))
        rows.append(row)
        print(f"n_steps={ns:4d} n_resid={row['n_resid']} build={t_build:5.2f}s "
              f"map({args.map_iters}it)={t_map:7.2f}s laplace={t_lap:6.2f}s "
              f"total={row['t_total_s']:7.2f}s peakRSS={row['peak_rss_mb']}MB")
    print(f"PbP applicable={pbp['applicable']} (distinct_slopes={pbp['n_distinct_slopes']})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(product=args.product, n_protein=n, dim=4 * n,
                                   map_iters=args.map_iters, pbp=pbp, rows=rows), indent=1),
                   encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
