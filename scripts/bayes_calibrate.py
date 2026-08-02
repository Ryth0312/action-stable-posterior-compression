"""Per-product Bayesian SMA calibration (Direction A, slice 1: MAP + Laplace).

Loads a product's REAL curves, builds the Gaussian likelihood on the
differentiable solver, runs the MAP fit + Laplace posterior, prints
identifiability diagnostics, saves the posterior, and prints a GATE summary.

Heavy ODE -> run on Colab (high-CPU; GPU does not help the float64 solver).
The Laplace Jacobian is the dominant cost: one REVERSE pass per residual point
(HLXSYN ~315), so wall time ~= n_resid x one backward at the chosen --n-steps.
Start at a modest --n-steps for this machinery/identifiability gate; accuracy of
the posterior vs the RK23 truth path is checked separately by the (slice-2)
posterior-predictive step. Use --no-laplace for a fast MAP-only pass.

Examples
--------
    # fast MAP-only smoke (minutes)
    OMP_NUM_THREADS=4 python scripts/bayes_calibrate.py --product HLXSYN \
        --n-steps 200 --map-iters 80 --no-laplace
    # full MAP + Laplace posterior + diagnostics (the slice-1 gate)
    OMP_NUM_THREADS=4 python scripts/bayes_calibrate.py --product HLXSYN \
        --n-steps 300 --map-iters 150
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import cex_model.app_support as A
from cex_model.bayes import (
    AKTA_NOISE_FLOOR_G_L,
    Posterior,
    components_to_u,
    correlation_pairs,
    eigen_identifiability,
    identifiability_report,
    laplace_posterior,
    map_fit,
    mechanistic_model,
    physical_prior,
    posterior_predictive_rk23,
    shrinkage,
    svi_posterior,
)
from cex_model.bayes.plots import plot_identifiability, plot_posterior_predictive
from cex_model.diffsolver.calibrate_diff import targets_from_bundle


def _rmse(predict_fn, obs, u) -> float:
    """Curve RMSE (g/L) at parameter vector ``u`` on the differentiable solver."""
    with torch.no_grad():
        r = predict_fn(torch.tensor(np.asarray(u, float), dtype=torch.float64)) - obs
    return float(torch.sqrt(torch.mean(r**2)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--product", default="HLXSYN")
    ap.add_argument("--n-steps", type=int, default=300, help="BDF time-grid resolution")
    ap.add_argument("--init", choices=["config", "generic"], default="config",
                    help="MAP start: config = existing calibration; generic = product-agnostic cold")
    ap.add_argument("--init-from", default=None,
                    help="path to a saved {product}_posterior.npz to resume the MAP fit from its "
                         "u_map, instead of --init; lets --map-iters extend a previous run without "
                         "re-optimizing the iterations it already did (Adam has no other state to "
                         "resume -- momentum restarts fresh, only the parameter start point carries over)")
    ap.add_argument("--map-iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--engine", choices=["laplace", "svi"], default="laplace",
                    help="posterior engine: laplace (cheap, Gaussian) or svi (Pyro, full-cov)")
    ap.add_argument("--svi-steps", type=int, default=1500)
    ap.add_argument("--svi-lr", type=float, default=0.02)
    ap.add_argument("--no-laplace", action="store_true", help="MAP only (skip the posterior)")
    ap.add_argument("--predict", action="store_true", help="RK23 posterior-predictive + figures")
    ap.add_argument("--predict-samples", type=int, default=100)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="drop experiments whose name contains any of these substrings (fit AND predict)")
    ap.add_argument("--out-dir", default="results/bayes")
    args = ap.parse_args()

    b = A.load_product(args.product)
    if args.exclude:
        from cex_model.bayes.compare import filter_experiments
        kept, dropped = filter_experiments(b.experiments, args.exclude)
        if dropped:
            print(f"excluded {len(dropped)} experiment(s): {[e.name for e in dropped]}")
            b.experiments = kept
    n = b.components.n_protein
    tgs = targets_from_bundle(b, n_steps=args.n_steps)
    predict_fn, obs = mechanistic_model(tgs, n)
    prior = physical_prior(n)

    print(f"[{args.product}] n_protein={n}, n_exp={len(tgs)}, n_resid={int(obs.numel())}, "
          f"sigma_obs={AKTA_NOISE_FLOOR_G_L} g/L, n_steps={args.n_steps}")
    for j, c in enumerate(b.components.components):
        print(f"  c{j + 1} = {c.name} ({c.component_type.value}, feed {c.fraction:.1f}%)")

    u_config = components_to_u(b.components)
    if args.init_from:
        u_init = Posterior.load(args.init_from).u_map
        init_label = f"resumed from {args.init_from}"
    elif args.init == "config":
        u_init = u_config
        init_label = "config init"
    else:
        u_init = components_to_u(A.generic_components(n, b.components.fraction_array().tolist()))
        init_label = "generic init"
    rmse_init = _rmse(predict_fn, obs, u_init)

    t0 = time.time()
    u_map, hist = map_fit(u_init, predict_fn, obs, prior, sigma_obs=AKTA_NOISE_FLOOR_G_L,
                          iters=args.map_iters, lr=args.lr, progress=True)
    t_map = time.time() - t0
    rmse_map = _rmse(predict_fn, obs, u_map)
    rmse_config = _rmse(predict_fn, obs, u_config)
    print(f"\nMAP ({init_label}): nll {hist['nll'][0]:.3e} -> {hist['nll'][-1]:.3e} "
          f"in {t_map:.1f}s / {args.map_iters} iters")
    print(f"curve RMSE (g/L): init={rmse_init:.4f}  MAP={rmse_map:.4f}  config-ref={rmse_config:.4f}")

    post = None
    res = None
    out = Path(args.out_dir)
    if not args.no_laplace:
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if args.engine == "laplace":
            post = laplace_posterior(u_map, predict_fn, obs, prior, template=b.components,
                                     sigma_obs=AKTA_NOISE_FLOOR_G_L)
            built = f"Laplace in {time.time() - t0:.1f}s ({int(obs.numel())} Jacobian passes)"
        else:
            post, svi_hist = svi_posterior(predict_fn, obs, prior, template=b.components,
                                           sigma_obs=AKTA_NOISE_FLOOR_G_L, steps=args.svi_steps,
                                           lr=args.svi_lr, init_loc=u_map, progress=True)
            built = (f"SVI in {time.time() - t0:.1f}s (ELBO {svi_hist['loss'][0]:.3e} -> "
                     f"{svi_hist['loss'][-1]:.3e}, {args.svi_steps} steps)")
        path = out / f"{args.product}_posterior.npz"
        post.save(path)
        mineig = float(np.linalg.eigvalsh(post.cov).min())
        print(f"\n{args.engine} posterior: {built} -> {path}")
        print(f"posterior cov: dim={post.dim}, min eigval={mineig:.2e} (PSD if > 0)")

        sh = shrinkage(post, prior)
        print("shrinkage posterior_std/prior_std (sorted; < 1 = data-informed):")
        for name, r in sorted(sh.items(), key=lambda kv: kv[1]):
            print(f"   {name:14s} {r:.3f}")
        print("top posterior correlations (degeneracies):")
        for d in correlation_pairs(post, k=5):
            print(f"   {d['params'][0]:14s} <-> {d['params'][1]:14s} corr={d['corr']:+.3f}")
        print("widest (sloppiest) posterior directions:")
        for dct in eigen_identifiability(post)["directions"][:3]:
            tops = ", ".join(f"{p}({w:+.2f})" for p, w in zip(dct["top_params"], dct["weights"]))
            print(f"   std={dct['std']:.3f} ~ {tops}")

        if args.predict:
            t0 = time.time()
            res = posterior_predictive_rk23(post, b, n_samples=args.predict_samples)
            print(f"\nRK23 posterior-predictive ({args.predict_samples} samples) in {time.time() - t0:.1f}s:")
            for e in res["per_experiment"]:
                print(f"   {e['name']:24s} coverage={e['coverage']:.2f}  rmse={e['rmse_total']:.4f} g/L")
            agg = res["aggregate"]
            print(f"   aggregate: coverage={agg['coverage']:.2f}  rmse={agg['rmse_total']:.4f} g/L  "
                  f"model_adequate={agg['model_adequate']}")
            pngs = plot_posterior_predictive(res, out, product=args.product)
            plot_identifiability(shrinkage(post, prior), out / f"{args.product}_identifiability.png",
                                 title=f"{args.product} identifiability")
            print(f"   figures -> {out}/ ({len(pngs)} predictive PNG + identifiability bar)")

        # machine-readable summary for the cross-product comparison table (bayes_compare)
        rep = identifiability_report(post, prior)
        summary = {
            "product": args.product, "engine": args.engine, "n_steps": args.n_steps,
            "n_protein": n, "dim": int(post.dim),
            "curve_rmse": {"map": rmse_map, "config_ref": rmse_config, "init": rmse_init},
            "identifiability": {
                "met": rep["met"], "worst_dir_shrinkage": rep["worst_dir_shrinkage"],
                "n_met": int(post.dim - len(rep["unmet_params"])), "unmet_params": rep["unmet_params"],
                "max_corr": rep["max_corr"], "min_eig_precision": rep["min_eig_precision"],
                "shrinkage": rep["shrinkage"],
                "top_correlations": [{"params": list(d["params"]), "corr": d["corr"]}
                                     for d in correlation_pairs(post, k=5)],
            },
        }
        if res is not None:
            summary["predictive"] = {
                "level": res["level"], "n_samples": res["n_samples"], "aggregate": res["aggregate"],
                "per_experiment": [{k: v for k, v in e.items() if k != "band"}
                                   for e in res["per_experiment"]],
            }
        spath = out / f"{args.product}_summary.json"
        spath.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"summary -> {spath}")

    # ---- gate summary (thresholds documented in the plan / handoff) -------
    print("\n=== GATE SUMMARY ===")
    g1 = (rmse_map <= 1.2 * rmse_config + 1e-12) and (hist["nll"][-1] <= hist["nll"][0])
    print(f"G1 MAP fit {'PASS' if g1 else 'FAIL'}: RMSE_map={rmse_map:.4f} <= 1.2*config-ref"
          f"({rmse_config:.4f}); nll decreased={hist['nll'][-1] <= hist['nll'][0]}")
    if post is not None:
        ci_ok = all(np.all(np.isfinite(v)) for v in post.credible_interval(0.9).values())
        g2 = mineig > 0 and ci_ok
        print(f"G2 posterior {'PASS' if g2 else 'FAIL'}: PSD={mineig > 0}, all 90% CI finite={ci_ok}")
        n_pinned = sum(1 for r in sh.values() if r < 0.5)
        max_corr = max(abs(d["corr"]) for d in correlation_pairs(post, k=10))
        g3 = n_pinned >= 1 and max_corr > 0.7
        print(f"G3 identifiability {'PASS' if g3 else 'FAIL'}: {n_pinned}/{post.dim} params pinned "
              f"(<0.5); strongest |corr|={max_corr:.2f} (>0.7 => a degeneracy is present)")
    if res is not None:
        agg = res["aggregate"]
        g4 = agg["coverage"] >= res["level"] - 0.15
        print(f"G4 posterior-predictive {'PASS' if g4 else 'FAIL'}: RK23 coverage={agg['coverage']:.2f} "
              f"(nominal {res['level']}); model_adequate={agg['model_adequate']}")
    print("====================")


if __name__ == "__main__":
    main()
