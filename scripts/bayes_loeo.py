"""P0 validation bundle for one product (Direction A reviewer responses).

Runs, on a product's REAL curves, the four held-out / robustness checks that turn
the in-sample posterior-predictive *check* into genuine validation:

  * auto-OOD outlier flag    -- which experiments are protocol-mismatch / OOD
                                (so a drop is justified, not cherry-picked);
  * empirical-Bayes sigma_obs -- RMS of the MAP residuals (does the assumed
                                0.148 g/L noise floor match the data?);
  * retrospective BOED       -- credit each EXISTING experiment by its information
                                content (real-data complement to the synthetic loop);
  * leave-one-experiment-out -- fit on n-1, predict the held-out curve on RK23
                                (the genuine out-of-sample number)        [--loeo];
  * decision-level LOEO      -- fit on n-1, predict the held-out OP's pooled
                                purity/yield and compare to the value read from the
                                held-out measured curve (out-of-sample DECISION) [--decision];
  * sigma_obs sensitivity    -- refit across a sigma grid                  [--sigma-sweep];
  * n_steps sensitivity      -- refit across BDF time-grid resolutions     [--n-steps-sweep].

Heavy ODE / many refits -> Colab.  Writes ``results/bayes/{product}_validation.json``.

Examples
--------
    # core checks (1 MAP + 1 Laplace + per-exp Jacobians + 1 predictive)
    OMP_NUM_THREADS=4 python scripts/bayes_loeo.py --product HLXSYN --n-steps 300
    # add the genuine out-of-sample LOEO (n refits) and the sigma sweep
    OMP_NUM_THREADS=4 python scripts/bayes_loeo.py --product HLXSYN --n-steps 300 \
        --loeo --sigma-sweep
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
    components_to_u,
    decision_leave_one_experiment_out,
    empirical_sigma_obs,
    flag_outlier_experiments,
    laplace_posterior,
    leave_one_experiment_out,
    map_fit,
    mechanistic_model,
    n_steps_sensitivity,
    physical_prior,
    Posterior,
    posterior_predictive_rk23,
    retrospective_design,
    sigma_obs_sensitivity,
)
from cex_model.diffsolver.calibrate_diff import targets_from_bundle


def _op_matrix(experiments) -> list[list[float]]:
    return [[e.loading_g_l, e.gradient_start_pct, e.gradient_end_pct, e.elution_cv] for e in experiments]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--product", default="HLXSYN")
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--map-iters", type=int, default=150)
    ap.add_argument("--init-from", default=None,
                    help="warm-start every MAP fit (base + LOEO/sweep folds) from a committed "
                         "posterior's u_map (e.g. results/bayes/HLXSYN_posterior.npz); LOEO/sweeps are "
                         "small perturbations of the full fit, so this converges in far fewer --map-iters "
                         "than the generic-init default (which needs thousands of iters for HLXSYN)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--predict-samples", type=int, default=100)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="drop experiments whose name contains any of these substrings")
    ap.add_argument("--loeo", action="store_true", help="run leave-one-experiment-out (n refits)")
    ap.add_argument("--decision", action="store_true",
                    help="run DECISION-level leave-one-experiment-out: predict the held-out OP's pooled "
                         "purity/yield from the n-1 posterior and compare to the value read from the "
                         "held-out measured curve (n refits + decision Jacobian/MC per fold)")
    ap.add_argument("--mc-samples", type=int, default=200,
                    help="Monte-Carlo posterior-predictive draws for the --decision band")
    ap.add_argument("--sigma-sweep", action="store_true", help="run sigma_obs sensitivity (refit per sigma)")
    ap.add_argument("--n-steps-sweep", action="store_true",
                    help="run discretization (n_steps) sensitivity (refit per resolution)")
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
    prior = physical_prior(n)
    tgs = targets_from_bundle(b, n_steps=args.n_steps)
    predict_fn, obs = mechanistic_model(tgs, n)
    print(f"[{args.product}] n_protein={n}, n_exp={len(tgs)}, n_resid={int(obs.numel())}, "
          f"n_steps={args.n_steps}")

    # --- full-data MAP + Laplace (shared by the outlier flag / retrospective / empirical sigma)
    u_init = None
    if args.init_from:
        u_init = Posterior.load(args.init_from).u_map
        print(f"warm-starting MAP fits from {args.init_from} (u_map)")
    u0 = components_to_u(b.components) if u_init is None else np.asarray(u_init, float)
    t0 = time.time()
    u_map, _ = map_fit(u0, predict_fn, obs, prior, sigma_obs=AKTA_NOISE_FLOOR_G_L,
                       iters=args.map_iters, lr=args.lr, progress=True)
    post = laplace_posterior(u_map, predict_fn, obs, prior, template=b.components,
                             sigma_obs=AKTA_NOISE_FLOOR_G_L)
    with torch.no_grad():
        resid = (predict_fn(torch.tensor(u_map, dtype=torch.float64)) - obs).numpy()
    sigma_hat = empirical_sigma_obs(resid)
    print(f"MAP + Laplace in {time.time() - t0:.1f}s; empirical sigma_obs={sigma_hat:.4f} g/L "
          f"(floor {AKTA_NOISE_FLOOR_G_L})")

    res = posterior_predictive_rk23(post, b, n_samples=args.predict_samples)
    per_cov = [e["coverage"] for e in res["per_experiment"]]
    flags = flag_outlier_experiments(_op_matrix(b.experiments), per_cov)
    for e, f in zip(b.experiments, flags):
        tag = " <- FLAGGED" if f["flagged"] else ""
        print(f"   {e.name[:28]:28s} coverage={f['coverage']:.2f} op_z={f['op_leverage_z']:.1f}{tag}")

    print("retrospective BOED (info credit of each existing experiment):")
    retro = retrospective_design(b, prior, u_map, n_steps=args.n_steps)
    for r in retro:
        print(f"   rank {r['rank']}: {r['name'][:28]:28s} eig={r['eig']:.3f} nats  {r['op']}")

    out = {
        "product": args.product, "n_steps": args.n_steps, "n_exp": len(tgs),
        "empirical_sigma_obs": sigma_hat, "noise_floor": AKTA_NOISE_FLOOR_G_L,
        "predictive_coverage": res["aggregate"]["coverage"],
        "outlier_flags": [{"name": e.name, **f} for e, f in zip(b.experiments, flags)],
        "retrospective_eig": [{"name": r["name"], "rank": r["rank"], "eig": r["eig"], "op": r["op"]}
                              for r in retro],
    }

    if args.loeo:
        t0 = time.time()
        loeo = leave_one_experiment_out(b, prior, n_steps=args.n_steps, map_iters=args.map_iters,
                                        lr=args.lr, predict_samples=args.predict_samples,
                                        u_init=u_init, progress=True)
        out["loeo"] = loeo
        print(f"LOEO out-of-sample coverage={loeo['aggregate']['coverage']:.2f} "
              f"(in-sample {res['aggregate']['coverage']:.2f}) in {time.time() - t0:.1f}s")

    if args.decision:
        t0 = time.time()
        dloeo = decision_leave_one_experiment_out(
            b, prior, n_steps=args.n_steps, map_iters=args.map_iters, lr=args.lr,
            u_init=u_init, mc_samples=args.mc_samples, progress=True)
        out["decision_loeo"] = dloeo
        a = dloeo["aggregate"]
        print(f"decision-LOEO: {a['n_observable']}/{a['n_folds']} folds observable, "
              f"frac in-band={a['frac_in_band']:.2f}, worst_dec range "
              f"[{a['worst_dec_range'][0]:.3f}, {a['worst_dec_range'][1]:.3f}] in {time.time() - t0:.1f}s")

    if args.sigma_sweep:
        t0 = time.time()
        sweep = sigma_obs_sensitivity(b, prior, n_steps=args.n_steps, map_iters=args.map_iters,
                                      lr=args.lr, predict_samples=args.predict_samples,
                                      u_init=u_init, progress=True)
        out["sigma_obs_sweep"] = sweep
        print(f"sigma_obs sweep in {time.time() - t0:.1f}s: "
              + ", ".join(f"sigma={r['sigma_obs']:.3f}->cov {r.get('coverage', float('nan')):.2f}"
                          for r in sweep["grid"]))

    if args.n_steps_sweep:
        t0 = time.time()
        nsweep = n_steps_sensitivity(b, prior, sigma_obs=AKTA_NOISE_FLOOR_G_L, map_iters=args.map_iters,
                                     lr=args.lr, predict_samples=args.predict_samples,
                                     u_init=u_init, progress=True)
        out["n_steps_sweep"] = nsweep
        print(f"n_steps sweep in {time.time() - t0:.1f}s: "
              + ", ".join(f"n={r['n_steps']}->drift {r['map_drift_vs_coarsest']:.3f}"
                          for r in nsweep["grid"]))

    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    path = od / f"{args.product}_validation.json"
    # merge into any existing artefact so the heavy sub-analyses (--loeo / --sigma-sweep /
    # --n-steps-sweep) can be run in separate Colab sessions without losing earlier results.
    if path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        prev.update(out)
        out = prev
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"validation -> {path} (keys: {', '.join(sorted(out))})")


if __name__ == "__main__":
    main()
