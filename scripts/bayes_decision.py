"""Decision-relevant identifiability on a REAL product (Direction A, decision tool).

Loads a product's real curves, builds the MAP + Laplace posterior exactly like
``bayes_calibrate.py``, then reports the DECISION-space diagnostic at a chosen
operating window: pooled main-group purity + yield, their posterior std, the
tolerance-whitened decision worst-direction ``worst_dec``, and -- the headline --
the parameter worst-direction ``worst_dir`` alongside it.  The story: theta is not
identified (``worst_dir`` large) yet the DECISION is well-determined (``worst_dec``
small), because the unidentified directions are the sloppy ones that barely move
purity/yield.

DEFERRED: run on Colab (heavy ODE; Laplace Jacobian is the cost) once the P1 posterior
numbers are available; not executed in the implementation round.  Example:
    OMP_NUM_THREADS=4 python scripts/bayes_decision.py --product HLXSYN --n-steps 300 \
        --map-iters 150 --mc-samples 200
    # optional explicit production window:
    OMP_NUM_THREADS=4 python scripts/bayes_decision.py --product HLXSYN --decision-op 30 40 80 12

By default this reuses a saved ``{product}_posterior.npz`` (from ``bayes_calibrate.py``) if
one exists at ``--out-dir`` / ``--posterior``, skipping the MAP+Laplace refit entirely --
same ``_load_or_fit`` convention as ``bayes_design_experiments.py`` / ``bayes_fisher_ablation.py``.
Pass ``--refit`` to force a fresh MAP fit even when a saved posterior is present.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import cex_model.app_support as A
from cex_model.bayes import (
    AKTA_NOISE_FLOOR_G_L,
    Posterior,
    components_to_u,
    decision_covariance,
    decision_covariance_mc,
    decision_jacobian,
    decision_report,
    identifiability_report,
    laplace_posterior,
    map_fit,
    mechanistic_model,
    physical_prior,
)
from cex_model.diffsolver.calibrate_diff import targets_from_bundle


def _load_or_fit_posterior(bundle, prior, args, out: Path) -> Posterior:
    path = Path(args.posterior) if args.posterior else out / f"{args.product}_posterior.npz"
    if path.exists() and not args.refit:
        print(f"loaded posterior: {path} (pass --refit to force a fresh MAP fit instead)")
        return Posterior.load(path)
    print(f"{'--refit set' if path.exists() else f'no saved posterior at {path}'}; "
          f"fitting MAP+Laplace fresh (n_steps={args.n_steps}, iters={args.map_iters}) ...")
    n = bundle.components.n_protein
    predict_fn, obs = mechanistic_model(targets_from_bundle(bundle, n_steps=args.n_steps), n)
    t0 = time.time()
    u_map, _ = map_fit(components_to_u(bundle.components), predict_fn, obs, prior,
                       sigma_obs=AKTA_NOISE_FLOOR_G_L, iters=args.map_iters, lr=args.lr, progress=True)
    post = laplace_posterior(u_map, predict_fn, obs, prior, template=bundle.components,
                             sigma_obs=AKTA_NOISE_FLOOR_G_L)
    print(f"MAP + Laplace in {time.time() - t0:.1f}s")
    return post


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--product", default="HLXSYN")
    ap.add_argument("--posterior", default=None, help="path to a saved {product}_posterior.npz")
    ap.add_argument("--refit", action="store_true",
                    help="force a fresh MAP+Laplace fit even if a saved posterior exists")
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--map-iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--decision-op", type=float, nargs=4, default=None,
                    metavar=("LOADING", "GSTART", "GEND", "ECV"),
                    help="operating window for the decision; default = first experiment's OP")
    ap.add_argument("--tol-purity", type=float, default=0.02)
    ap.add_argument("--tol-yield", type=float, default=0.05)
    ap.add_argument("--tau-dec", type=float, default=1.0)
    ap.add_argument("--mc-samples", type=int, default=0, help="if >0, also run the MC cross-check")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="drop experiments whose name contains any of these substrings "
                         "(e.g. DT for HLXSYN's non-standard run, to match the section 3.3 posterior)")
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
    grp = A.group_indices(b.components)
    print("components:", [(c.name, c.component_type.value, round(float(c.fraction), 1))
                          for c in b.components.components])
    print(f"group_indices (acid/main/basic) = {grp}")
    if not grp["main"]:
        print("WARNING: no MAIN component -> pooled purity (g[0]) is degenerate for this product")
    prior = physical_prior(n)
    e0 = b.experiments[0]
    op = (list(args.decision_op) if args.decision_op is not None
          else [e0.loading_g_l, e0.gradient_start_pct, e0.gradient_end_pct, e0.elution_cv])
    tol = (args.tol_purity, args.tol_yield)
    print(f"[{args.product}] n_protein={n}, n_exp={len(b.experiments)}, decision OP={op}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    post = _load_or_fit_posterior(b, prior, args, out)

    param = identifiability_report(post, prior)
    dec = decision_report(post, b, op, tol=tol, tau_dec=args.tau_dec, n_steps=args.n_steps)
    print(f"\nPARAMETER worst_dir = {param['worst_dir_shrinkage']:.3f} (identified={param['met']})")
    print(f"DECISION  worst_dec = {dec['worst_dec']:.3f} (met={dec['met']}; "
          f"window={dec['window_mode']} [{dec['window_s'][0]:.0f},{dec['window_s'][1]:.0f}]s)")
    print(f"  g@MAP: purity={dec['g_map']['pool_purity']:.3f} yield={dec['g_map']['pool_yield']:.3f}")
    for k in dec["decision_shrinkage"]:
        print(f"  {k:11s} std={dec['decision_std'][k]:.4f}  std/tol={dec['decision_shrinkage'][k]:.3f}")

    # Decision Jacobian G = d(purity,yield)/du at the MAP (cheap: one reverse pass per
    # quantity). Saved unconditionally so a downstream tool can recompute worst_dec under
    # a modified prior (e.g. the empirical-Bayes population prior, scripts/bayes_eb_pooling.py
    # --apply-to) without re-running the solver.
    G = decision_jacobian(b, op, post.u_map, n_steps=args.n_steps)

    payload = {"product": args.product, "decision_op": op, "tol": list(tol), "tau_dec": args.tau_dec,
               "param_worst_dir": param["worst_dir_shrinkage"], "param_identified": param["met"],
               "decision_quantities": list(dec["decision_std"].keys()),
               "decision_jacobian": np.asarray(G, float).tolist(),
               "decision": {k: dec[k] for k in ("worst_dec", "met", "decision_std",
                                                "decision_shrinkage", "g_map", "window_s", "window_mode")}}

    if args.mc_samples > 0:
        C_lin = decision_covariance(G, post.cov)
        mc = decision_covariance_mc(post, b, op, n_samples=args.mc_samples, n_steps=args.n_steps)
        C_mc = np.asarray(mc["C_mc"], float)
        rel = float(np.linalg.norm(C_lin - C_mc) / max(np.linalg.norm(C_mc), 1e-30))
        print(f"\nlinearized vs MC decision covariance: rel_Frobenius={rel:.2f} (n_used={mc['n_used']})")
        payload["crosscheck"] = {"C_lin": C_lin.tolist(), "C_mc": C_mc.tolist(),
                                 "rel_frobenius": rel, "n_used": mc["n_used"]}

    jpath = out / f"{args.product}_decision.json"
    jpath.write_text(json.dumps(payload, indent=2))
    print(f"\njson -> {jpath}")


if __name__ == "__main__":
    main()
