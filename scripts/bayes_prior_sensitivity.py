"""Prior-scale / -location robustness of the identifiability + decision verdicts.

Refits MAP+Laplace under a grid of prior variants (std x ``scale``, mean + ``shift``
prior-std) and reports, per variant, the prior-whitened parameter ``worst_dir`` and the
tolerance-whitened decision ``worst_dec`` at the historical window.  If the verdicts
(not-identified / determined) hold across the grid, the conclusion is not an artifact of
the specific weak prior -- the reviewer-facing "is the result prior-driven?" check.

Heavy (one refit per prior variant on the real solver; Laplace Jacobian is the cost)
-> Colab.  By default it WARM-STARTS every refit from a committed ``{product}_posterior.npz``
(``Posterior.load(...).u_map``), so the perturbed-prior fits start near the optimum and stay
cheap -- reusing the posterior already in the repo instead of a cold MAP fit.  Example:

    OMP_NUM_THREADS=4 python scripts/bayes_prior_sensitivity.py --product HLXSYN \
        --n-steps 300 --scales 0.5 0.7 1.0 1.5 2.0
    OMP_NUM_THREADS=4 python scripts/bayes_prior_sensitivity.py --product HLXSYN --exclude DT \
        --n-steps 300 --shifts -0.5 0.0 0.5
    # cold start (ignore the saved posterior) if none exists:
    OMP_NUM_THREADS=4 python scripts/bayes_prior_sensitivity.py --product HLXSYN --no-warm-start

Writes ``results/bayes/{product}_prior_sensitivity.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import cex_model.app_support as A
from cex_model.bayes import Posterior, physical_prior, prior_sensitivity


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--product", default="HLXSYN")
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--map-iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.7, 1.0, 1.5, 2.0],
                    help="prior std multipliers (1.0 reproduces the committed posterior)")
    ap.add_argument("--shifts", type=float, nargs="+", default=[0.0],
                    help="prior mean shifts in units of prior std")
    ap.add_argument("--decision-op", type=float, nargs=4, default=None,
                    metavar=("LOADING", "GSTART", "GEND", "ECV"),
                    help="operating window for worst_dec; default = first experiment's OP")
    ap.add_argument("--tol-purity", type=float, default=0.02)
    ap.add_argument("--tol-yield", type=float, default=0.05)
    ap.add_argument("--tau-dec", type=float, default=1.0)
    ap.add_argument("--no-decision", action="store_true", help="parameter worst_dir only (no worst_dec)")
    ap.add_argument("--init-from", default=None,
                    help="posterior .npz to warm-start every refit from; default = "
                         "{out-dir}/{product}_posterior.npz if present")
    ap.add_argument("--no-warm-start", action="store_true",
                    help="cold-start each fit from the components' u (ignore the saved posterior)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="drop experiments whose name contains any of these substrings (e.g. DT)")
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

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    u_init = None
    if not args.no_warm_start:
        init_path = Path(args.init_from) if args.init_from else out / f"{args.product}_posterior.npz"
        if init_path.exists():
            u_init = Posterior.load(init_path).u_map
            print(f"warm-starting every refit from {init_path} (u_map)")
        else:
            print(f"no saved posterior at {init_path}; cold-starting from the components' u")

    op = list(args.decision_op) if args.decision_op is not None else None
    print(f"[{args.product}] n_protein={n}, n_exp={len(b.experiments)}, "
          f"scales={args.scales}, shifts={args.shifts}, decision={not args.no_decision}")

    t0 = time.time()
    res = prior_sensitivity(
        b, prior, scale_grid=tuple(args.scales), shift_grid=tuple(args.shifts), op=op,
        tol=(args.tol_purity, args.tol_yield), tau_dec=args.tau_dec, n_steps=args.n_steps,
        map_iters=args.map_iters, lr=args.lr, decision=not args.no_decision,
        u_init=u_init, progress=True)
    print(f"\nprior sweep ({len(res['grid'])} refits) in {time.time() - t0:.1f}s")

    # Stability summary: do the verdicts hold across the whole grid?
    dirs = [r["worst_dir"] for r in res["grid"]]
    res["worst_dir_range"] = [float(min(dirs)), float(max(dirs))]
    res["identified_stable"] = len({r["identified"] for r in res["grid"]}) == 1
    if not args.no_decision:
        decs = [r["worst_dec"] for r in res["grid"]]
        res["worst_dec_range"] = [float(min(decs)), float(max(decs))]
        res["decision_met_stable"] = len({r["decision_met"] for r in res["grid"]}) == 1
        print(f"worst_dir in [{res['worst_dir_range'][0]:.3f}, {res['worst_dir_range'][1]:.3f}] "
              f"(identified stable={res['identified_stable']}); "
              f"worst_dec in [{res['worst_dec_range'][0]:.3f}, {res['worst_dec_range'][1]:.3f}] "
              f"(decision met stable={res['decision_met_stable']})")

    res["product"] = args.product
    res["config"] = {"n_steps": args.n_steps, "map_iters": args.map_iters,
                     "exclude": list(args.exclude), "warm_started": u_init is not None}
    jpath = out / f"{args.product}_prior_sensitivity.json"
    jpath.write_text(json.dumps(res, indent=2))
    print(f"json -> {jpath}")


if __name__ == "__main__":
    main()
