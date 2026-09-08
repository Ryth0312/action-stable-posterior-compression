"""What the two committed hierarchy draw sets change, and what they cannot change.

Supplement B records that two draw sets are committed and that every deployed read names the capped one.
This quantifies the difference on the objects the article reports, at each product's historical operating
condition and at the correlated refit both sets sit on:

  * the discrepancy scales and the per-product bias means, read from the draws themselves;
  * the predictive worst-direction decision width, which the draws enter through Sigma_delta;
  * the predictive P(meet), which is the reported quantity the draws can actually move.

The CONDITIONAL decision width is not compared, because it is a functional of theta | M alone and no draw
set can move it. The capped column reproduces the committed deployed scan, which is the check that this
recomputation is the same object the article reads.

Run:
  python scripts/bayes_draw_set_comparison.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

import cex_model.app_support as A
from cex_model.bayes.compare import filter_experiments
from cex_model.bayes.decision import decision_jacobian
from cex_model.bayes.posterior import Posterior

SPEC = np.array([0.70, 0.50])
TOL = np.array([0.02, 0.05])
C_MEAS = np.diag(np.array([0.005, 0.008]) ** 2)
_PRODUCT_MAP = {"HLXSYN": ("HLXSYN", ["DT"])}
NAME = {'HLXSYN': 'mAb A'}


def p_meet(g, C, bias, sd):
    return float(np.mean([
        multivariate_normal(mean=-(g + bias[k]), cov=C + sd[k] + C_MEAS + 1e-12 * np.eye(2),
                            allow_singular=True).cdf(-SPEC) for k in range(len(sd))]))


def w_pred(C, sd):
    Ti = np.diag(1.0 / TOL)
    return float(np.sqrt(np.linalg.eigvalsh(Ti @ (C + sd.mean(0) + C_MEAS) @ Ti).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=['HLXSYN'])
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--primary", default="results/bayes/hier_draws.npz")
    ap.add_argument("--deployed", default="results/bayes/hier_draws_capB.npz")
    ap.add_argument("--out", default="results/bayes/draw_set_comparison.json")
    args = ap.parse_args()

    a, b = np.load(args.primary), np.load(args.deployed)
    Ma, Mb = a["Sd_draws"].mean(0), b["Sd_draws"].mean(0)
    out = {"primary": args.primary, "deployed": args.deployed, "n_draws": int(len(a["Sd_draws"])),
           "sd_primary": [float(np.sqrt(Ma[0, 0])), float(np.sqrt(Ma[1, 1]))],
           "sd_deployed": [float(np.sqrt(Mb[0, 0])), float(np.sqrt(Mb[1, 1]))], "rows": []}

    for pid in args.products:
        post = Posterior.load(Path(args.in_dir) / f"{pid}_correlated_posterior.npz")
        op = json.loads((Path(args.in_dir) / f"{pid}_decision.json").read_text())["decision_op"]
        prod, drop = _PRODUCT_MAP.get(pid, (pid, []))
        bundle = A.load_product(prod)
        if drop:
            bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
        G, g, _ = decision_jacobian(bundle, list(map(float, op)), post.u_map, n_steps=300,
                                    return_extra=True)
        G = np.atleast_2d(G)
        C = G @ post.cov @ G.T
        r = {"product": pid, "name": NAME.get(pid, pid), "op": [float(v) for v in op],
             "bias_gap_max": float(np.abs(a[f"b_{pid}"].mean(0) - b[f"b_{pid}"].mean(0)).max()),
             "w_pred_primary": w_pred(C, a["Sd_draws"]), "w_pred_deployed": w_pred(C, b["Sd_draws"]),
             "p_meet_primary": p_meet(g, C, a[f"b_{pid}"], a["Sd_draws"]),
             "p_meet_deployed": p_meet(g, C, b[f"b_{pid}"], b["Sd_draws"])}
        r["p_meet_shift"] = r["p_meet_deployed"] - r["p_meet_primary"]
        r["w_pred_shift"] = r["w_pred_deployed"] - r["w_pred_primary"]
        out["rows"].append(r)
        print(f"{r['name']:6s} bias gap={r['bias_gap_max']:.4f}  w_pred {r['w_pred_primary']:.4f}"
              f"->{r['w_pred_deployed']:.4f}  P(meet) {r['p_meet_primary']:.5f}"
              f"->{r['p_meet_deployed']:.5f}  ({r['p_meet_shift']:+.4f})")

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nsd  primary {tuple(round(x,4) for x in out['sd_primary'])}  "
          f"deployed {tuple(round(x,4) for x in out['sd_deployed'])}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
