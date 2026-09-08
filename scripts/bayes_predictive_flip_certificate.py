"""The candidate-level linearisation certificate at the DEPLOYED law, in its exact (flip) form.

``step4_r2_paired_coupling.py --hier-draws`` records the two directional disagreement counts between the
nonlinear and the linearised arm under the predictive mixture. For an indicator loss main text Lemma 3.2 is
an identity rather than an inequality,

    P_nl - P_lin = q^+ - q^-,     q^+ = P{X' in Abar, Y' notin Abar},  q^- = P{Y' in Abar, X' notin Abar},

so exact one-sided Clopper-Pearson limits on the two counts bound the gap in each direction separately. That
is strictly sharper than the union form P{tube} + P{|X-Y| > t}, which charges the whole boundary tube whether
or not a paired difference reaches it. This assembles the two limits with the support-gap bridge, centres the
interval on the deployed P(meet), and asks whether the three-way threshold classification is preserved.

It also re-derives the flip counts from the committed paired draws, so the artifact is checked and not
trusted.

Run:
  python scripts/bayes_predictive_flip_certificate.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

SPEC = np.array([0.70, 0.50])
C_MEAS = np.diag(np.array([0.005, 0.008]) ** 2)
DELTA, M_GRID = 0.05, 50
TAU_HI, TAU_LO = 0.95, 0.05
NAME = {'HLXSYN': 'mAb A'}


def cp_upper(k, n, alpha):
    return 1.0 if k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))


def classify(p):
    return "decisive-meet" if p >= TAU_HI else ("decisive-miss" if p <= TAU_LO else "ambiguous")


def flips(X, Y, tol, U):
    inX = np.all((X + U) / tol >= SPEC / tol, axis=1)
    inY = np.all((Y + U) / tol >= SPEC / tol, axis=1)
    return int(np.sum(inX & ~inY)), int(np.sum(inY & ~inX))


def predictive_draw(product, hier, n, rng):
    bias, sd = hier[f"b_{product}"], hier["Sd_draws"]
    L = np.linalg.cholesky(sd + C_MEAS + 1e-14 * np.eye(2))
    k = rng.integers(0, len(sd), n)
    return bias[k] + np.einsum("nij,nj->ni", L[k], rng.standard_normal((n, 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/bayes/r2_paired_coupling_*correlated*_op*.json")
    ap.add_argument("--draws-dir", default="results/bayes")
    ap.add_argument("--hier-draws", default="results/bayes/hier_draws_capB.npz")
    ap.add_argument("--predictive-cert", default="results/bayes/predictive_linearisation_certificate.json")
    ap.add_argument("--seed", type=int, default=0, help="the seed the paired run used; flips use seed+1")
    ap.add_argument("--out", default="results/bayes/predictive_flip_certificate.json")
    args = ap.parse_args()

    alpha = DELTA / (2 * M_GRID + 2)          # the budget the paired artifact was computed at
    pred = {(r["product"], round(r["op"][0], 4)): r
            for r in json.load(open(args.predictive_cert))["rows"]}
    hier = np.load(args.hier_draws)
    rows = []
    for f in sorted(glob.glob(args.glob)):
        d = json.load(open(f))[0]
        pf = d.get("predictive_flips")
        if not pf:
            print(f"skip {Path(f).name}: no flip block (re-run with --hier-draws)")
            continue
        pr = pred[(d["product"], round(d["op"][0], 4))]
        z = np.load(Path(args.draws_dir) / f"paired_draws_{d['product']}_op{d['op'][0]:g}.npz")
        X, Y, tol, n = z["X"], z["Y"], z["tol"], z["X"].shape[0]

        # the draws must be the ones the conditional certificate was computed from
        pX = float(np.mean(np.all(X / tol >= SPEC / tol, axis=1)))
        pY = float(np.mean(np.all(Y / tol >= SPEC / tol, axis=1)))
        matched = (abs(pX - d["P_X_meet_nonlinear"]) < 1e-12
                   and abs(pY - d["P_Y_meet_linear"]) < 1e-12 and np.allclose(z["op"], d["op"]))

        up, dn = flips(X, Y, tol, predictive_draw(d["product"], hier, n,
                                                  np.random.default_rng(args.seed + 1)))
        reproduced = (up == round(pf["flip_up_rate"] * n) and dn == round(pf["flip_down_rate"] * n))
        # the same identity at the conditional law, on the same draws, isolates form from law
        cup, cdn = flips(X, Y, tol, np.zeros_like(X))

        q_up, q_dn = cp_upper(up, n, alpha), cp_upper(dn, n, alpha)
        p_dep, bridge = pr["P_meet_deployed"], pr["support_gap_upper"]
        lo, hi = max(0.0, p_dep - q_dn - bridge), min(1.0, p_dep + q_up + bridge)
        state = classify(p_dep)
        rows.append({
            "product": d["product"], "name": NAME[d["product"]], "loading": float(d["op"][0]),
            "op": [float(v) for v in d["op"]], "n_paired": n,
            "P_meet_deployed": p_dep, "flip_up_count": up, "flip_down_count": dn,
            "flip_up_upper": q_up, "flip_down_upper": q_dn, "bridge_upper": bridge,
            "certified_bound": float(max(q_up, q_dn) + bridge), "certified_interval": [lo, hi],
            "state": state, "classification_preserved": bool(classify(lo) == state == classify(hi)),
            "flip_counts_conditional": [cup, cdn],
            "bound_conditional_union": d["certified_bound"],
            "bound_deployed_union": pr["certified_bound"],
            "bound_conditional_flip": float(max(cp_upper(cup, n, alpha), cp_upper(cdn, n, alpha))
                                            + d["support_gap_upper"]),
            "draws_match_conditional_artifact": bool(matched), "flips_reproduced": bool(reproduced),
        })
        r = rows[-1]
        print(f"{r['name']} {r['loading']:6.1f} g/L  P_dep={p_dep:.4f} flips={up}/{dn} "
              f"b={r['certified_bound']:.4f} [{lo:.4f},{hi:.4f}] {state:14s} "
              f"kept={'yes' if r['classification_preserved'] else 'NO':3s} "
              f"(union {pr['certified_bound']:.4f}, {pr['certified_bound']/r['certified_bound']:.1f}x looser) "
              f"| checked={matched and reproduced}")

    n_tests = 4 * len(rows)                   # two flip limits and two bridge limits per condition
    out = {"alpha_per_test": alpha, "n_tests": n_tests, "joint_level": 1 - n_tests * alpha,
           "tau_dec": TAU_HI, "tau_miss": TAU_LO,
           "n_preserved": sum(r["classification_preserved"] for r in rows),
           "n_preserved_union": sum(1 for r in rows if r["bound_deployed_union"] + 0 and
                                    classify(max(0.0, r["P_meet_deployed"] - r["bound_deployed_union"]))
                                    == r["state"] ==
                                    classify(min(1.0, r["P_meet_deployed"] + r["bound_deployed_union"]))),
           "all_checked": all(r["draws_match_conditional_artifact"] and r["flips_reproduced"]
                              for r in rows),
           "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\npreserved {out['n_preserved']}/{len(rows)} at the deployed law by the flip form, "
          f"{out['n_preserved_union']}/{len(rows)} by the union form")
    print(f"{n_tests} tests at alpha={alpha:.4e} -> joint level >= {out['joint_level']:.4f}; "
          f"provenance checked: {out['all_checked']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
