"""Is any pool candidate under more linearisation stress than the condition the certificate was earned at?

THE QUESTION.  The joint-region linearisation certificate of main text Section 5.1 is computed at each
product's HISTORICAL operating condition.  Every deployed read -- the shortfall action, the meet
probabilities, the compression certificates -- is taken across a 24-point pool that spans a four-dimensional
engineering box, including low-loading validation points and candidates at the high-loading adequacy cap.
Nothing forces a linearisation validated at one condition to hold across that box, and a referee is entitled
to ask why it should.

WHAT THIS IS, AND WHAT IT IS NOT.  This is a SCREEN, not a certificate.  The formal paired certificate costs
about 4e3 nonlinear solver draws per condition; at the pool's size that is days of compute, and it is left to
the pre-submission run.  What is cheap is a deterministic curvature probe.  At each candidate the linearised
map is exact at the fitted reference point by construction, so its error is second order and is largest along
the directions the posterior is widest in and the directions the decision is most sensitive to.  Probing
those directions at two posterior standard deviations gives, per candidate,

    d(a) = max over probes of || T^{-1}{ g(theta_p, a) - g_lin(theta_p, a) } ||_inf ,

a whitened worst-probe discrepancy.  The screen reports d(a) for every candidate BESIDE d(hist), the same
statistic at the condition where the formal certificate exists.  Candidates with d(a) <= d(hist) are ones the
inherited certificate covers no worse than the condition it was earned at; candidates above it are named, and
are what the pre-submission certificate run should target.

The probe directions are the leading eigenvectors of the posterior covariance -- the practically
nonidentified ridge, where a local approximation is most exposed -- together with the direction that moves
the decision most, the leading right singular vector of ``G Sigma^{1/2}``, which the ridge directions need not
contain.  Probes are clipped to the support the solver is defined on, the same guard the deployed runs use.

    python scripts/bayes_candidate_linearisation_screen.py --jac-tag _N48 \
        --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 45 --n-dir 3 --z 2.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import cex_model.app_support as A
from cex_model.bayes.active import _sim_for_op
from cex_model.bayes.compare import filter_experiments
from cex_model.bayes.decision import _pool_quantities, decision_forward
from cex_model.bayes.likelihood import unpack_u
from cex_model.bayes.posterior import Posterior
from cex_model.bayes.prior import solver_guard_bounds
from cex_model.diffsolver.torch_solver import DTYPE

PRODUCTS = ['HLXSYN']
TOL = np.array([0.02, 0.05])
_MAP = {"HLXSYN": ("HLXSYN", ["DT"])}


def _bundle(product):
    prod, drop = _MAP.get(product, (product, []))
    b = A.load_product(prod)
    if drop:
        b.experiments, _ = filter_experiments(b.experiments, drop)
    return b


def probe_directions(cov, G, n_dir):
    """Leading posterior directions plus the leading decision direction, each with its own scale."""
    ev, V = np.linalg.eigh(np.asarray(cov, float))
    order = np.argsort(ev)[::-1]
    dirs = [(V[:, k] * np.sqrt(max(ev[k], 0.0))) for k in order[:n_dir]]   # one posterior sd along each
    S = np.linalg.cholesky(np.asarray(cov, float) + 1e-12 * np.eye(len(cov)))
    _, _, Wt = np.linalg.svd(np.asarray(G, float) @ S, full_matrices=False)
    d = S @ Wt[0]                                                          # the decision-heaviest direction
    return dirs + [d]


def screen_one(bundle, post, op, G, g_map, n_dir, z, n_steps, guard):
    """Probe the nonlinear map on the SAME estimand the deployed Monte-Carlo pushforward uses.

    The window is selected once on the reference curve and held fixed, and the curve is then evaluated on the
    non-differentiable solver path with the numpy collection metrics -- exactly what
    ``decision_covariance_mc(reselect=False)`` does per draw.  Taking the differentiable objective instead
    would measure the same quantity through a much slower path.
    """
    _, sel, grp = decision_forward(bundle, list(map(float, op)), post.u_map, n_steps=n_steps)
    sim = _sim_for_op(bundle, list(map(float, op)), n_steps)
    n = bundle.components.n_protein
    lo, hi = guard
    u0 = np.asarray(post.u_map, float)

    def g_nl(u):
        with torch.no_grad():
            curve = sim.elution_curve(*unpack_u(torch.tensor(u, dtype=DTYPE), n),
                                      differentiable=False).numpy()
        return np.asarray(_pool_quantities(curve, sel.start_idx, sel.end_idx, grp["main"]), float)

    out = []
    for k, d in enumerate(probe_directions(post.cov, G, n_dir)):
        for s in (+1.0, -1.0):
            u = np.clip(u0 + s * z * d, lo, hi)
            x = g_nl(u)
            y = np.asarray(g_map, float) + np.asarray(G, float) @ (u - u0)
            out.append({"direction": ("posterior" if k < n_dir else "decision"), "k": int(k),
                        "sign": float(s), "clipped": bool(np.any(u != u0 + s * z * d)),
                        "d_inf": float(np.max(np.abs(x - y) / TOL)),
                        "x": x.tolist(), "y": y.tolist()})
    return out


def run(product, *, in_dir, jac_tag, n_dir, z, n_steps, indices=None):
    post = Posterior.load(Path(in_dir) / f"{product}_correlated_posterior.npz")
    zf = np.load(Path(in_dir) / f"inflation_jacobians_{product}{jac_tag}.npz")
    ops, Gs, gmaps = zf["ops"], zf["G"], zf["g_map"]
    bundle = _bundle(product)
    guard = solver_guard_bounds(bundle.components.n_protein)
    keep = set(range(len(ops))) if indices is None else set(indices) | {len(ops) - 1}
    rows = []
    for i, op in enumerate(ops):                      # the last row is the historical condition
        if i not in keep:
            continue
        probes = screen_one(bundle, post, op, Gs[i], gmaps[i], n_dir, z, n_steps, guard)
        d = max(p["d_inf"] for p in probes)
        rows.append({"index": i, "is_historical": bool(i == len(ops) - 1),
                     "loading": float(op[0]), "op": [float(v) for v in op],
                     "d_inf_worst": d, "d_inf_by_probe": [p["d_inf"] for p in probes],
                     "any_clipped": bool(any(p["clipped"] for p in probes))})
        print(f"  [{product}] {i + 1}/{len(ops)} loading={op[0]:6.2f}  d_inf={d:.4f}"
              f"{'   <- historical' if i == len(ops) - 1 else ''}", flush=True)
    hist = rows[-1]["d_inf_worst"]
    pool = rows[:-1]
    if not pool:
        raise SystemExit(f"{product}: nothing but the historical condition was screened")
    above = [r["index"] for r in pool if r["d_inf_worst"] > hist]
    return {"product": product, "n_candidates": len(pool), "z_sd": z, "n_dir": n_dir,
            "d_hist": hist, "d_pool_max": max(r["d_inf_worst"] for r in pool),
            "d_pool_median": float(np.median([r["d_inf_worst"] for r in pool])),
            "n_above_historical": len(above), "indices_above_historical": above, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="*", default=PRODUCTS)
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--jac-tag", default="")
    ap.add_argument("--n-dir", type=int, default=3)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--indices", type=int, nargs="*", default=None,
                    help="screen only these candidate indices (the historical condition is always kept)")
    ap.add_argument("--out", default="results/bayes/candidate_linearisation_screen.json")
    a = ap.parse_args()
    res = [run(p, in_dir=a.in_dir, jac_tag=a.jac_tag, n_dir=a.n_dir, z=a.z, n_steps=a.n_steps,
               indices=a.indices) for p in a.products]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\n{'product':14}{'d(hist)':>10}{'pool median':>13}{'pool max':>10}{'above hist':>12}")
    for r in res:
        print(f"{r['product']:14}{r['d_hist']:10.4f}{r['d_pool_median']:13.4f}{r['d_pool_max']:10.4f}"
              f"{r['n_above_historical']:>6}/{r['n_candidates']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
