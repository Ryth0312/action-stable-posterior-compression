"""Does a different fitted point change the ACTION, not just the width?

The multistart audit reports that the worst-direction decision width is stable across restarts. That
answers a question about the covariance. The fitted points themselves move by up to 3.85 prior-whitened
units, and on mAb A the decision mean at the historical condition already moves by 1.1 purity tolerances
across the six starts, so the question a reader actually has is whether the SELECTION moves with them.

This re-scores the deployed 24-candidate pool at each restart's own law: the shortfall minimizer, the
best in-domain specification probability, and the engineering branch the stratified rule returns. It
needs no new fit -- the restarts' posteriors are saved by bayes_correlated_refit.py --multistart -- but
it does need one decision Jacobian per (candidate, restart), about 25 s each.

Run:  python scripts/bayes_multistart_action_stability.py [--products HLXSYN HLXSYN HLXSYN]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bayes_pool_refinement import C_MEAS, DEPLOY, SPEC, _bundle, branch_of      # noqa: E402
from bayes_route_b_predictive_action import TOL                                # noqa: E402
from cex_model.bayes.decision import decision_covariance, decision_jacobian      # noqa: E402
from cex_model.bayes.decision_window import gaussian_meet_prob                   # noqa: E402
from cex_model.bayes.posterior import Posterior                                  # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results" / "bayes"


def shortfall(g, C, bias, sd_draws):
    """Expected tolerance-scaled purity--yield shortfall under the committed predictive mixture."""
    mu = (g + bias) / TOL
    var = np.diagonal(sd_draws + C + C_MEAS, axis1=1, axis2=2) / TOL ** 2
    sd = np.sqrt(np.clip(var, 1e-300, None))
    z = (SPEC / TOL - mu) / sd
    from scipy.stats import norm
    return float(((SPEC / TOL - mu) * norm.cdf(z) + sd * norm.pdf(z)).sum(axis=1).mean())


def score(product, post, ops, bias, sd_draws, n_steps):
    """(g, C, P(meet), risk) at every candidate under one restart's law."""
    gs, ps, rs = [], [], []
    for op in ops:
        G, g_map, _ = decision_jacobian(_BUNDLE[product], list(op), post.u_map,
                                        n_steps=n_steps, return_extra=True)
        g = np.asarray([g_map["pool_purity"], g_map["pool_yield"]], float) \
            if isinstance(g_map, dict) else np.asarray(g_map, float).ravel()[:2]
        C = np.asarray(decision_covariance(G, post.cov), float)
        gs.append(g)
        ps.append(float(np.mean([gaussian_meet_prob(g + bias[k], C + sd_draws[k] + C_MEAS, SPEC)
                                 for k in range(len(sd_draws))])))
        rs.append(shortfall(g, C, bias, sd_draws))
    return np.array(gs), np.array(ps), np.array(rs)


_BUNDLE = {}


def run(product, *, hier_draws, n_steps, n_restarts):
    _BUNDLE[product] = _bundle(product)
    # the deployed scan's pool, so the candidates are the ones the article reports on
    scan = json.loads((RES / f"{product}_decision_window_predictive_hier.json").read_text())
    ops = np.asarray([r["op"] for r in scan["rows"]], float)
    hd = np.load(hier_draws)
    bias, sd_draws = hd[f"b_{product}"], hd["Sd_draws"]
    lo, hi = DEPLOY[product]
    loading = ops[:, 0]

    rows, G_all = [], []
    for s in range(n_restarts):
        f = RES / "multistart" / f"{product}_r{s}_posterior.npz"
        if not f.exists():
            raise SystemExit(f"{f} absent; re-run bayes_correlated_refit.py --multistart to save it")
        gs, ps, rs = score(product, Posterior.load(str(f)), ops, bias, sd_draws, n_steps)
        G_all.append(gs)
        a_star = int(np.argmin(rs))
        br, best = branch_of(ps, loading, lo, hi)
        indom = (loading >= lo) & (loading <= hi)
        rows.append(dict(restart=s, a_star=a_star, loading_a_star=float(loading[a_star]),
                         branch=br, best_candidate=best, loading_best=float(loading[best]),
                         p_best=float(ps[best]),
                         p_max_deployment=float(ps[indom].max()) if indom.any() else None,
                         p_max_all=float(ps.max())))
        print(f"  {product:14} restart {s}: minimizer @{loading[a_star]:7.4f}  "
              f"branch={br:16} max-domain P={rows[-1]['p_max_deployment']:.4f}", flush=True)

    G_all = np.array(G_all)                                    # (S, A, 2)
    d = np.abs(G_all[:, None, :, :] - G_all[None, :, :, :]) / TOL
    return dict(product=product, n_restarts=n_restarts, n_candidates=len(ops),
                deployment_domain=[lo, hi], hier_draws=str(hier_draws), rows=rows,
                branches=sorted({r["branch"] for r in rows}),
                minimizers=sorted({r["loading_a_star"] for r in rows}),
                p_max_deployment_range=[min(r["p_max_deployment"] for r in rows),
                                        max(r["p_max_deployment"] for r in rows)],
                max_decision_mean_shift=float(np.linalg.norm(d, axis=-1).max()),
                max_purity_shift=float(d[..., 0].max()), max_yield_shift=float(d[..., 1].max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+", default=['HLXSYN'])
    ap.add_argument("--hier-draws", default=str(RES / "hier_draws_capB.npz"))
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--n-restarts", type=int, default=6)
    ap.add_argument("--out", default=str(RES / "multistart_action_stability.json"))
    a = ap.parse_args()

    out = [run(p, hier_draws=a.hier_draws, n_steps=a.n_steps, n_restarts=a.n_restarts)
           for p in a.products]
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("\n" + "=" * 92)
    for b in out:
        print(f"{b['product']:14} branches={b['branches']}  minimizers={b['minimizers']}  "
              f"max-domain P in [{b['p_max_deployment_range'][0]:.4f},{b['p_max_deployment_range'][1]:.4f}]")
        print(f"{'':14} largest decision-mean shift across restarts: "
              f"{b['max_decision_mean_shift']:.3f} tolerance units "
              f"(purity {b['max_purity_shift']:.3f}, yield {b['max_yield_shift']:.3f})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
