"""Finite-pool stability of the meet-probability gate (Colab / torch), the deployed rule's counterpart to R1.

Theorem 2.2 certifies the posterior-shortfall-risk minimiser. The rule actually deployed is a threshold on
P(meet), and picking the highest-P(meet) candidate is itself the Bayes action of the 0-1 failure loss
``ell = 1{Z not in M}``, since ``R_fail(a) = 1 - P(Z_a in M)``. The 0.95 is a chance constraint, not a loss, so
what has to be certified is that compression moves no candidate across it.

PRIMARY CERTIFICATE: directional flip rates. Full and compressed decisions can both be drawn, so the
disagreement of the two meet-indicators is observable and there is no need to bound it through a distance.
Writing

    q-_{a,j} = P(X_a in A, Y_{a,j} not in A),      q+_{a,j} = P(X_a not in A, Y_{a,j} in A),

we have exactly ``p_{a,j} = p_a - q- + q+``, so with simultaneous upper bounds on each rate

    p_a - qbar-  >= tau   =>   p_{a,j} >= tau        (a decisive candidate stays decisive)
    p_a + qbar+  <  tau   =>   p_{a,j} <  tau        (a non-decisive one stays non-decisive).

Keeping the directions apart is what makes this tight: a decisive candidate is only at risk from q- and a
non-decisive one only from q+, so neither class pays for the other's rate.

The couplings are the canonical ones, matched to the two identities of Lemma 2.1 rather than chosen for
convenience, which is what makes this the finite-sample realisation of R1's covariance residual rather than a
parallel empirical procedure. Both were verified numerically against C - C_S and C - C_D.

DIAGNOSTIC: the analytic route. The same joint-region coupling bound Section 4.1 uses for the linearisation,
|p_a - p_{a,j}| <= P{d_inf(Z, dA) <= t} + B_{a,j}/t with B from Theorem 2.2, minimised over t. It is computed
for comparison and is expected to be vacuous whenever the decision sits near the specification.

WHAT THE CONFIDENCE STATEMENT COVERS. The hierarchy draws are a committed finite set from a Gibbs run, not
independent samples from the hierarchy posterior. Each paired draw resamples one of them uniformly with
replacement and then draws parameters and noise independently, so the binomial bounds are exact for the
empirical mixture the deployed probability is actually computed on -- the same object Table 4 reports. They do
not cover sampling error in the hierarchy posterior itself, which Section 4.3 handles separately.

PREDICTIVE LAYER. The deployed law adds a shared product bias, hierarchical discrepancy and measurement noise.
The same draw is added to both arms, so it cancels from X - Y while still moving both draws relative to the
boundary; the flip rates are therefore those of the deployed probability. This assumes the hierarchy is not
re-estimated after compression, which is the question the paper asks.

WHICH DRAWS FILE. hier_draws_capB.npz, not hier_draws.npz. The committed deployed scans predate provenance
recording, so the file they integrated had to be recovered from them. For mAb A the scan uses method="gauss",
which makes C_par deterministic, and it stores g_map verbatim in row["mean_g"], so per candidate the only
unknown is the 3-parameter C_par: fit it to (p_meet, p5, p95) and then predict p_action_decisive and
worst_dec, which took no part in the fit. Under capB the inverse problem is consistent -- 22 of 24 candidates
fit to 2.5e-5 and 8 reproduce all five observables to printed precision -- while under hier_draws.npz it is
inconsistent, no admissible C_par fitting even the three probabilities on 23 of 24 candidates (mean residual
5.1e-3) and none reproducing all five. The three prior-sensitivity replicas are excluded by one to two orders.
This agrees with paper_SI.tex, which records the primary-to-capB shift in the shared discrepancy covariance
(purity 0.0219->0.0221, yield 0.0637->0.0658, matching the two files' sqrt(diag) exactly) and states that the
deployed full-law tables report the capped rho<=0.9 reads as the canonical model version.

Run (Colab, after `pip install -e ".[bayes]"`):
  OMP_NUM_THREADS=4 python scripts/step2d_meet_margin_certificate.py \
      --products HLXSYN HLXSYN HLXSYN --n-steps 300 \
      --posterior correlated_posterior --hier-draws results/bayes/hier_draws_capB.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cex_model.app_support as A
from cex_model.bayes.compare import filter_experiments
from cex_model.bayes.decision import decision_covariance_mc, decision_jacobian
from cex_model.bayes.decision_compression import (
    decision_cov,
    rotate_sigma_block,
    schur_cov,
    variant_d_cov,
    whiten,
)
from cex_model.bayes.decision_window import gaussian_meet_prob
from cex_model.bayes.design import candidate_pool_for
from cex_model.bayes.posterior import Posterior

_PRODUCT_MAP = {"HLXSYN": ("HLXSYN", ["DT"])}
TAU = 0.95
T_GRID = np.concatenate([np.linspace(0.02, 2.0, 100), np.linspace(2.1, 12.0, 100)])


def _load(product, in_dir, posterior):
    post = Posterior.load(Path(in_dir) / f"{product}_{posterior}.npz")
    dj = json.loads((Path(in_dir) / f"{product}_decision.json").read_text())
    tol = np.asarray(dj.get("tol", [0.02, 0.05]), float)
    prod, drop = _PRODUCT_MAP.get(product, (product, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
    return post, bundle, tol


def _tube(mean_w, cov_w, spec_w, rng, n):
    """Sample d_inf(Z, dM) once, so every t on the grid is scored on the same draws."""
    Z = rng.multivariate_normal(mean_w, cov_w, size=n)
    marg = Z - spec_w
    inside = np.all(marg >= 0, axis=1)
    return np.where(inside, marg.min(axis=1), np.clip(-marg, 0.0, None).max(axis=1))


def _eps(d_full, d_comp, B):
    """inf_t [ min(tube_full(t), tube_comp(t)) + B/t ], both tubes on shared draws."""
    best, best_t = np.inf, np.nan
    for t in T_GRID:
        tube = min(float(np.mean(d_full <= t)), float(np.mean(d_comp <= t)))
        v = tube + B / t
        if v < best:
            best, best_t = v, float(t)
    return min(best, 1.0), best_t


def _paired_flips(Sr, Gr, ui, vi, g_map, spec, rng, n, alpha, bias=None, sd=None):
    """Canonical couplings, matched one-to-one to the two identities of Lemma 2.1, and the two flip rates.

    Schur:  theta = (u, E[v|u] + r) and theta_S = (u, E[v|u]) share u and differ only by the conditional
            residual, so X - Y = G_v r with Var = G_v Sigma_{v|u} G_v^T = C - C_S.
    Var-D:  writing x_u = L x_v + e with L = Sigma_uv Sigma_vv^{-1} and e ~ N(0, Sigma_{u|v}), the paired draw
            is theta_D = (mu_u + e, mu_v), so X - Y = Gamma_v x_v with Var = Gamma_v Sigma_vv Gamma_v^T = C - C_D.

    Both were checked numerically against the two residuals before being used here. Using an arbitrary coupling
    would still give a valid bound, but not one that is the finite-sample realisation of R1's covariance residual.

    The two directions are kept apart. A decisive candidate can only be lost through
    q- = P(X in A, Y not in A) and a non-decisive one can only be gained through q+ = P(X not in A, Y in A),
    with p_{a,j} - p_a = q+ - q-, so certifying each class needs only its own one-sided rate.
    """
    from scipy.stats import beta as _beta
    pinv = np.linalg.pinv
    Suu = Sr[np.ix_(ui, ui)]; Svv = Sr[np.ix_(vi, vi)]; Svu = Sr[np.ix_(vi, ui)]; Suv = Svu.T
    Gu, Gv = Gr[:, ui], Gr[:, vi]
    gm = np.asarray(g_map, float)
    zu, zv = np.zeros(len(ui)), np.zeros(len(vi))

    K = Svu @ pinv(Suu)
    xu = rng.multivariate_normal(zu, Suu, size=n)
    r = rng.multivariate_normal(zv, Svv - K @ Suv, size=n)
    shared = gm + xu @ (Gu + Gv @ K).T
    XY = {"S": (shared + r @ Gv.T, shared)}

    L = Suv @ pinv(Svv)
    xv = rng.multivariate_normal(zv, Svv, size=n)
    e = rng.multivariate_normal(zu, Suu - L @ Svu, size=n)
    base = gm + e @ Gu.T
    XY["D"] = (base + xv @ (Gv + Gu @ L).T, base)

    out = {}
    for name, (X, Y) in XY.items():
        if bias is not None:              # the shared predictive layer cancels in X - Y but moves both draws
            k = rng.integers(0, len(bias), size=n)
            xi = np.empty((n, X.shape[1]))
            for j in np.unique(k):
                m = k == j
                xi[m] = rng.multivariate_normal(bias[j], sd[j], size=int(m.sum()))
            X, Y = X + xi, Y + xi
        mx = np.all(X >= spec, axis=1); my = np.all(Y >= spec, axis=1)
        qm = int(np.sum(mx & ~my))        # decisive under the full law, lost under compression
        qp = int(np.sum(~mx & my))        # gained under compression
        cp = lambda d: 1.0 if d >= n else float(_beta.ppf(1.0 - alpha, d + 1, n - d))
        out[name] = {"q_minus": qm / n, "q_plus": qp / n,
                     "q_minus_upper": cp(qm), "q_plus_upper": cp(qp),
                     "eps_cp": cp(qm + qp),   # the two-sided rate, for comparison with the analytic route
                     "p_full_mc": float(mx.mean()), "p_comp_mc": float(my.mean())}
    return out


def run(product, *, in_dir, n_candidates, n_steps, n_samples, spec, seed, posterior, n_mc, n_paired,
        delta, sigma_meas, n_tube_draws, hier_draws):
    post, bundle, tol = _load(product, in_dir, posterior)
    dj = json.loads((Path(in_dir) / f"{product}_decision.json").read_text())
    _cc = dj.get("crosscheck")
    relf = float(_cc["rel_frobenius"]) if _cc and "rel_frobenius" in _cc else None
    method = "mc" if (relf or 0.0) > 0.5 else "gauss"
    c_meas = np.diag(np.asarray(sigma_meas, float) ** 2)
    print(f"[{product}] decision covariance: {method} (rel-Frob {'n/a' if relf is None else f'{relf:.2f}'})")
    ops = [list(o) for o in candidate_pool_for(bundle, n_candidates=n_candidates, seed=seed)]
    spec = np.asarray(spec, float)
    spec_w = spec / tol
    rng = np.random.default_rng(seed)
    n = post.n_protein
    # familywise over candidates x reductions x laws, so the whole pool is certified simultaneously
    alpha = delta / (n_candidates * 2 * 2 * (2 if hier_draws else 1))

    draws = None
    if hier_draws:
        # The predictive certificate is only comparable to the deployed scan if both integrate the same
        # committed draws. Nothing checked that before, and the scan's own writer records having once been
        # run on the low-discrepancy draws for two products and the primary ones for a third.
        dep = Path(in_dir) / f"{product}_decision_window_predictive_hier.json"
        want = ((json.loads(dep.read_text()).get("provenance") or {}).get("hier_draws")
                if dep.exists() else None)
        if want is None:
            print(f"[{product}] WARNING: the deployed scan carries no provenance block, so --hier-draws "
                  f"{hier_draws} cannot be cross-checked against the scan this certificate must match")
        elif str(want) != str(hier_draws):
            raise SystemExit(f"[{product}] --hier-draws {hier_draws} != deployed scan's {want}")
        z = np.load(hier_draws, allow_pickle=True)
        draws = {"b": z[f"b_{product}"], "Sd": z["Sd_draws"]}
        print(f"[{product}] predictive layer: {len(draws['Sd'])} hierarchy draws")

    rows = []
    for idx, op in enumerate(ops):
        G, g_map, _ = decision_jacobian(bundle, op, post.u_map, n_steps=n_steps, return_extra=True)
        G = np.atleast_2d(G)
        Sr, Gr, ui, vi = rotate_sigma_block(post.cov, G, n)
        C = {"full": decision_cov(Gr, Sr),
             "S": decision_cov(Gr, schur_cov(Sr, ui, vi)),
             "D": variant_d_cov(Sr, Gr, ui, vi)}
        # Compression is defined in parameter space and reaches the decision through G, so the certificate is
        # computed with BOTH laws linearised: swapping only C_full for the deployed Monte-Carlo pushforward
        # would make eps measure linearisation error on top of compression. Where the deployed scan does use
        # that pushforward (mAb C, screen fired), the deployed probability is reported alongside for
        # comparison, and the gap between the two laws is what R2 -- not this certificate -- has to carry.
        C_dep = None
        if method == "mc":
            # Same construction the deployed scan uses: the sample covariance of the raw per-draw decisions,
            # not the routine's own C_mc, so the comparison probability is the one the paper reports.
            mc = decision_covariance_mc(post, bundle, op, n_samples=n_samples, seed=seed,
                                        n_steps=n_steps, return_gs=True)
            gs = np.asarray(mc.get("gs", []), float)
            C_dep = np.cov(gs.T) if gs.ndim == 2 and gs.shape[0] > 2 else C["full"]
        rec = {"idx": idx, "op": [float(x) for x in op], "loading": float(op[0])}

        for law in (("conditional",) + (("predictive",) if draws is not None else ())):
            # p_a and the tube are law-specific; B is not -- the shared predictive layer cancels in the coupling.
            if law == "conditional":
                mu = {k: np.asarray(g_map, float) for k in C}
                cov = {k: C[k] for k in C}
                p = {k: gaussian_meet_prob(g_map, C[k], spec) for k in C}
                dset = {k: _tube(mu[k] / tol, whiten(cov[k], tol), spec_w, rng, n_mc) for k in C}
            else:
                p, dset = {}, {}
                sub = np.linspace(0, len(draws["Sd"]) - 1, min(n_tube_draws, len(draws["Sd"]))).astype(int)
                for k in C:
                    # The reported probability is the mixture mean over ALL committed draws, which is the
                    # object the deployed scan reports and the object the paired flip rates resample. Taking
                    # it on the tube's subsample instead applies the identity p_j = p - q- + q+ across two
                    # different empirical mixtures, and the resulting error in p is of the same size as the
                    # flip rates it is netted against. Only the tube is subsampled; it needs MC draws.
                    p[k] = float(np.mean([
                        gaussian_meet_prob(np.asarray(g_map, float) + draws["b"][j],
                                           C[k] + draws["Sd"][j] + c_meas, spec)
                        for j in range(len(draws["Sd"]))]))
                    ds = []
                    for j in sub:
                        m = np.asarray(g_map, float) + draws["b"][j]
                        cv = C[k] + draws["Sd"][j] + c_meas
                        ds.append(_tube(m / tol, whiten(cv, tol), spec_w, rng,
                                        max(64, n_mc // len(sub))))
                    dset[k] = np.concatenate(ds)          # mixture tube = pooled draws
            pe = _paired_flips(Sr, Gr, ui, vi, g_map, spec, rng, n_paired, alpha,
                               bias=(draws["b"] if law == "predictive" else None),
                               sd=([s + c_meas for s in draws["Sd"]] if law == "predictive" else None))
            for j in ("S", "D"):
                Dj = float(np.linalg.norm(whiten(C["full"] - C[j], tol), "fro"))
                cj = float(np.linalg.eigvalsh(whiten(C[j], tol)).min())
                B = Dj / (2.0 * np.sqrt(max(cj, 1e-12)))
                e, t_star = _eps(dset["full"], dset[j], B)
                f = pe[j]
                qm, qp = f["q_minus_upper"], f["q_plus_upper"]
                # |p_j - p| = |q+ - q-| <= max(q+, q-), so the two-sided figure needs only the two limits
                # already allocated. eps_cp = cp(K- + K+) is never smaller, since cp is increasing in the
                # count, and eps_analytic is a raw grid minimum with no confidence correction, so neither may
                # be allowed to win; both stay in the record as diagnostics.
                eb = max(qm, qp)
                rec[f"{law}_{j}"] = {
                    "p_full": p["full"], "p_comp": p[j], "obs_dP": abs(p["full"] - p[j]),
                    "B": B, "eps_analytic": e, "t_star": t_star,
                    "eps_paired": f["eps_cp"], "eps": eb,
                    "q_minus": f["q_minus"], "q_plus": f["q_plus"],
                    "q_minus_upper": qm, "q_plus_upper": qp,
                    # one-sided, which is what each class actually needs
                    "certified_decisive": bool(p["full"] - qm >= TAU),
                    "certified_not_decisive": bool(p["full"] + qp < TAU),
                    "lo": max(0.0, p["full"] - qm), "hi": min(1.0, p["full"] + qp),
                }
            rec[f"{law}_p_full"] = p["full"]
            if C_dep is not None:                       # deployed law, for comparison only
                if law == "conditional":
                    rec["conditional_p_deployed"] = gaussian_meet_prob(g_map, C_dep, spec)
                else:
                    rec["predictive_p_deployed"] = float(np.mean([
                        gaussian_meet_prob(np.asarray(g_map, float) + draws["b"][j],
                                           C_dep + draws["Sd"][j] + c_meas, spec)
                        for j in range(len(draws["Sd"]))]))
        rows.append(rec)
        print(f"  [{idx:2d}] load={op[0]:6.2f} " + " ".join(
            f"{law[:4]}/{j}: p={rec[f'{law}_{j}']['p_full']:.3f} eps={rec[f'{law}_{j}']['eps']:.3f}"
            for law in (("conditional",) + (("predictive",) if draws is not None else ()))
            for j in ("S", "D")))

    out = {"product": product, "tau": TAU, "spec": spec.tolist(), "tol": tol.tolist(),
           "method": method, "rel_frobenius": relf, "sigma_meas": list(map(float, sigma_meas)),
           "posterior": posterior, "n_candidates": len(ops), "n_mc": n_mc,
           "n_paired": n_paired, "delta": delta, "alpha_per_test": alpha,
           "hier_draws": str(hier_draws) if hier_draws else None,
           "n_tube_draws": (int(min(n_tube_draws, len(draws["Sd"]))) if draws is not None else None),
           "rows": rows}
    for law in (("conditional",) + (("predictive",) if draws is not None else ())):
        for j in ("S", "D"):
            cells = [r[f"{law}_{j}"] for r in rows]
            straddle = [c for c in cells if not (c["certified_decisive"] or c["certified_not_decisive"])]
            p = np.array([c["p_full"] for c in cells]); e = np.array([c["eps"] for c in cells])
            # the selection and the no-decisive claim are one-sided in opposite directions, so each uses its
            # own rate rather than the two-sided eps: the leader can only be lost through q-, a rival can only
            # be gained through q+.
            qmv = np.array([c["q_minus_upper"] for c in cells])
            qpv = np.array([c["q_plus_upper"] for c in cells])
            a = int(np.argmax(p))
            out[f"summary_{law}_{j}"] = {
                "max_eps": float(e.max()),
                "max_eps_analytic": float(max(c["eps_analytic"] for c in cells)),
                "max_eps_paired": float(max(c["eps_paired"] for c in cells)),
                "max_q_minus_upper": float(qmv.max()),
                "max_q_plus_upper": float(qpv.max()),
                "max_obs_dP": float(max(c["obs_dP"] for c in cells)),
                "n_straddling_tau": len(straddle),
                "decisive_set_certified": len(straddle) == 0,
                "argmax_certified": bool(p[a] - qmv[a] > np.max(np.delete(p + qpv, a))),
                "argmax_gap": float(p[a] - np.max(np.delete(p, a))),
                "argmax_margin": float(p[a] - qmv[a] - np.max(np.delete(p + qpv, a))),
                "no_decisive_candidate_certified": bool(all(c["certified_not_decisive"] for c in cells)),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=["HLXSYN", "HLXSYN", "HLXSYN"])
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--n-candidates", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--spec", type=float, nargs=2, default=[0.70, 0.50])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--posterior", default="correlated_posterior",
                    help="correlated_posterior (deployed) or posterior (independent-residual)")
    ap.add_argument("--n-mc", type=int, default=200_000)
    ap.add_argument("--n-paired", type=int, default=400_000)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--n-samples", type=int, default=200)   # matches bayes_decision_window.py's default
    ap.add_argument("--sigma-meas", type=float, nargs=2, default=[0.005, 0.008])
    ap.add_argument("--n-tube-draws", type=int, default=200,
                    help="hierarchy draws used for the analytic tube only; the paired eps and the "
                         "reported probabilities always use all of them")
    ap.add_argument("--hier-draws", default=None,
                    help="e.g. results/bayes/hier_draws.npz; omit for the conditional law only")
    ap.add_argument("--out", default="results/bayes/meet_margin_certificate.json")
    a = ap.parse_args()

    res = [run(p, in_dir=a.in_dir, n_candidates=a.n_candidates, n_steps=a.n_steps,
               n_samples=a.n_samples, spec=a.spec, sigma_meas=a.sigma_meas,
               seed=a.seed, posterior=a.posterior, n_mc=a.n_mc, n_paired=a.n_paired,
               delta=a.delta, n_tube_draws=a.n_tube_draws, hier_draws=a.hier_draws)
           for p in a.products]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}\n")
    for r in res:
        print(f"=== {r['product']} ===")
        for k, v in r.items():
            if k.startswith("summary_"):
                print(f"  {k[8:]:16} q-<={v['max_q_minus_upper']:.5f} q+<={v['max_q_plus_upper']:.5f} "
                      f"(analytic eps {v['max_eps_analytic']:.3f})  straddling={v['n_straddling_tau']:2d}  "
                      f"set={str(v['decisive_set_certified']):5} none-decisive={v['no_decisive_candidate_certified']}")


if __name__ == "__main__":
    main()
