"""Alignment check (Colab / torch): shortfall-risk argmin vs P(meet) argmax on the REAL candidate pool.

Cor R1.3 scopes R1 to the shortfall-risk Bayes action. This script tests the *empirical bridge* to the
paper's deployed P(meet)-driven action, under the SAME local-Gaussian mechanistic posterior (NOT the OU /
hierarchical / empirical-convolution predictive law), on the committed Sobol candidate pool, for the full,
Schur-compressed, and variant-D posteriors.

Honest reporting (per the plan): a product is classified as
  * unique-action  : P(meet) has a well-separated argmax -> report whether argmin R = argmax P(meet)
                     and whether that action is the SAME under full / Schur / variant-D;
  * level-failure  : max_a P(meet) < p_lo -> report whether the shortfall analysis also finds no decisive
                     action (feasibility alignment, NOT candidate-index identity);
  * near-tie       : top-two P(meet) gap small -> report the gap and the top-set overlap.
It does NOT force candidate-index identity, and it does NOT claim R1 preserves the deployed P(meet) action.

POSTERIOR. ``--posterior`` selects the base law. ``correlated_posterior`` is the fit every deployment read in
the application uses and is what main-text Table 3 should be computed on; ``posterior`` is the
independent-residual Laplace fit, kept as a residual-model sensitivity. The two differ by 5-22% in relative
Frobenius norm on the decision covariance, so the choice moves the numbers. Output paths carry a corr/iid tag
so the two runs do not overwrite each other.

Also emitted: ``w_par`` and ``w_v``, the prior-whitened widths of the full parameter vector and of the
compressed block, so every quantity in Table 3 comes from one script and one posterior.

Run (Colab, after `pip install -e ".[bayes]"`):
  OMP_NUM_THREADS=4 python scripts/step2_pmeet_alignment.py --products HLXSYN HLXSYN HLXSYN \
      --n-steps 300 --posterior correlated_posterior
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cex_model.app_support as A
from cex_model.bayes.compare import filter_experiments
from cex_model.bayes.decision import decision_jacobian
from cex_model.bayes.decision_compression import (
    bures_w2,
    decision_cov,
    rotate_sigma_block,
    schur_cov,
    shortfall_risk,
    variant_d_cov,
    whiten,
)
from cex_model.bayes.decision_window import gaussian_meet_prob
from cex_model.bayes.design import candidate_pool_for
from cex_model.bayes.posterior import Posterior

_PRODUCT_MAP = {"HLXSYN": ("HLXSYN", ["DT"])}
P_HI = 0.90          # decisive-meet threshold on P(meet)
P_LO = 0.10          # infeasible-everywhere threshold on max P(meet)
GAP_TIE = 0.05       # near-tie threshold on the top-two P(meet) gap
DACT_FLOOR = 1e-3    # below this the shortfall action gap is degenerate (operate-anywhere) -> use P(meet)-preservation


def _load(product, in_dir, posterior="posterior"):
    post = Posterior.load(Path(in_dir) / f"{product}_{posterior}.npz")
    dj = json.loads((Path(in_dir) / f"{product}_decision.json").read_text())
    tol = np.asarray(dj.get("tol", [0.02, 0.05]), float)
    prod, drop = _PRODUCT_MAP.get(product, (product, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
    return post, bundle, tol


_PRIOR_STD = (1.5, 2.0, 4.25, 24.75)   # keq(log10), kkin(log10), nu, sigma: (hi-lo)/4 from prior.DEFAULT_BOUNDS


def _prior_whitened_widths(Sigma, n):
    """(w_par, w_v): prior-whitened widths of the whole parameter vector and of the compressed block v.

    The sigma-block rotation depends only on the common-mode direction, not on G, so Sigma_rot and v_idx are
    the same for every candidate and these are computed once per product. v is the (n-1)-dimensional
    sigma-differential space; the data-informed common mode stays in u.
    """
    Sigma = np.asarray(Sigma, float)
    Sr, _, _, vi = rotate_sigma_block(Sigma, np.zeros((1, Sigma.shape[0])), n)
    D = np.diag(1.0 / np.concatenate([[s] * n for s in _PRIOR_STD]))
    W = D @ Sr @ D
    W = 0.5 * (W + W.T)
    Wvv = W[np.ix_(vi, vi)]
    return (float(np.sqrt(np.linalg.eigvalsh(W).max())),
            float(np.sqrt(np.linalg.eigvalsh(0.5 * (Wvv + Wvv.T)).max())),
            int(len(vi)))


def _decision_laws(post, bundle, ops, *, n_steps):
    """Per-candidate g_map and the full/Schur/variant-D decision covariances."""
    n = post.n_protein
    Sigma = post.cov
    out = []
    for op in ops:
        G, g_map, _ = decision_jacobian(bundle, list(op), post.u_map, n_steps=n_steps, return_extra=True)
        G = np.atleast_2d(G)
        Sr, Gr, ui, vi = rotate_sigma_block(Sigma, G, n)
        C_full = decision_cov(Gr, Sr)
        C_S = decision_cov(Gr, schur_cov(Sr, ui, vi))
        C_D = variant_d_cov(Sr, Gr, ui, vi)
        out.append({"op": list(map(float, op)), "g_map": g_map, "C": {"full": C_full, "S": C_S, "D": C_D}})
    return out


def _lambda_stability(A, beta, lam0, lam_hi=1e4):
    """Exact interval of the RELATIVE weight lambda = w_1/w_2 around ``lam0`` on which (i) the
    full-posterior Bayes action keeps its index and (ii) the one-sided certificate eta_j < Delta_act fires.

    The loss is separable, so with w = w_2 (lambda, 1) the full-posterior risk of candidate a is the affine
    function R(a; lambda) = lambda A[a,0] + A[a,1], and eta_j = max_a bound_{a,j} = w_2 sqrt(1+lambda^2)
    beta_j with beta_j = max_a W2(Cbar_a, Cbar_{a,j}) free of w -- the transport bound and the matrix-constant
    one share this factorisation, since both are ||w||_2 times a w-free covariance functional.  The common factor
    w_2 cancels in eta_j / Delta_act, so only lambda matters.  Between consecutive pairwise crossings of the
    affine risks the whole candidate ordering is fixed, so Delta_act = alpha + b*lambda there and the
    certificate boundary alpha + b*lambda = beta_j sqrt(1+lambda^2) is a quadratic.  Pure arithmetic on A
    and beta: no solver, no posterior work.
    """
    A = np.asarray(A, float)
    pts = [0.0, float(lam_hi)]
    for i in range(A.shape[0]):                       # ordering breakpoints
        for m in range(i + 1, A.shape[0]):
            dp = A[i, 0] - A[m, 0]
            if dp != 0.0:
                x = -(A[i, 1] - A[m, 1]) / dp
                if 0.0 < x < lam_hi:
                    pts.append(float(x))
    cells = np.unique(np.asarray(pts, float))
    roots = []
    for lo, hi in zip(cells[:-1], cells[1:]):         # certificate breakpoints inside each cell
        o = np.argsort(A @ np.array([0.5 * (lo + hi), 1.0]))
        b = A[o[1], 0] - A[o[0], 0]
        al = A[o[1], 1] - A[o[0], 1]
        for bj in beta.values():
            qa, qb, qc = b * b - bj * bj, 2.0 * al * b, al * al - bj * bj
            if qa == 0.0:
                cand = [-qc / qb] if qb != 0.0 else []
            else:
                disc = qb * qb - 4.0 * qa * qc
                cand = [(-qb + s * np.sqrt(disc)) / (2.0 * qa) for s in (1.0, -1.0)] if disc >= 0 else []
            roots += [float(x) for x in cand if lo < x < hi and al + b * x >= -1e-12]
    bps = np.unique(np.concatenate([cells, np.asarray(roots or [0.0], float)]))
    mid = 0.5 * (bps[:-1] + bps[1:])
    Rm = A @ np.vstack([mid, np.ones_like(mid)])
    srt = np.sort(Rm, axis=0)
    Dact = srt[1] - srt[0]
    a0 = int(np.argmin(A @ np.array([lam0, 1.0])))
    dact0 = float(np.diff(np.sort(A @ np.array([lam0, 1.0])))[0])
    o = np.argsort(A @ np.array([lam_hi, 1.0]))       # terminal-cell purity slope: the lambda->inf limit
    b_inf = A[o[1], 0] - A[o[0], 0]                   # eta_j/Delta_act -> beta_j / b_inf

    def runs(mask, unbounded_ok):
        """Every maximal lambda-interval on which ``mask`` holds, plus the one containing lam0.

        ``at_lambda0 is None`` means the condition fails AT lam0 -- NOT that it fails everywhere; read
        ``all`` for the intervals where it does hold (mAb B's variant D is exactly that case).
        """
        iv, i = [], 0
        while i < mask.size:
            if mask[i]:
                j = i
                while j + 1 < mask.size and mask[j + 1]:
                    j += 1
                hi = float(bps[j + 1])
                iv.append([float(bps[i]), float("inf") if (hi >= lam_hi and unbounded_ok) else hi])
                i = j + 1
            else:
                i += 1
        c = min(max(int(np.searchsorted(bps, lam0, side="right")) - 1, 0), mask.size - 1)
        at0 = None
        if mask[c]:
            for r in iv:
                if r[0] <= lam0 <= (lam_hi if r[1] == float("inf") else r[1]):
                    at0 = r
                    break
        return {"at_lambda0": at0, "all": iv, "lambda_max_scanned": float(lam_hi)}

    out = {"lambda0": float(lam0), "action_index": a0, "Delta_act_at_lambda0": dact0,
           "action_gap_meaningful": bool(dact0 > DACT_FLOOR),
           "beta": {j: float(b) for j, b in beta.items()},
           # a degenerate gap makes argmin a floating-point tie-break, so the interval would mean nothing
           "action_interval": runs(np.argmin(Rm, axis=0) == a0, True) if dact0 > DACT_FLOOR else None}
    for j, bj in beta.items():
        out[f"cert_interval_{j}"] = runs(np.sqrt(1.0 + mid ** 2) * bj < Dact, bool(bj < b_inf))
    return out


def _analyze(laws, tol, spec, w):
    """argmin shortfall & argmax P(meet) under each posterior variant + classification."""
    k = np.asarray(spec).size
    eye = np.eye(k)
    res = {}
    for j in ("full", "S", "D"):
        # A[a, q] = E(sbar_q - Z_{a,q})_+ , the unit-weight coordinate shortfall: the loss is separable, so
        # risks = A @ w exactly and every w-dependence of the analysis passes through this one product.
        A = np.array([[shortfall_risk(np.asarray(d["g_map"]) / tol, whiten(d["C"][j], tol),
                                      np.asarray(spec) / tol, eye[q]) for q in range(k)] for d in laws])
        risks = A @ np.asarray(w, float)
        pmeet = np.array([gaussian_meet_prob(d["g_map"], d["C"][j], spec) for d in laws])
        res[j] = {"argmin_R": int(np.argmin(risks)), "argmax_P": int(np.argmax(pmeet)),
                  "maxP": float(pmeet.max()),
                  "top2gap": float(np.diff(np.sort(pmeet)[-2:])[0]) if pmeet.size >= 2 else float("nan"),
                  "A": A, "risks": risks, "pmeet": pmeet}
    # theorem-facing quantities on the REAL pool. Theorem R1 gives the generic two-sided 2*eta_j < Delta_act;
    # Corollary R1b (convex order) halves it to eta_j < Delta_act and adds the envelope C_j, which needs no gap.
    L = float(np.linalg.norm(w))
    rf = res["full"]["risks"]
    order = np.argsort(rf)
    Delta_act = float(rf[order[1]] - rf[order[0]]) if rf.size >= 2 else float("inf")
    r_full = rf - rf[order[0]]                       # full-posterior regret per candidate
    argmin_full = set(np.flatnonzero(r_full <= 0.0).tolist())
    tf = {}
    for j in ("S", "D"):
        Dj = np.array([np.linalg.norm(whiten(d["C"]["full"] - d["C"][j], tol), "fro") for d in laws])
        cj = np.array([np.linalg.eigvalsh(whiten(d["C"][j], tol)).min() for d in laws])
        # eq:riskbound is |R_pi - R_pi_j| <= L W2(P_a, P_a,j) <= L Delta_{a,j} / (2 sqrt(c_{a,j})).  The
        # transport term is the tighter one and, the two laws being equal-mean Gaussians, W2 IS the Bures
        # metric, available in closed form.  Report it: the matrix step that follows only exists to make the
        # bound expressible in the residual, and paying for it costs a factor of 1.5-5 here.  Nothing
        # downstream uses anything about b_{a,j} except that it bounds |R_pi(a) - R_pi_j(a)|, so Theorem R1,
        # its one-sided refinement and the envelope all carry over verbatim, and the bound stays valid for any
        # L-Lipschitz loss rather than only for this one.
        bnd = L * np.array([bures_w2(whiten(d["C"]["full"], tol), whiten(d["C"][j], tol)) for d in laws])
        bnd_frob = L / (2.0 * np.sqrt(np.clip(cj, 1e-12, None))) * Dj
        signed = rf - res[j]["risks"]                # >= 0 by the convex order; checked, not assumed
        eta_obs = float(np.max(np.abs(signed)))
        env = np.flatnonzero(r_full <= bnd)          # C_j = {a : r_pi(a) <= b_{a,j}}
        tf[j] = {"max_Delta": float(Dj.max()), "eta_bound": float(bnd.max()), "eta_obs": eta_obs,
                 # the matrix-constant route, kept so the cost of that step stays visible
                 "eta_bound_frobenius": float(bnd_frob.max()),
                 "tightening_vs_frobenius": float(bnd_frob.max() / bnd.max()) if bnd.max() > 0 else float("nan"),
                 "ratio_bound": float(2 * bnd.max() / Delta_act) if Delta_act > 0 else float("inf"),
                 "ratio_obs": float(2 * eta_obs / Delta_act) if Delta_act > 0 else float("inf"),
                 # Corollary R1b
                 "one_sided_min": float(signed.min()), "one_sided_holds": bool(signed.min() >= -1e-9),
                 "ratio_bound_onesided": float(bnd.max() / Delta_act) if Delta_act > 0 else float("inf"),
                 "envelope_size": int(env.size),
                 "envelope_max_regret": float(r_full[env].max()) if env.size else float("nan"),
                 "envelope_in_argmin": bool(set(env.tolist()) <= argmin_full),
                 "b_at_compressed_argmin": float(bnd[int(np.argmin(res[j]["risks"]))]),
                 "regret_at_compressed_argmin": float(r_full[int(np.argmin(res[j]["risks"]))]),
                 # bound_{a,j} / L : the w-free part of the certificate, used by the lambda sweep
                 "beta": float((bnd / L).max())}
    tf["Delta_act"] = Delta_act
    if k == 2 and w[1] > 0:
        tf["lambda_stability"] = _lambda_stability(res["full"]["A"], {j: tf[j]["beta"] for j in ("S", "D")},
                                                   float(w[0]) / float(w[1]))
    tf["pool_max_regret"] = float(r_full.max())      # so a wide envelope can be told from a vacuous one
    action_gap_meaningful = Delta_act > DACT_FLOOR   # else the shortfall action is degenerate (operate-anywhere)
    tf["action_gap_meaningful"] = bool(action_gap_meaningful)
    tf["action_gap_certified"] = bool(action_gap_meaningful
                                      and tf["S"]["ratio_bound_onesided"] < 1
                                      and tf["D"]["ratio_bound_onesided"] < 1)
    # the envelope route certifies exact preservation without any gap condition
    tf["action_certified_by_envelope"] = bool(tf["S"]["envelope_in_argmin"] and tf["D"]["envelope_in_argmin"])

    # P(meet)-preservation: the correct certificate when the decision is DECISIVE at every candidate
    # (operate-anywhere), where the shortfall action gap Δ_act≈0 makes 2η<Δ_act inapplicable.
    Pfull = res["full"]["pmeet"]
    decisive_full = Pfull >= P_HI
    pmeet_pres = {}
    for j in ("S", "D"):
        Pj = res[j]["pmeet"]
        dP = float(np.max(np.abs(Pfull - Pj)))
        min_dec = float(np.min(Pj[decisive_full])) if decisive_full.any() else float("nan")
        pmeet_pres[j] = {"max_abs_dP": dP, "min_Pj_over_full_decisive": min_dec,
                         "decisiveness_preserved": bool(decisive_full.any() and min_dec >= P_HI and dP < 0.05)}
    tf["pmeet_preservation"] = pmeet_pres

    maxP = res["full"]["maxP"]
    frac_decisive = float(np.mean(decisive_full))
    if maxP < P_LO:
        regime = "infeasible-everywhere"          # redesign-pool: shortfall AND P(meet) find no feasible action
    elif not action_gap_meaningful and frac_decisive >= 0.5:
        regime = "operate-anywhere"               # decisive everywhere; shortfall action degenerate (Δ_act≈0)
    elif res["full"]["top2gap"] < GAP_TIE:
        regime = "near-tie"
    else:
        regime = "separation"                     # a genuine argmax; the action-gap certificate applies
    aligned_idx = all(res[j]["argmin_R"] == res[j]["argmax_P"] for j in ("full", "S", "D"))
    same_action = len({res[j]["argmax_P"] for j in ("full", "S", "D")}) == 1
    # which certificate is the operative one for this regime:
    if regime == "operate-anywhere":
        cert_ok = bool(pmeet_pres["S"]["decisiveness_preserved"] and pmeet_pres["D"]["decisiveness_preserved"])
        cert_kind = "P(meet)-decisiveness-preserved"
    elif regime in ("infeasible-everywhere", "separation"):
        cert_ok = tf["action_gap_certified"]
        cert_kind = "action-gap (2η<Δ_act)"
    else:
        cert_ok, cert_kind = False, "near-tie (report top-set overlap)"
    return {"regime": regime, "maxP": maxP, "frac_decisive": frac_decisive, "w": np.asarray(w, float).tolist(),
            "L": L, "candidates": [d["op"] for d in laws],
            "coord_shortfall": {j: res[j]["A"].tolist() for j in ("full", "S", "D")},
            "top2gap": res["full"]["top2gap"], "theorem_facing": tf, "certificate_kind": cert_kind,
            "certificate_ok": bool(cert_ok), "argmin_R_argmax_P_agree": bool(aligned_idx),
            "argmax_P_stable_across_compressions": bool(same_action),
            "both_conclude_infeasible": bool(regime == "infeasible-everywhere"),
            "per_variant": {j: {"argmin_R": res[j]["argmin_R"], "argmax_P": res[j]["argmax_P"],
                                "maxP": res[j]["maxP"]} for j in ("full", "S", "D")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=["HLXSYN", "HLXSYN", "HLXSYN"])
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--n-candidates", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--spec", type=float, nargs=2, default=[0.70, 0.50], help="purity_min yield_min")
    ap.add_argument("--w", type=float, nargs=2, default=[1.0, 1.0],
                    help="shortfall importance weights (purity yield); L=||w||_2. Both quantities are already "
                         "in tolerance units, so the application fixes w=(1,1); only w[0]/w[1] moves the ratios")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--posterior", default="posterior",
                    help="posterior (independent-residual) or correlated_posterior (the deployed fit)")
    ap.add_argument("--out", default=None, help="default: results/bayes/pmeet_alignment_<tag>_<products>.json")
    args = ap.parse_args()
    w = np.asarray(args.w, float)
    tag = "corr" if args.posterior.startswith("correlated") else "iid"
    out_path = args.out or f"results/bayes/pmeet_alignment_{tag}_{'_'.join(args.products)}.json"

    rows = []
    for product in args.products:
        post, bundle, tol = _load(product, args.in_dir, args.posterior)
        ops = candidate_pool_for(bundle, n_candidates=args.n_candidates, seed=args.seed)
        laws = _decision_laws(post, bundle, ops, n_steps=args.n_steps)
        a = _analyze(laws, tol, args.spec, w)
        a["product"] = product
        a["posterior"] = args.posterior
        wp, wv, kv = _prior_whitened_widths(post.cov, post.n_protein)
        a["w_par"], a["w_v"], a["dim_v"] = wp, wv, kv
        rows.append(a)
        tf = a["theorem_facing"]
        pp = tf["pmeet_preservation"]
        print(f"{product:14} w_par={wp:.3f} w_v={wv:.3f} (dim {kv})")
        print(f"{product:14} regime={a['regime']:22} maxP={a['maxP']:.3f} frac_decisive={a['frac_decisive']:.2f} "
              f"| certificate[{a['certificate_kind']}]={a['certificate_ok']}")
        if a["regime"] == "operate-anywhere":
            print(f"{'':14}   P(meet)-preservation: min P_S={pp['S']['min_Pj_over_full_decisive']:.3f} "
                  f"min P_D={pp['D']['min_Pj_over_full_decisive']:.3f}  max|ΔP|(S/D)={pp['S']['max_abs_dP']:.3f}/{pp['D']['max_abs_dP']:.3f}")
        elif tf["action_gap_meaningful"]:
            print(f"{'':14}   action-gap: Δ_act={tf['Delta_act']:.3e}  η_S^bnd/Δ={tf['S']['ratio_bound_onesided']:.2f} "
                  f"η_D^bnd/Δ={tf['D']['ratio_bound_onesided']:.2f}  (obs η_D/Δ={tf['D']['ratio_obs'] / 2:.2f})  "
                  f"argmin_R=argmax_P:{a['argmin_R_argmax_P_agree']} stable:{a['argmax_P_stable_across_compressions']}")
        else:
            print(f"{'':14}   near-tie: Δ_act={tf['Delta_act']:.3e} (shortfall action gap degenerate; report top-set overlap)")
        # Corollary R1b: holds with or without a gap, so it is reported for every regime.
        print(f"{'':14}   envelope: pool max regret={tf['pool_max_regret']:.3e}  " + "  ".join(
            f"|C_{j}|={tf[j]['envelope_size']}/{len(ops)} maxreg={tf[j]['envelope_max_regret']:.2e} "
            f"⊆argmin:{tf[j]['envelope_in_argmin']}" for j in ("S", "D")))
        if "lambda_stability" in tf:
            ls = tf["lambda_stability"]

            def fmt(d):
                if d is None:
                    return "n/a (degenerate gap)"
                a0_ = d["at_lambda0"]
                s = "not at lambda0" if a0_ is None else f"[{a0_[0]:.3g}, {a0_[1]:.3g}]"
                return s + f" (all: {len(d['all'])} interval(s))"
            print(f"{'':14}   lambda=w_p/w_y at {ls['lambda0']:.3g}: action a{ls['action_index']} stable on "
                  f"{fmt(ls['action_interval'])}  cert_S on {fmt(ls['cert_interval_S'])}  "
                  f"cert_D on {fmt(ls['cert_interval_D'])}")
        for j in ("S", "D"):
            if not tf[j]["one_sided_holds"]:
                print(f"{'':14}   WARNING: convex order violated for {j}: min(R_full - R_{j}) = "
                      f"{tf[j]['one_sided_min']:.3e} < 0; Corollary R1b's premise fails numerically")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(rows, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o))
    print(f"\nwrote {out_path}")
    print("Regimes: operate-anywhere => P(meet) decisive at every candidate; the shortfall action gap is degenerate,")
    print("  so the certificate is P(meet)-decisiveness preservation (min P under S/D stays >= P_HI, |ΔP| small).")
    print("  infeasible-everywhere / separation => the action-gap certificate η<Δ_act applies (one-sided, R1b).")
    print("  The envelope C_j = {a : r_pi(a) <= b_a,j} needs no gap and is reported in every regime: it contains")
    print("  every compressed Bayes action, so |C_j|=1 certifies exact preservation on its own. R1/R1b formally")
    print("  preserve the shortfall-risk action; the P(meet)-based deployment is the empirical bridge above.")


if __name__ == "__main__":
    main()
