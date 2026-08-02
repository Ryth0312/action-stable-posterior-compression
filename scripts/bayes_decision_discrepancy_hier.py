"""Hierarchical / partial-pooling decision-level model discrepancy (single primary model).

Replaces the two plug-in extremes of ``bayes_decision_discrepancy.py`` (own-fold n=2 and
cross-product-pooled LOPO) with ONE principled estimator: a low-dimensional Bayesian
hierarchical model over the in-domain decision-LOEO residuals, propagating the estimation
uncertainty of the bias, the shared discrepancy covariance, and the shrinkage into P(meet).

Model (per product p, in-domain fold i; decision quantity = (pooled purity, yield) in R^2):

    r_{p,i} = d_{p,i} + eta_{p,i},    eta_{p,i} ~ N(0, N_{p,i})   (known fold noise floor)
    d_{p,i} ~ N(b_p, Sigma_delta)                                (shared discrepancy cov)
    b_p     ~ N(mu_b, Sigma_b)                                   (partial-pooled product bias)

with N_{p,i} = diag(pstd_{p,i}^2 + sigma_meas^2) the held-out parameter-epistemic + measurement
variance removed from the residual (so the empirical variance is not double counted). Sigma_delta
is SHARED across products (the reviewer's "shared / strong-shrinkage 2x2 covariance"); b_p is
partially pooled toward mu_b through Sigma_b.

Weakly-informative conjugate priors, sampled by Gibbs (pure numpy/scipy):
    Sigma_delta ~ IW(nu_d, Psi_d),  Sigma_b ~ IW(nu_b, Psi_b),  mu_b ~ N(0, V_mu).
Deployed discrepancy for product p at a new in-domain condition is the posterior predictive
d(c) ~ N(b_p, Sigma_delta); P(meet) integrates over the joint posterior of (b_p, Sigma_delta),
so bias- and covariance-estimation uncertainty enters the action. own-fold and LOPO remain the
two-end sensitivity, reported alongside.

Reads only committed results/bayes/*.json (same inputs as bayes_decision_discrepancy.py).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy.stats import multivariate_normal

RES = os.path.join(os.path.dirname(__file__), "..", "results", "bayes")
SPEC = np.array([0.70, 0.50])
TOL = np.array([0.02, 0.05])
LOAD_CAP = 35.0
SIGMA_MEAS = np.array([0.005, 0.008])
DELTA_DECISIVE = 0.05
DELTA_STAR = 0.10                      # material worst_dec reduction threshold

PRODUCTS = [
    ("mAb A", "HLXSYN"),
    ("mAb B", "HLXSYN"),
    ("mAb C", "HLXSYN"),
]


def _load(fn):
    with open(os.path.join(RES, fn)) as f:
        return json.load(f)


def load_folds(pid, sigma_meas):
    """In-domain decision-LOEO residuals with per-fold known noise floor N = diag(pstd^2+meas^2)."""
    d = _load(f"{pid}_correlated_loeo.json")
    out = []
    for fo in d["decision_loeo"]["folds"]:
        if fo["op"][0] > LOAD_CAP:
            continue
        r = np.array([fo["observed"]["pool_purity"] - fo["g_pred"]["pool_purity"],
                      fo["observed"]["pool_yield"] - fo["g_pred"]["pool_yield"]])
        ps = np.array([fo["decision_std"]["pool_purity"], fo["decision_std"]["pool_yield"]])
        out.append(dict(r=r, N=np.diag(ps ** 2 + sigma_meas ** 2)))
    return out


def c_param(pid):
    """Correlated-refit conditional decision covariance G Sigma_corr G^T (MC base if lin flagged)."""
    dec = _load(f"{pid}_decision.json")
    relf = float(dec["crosscheck"].get("rel_frobenius", 0.0))
    cp = json.load(open(os.path.join(RES, "c_param_correlated.json")))
    if relf > 0.5:                       # correlated linearization unreliable -> iid MC base
        return np.array(dec["crosscheck"].get("C_mc", dec["crosscheck"]["C_lin"])), relf
    return np.array(cp[pid]), relf


def g_map_corr(pid):
    rep = _load(f"{pid}_correlated.json")["decision_report"]["g_map"]
    return np.array([rep["pool_purity"], rep["pool_yield"]])


def deployed_wdec_cond(pid):
    """Conditional wdec_theta|M from the SAME deployed conditional covariance base c_param(pid)
    uses for the predictive read (post-B0 self-consistency). The pre-B0 decision_discrepancy_ou_loeo.json
    carried a stale value for mAb A (0.72 vs the fresh rho<=0.9 refit's 0.54) from an earlier refit; we
    read the fresh base so wdec_cond, wdec_pred, and P(meet) all trace to one correlated posterior."""
    return worst_dec(c_param(pid)[0])


def iw_sample(nu, Psi, rng):
    """Inverse-Wishart draw via Bartlett: Sigma = (L A A^T L^T)^{-1}, Psi^{-1}=L L^T."""
    p = Psi.shape[0]
    L = np.linalg.cholesky(np.linalg.inv(Psi))
    A = np.zeros((p, p))
    for i in range(p):
        A[i, i] = np.sqrt(rng.chisquare(nu - i))
    for i in range(1, p):
        for j in range(i):
            A[i, j] = rng.standard_normal()
    W = L @ A @ A.T @ L.T                 # Wishart(nu, Psi^{-1})
    return np.linalg.inv(W)


def worst_dec(C):
    Ti = np.diag(1.0 / TOL)
    return float(np.sqrt(np.max(np.linalg.eigvalsh(Ti @ C @ Ti))))


def gibbs(products_folds, *, prior, n_iter=40000, burn=8000, thin=8, seed=0):
    """Gibbs sampler; returns posterior draws of (b_p per product, Sigma_delta)."""
    rng = np.random.default_rng(seed)
    pids = list(products_folds.keys())
    nu_d, s_d = prior["nu_d"], prior["s_d"]
    nu_b, s_b = prior["nu_b"], prior["s_b"]
    V_mu = prior["v_mu"] * np.eye(2)
    Psi_d = (nu_d - 3) * np.diag(s_d ** 2)     # E[Sigma_delta] = diag(s_d^2)
    Psi_b = (nu_b - 3) * np.diag(s_b ** 2)

    # init
    b = {p: np.zeros(2) for p in pids}
    d = {p: [f["r"].copy() for f in fl] for p, fl in products_folds.items()}
    Sig_d = np.diag(s_d ** 2)
    Sig_b = np.diag(s_b ** 2)
    mu_b = np.zeros(2)

    draws = {p: [] for p in pids}
    draws_Sd = []
    for it in range(n_iter):
        Sd_inv = np.linalg.inv(Sig_d)
        Sb_inv = np.linalg.inv(Sig_b)
        # 1) latent clean discrepancy d_{p,i} | r, b_p, Sigma_delta
        for p, fl in products_folds.items():
            for i, f in enumerate(fl):
                Ninv = np.linalg.inv(f["N"])
                prec = Ninv + Sd_inv
                cov = np.linalg.inv(prec)
                mean = cov @ (Ninv @ f["r"] + Sd_inv @ b[p])
                d[p][i] = rng.multivariate_normal(mean, cov)
        # 2) b_p | d_{p,.}, mu_b, Sigma_b
        for p in pids:
            n = len(d[p])
            dbar = np.mean(np.array(d[p]), axis=0)
            prec = n * Sd_inv + Sb_inv
            cov = np.linalg.inv(prec)
            mean = cov @ (n * Sd_inv @ dbar + Sb_inv @ mu_b)
            b[p] = rng.multivariate_normal(mean, cov)
        # 3) Sigma_delta | d, b  ~ IW(nu_d + N, Psi_d + sum (d-b)(d-b)^T)
        S = np.zeros((2, 2))
        N_tot = 0
        for p in pids:
            for di in d[p]:
                e = (di - b[p]).reshape(2, 1)
                S += e @ e.T
                N_tot += 1
        Sig_d = iw_sample(nu_d + N_tot, Psi_d + S, rng)
        # 4) mu_b | b, Sigma_b
        Kp = len(pids)
        prec = Kp * Sb_inv + np.linalg.inv(V_mu)
        cov = np.linalg.inv(prec)
        bbar = np.mean([b[p] for p in pids], axis=0)
        mean = cov @ (Kp * Sb_inv @ bbar)
        mu_b = rng.multivariate_normal(mean, cov)
        # 5) Sigma_b | b, mu_b ~ IW(nu_b + K, Psi_b + sum (b-mu)(b-mu)^T)
        Sb = np.zeros((2, 2))
        for p in pids:
            e = (b[p] - mu_b).reshape(2, 1)
            Sb += e @ e.T
        Sig_b = iw_sample(nu_b + Kp, Psi_b + Sb, rng)

        if it >= burn and (it - burn) % thin == 0:
            for p in pids:
                draws[p].append(b[p].copy())
            draws_Sd.append(Sig_d.copy())
    return {p: np.array(v) for p, v in draws.items()}, np.array(draws_Sd)


def p_meet_hier(g, C_par, b_draws, Sd_draws, sigma_meas, rng, M_inner=400):
    """P(meet) integrating over posterior (b_p, Sigma_delta); full MC (nonlinearity + estim. unc.)."""
    Cm = np.diag(sigma_meas ** 2)
    hit = 0
    tot = 0
    wdec_draws = []
    for b, Sd in zip(b_draws, Sd_draws):
        C = C_par + Sd + Cm
        wdec_draws.append(worst_dec(C))
        L = np.linalg.cholesky(C + 1e-15 * np.eye(2))
        z = rng.standard_normal((M_inner, 2))
        x = (g + b) + z @ L.T
        hit += np.sum((x[:, 0] >= SPEC[0]) & (x[:, 1] >= SPEC[1]))
        tot += M_inner
    return hit / tot, np.array(wdec_draws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shrink-note", default="hierarchical-partial-pooling")
    ap.add_argument("--nu-d", type=float, default=4.0)
    ap.add_argument("--sd-pur", type=float, default=0.03)
    ap.add_argument("--sd-yld", type=float, default=0.05)
    ap.add_argument("--nu-b", type=float, default=4.0)
    ap.add_argument("--sb-pur", type=float, default=0.02)
    ap.add_argument("--sb-yld", type=float, default=0.03)
    ap.add_argument("--v-mu", type=float, default=0.05 ** 2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="product id(s) to drop from the SHARED-covariance pool (reference-class sensitivity end, "
                         "e.g. --exclude HLXSYN excludes the level-failure product's yield pathology)")
    ap.add_argument("--out", default=os.path.join(RES, "decision_discrepancy_hier.json"))
    ap.add_argument("--dump-draws", default=None,
                    help="also save the Gibbs (b_p, Sigma_delta) draws to this .npz for the predictive "
                         "scan (bayes_decision_window.py --hier-draws), so B2 integrates over the hierarchy "
                         "posterior instead of the posterior-mean plug-in.")
    a = ap.parse_args()

    prior = dict(nu_d=a.nu_d, s_d=np.array([a.sd_pur, a.sd_yld]),
                 nu_b=a.nu_b, s_b=np.array([a.sb_pur, a.sb_yld]), v_mu=a.v_mu)
    prods = [(name, pid) for name, pid in PRODUCTS if pid not in a.exclude]
    pf = {pid: load_folds(pid, SIGMA_MEAS) for _, pid in prods}
    if a.exclude:
        print(f"[reference-class end] excluding {a.exclude} from the shared-covariance pool")
    b_draws, Sd_draws = gibbs(pf, prior=prior, seed=a.seed)
    rng = np.random.default_rng(a.seed + 1)
    if a.dump_draws:
        # save the aligned (b_p per product, shared Sigma_delta) draws for the predictive scan (B2)
        np.savez(a.dump_draws, Sd_draws=Sd_draws, pids=np.array(list(b_draws.keys())),
                 **{f"b_{p}": v for p, v in b_draws.items()})
        print(f"[dump] wrote {len(Sd_draws)} hierarchy draws -> {a.dump_draws}")

    print(f"\n{'='*100}\nHIERARCHICAL decision-discrepancy (single primary model)   "
          f"prior: nu_d={a.nu_d} s_d={prior['s_d'].tolist()} nu_b={a.nu_b} s_b={prior['s_b'].tolist()}\n{'='*100}")
    # shared Sigma_delta posterior
    Sd_mean = Sd_draws.mean(axis=0)
    print(f"shared Sigma_delta posterior mean diag = "
          f"[{Sd_mean[0,0]:.2e}, {Sd_mean[1,1]:.2e}] (sd purity {np.sqrt(Sd_mean[0,0]):.4f}, "
          f"yield {np.sqrt(Sd_mean[1,1]):.4f}); corr {Sd_mean[0,1]/np.sqrt(Sd_mean[0,0]*Sd_mean[1,1]):+.2f}")
    print(f"{'product':<8}{'wdec_cond':>10}{'wdec_pred(post-mean)':>22}{'wdec_pred[5,95]':>20}"
          f"{'P(meet)':>10}{'P(meet)[5,95]':>18}  action")
    out = {"spec": SPEC.tolist(), "tol": TOL.tolist(), "load_cap": LOAD_CAP,
           "sigma_meas": SIGMA_MEAS.tolist(), "prior": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                        for k, v in prior.items()},
           "Sigma_delta_postmean": Sd_mean.tolist(), "excluded_from_pool": a.exclude, "products": {}}
    for name, pid in prods:
        C_par, relf = c_param(pid)
        g = g_map_corr(pid)
        # deployed conditional wdec_theta|M (committed OU-LOEO artifact) drives the Weyl ceiling, so the
        # hierarchical ceiling is consistent with the wdec_theta|M the paper reports in Table S17.
        wd_cond = deployed_wdec_cond(pid)
        # bias posterior for this product; per-draw P(meet) band via block bootstrap over draws
        b_p = b_draws[pid]
        pmeet, wdec_draws = p_meet_hier(g, C_par, b_p, Sd_draws, SIGMA_MEAS, rng)
        # credible band on wdec_pred and on P(meet) (draw-level, coarse)
        wd_lo, wd_hi = np.percentile(wdec_draws, [5, 95])
        wd_mean = wdec_draws.mean()
        # P(meet) band: recompute P(meet) per posterior draw (single-draw MC) then take percentiles
        pm_draws = []
        for b, Sd in zip(b_p[::4], Sd_draws[::4]):
            C = C_par + Sd + np.diag(SIGMA_MEAS ** 2)
            pm_draws.append(float(multivariate_normal(mean=-(g + b), cov=C, allow_singular=True).cdf(-SPEC)))
        pm_draws = np.array(pm_draws)
        pm_lo, pm_hi = np.percentile(pm_draws, [5, 95])
        # Weyl irreducible-floor ceiling under hierarchical C_delta (posterior mean)
        C_pred_mean = C_par + Sd_mean + np.diag(SIGMA_MEAS ** 2)
        wd_pred_mean_full = worst_dec(C_pred_mean)
        weyl = wd_pred_mean_full - np.sqrt(max(wd_pred_mean_full ** 2 - wd_cond ** 2, 0.0))
        # action logic (predictive P(meet), decisive band; take-data gated by Weyl>Delta*)
        if pmeet >= 1 - DELTA_DECISIVE:
            act = "operate-as-is"
        elif pmeet <= DELTA_DECISIVE:
            act = "redesign-pool(level)"
        else:
            act = "take-data" if weyl >= DELTA_STAR else "abstain"
        print(f"{name:<8}{wd_cond:>10.3f}{wd_mean:>22.3f}{f'[{wd_lo:.2f},{wd_hi:.2f}]':>20}"
              f"{pmeet:>10.3f}{f'[{pm_lo:.2f},{pm_hi:.2f}]':>18}  {act}   (Weyl ceil {weyl:.3f})")
        out["products"][pid] = dict(name=name, relFrob=relf, wdec_cond=wd_cond,
                                    wdec_pred_mean=float(wd_mean), wdec_pred_ci=[float(wd_lo), float(wd_hi)],
                                    p_meet=float(pmeet), p_meet_ci=[float(pm_lo), float(pm_hi)],
                                    bias_postmean=b_p.mean(axis=0).tolist(),
                                    weyl_ceiling=float(weyl), action=act)
    print(f"{'='*100}")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
