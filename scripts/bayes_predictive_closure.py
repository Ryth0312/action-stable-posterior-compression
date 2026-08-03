#!/usr/bin/env python3
"""Predictive-law closure for the hierarchical decision-discrepancy layer (SI S17).

Purpose (reviewer item 1 / statistical closure). The deployed marginal decision law is a
mixture over the hierarchy posterior h = (b_p, Sigma_delta):

    g | D, c  ~  int  N( g(theta_hat^corr, c) + b_p ,  C_par(c) + Sigma_delta + C_meas )  dp(h | D).

Its total covariance is, by the law of total variance,

    C_pred(c) = E_h[ C_par + Sigma_delta + C_meas ]  +  Var_h( g(theta_hat) + b_p )
              = ( C_par + E[Sigma_delta] + C_meas )  +  Var_h( b_p ),

since g(theta_hat), C_par and C_meas do not depend on h. The previously reported
`wdec_pred` was E_h[ worst_dec(C_h) ] (the mean over draws of the per-draw worst direction),
which is NOT the worst direction of a single predictive covariance and omits Var_h(b_p).
This script reports the single-predictive-law `wdec_pred_full` = worst_dec(C_pred), i.e. the
spread of the SAME law whose mixture gives P(meet), so every decision quantity is read off one
predictive distribution.

It also emits the posterior summaries the reviewer asks for (b_A,b_B,b_C, mu_b, Sigma_b,
Sigma_delta) which the primary script does not persist, and repeats the full law on the
without-B reference-class pool.

Pure numpy/scipy on the committed folds; reuses the canonical Gibbs sampler from
bayes_decision_discrepancy_hier so the (b_p, Sigma_delta) draws are byte-identical to the
deployed run. Writes results/bayes/predictive_closure.json.
"""
from __future__ import annotations
import json, os
import numpy as np

import bayes_decision_discrepancy_hier as H  # canonical constants, loaders, gibbs, worst_dec

RES = H.RES
SPEC, TOL, SIGMA_MEAS = H.SPEC, H.TOL, H.SIGMA_MEAS
CM = np.diag(SIGMA_MEAS ** 2)


def _instrumented_gibbs(products_folds, *, prior, n_iter=40000, burn=8000, thin=8, seed=0):
    """Byte-identical to H.gibbs (same rng call order) but also collects mu_b and Sigma_b draws.

    Verified against H.gibbs below; the extra collection happens only at the thinning step and
    does not touch the rng stream, so b_p / Sigma_delta draws match H.gibbs exactly.
    """
    rng = np.random.default_rng(seed)
    pids = list(products_folds.keys())
    nu_d, s_d = prior["nu_d"], prior["s_d"]
    nu_b, s_b = prior["nu_b"], prior["s_b"]
    V_mu = prior["v_mu"] * np.eye(2)
    Psi_d = (nu_d - 3) * np.diag(s_d ** 2)
    Psi_b = (nu_b - 3) * np.diag(s_b ** 2)

    b = {p: np.zeros(2) for p in pids}
    d = {p: [f["r"].copy() for f in fl] for p, fl in products_folds.items()}
    Sig_d = np.diag(s_d ** 2)
    Sig_b = np.diag(s_b ** 2)
    mu_b = np.zeros(2)

    draws = {p: [] for p in pids}
    draws_Sd, draws_mu, draws_Sb = [], [], []
    for it in range(n_iter):
        Sd_inv = np.linalg.inv(Sig_d)
        Sb_inv = np.linalg.inv(Sig_b)
        for p, fl in products_folds.items():
            for i, f in enumerate(fl):
                Ninv = np.linalg.inv(f["N"])
                prec = Ninv + Sd_inv
                cov = np.linalg.inv(prec)
                mean = cov @ (Ninv @ f["r"] + Sd_inv @ b[p])
                d[p][i] = rng.multivariate_normal(mean, cov)
        for p in pids:
            n = len(d[p])
            dbar = np.mean(np.array(d[p]), axis=0)
            prec = n * Sd_inv + Sb_inv
            cov = np.linalg.inv(prec)
            mean = cov @ (n * Sd_inv @ dbar + Sb_inv @ mu_b)
            b[p] = rng.multivariate_normal(mean, cov)
        S = np.zeros((2, 2)); N_tot = 0
        for p in pids:
            for di in d[p]:
                e = (di - b[p]).reshape(2, 1); S += e @ e.T; N_tot += 1
        Sig_d = H.iw_sample(nu_d + N_tot, Psi_d + S, rng)
        Kp = len(pids)
        prec = Kp * Sb_inv + np.linalg.inv(V_mu)
        cov = np.linalg.inv(prec)
        bbar = np.mean([b[p] for p in pids], axis=0)
        mean = cov @ (Kp * Sb_inv @ bbar)
        mu_b = rng.multivariate_normal(mean, cov)
        Sb = np.zeros((2, 2))
        for p in pids:
            e = (b[p] - mu_b).reshape(2, 1); Sb += e @ e.T
        Sig_b = H.iw_sample(nu_b + Kp, Psi_b + Sb, rng)

        if it >= burn and (it - burn) % thin == 0:
            for p in pids:
                draws[p].append(b[p].copy())
            draws_Sd.append(Sig_d.copy())
            draws_mu.append(mu_b.copy())
            draws_Sb.append(Sig_b.copy())
    return ({p: np.array(v) for p, v in draws.items()},
            np.array(draws_Sd), np.array(draws_mu), np.array(draws_Sb))


def _summ(x):
    x = np.asarray(x)
    return dict(mean=x.mean(0).tolist(), sd=x.std(0, ddof=1).tolist())


def _closure_for_pool(pf, prior, tag):
    b_draws, Sd, mu, Sb = _instrumented_gibbs(pf, prior=prior, seed=0)
    # cross-check against the canonical sampler (b_p / Sigma_delta must be identical)
    b_ref, Sd_ref = H.gibbs(pf, prior=prior, seed=0)
    for p in b_draws:
        assert np.allclose(b_draws[p], b_ref[p]), f"b draws diverge from H.gibbs for {p}"
    assert np.allclose(Sd, Sd_ref), "Sigma_delta draws diverge from H.gibbs"

    ESd = Sd.mean(0)
    out = {"tag": tag, "E_Sigma_delta": ESd.tolist(),
           "Sigma_delta_sd": np.array([Sd[:, 0, 0], Sd[:, 1, 1]]).std(1, ddof=1).tolist(),
           "Sigma_delta_corr_mean": float(np.mean(Sd[:, 0, 1] / np.sqrt(Sd[:, 0, 0] * Sd[:, 1, 1]))),
           "mu_b": _summ(mu),
           "Sigma_b_diag_mean": [float(Sb[:, 0, 0].mean()), float(Sb[:, 1, 1].mean())],
           "products": {}}
    for pid in pf:
        C_par, relf = H.c_param(pid)
        g = H.g_map_corr(pid)
        b_p = b_draws[pid]
        Var_b = np.cov(b_p.T)
        C_mean = C_par + ESd + CM                      # E_h[C_h]  (what the SI formula writes)
        C_pred = C_mean + Var_b                         # full marginal predictive law
        perdraw = np.array([H.worst_dec(C_par + Sd[s] + CM) for s in range(len(Sd))])
        out["products"][pid] = dict(
            relFrob=relf,
            wdec_cond=H.worst_dec(C_par),
            wdec_pred_perdraw_mean=float(perdraw.mean()),      # previously reported quantity
            wdec_pred_of_mean_cov=H.worst_dec(C_mean),         # worst_dec(E_h[C_h]) literal SI formula
            wdec_pred_full=H.worst_dec(C_pred),                # single-predictive-law spread (deployed)
            base_is_nonlinear=bool(relf > 0.5),
            b_p=_summ(b_p),
            Var_b_diag_sd=np.sqrt(np.diag(Var_b)).tolist(),
            sd_Ccond=np.sqrt(np.diag(C_par)).tolist(),
            sd_ESd=np.sqrt(np.diag(ESd)).tolist(),
            sd_Cpred=np.sqrt(np.diag(C_pred)).tolist(),
        )
    return out


def main():
    prior = dict(nu_d=4.0, s_d=np.array([0.03, 0.05]), nu_b=4.0,
                 s_b=np.array([0.02, 0.03]), v_mu=0.05 ** 2)
    pf_full = {pid: H.load_folds(pid, SIGMA_MEAS) for _, pid in H.PRODUCTS}
    full = _closure_for_pool(pf_full, prior, "full-cohort")

    # without-B reference-class end (drop mAb B / HLXSYN from the shared pool)
    pf_noB = {pid: fl for pid, fl in pf_full.items() if pid != "HLXSYN"}
    noB = _closure_for_pool(pf_noB, prior, "without-B")

    result = {"spec": SPEC.tolist(), "tol": TOL.tolist(), "sigma_meas": SIGMA_MEAS.tolist(),
              "note": ("wdec_pred_full = worst_dec(E_h[C_h] + Var_h(b_p)) is the single "
                       "predictive-law spread consistent with the mixture P(meet); "
                       "wdec_pred_perdraw_mean is the previously reported E_h[worst_dec(C_h)]."),
              "full_cohort": full, "without_B": noB}
    outp = os.path.join(RES, "predictive_closure.json")
    with open(outp, "w") as f:
        json.dump(result, f, indent=2)

    # human-readable summary
    print("=" * 96)
    print("PREDICTIVE-LAW CLOSURE (item 1): single predictive covariance C_pred = E_h[C_h] + Var_h(b_p)")
    print("=" * 96)
    print(f"shared Sigma_delta: E-diag sd = {[round(x,4) for x in np.sqrt(full['E_Sigma_delta'][0][0:1]+[full['E_Sigma_delta'][1][1]])]}"
          f"  corr {full['Sigma_delta_corr_mean']:.2f}")
    print(f"mu_b: mean {np.round(full['mu_b']['mean'],4).tolist()}  sd {np.round(full['mu_b']['sd'],4).tolist()}")
    print(f"Sigma_b diag mean {np.round(full['Sigma_b_diag_mean'],6).tolist()}")
    hdr = f"{'product':<14}{'wdec_cond':>10}{'perdraw_mean':>14}{'wdec(E[C])':>12}{'wdec_pred_FULL':>16}{'base':>10}"
    print("\n[full cohort]"); print(hdr)
    for pid, v in full["products"].items():
        print(f"{pid:<14}{v['wdec_cond']:>10.3f}{v['wdec_pred_perdraw_mean']:>14.3f}"
              f"{v['wdec_pred_of_mean_cov']:>12.3f}{v['wdec_pred_full']:>16.3f}"
              f"{'nonlin' if v['base_is_nonlinear'] else 'lin':>10}")
        print(f"    b_p mean {np.round(v['b_p']['mean'],4).tolist()} sd {np.round(v['b_p']['sd'],4).tolist()}"
              f"   Var_h(b_p) diag sd {np.round(v['Var_b_diag_sd'],4).tolist()}")
    print("\n[without B]"); print(hdr)
    for pid, v in noB["products"].items():
        print(f"{pid:<14}{v['wdec_cond']:>10.3f}{v['wdec_pred_perdraw_mean']:>14.3f}"
              f"{v['wdec_pred_of_mean_cov']:>12.3f}{v['wdec_pred_full']:>16.3f}"
              f"{'nonlin' if v['base_is_nonlinear'] else 'lin':>10}")
    print("=" * 96)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
