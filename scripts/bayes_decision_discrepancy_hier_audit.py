"""Small-sample statistical audit of the hierarchical decision-discrepancy Gibbs sampler.

Addresses the Technometrics reviewer's item #4 ("minimal statistical audit of the hierarchy"),
entirely as post-processing of the SAME committed folds the primary fit reads
(results/bayes/{pid}_correlated_loeo.json etc.) -- no solver, no proprietary curves.

Reports:
  (1) multi-chain convergence: split-Rhat and bulk-ESS for Sigma_delta entries, each b_p, and the
      derived per-product P(meet); Monte-Carlo standard error of P(meet) and wdec_pred.
  (2) discrepancy-prior scale sensitivity: sweep (nu_d, s_d) and report P(meet)/wdec_pred/action.
  (3) leave-one-product influence: drop each product from the SHARED-covariance pool.

The hierarchical draws of (b_p, Sigma_delta) are g_map-independent, so the convergence audit is
authoritative regardless of the mAb-A g_map artifact inconsistency; the derived P(meet) is reported
for BOTH committed A g_map candidates (rho<=0.9 cap vs rho=0.99) to make the rho-cap sensitivity explicit.
"""
from __future__ import annotations
import json, os
import numpy as np

import bayes_decision_discrepancy_hier as H  # reuse gibbs/load_folds/iw_sample/worst_dec/p_meet_hier

RES = H.RES
SPEC, TOL, SIGMA_MEAS = H.SPEC, H.TOL, H.SIGMA_MEAS
PRODUCTS = H.PRODUCTS


def split_rhat(chains):
    """Split-Rhat (Gelman-Rubin) for a list of 1-D chains of equal length."""
    chains = [np.asarray(c, float) for c in chains]
    n = min(len(c) for c in chains)
    halves = []
    for c in chains:
        c = c[:n]
        halves.append(c[: n // 2]); halves.append(c[n // 2 : 2 * (n // 2)])
    m = len(halves); nn = len(halves[0])
    means = np.array([h.mean() for h in halves])
    vars = np.array([h.var(ddof=1) for h in halves])
    W = vars.mean()
    B = nn * means.var(ddof=1)
    var_plus = (nn - 1) / nn * W + B / nn
    return float(np.sqrt(var_plus / W)) if W > 0 else np.nan


def ess_bulk(chains):
    """Bulk effective sample size via Geyer initial-monotone-sequence autocorrelation."""
    chains = [np.asarray(c, float) for c in chains]
    n = min(len(c) for c in chains)
    chains = [c[:n] for c in chains]
    m = len(chains)
    means = np.array([c.mean() for c in chains])
    vars = np.array([c.var(ddof=1) for c in chains])
    W = vars.mean()
    B = n * means.var(ddof=1) if m > 1 else 0.0
    var_plus = (n - 1) / n * W + (B / n if m > 1 else 0.0)
    if var_plus <= 0:
        return float(m * n)
    # mean autocorrelation across chains via FFT
    def acf(x):
        x = x - x.mean()
        f = np.fft.rfft(x, n=2 * len(x))
        a = np.fft.irfft(f * np.conj(f))[: len(x)].real
        return a / a[0] if a[0] > 0 else a
    rho = np.mean([acf(c) for c in chains], axis=0)
    # Geyer: sum paired autocorrs while positive & decreasing
    t = 1; s = 0.0
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        s += pair
        t += 2
    tau = 1.0 + 2.0 * s
    return float(m * n / tau) if tau > 0 else float(m * n)


def g_map_candidates(pid):
    """Return committed g_map candidates for the historical decision OP.

    After B0 the deployed refit is the rho<=0.9 correlated posterior, whose g_map lives in
    {pid}_correlated.json (fresh); decision_discrepancy_ou_loeo.json still carries the pre-B0
    g_map for A (stale) and is kept only to expose that drift.
    """
    out = {}
    rep = H._load(f"{pid}_correlated.json")["decision_report"]["g_map"]
    out["correlated.json(rho<=.9 deployed)"] = np.array([rep["pool_purity"], rep["pool_yield"]])
    try:
        out["ou_loeo(pre-B0, stale for A)"] = np.array(H._load("decision_discrepancy_ou_loeo.json")["products"][pid]["g_map"])
    except Exception:
        pass
    return out


def run_chains(pf, prior, seeds):
    b_chains = {p: [] for p in pf}
    sd_chains = {"pur_var": [], "yld_var": [], "corr": []}
    all_b, all_sd = {p: [] for p in pf}, []
    for s in seeds:
        bd, sd = H.gibbs(pf, prior=prior, seed=s)
        for p in pf:
            b_chains[p].append(bd[p])
            all_b[p].append(bd[p])
        pv = sd[:, 0, 0]; yv = sd[:, 1, 1]; cr = sd[:, 0, 1] / np.sqrt(pv * yv)
        sd_chains["pur_var"].append(pv); sd_chains["yld_var"].append(yv); sd_chains["corr"].append(cr)
        all_sd.append(sd)
    all_b = {p: np.concatenate(v) for p, v in all_b.items()}
    all_sd = np.concatenate(all_sd)
    return b_chains, sd_chains, all_b, all_sd


def main():
    prior = dict(nu_d=4.0, s_d=np.array([0.03, 0.05]), nu_b=4.0,
                 s_b=np.array([0.02, 0.03]), v_mu=0.05 ** 2)
    pids = [pid for _, pid in PRODUCTS]
    pf = {pid: H.load_folds(pid, SIGMA_MEAS) for pid in pids}
    seeds = [0, 1, 2, 3]
    print(f"folds per product: " + ", ".join(f"{p}={len(pf[p])}" for p in pids)
          + f"  (total {sum(len(v) for v in pf.values())})")

    # ---- (1) convergence ----
    b_chains, sd_chains, all_b, all_sd = run_chains(pf, prior, seeds)
    print("\n=== (1) MULTI-CHAIN CONVERGENCE (4 chains, 4000 draws each) ===")
    print(f"{'scalar':<22}{'Rhat':>8}{'ESS_bulk':>10}{'post.mean':>12}{'MCSE':>12}")
    for name, ch in [("Sigma_delta[pur var]", sd_chains["pur_var"]),
                     ("Sigma_delta[yld var]", sd_chains["yld_var"]),
                     ("Sigma_delta[corr]", sd_chains["corr"])]:
        rh = split_rhat(ch); ess = ess_bulk(ch)
        pooled = np.concatenate(ch); mcse = pooled.std(ddof=1) / np.sqrt(ess)
        print(f"{name:<22}{rh:>8.3f}{ess:>10.0f}{pooled.mean():>12.5g}{mcse:>12.3g}")
    for p in pids:
        for j, q in enumerate(["pur", "yld"]):
            ch = [b_chains[p][k][:, j] for k in range(len(seeds))]
            rh = split_rhat(ch); ess = ess_bulk(ch)
            pooled = np.concatenate(ch); mcse = pooled.std(ddof=1) / np.sqrt(ess)
            print(f"{'b['+p[:8]+','+q+']':<22}{rh:>8.3f}{ess:>10.0f}{pooled.mean():>12.5g}{mcse:>12.3g}")

    # derived P(meet) convergence (per chain) at deployed rho<=.9 g_map
    print("\n=== (1b) derived P(meet) across chains (deployed rho<=.9 g_map) ===")
    for _, pid in PRODUCTS:
        gm = g_map_candidates(pid)["correlated.json(rho<=.9 deployed)"]
        Cpar, _ = H.c_param(pid)
        pms = []
        for k, s in enumerate(seeds):
            rng = np.random.default_rng(1000 + s)
            pm, _ = H.p_meet_hier(gm, Cpar, b_chains[pid][k], all_sd[: len(b_chains[pid][k])], SIGMA_MEAS, rng)
            pms.append(pm)
        pms = np.array(pms)
        print(f"  {pid:<14} P(meet) per chain = {np.round(pms,4)}  mean={pms.mean():.4f}  across-chain sd={pms.std(ddof=1):.4f}")

    # ---- (1d) action frequency at the historical OP: P(P(meet|h) >= 0.95 | D) ----
    print("\n=== (1d) ACTION FREQUENCY at historical OP (over hierarchy draws, deployed rho<=.9 g_map) ===")
    Cm = np.diag(SIGMA_MEAS ** 2)
    for _, pid in PRODUCTS:
        gm = g_map_candidates(pid)["correlated.json(rho<=.9 deployed)"]
        Cpar, _ = H.c_param(pid)
        bd = all_b[pid]; sd = all_sd[: len(bd)]
        idx = np.arange(0, len(bd), 8)
        rng = np.random.default_rng(3)
        pcond = []
        for i in idx:
            C = Cpar + sd[i] + Cm
            z = rng.multivariate_normal(gm + bd[i], C, 4000)
            pcond.append(np.mean((z[:, 0] >= SPEC[0]) & (z[:, 1] >= SPEC[1])))
        pcond = np.array(pcond)
        pdec = np.mean(pcond >= 0.95)
        print(f"  {pid:<14} P(meet|h) mean={pcond.mean():.3f} 10-90%=[{np.percentile(pcond,10):.3f},"
              f"{np.percentile(pcond,90):.3f}]  P(decisive>=.95|D)={pdec:.3f}  P(non-decisive|D)={1-pdec:.3f}")

    # ---- (1c) A g_map (rho-cap) sensitivity of the derived action number ----
    print("\n=== (1c) mAb A: P(meet) under the two committed g_map artifacts (rho-cap conflict) ===")
    Cpar_A, _ = H.c_param("HLXSYN")
    for lab, gm in g_map_candidates("HLXSYN").items():
        rng = np.random.default_rng(7)
        pm, wd = H.p_meet_hier(gm, Cpar_A, all_b["HLXSYN"], all_sd, SIGMA_MEAS, rng)
        mcse = pm * (1 - pm) / np.sqrt(len(all_b["HLXSYN"]) * 400)
        print(f"  g_map={lab:<32} purity/yield={np.round(gm,3)}  P(meet)={pm:.4f} (MCSE~{mcse:.4f})  wdec_pred={wd.mean():.3f}")

    # ---- (2) discrepancy-prior scale sensitivity ----
    print("\n=== (2) DISCREPANCY-PRIOR SCALE SENSITIVITY (shared Sigma_delta prior) ===")
    print(f"{'prior(nu_d, s_d)':<26}{'Sd_pur_sd':>10}{'Sd_yld_sd':>10}{'A P(meet)':>11}{'C P(meet)':>11}")
    grids = [(4.0, [0.03, 0.05]), (4.0, [0.015, 0.025]), (4.0, [0.06, 0.10]),
             (6.0, [0.03, 0.05]), (10.0, [0.03, 0.05])]
    for nu_d, s_d in grids:
        pr = dict(prior); pr["nu_d"] = nu_d; pr["s_d"] = np.array(s_d)
        bd, sd = H.gibbs(pf, prior=pr, seed=0, n_iter=12000, burn=2000)
        spur = np.sqrt(sd[:, 0, 0]).mean(); syld = np.sqrt(sd[:, 1, 1]).mean()
        row = f"{f'({nu_d}, {s_d})':<26}{spur:>10.4f}{syld:>10.4f}"
        for pid in ['HLXSYN']:
            gm = g_map_candidates(pid)["correlated.json(rho<=.9 deployed)"]
            Cpar, _ = H.c_param(pid)
            rng = np.random.default_rng(5)
            pm, _ = H.p_meet_hier(gm, Cpar, bd[pid], sd, SIGMA_MEAS, rng)
            row += f"{pm:>11.4f}"
        print(row)

    # ---- (3) leave-one-product influence ----
    print("\n=== (3) LEAVE-ONE-PRODUCT INFLUENCE (drop from shared-covariance pool) ===")
    print(f"{'pool':<20}{'Sd_pur_sd':>10}{'Sd_yld_sd':>10}{'corr':>8}{'A P(meet)':>11}{'C P(meet)':>11}")
    for drop in [None, "HLXSYN", "HLXSYN", "HLXSYN"]:
        sub = {p: v for p, v in pf.items() if p != drop}
        bd, sd = H.gibbs(sub, prior=prior, seed=0, n_iter=12000, burn=2000)
        spur = np.sqrt(sd[:, 0, 0]).mean(); syld = np.sqrt(sd[:, 1, 1]).mean()
        cr = (sd[:, 0, 1] / np.sqrt(sd[:, 0, 0] * sd[:, 1, 1])).mean()
        row = f"{('full' if drop is None else 'drop '+drop):<20}{spur:>10.4f}{syld:>10.4f}{cr:>8.2f}"
        for pid in ['HLXSYN']:
            if pid == drop:
                row += f"{'--':>11}"; continue
            gm = g_map_candidates(pid)["correlated.json(rho<=.9 deployed)"]
            Cpar, _ = H.c_param(pid)
            rng = np.random.default_rng(5)
            pm, _ = H.p_meet_hier(gm, Cpar, bd[pid], sd, SIGMA_MEAS, rng)
            row += f"{pm:>11.4f}"
        print(row)


if __name__ == "__main__":
    main()
