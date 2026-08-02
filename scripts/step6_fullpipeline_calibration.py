"""Phase 2: does the ESTIMATED hierarchy---not an inserted oracle floor---restore decision calibration?

R5b showed that adding a covariance floor matched to the generative discrepancy repairs the conditional law's
dispersion failure. That is a self-consistency check, not evidence about the deployed procedure, which does not
know the true discrepancy: it estimates a shared 2x2 ``Sigma_delta`` and partially-pooled product biases ``b_p``
by Gibbs from a handful of decision-LOEO fold residuals (here 2/5/2 in-domain folds -- nine in total). This study
asks the question an AOAS referee would: with that estimator in the loop, and with the generative discrepancy
deliberately mis-scaled and mis-oriented relative to what the estimator assumes, is the deployed predictive law
calibrated?

Per replicate:
  1. draw the TRUE hierarchy -- ``Sigma_delta^true`` (isotropic or anisotropic/rotated, at a chosen whitened
     size) and ``b_p^true`` per product;
  2. generate synthetic LOEO residuals on the REAL fold structure, ``r_{p,i} = d_{p,i} + eta_{p,i}`` with
     ``d_{p,i} ~ N(b_p^true, Sigma_delta^true)`` and the committed per-fold noise floors ``N_{p,i}``;
  3. RE-ESTIMATE the hierarchy from those synthetic folds with the deployed Gibbs sampler
     (``bayes_decision_discrepancy_hier.gibbs``, imported, not reimplemented);
  4. draw a fresh deployment discrepancy ``d_new ~ N(b_p^true, Sigma_delta^true)`` and form the ground-truth
     decision ``g_obs = g_lin(theta) + d_new + eps_meas``;
  5. score every arm's ``P(meet)`` against the realised outcome.

Arms
  ``conditional``   N(g, C_par)                              -- no discrepancy layer
  ``hier-est``      integrates over the Gibbs posterior of (b_p, Sigma_delta)  -- THE DEPLOYED LAW
  ``oracle``        C_par + Sigma_delta^true, bias known     -- R5b's arm, an upper reference
  ``floor-k``       C_par + k^2 Sigma_delta^true, no bias    -- inserted floor mis-scaled by k in {0,.5,1,1.5,2}

The parameter layer is the frozen linear-Gaussian surrogate of ``step5_decision_sbc.py`` (the Laplace posterior IS
the Gauss--Newton posterior at the MAP), so the whole study is torch-free and runs locally; the decision-map
linearisation gap it omits is separately certified in the R2 study. Reported per arm: reliability bins with
Wilson intervals, ECE, Brier, log score, calibration intercept/slope, tail counts with binomial intervals, SBC
rank uniformity for g, and realised selected-action regret over the committed candidate pool.

Usage:
  python scripts/step6_fullpipeline_calibration.py --n-replicates 300 --shape iso
  python scripts/step6_fullpipeline_calibration.py --n-replicates 300 --shape aniso
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import stats

_HERE = Path(__file__).resolve().parent
RES = _HERE.parent / "results" / "bayes"
SPEC = np.array([0.70, 0.50])
TOL = np.array([0.02, 0.05])
SIGMA_MEAS = np.array([0.005, 0.008])
LOAD_CAP = 35.0
P_HI = 0.95
PRODUCTS = [("mAb A", "HLXSYN"), ("mAb B", "HLXSYN"), ("mAb C", "HLXSYN")]
FLOOR_RATIOS = (0.0, 0.5, 1.0, 1.5, 2.0)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_HIER = _load_module(_HERE / "bayes_decision_discrepancy_hier.py", "hier")   # the DEPLOYED estimator
_SBC = _load_module(_HERE / "_calibration_metrics.py", "calmetrics")         # shared scoring helpers


# --------------------------------------------------------------------------- inputs
def _fold_noise(pid):
    """Committed per-fold known noise floors N_{p,i} = diag(pstd^2 + meas^2) for the in-domain folds."""
    d = json.loads((RES / f"{pid}_correlated_loeo.json").read_text())
    out = []
    for fo in d["decision_loeo"]["folds"]:
        if fo["op"][0] > LOAD_CAP:
            continue
        ps = np.array([fo["decision_std"]["pool_purity"], fo["decision_std"]["pool_yield"]])
        out.append(np.diag(ps**2 + SIGMA_MEAS**2))
    return out


def _c_par(pid):
    dec = json.loads((RES / f"{pid}_decision.json").read_text())
    relf = float((dec.get("crosscheck") or {}).get("rel_frobenius") or 0.0)
    if relf > 0.5:                                  # linearisation flagged -> the MC base, as deployed
        cc = dec["crosscheck"]
        return np.array(cc.get("C_mc") or cc["C_lin"], float)
    return np.array(json.loads((RES / "c_param_correlated.json").read_text())[pid], float)


def _g_map(pid):
    r = json.loads((RES / f"{pid}_decision.json").read_text())["decision"]["g_map"]
    return np.array([r["pool_purity"], r["pool_yield"]])


def _pool(pid, n_max):
    """Committed in-domain candidate decisions (mean_g per operating condition)."""
    rows = json.loads((RES / f"{pid}_decision_window.json").read_text())["operating_window_map"]["rows"]
    gs = [np.array([r["mean_g"]["pool_purity"], r["mean_g"]["pool_yield"]])
          for r in rows if r["op"][0] <= LOAD_CAP]
    return gs[:n_max] if gs else [_g_map(pid)]


def _sigma_delta_true(shape, wdec_size, ratio, angle):
    """Sigma_delta with whitened worst-direction ``wdec_size``; anisotropic = unequal eigenvalues + rotation."""
    lam = np.array([1.0, 1.0]) if shape == "iso" else np.array([1.0, 1.0 / max(ratio, 1e-9)])
    th = 0.0 if shape == "iso" else angle
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    S = R @ np.diag(lam) @ R.T
    Ti = np.diag(1.0 / TOL)
    S = S * (wdec_size**2 / float(np.max(np.linalg.eigvalsh(Ti @ S @ Ti))))
    return S


# --------------------------------------------------------------------------- one replicate
def _replicate(rng, folds_N, C_par, g0, pools, Sd_true, Sb_true, gibbs_kw, n_post, spec, icc=0.0):
    pids = [p for _, p in PRODUCTS]
    b_true = {p: rng.multivariate_normal(np.zeros(2), Sb_true) for p in pids}

    # (2) synthetic decision-LOEO residuals on the REAL fold structure
    synth = {}
    for p in pids:
        fl = []
        shared = rng.multivariate_normal(np.zeros(2), icc * Sd_true) if icc > 0 else np.zeros(2)
        for N in folds_N[p]:
            # icc>0 gives folds a shared within-product component, a proxy for LOEO training-set overlap
            d = b_true[p] + shared + rng.multivariate_normal(np.zeros(2), (1.0 - icc) * Sd_true)
            fl.append({"r": d + rng.multivariate_normal(np.zeros(2), N), "N": N})
        synth[p] = fl

    # (3) re-estimate the hierarchy with the DEPLOYED sampler
    b_draws, Sd_draws = _HIER.gibbs(synth, seed=int(rng.integers(1 << 31)), **gibbs_kw)

    Cm = np.diag(SIGMA_MEAS**2)
    out = {}
    for _, p in PRODUCTS:
        Cp = C_par[p]
        # (4) truth: parameter draw (frozen linear-Gaussian) + fresh discrepancy + measurement noise
        g_par = rng.multivariate_normal(g0[p], Cp)                 # the realised decision under the model
        d_new = rng.multivariate_normal(b_true[p], Sd_true)
        g_obs = g_par + d_new + rng.multivariate_normal(np.zeros(2), Cm)
        y = float(np.all(g_obs >= spec))

        laws = {}
        laws["conditional"] = (g0[p], Cp, None)
        laws["oracle"] = (g0[p] + b_true[p], Cp + Sd_true + Cm, None)
        for k in FLOOR_RATIOS:
            laws[f"floor-{k:g}"] = (g0[p], Cp + (k**2) * Sd_true + Cm, None)
        laws["hier-est"] = (g0[p], Cp, (b_draws[p], Sd_draws))     # integrated below

        rec = {}
        for name, (mu, C, mix) in laws.items():
            if mix is None:
                pm = _pmeet_gauss(mu, C, spec)
                draws = mu + rng.standard_normal((n_post, 2)) @ np.linalg.cholesky(C + 1e-18 * np.eye(2)).T
            else:
                bd, Sd = mix
                pm, draws = _pmeet_hier(mu, C, bd, Sd, Cm, rng, n_post, spec)
            rk = (int(np.sum(draws[:, 0] < g_obs[0])), int(np.sum(draws[:, 1] < g_obs[1])))
            # joint 2-D PIT: the deployed event is a JOINT orthant, so a per-quantity rank is not enough
            mu_d, Cd = draws.mean(axis=0), np.cov(draws.T) + 1e-18 * np.eye(2)
            q = float((g_obs - mu_d) @ np.linalg.solve(Cd, (g_obs - mu_d)))
            pit = float(stats.chi2(2).cdf(q)) if np.isfinite(q) else float("nan")
            rec[name] = {"p": pm, "y": y, "rank": rk, "pit": pit}

        eps_sel = rng.multivariate_normal(np.zeros(2), Cm)
        # (9) selected action over the committed candidate pool (the discrepancy layer is candidate-independent)
        sel = {}
        for name, (mu, C, mix) in laws.items():
            ps, gs = [], pools[p]
            for gc in gs:
                if mix is None:
                    ps.append(_pmeet_gauss(gc + (mu - g0[p]), C, spec))
                else:
                    ps.append(_pmeet_hier(gc, C, *mix, Cm, rng, 0, spec)[0])
            ps = np.asarray(ps, float)
            spread = float(ps.max() - ps.min())         # 0 => the law cannot discriminate among candidates
            ties = np.flatnonzero(ps >= ps.max() - 1e-12)
            k = int(rng.choice(ties))                    # random tie-break: never silently prefer index 0
            # one shared realisation (parameter draw, discrepancy, measurement noise) for EVERY candidate, so
            # the selected action and the best-in-hindsight are compared on the same footing (regret >= 0)
            real = [gc + (g_par - g0[p]) + d_new + eps_sel for gc in gs]
            shorts = [float(np.sum(np.clip((spec - r) / TOL, 0, None))) for r in real]
            sel[name] = {"p_sel": float(ps[k]), "y_sel": float(np.all(real[k] >= spec)),
                         "regret": shorts[k] - min(shorts), "p_spread": spread,
                         "n_tied": int(ties.size)}
        out[p] = {"fixed": rec, "selected": sel}
    return out


def _pmeet_gauss(mu, C, spec):
    from scipy.stats import multivariate_normal as mvn
    return float(mvn(mean=-np.asarray(mu, float), cov=np.asarray(C, float) + 1e-15 * np.eye(2),
                     allow_singular=True).cdf(-np.asarray(spec, float)))


def _pmeet_hier(g, C_par, b_draws, Sd_draws, Cm, rng, n_post, spec):
    """P(meet) integrating over the Gibbs posterior of (b_p, Sigma_delta) -- the deployed predictive law.

    Returns ``(p_meet, draws)``; the draws mix over the WHOLE posterior (one random (b, Sigma_delta) index per
    draw), so they are a proper sample of the predictive law rather than a few consecutive Gibbs blocks.
    """
    n_d = len(Sd_draws)
    idx = rng.choice(n_d, size=min(160, n_d), replace=False)
    hit, tot = 0, 0
    for i in idx:
        C = C_par + Sd_draws[i] + Cm
        L = np.linalg.cholesky(C + 1e-15 * np.eye(2))
        x = (g + b_draws[i]) + rng.standard_normal((48, 2)) @ L.T
        hit += int(np.sum((x[:, 0] >= spec[0]) & (x[:, 1] >= spec[1])))
        tot += x.shape[0]
    draws = np.zeros((0, 2))
    if n_post:
        js = rng.integers(0, n_d, size=n_post)
        z = rng.standard_normal((n_post, 2))
        draws = np.array([g + b_draws[j] + np.linalg.cholesky(C_par + Sd_draws[j] + Cm + 1e-15 * np.eye(2)) @ z[t]
                          for t, j in enumerate(js)])
    return hit / tot, draws


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-replicates", type=int, default=300)
    ap.add_argument("--shape", choices=["iso", "aniso"], default="iso")
    ap.add_argument("--disc-wdec", type=float, default=1.271, help="whitened worst-direction size of Sigma_delta")
    ap.add_argument("--aniso-ratio", type=float, default=6.0, help="eigenvalue ratio when --shape aniso")
    ap.add_argument("--aniso-angle", type=float, default=np.pi / 5, help="rotation of Sigma_delta (radians)")
    ap.add_argument("--bias-wdec", type=float, default=0.6, help="whitened size of the product-bias covariance")
    ap.add_argument("--n-post", type=int, default=200)
    ap.add_argument("--pool-size", type=int, default=8)
    ap.add_argument("--gibbs-iter", type=int, default=3000)
    ap.add_argument("--gibbs-burn", type=int, default=600)
    ap.add_argument("--gibbs-thin", type=int, default=6)
    ap.add_argument("--fold-icc", type=float, default=0.0,
                    help="shared within-product fold component as a fraction of Sigma_delta; a proxy for "
                         "LOEO training-set overlap (0 = conditionally independent folds)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = Path(args.out or RES / f"r6_fullpipeline_calibration_{args.shape}.json")
    folds_N = {p: _fold_noise(p) for _, p in PRODUCTS}
    C_par = {p: _c_par(p) for _, p in PRODUCTS}
    g0 = {p: _g_map(p) for _, p in PRODUCTS}
    pools = {p: _pool(p, args.pool_size) for _, p in PRODUCTS}
    Sd_true = _sigma_delta_true(args.shape, args.disc_wdec, args.aniso_ratio, args.aniso_angle)
    Sb_true = _sigma_delta_true("iso", args.bias_wdec, 1.0, 0.0)
    prior = dict(nu_d=8.0, s_d=np.array([0.02, 0.04]), nu_b=8.0, s_b=np.array([0.02, 0.04]), v_mu=0.05**2)
    gibbs_kw = dict(prior=prior, n_iter=args.gibbs_iter, burn=args.gibbs_burn, thin=args.gibbs_thin)

    print(f"[setup] shape={args.shape} Sigma_delta^true whitened wdec={args.disc_wdec} "
          f"(eig ratio {np.linalg.eigvalsh(Sd_true)[1]/np.linalg.eigvalsh(Sd_true)[0]:.1f}); "
          f"folds per product: {[len(folds_N[p]) for _, p in PRODUCTS]}")

    rng = np.random.default_rng(args.seed)
    acc = {p: {arm: {"p": [], "y": [], "rk0": [], "rk1": [], "pit": [], "psel": [], "ysel": [], "regret": [], "spread": [], "ntie": []}
               for arm in ["conditional", "hier-est", "oracle"] + [f"floor-{k:g}" for k in FLOOR_RATIOS]}
           for _, p in PRODUCTS}
    for r in range(args.n_replicates):
        res = _replicate(rng, folds_N, C_par, g0, pools, Sd_true, Sb_true, gibbs_kw, args.n_post, SPEC,
                         icc=args.fold_icc)
        for _, p in PRODUCTS:
            for arm, v in res[p]["fixed"].items():
                a = acc[p][arm]
                a["p"].append(v["p"]); a["y"].append(v["y"])
                a["rk0"].append(v["rank"][0]); a["rk1"].append(v["rank"][1])
                if np.isfinite(v["pit"]): a["pit"].append(v["pit"])
            for arm, v in res[p]["selected"].items():
                a = acc[p][arm]
                a["psel"].append(v["p_sel"]); a["ysel"].append(v["y_sel"]); a["regret"].append(v["regret"])
                a["spread"].append(v["p_spread"]); a["ntie"].append(v["n_tied"])
        if (r + 1) % 25 == 0:
            print(f"  ... {r + 1}/{args.n_replicates}")

    rows = []
    for name, p in PRODUCTS:
        for arm, a in acc[p].items():
            blk = _SBC._score_block(a["p"], a["y"], f"{p}:{arm}")
            n_hi = int(np.sum(np.asarray(a["p"]) >= P_HI))
            hits = int(np.sum(np.asarray(a["y"])[np.asarray(a["p"]) >= P_HI])) if n_hi else 0
            ci = (stats.beta.ppf(0.025, hits + 0.5, n_hi - hits + 0.5) if n_hi else float("nan"),
                  stats.beta.ppf(0.975, hits + 0.5, n_hi - hits + 0.5) if n_hi else float("nan"))
            def _boot(vals, fn, B=2000, seed=1):
                v = np.asarray(vals, float)
                rb = np.random.default_rng(seed)
                st = [fn(v[rb.integers(0, v.size, v.size)]) for _ in range(B)]
                return [float(np.quantile(st, 0.025)), float(np.quantile(st, 0.975))]

            pv, yv = np.asarray(a["p"], float), np.asarray(a["y"], float)
            idx = np.arange(pv.size)
            rb = np.random.default_rng(1)
            boot_ece, boot_slope = [], []
            for _ in range(1000):
                j = rb.integers(0, idx.size, idx.size)
                bb = _SBC._score_block(pv[j], yv[j], "b")
                boot_ece.append(bb["ece"])
                if np.isfinite(bb["slope"]):
                    boot_slope.append(bb["slope"])
            rows.append({
                **blk, "product": p, "label": name, "arm": arm,
                "ece_ci": [float(np.quantile(boot_ece, 0.025)), float(np.quantile(boot_ece, 0.975))],
                "slope_ci": ([float(np.quantile(boot_slope, 0.025)), float(np.quantile(boot_slope, 0.975))]
                             if len(boot_slope) > 50 else [float("nan"), float("nan")]),
                "regret_ci": _boot(a["regret"], np.mean),
                "sbc_ks_pit": (float(stats.kstest(a["pit"], "uniform").pvalue) if len(a["pit"]) > 2
                               else float("nan")),
                "tail_hi_ci": [float(ci[0]), float(ci[1])],
                "sbc_ks_purity": _SBC._rank_stats(a["rk0"], args.n_post, "purity")["ks_pvalue"],
                "sbc_ks_yield": _SBC._rank_stats(a["rk1"], args.n_post, "yield")["ks_pvalue"],
                "sel_mean_regret": float(np.mean(a["regret"])),
                "sel_regret_se": float(np.std(a["regret"], ddof=1) / np.sqrt(len(a["regret"]))),
                "sel_base_rate": float(np.mean(a["ysel"])), "sel_mean_p": float(np.mean(a["psel"])),
                "sel_p_spread": float(np.mean(a["spread"])), "sel_mean_ties": float(np.mean(a["ntie"])),
            })

    print(f"\n=== full-pipeline calibration ({args.shape}, {args.n_replicates} replicates) ===")
    print(f"{'product':7} {'arm':12} {'ECE':>6} {'Brier':>7} {'slope':>7} {'KS_pur':>7} "
          f"{'tail n':>7} {'tail hit':>9} {'regret':>8} {'spread':>8}")
    for r in rows:
        th = "---" if not np.isfinite(r["tail_hi_hit_freq"]) else f"{r['tail_hi_hit_freq']:.3f}"
        sl = "---" if not np.isfinite(r["slope"]) else f"{r['slope']:.2f}"
        print(f"{r['label']:7} {r['arm']:12} {r['ece']:6.3f} {r['brier']:7.4f} {sl:>7} "
              f"{r['sbc_ks_purity']:7.3f} {r['tail_hi_n']:7d} {th:>9} "
              f"{r['sel_mean_regret']:8.3f} {r['sel_p_spread']:8.4f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Record the run configuration alongside the rows: shape and fold ICC live only in the invocation,
    # so a bare row list cannot be audited after the fact. Readers accept either shape (see _rows below).
    out_path.write_text(json.dumps({"meta": vars(args) | {"folds_per_product": [len(folds_N[q]) for _, q in PRODUCTS]}, "rows": rows}, indent=2))
    print(f"\nwrote {out_path}")
    print("Read: 'hier-est' is the DEPLOYED law (estimated from 2/5/2 synthetic folds). Compare it with")
    print("  'oracle' (knows the truth) and the floor-k ladder (inserted floor mis-scaled by k).")


if __name__ == "__main__":
    main()
