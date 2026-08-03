"""Modular decision-level model discrepancy + discrepancy-aware action recompute.

Implements the reviewer's steps 3-4 at the DECISION level (no unrestricted KOH-GP):

  g^real(c) = g(theta, c) + d(c),   d ~ low-dim cross-fitted decision residual model
  C_g^total = G Sigma_theta G^T  +  C_delta  +  C_meas

and reports two DISTINCT worst-direction metrics (do NOT merge them):

  wdec_cond  = sqrt(lambda_max(T^-1 (G Sigma G^T)                 T^-1))   # given the model
  wdec_pred  = sqrt(lambda_max(T^-1 (G Sigma G^T + C_delta + C_meas) T^-1)) # true predictive

Action / P(meet) / regret use wdec_pred (predictive), NOT wdec_cond.

The discrepancy d(c) is fit from the committed cross-fitted leave-one-experiment-out
decision residuals (observed - predicted pooled purity/yield), pooled across products
with a shrunk 2x2 covariance, isolating the true discrepancy variance from the
held-out parameter-epistemic term and the observed-QoI measurement error (so the
empirical residual variance is NOT added once wholesale -- reviewer's caution).

Pure numpy/scipy; reads only committed results/bayes/*.json. The parameter covariance
G Sigma G^T is taken from the committed iid Laplace posterior by default; pass a
correlated-refit decision covariance via --c-param-json to fill the "correlated" column
once the Colab OU/AR(1)+nugget refit (scripts/bayes_correlated_refit.py) has run.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy.stats import multivariate_normal, norm

RES = os.path.join(os.path.dirname(__file__), "..", "results", "bayes")
SPEC = np.array([0.70, 0.50])       # illustrative purity/yield spec
TOL = np.array([0.02, 0.05])        # nominal tolerances
LOAD_CAP = 35.0                     # supported-domain loading cap (g/L)
# Observed-QoI (HPLC/integration) measurement std -- PLACEHOLDER; set from the data
# owner's assay precision. Purity from HPLC area%, yield from mass integration.
SIGMA_MEAS = np.array([0.005, 0.008])
DELTA_DECISIVE = 0.05               # P(meet) decisive band: <delta miss, >1-delta meet

PRODUCTS = [
    ("mAb A", "HLXSYN", "HLXSYN_decision.json", "HLXSYN_validation.json"),
    ("mAb B", "HLXSYN", "HLXSYN_decision.json", "HLXSYN_validation.json"),
    ("mAb C", "HLXSYN", "HLXSYN_decision.json", "HLXSYN_validation.json"),
]


def _load(fn):
    with open(os.path.join(RES, fn)) as f:
        return json.load(f)


def load_product(decf, valf, folds_file=None):
    dec = _load(decf)
    # folds source: OU-consistent decision-LOEO (bayes_correlated_refit.py --decision-loeo) when supplied,
    # else the committed i.i.d.-LOEO validation artifact. Both use the decision_loeo.folds schema.
    if folds_file and os.path.exists(folds_file):
        with open(folds_file) as f:
            val = json.load(f)
    else:
        val = _load(valf)
    g = np.array([dec["decision"]["g_map"]["pool_purity"],
                  dec["decision"]["g_map"]["pool_yield"]])
    C_lin = np.array(dec["crosscheck"]["C_lin"])
    C_mc = np.array(dec["crosscheck"].get("C_mc", C_lin))
    relf = float(dec["crosscheck"].get("rel_frobenius", 0.0))
    folds = []
    for fo in val["decision_loeo"]["folds"]:
        folds.append(dict(
            loading=fo["op"][0],
            resid=np.array([fo["observed"]["pool_purity"] - fo["g_pred"]["pool_purity"],
                            fo["observed"]["pool_yield"] - fo["g_pred"]["pool_yield"]]),
            pstd=np.array([fo["decision_std"]["pool_purity"], fo["decision_std"]["pool_yield"]]),
        ))
    return dict(g=g, C_lin=C_lin, C_mc=C_mc, relf=relf, folds=folds)


def indomain(folds):
    return [f for f in folds if f["loading"] <= LOAD_CAP]


def shrink_to_diag(C, lam):
    """Ledoit-Wolf-style shrinkage of a 2x2 covariance toward its diagonal."""
    return lam * np.diag(np.diag(C)) + (1.0 - lam) * C


def estimate_discrepancy(folds_in, sigma_meas, shrink=0.5):
    """Fit d(c) from in-domain cross-fitted residuals: bias + isolated shrunk 2x2 cov.

    Isolate the discrepancy 2nd moment from the residual 2nd moment by subtracting the
    (held-out) parameter-epistemic decision variance and the observed measurement
    variance, then clip to PSD and shrink toward diagonal.
    """
    R = np.array([f["resid"] for f in folds_in])          # (n,2)
    P = np.array([f["pstd"] for f in folds_in]) ** 2       # (n,2) param var
    n = len(R)
    bias = R.mean(axis=0)
    # 2nd moment about the bias (unbiased if n>1)
    S = np.cov(R.T, bias=False) if n > 1 else np.diag((R[0] - bias) ** 2)
    S = np.atleast_2d(S)
    # isolate discrepancy: subtract mean held-out param var + measurement var (diagonal)
    sub = np.diag(P.mean(axis=0) + sigma_meas ** 2)
    C_delta = S - sub
    # PSD clip
    w, V = np.linalg.eigh(0.5 * (C_delta + C_delta.T))
    C_delta = V @ np.diag(np.clip(w, 0.0, None)) @ V.T
    C_delta = shrink_to_diag(C_delta, shrink)
    return bias, C_delta, n


def worst_dec(C):
    Ti = np.diag(1.0 / TOL)
    return float(np.sqrt(np.max(np.linalg.eigvalsh(Ti @ C @ Ti))))


def p_meet_gauss(g, C):
    return float(multivariate_normal(mean=-g, cov=C, allow_singular=True).cdf(-SPEC))


def p_meet_samples(g, C, dist="normal", df=4, M=200000, seed=0):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(C + 1e-15 * np.eye(2))
    z = rng.standard_normal((M, 2))
    if dist == "t":
        # multivariate-t with cov C: scale so covariance == C (needs df>2)
        s = rng.chisquare(df, M) / df
        z = z / np.sqrt(s)[:, None] * np.sqrt((df - 2) / df)
    x = g + z @ L.T
    return float(np.mean((x[:, 0] >= SPEC[0]) & (x[:, 1] >= SPEC[1])))


def state(pmeet):
    if pmeet >= 1 - DELTA_DECISIVE:
        return "meet"
    if pmeet <= DELTA_DECISIVE:
        return "miss"
    return "AMBIGUOUS"


def run(c_param_override=None, sigma_meas=SIGMA_MEAS, shrink=0.5, label="iid", g_override=None,
        folds_template=None):
    prods = {pid: load_product(decf, valf,
                               folds_file=(folds_template.format(product=pid) if folds_template else None))
             for _, pid, decf, valf in PRODUCTS}
    # pooled in-domain residuals (all three products) for the pooled / cross-fit estimate
    all_in = {pid: indomain(p["folds"]) for pid, p in prods.items()}

    out = {"spec": SPEC.tolist(), "tol": TOL.tolist(), "load_cap": LOAD_CAP,
           "sigma_meas": sigma_meas.tolist(), "shrink": shrink, "c_param_label": label,
           "products": {}}
    print(f"\n{'='*94}\nDecision-level discrepancy-aware recompute   [C_param source: {label}]\n"
          f"spec purity>={SPEC[0]}, yield>={SPEC[1]}; in-domain <= {LOAD_CAP:.0f} g/L; "
          f"sigma_meas={sigma_meas.tolist()}; shrink={shrink}\n{'='*94}")
    hdr = (f"{'product':<14}{'wdec_cond':>10}{'wdec_pred':>10}{'wdec_pred(LOPO)':>16}"
           f"{'P(meet)cond':>12}{'P(meet)pred':>12}{'  t4':>7}{'  LOPO':>8}  state")
    print(hdr)

    for name, pid, decf, valf in PRODUCTS:
        p = prods[pid]
        # decision point: correlated-refit g_map overrides the committed iid g_map when given
        # (the refit MOVES g_map -- a fixed-MAP inflation cannot see this)
        p["g"] = np.array(g_override[pid]) if (g_override and pid in g_override) else p["g"]
        # choose the parameter covariance base
        if c_param_override and pid in c_param_override:
            C_par = np.array(c_param_override[pid])
        else:
            C_par = p["C_mc"] if p["relf"] > 0.5 else p["C_lin"]  # MC base if lin flagged

        # discrepancy: own-product and leave-one-product-out (cross-fitted)
        bias_o, Cd_o, n_o = estimate_discrepancy(all_in[pid], sigma_meas, shrink)
        pooled_lopo = [f for q, fl in all_in.items() if q != pid for f in fl]
        bias_l, Cd_l, n_l = estimate_discrepancy(pooled_lopo, sigma_meas, shrink)

        wd_cond = worst_dec(C_par)
        C_pred_o = C_par + Cd_o + np.diag(sigma_meas ** 2)
        C_pred_l = C_par + Cd_l + np.diag(sigma_meas ** 2)
        wd_pred_o = worst_dec(C_pred_o)
        wd_pred_l = worst_dec(C_pred_l)

        pm_cond = p_meet_gauss(p["g"], C_par)
        pm_pred_o = p_meet_samples(p["g"] + bias_o, C_pred_o, "normal")
        pm_pred_o_t = p_meet_samples(p["g"] + bias_o, C_pred_o, "t")
        pm_pred_l = p_meet_samples(p["g"] + bias_l, C_pred_l, "normal")

        print(f"{name:<14}{wd_cond:>10.3f}{wd_pred_o:>10.3f}{wd_pred_l:>16.3f}"
              f"{pm_cond:>12.3f}{pm_pred_o:>12.3f}{pm_pred_o_t:>7.2f}{pm_pred_l:>8.2f}"
              f"  {state(pm_pred_o)}")

        out["products"][pid] = dict(
            name=name, g_map=p["g"].tolist(), c_param_relFrob=p["relf"],
            wdec_cond=wd_cond, wdec_pred_own=wd_pred_o, wdec_pred_lopo=wd_pred_l,
            p_meet_cond=pm_cond, p_meet_pred_own=pm_pred_o, p_meet_pred_own_t4=pm_pred_o_t,
            p_meet_pred_lopo=pm_pred_l,
            disc_own=dict(bias=bias_o.tolist(), C_delta=Cd_o.tolist(), n_folds=n_o),
            disc_lopo=dict(bias=bias_l.tolist(), C_delta=Cd_l.tolist(), n_folds=n_l),
            state_pred=state(pm_pred_o))
    print(f"{'-'*94}\nwdec_cond = conditional (given model); wdec_pred = predictive "
          f"(+discrepancy+meas). Action uses PREDICTIVE.\n"
          f"LOPO = discrepancy fit from the OTHER products (cross-fitted); t4 = Student-t(4) tails.\n{'='*94}")
    return out


def report_loading_trend(folds_template=None):
    """Test whether the in-domain decision-LOEO residual has a loading trend (reviewer B3).

    A significant slope of (purity, yield) residual on loading argues for a loading-DEPENDENT
    discrepancy over the deployed constant one; a flat trend backs the constant-discrepancy model.
    Pure OLS across the pooled in-domain folds of all products.
    """
    from scipy import stats
    prods = {pid: load_product(decf, valf,
                               folds_file=(folds_template.format(product=pid) if folds_template else None))
             for _, pid, decf, valf in PRODUCTS}
    xs, rp, ry, tags = [], [], [], []
    for name, pid, _, _ in PRODUCTS:
        for f in indomain(prods[pid]["folds"]):
            xs.append(f["loading"]); rp.append(f["resid"][0]); ry.append(f["resid"][1]); tags.append(pid)
    xs = np.array(xs); rp = np.array(rp); ry = np.array(ry); n = len(xs)
    print(f"\n{'='*80}\nIn-domain decision-LOEO residual vs loading (constant- vs linear-discrepancy check)\n"
          f"n={n} in-domain folds (<= {LOAD_CAP:.0f} g/L); loadings {sorted(set(np.round(xs,1)))}\n{'='*80}")
    out = {"n": n, "load_cap": LOAD_CAP, "loadings": xs.tolist(), "by_quantity": {}}
    for qname, r in [("pool_purity", rp), ("pool_yield", ry)]:
        lr = stats.linregress(xs, r)
        out["by_quantity"][qname] = dict(slope=lr.slope, intercept=lr.intercept, r2=lr.rvalue ** 2,
                                         p_value=lr.pvalue, stderr=lr.stderr,
                                         resid_range=[float(r.min()), float(r.max())])
        sig = "SIGNIFICANT trend" if (n > 2 and lr.pvalue < 0.05) else "no significant trend"
        print(f"  {qname:<12}: slope={lr.slope:+.5f}/g/L  R^2={lr.rvalue**2:.3f}  p={lr.pvalue:.3f}  "
              f"resid=[{r.min():+.3f},{r.max():+.3f}]  -> {sig}")
    n_ld = len(set(np.round(xs, 1)))
    print(f"\nReading: {n_ld} distinct in-domain loadings; a linear-in-loading discrepancy needs >=3 per "
          f"product to be stably fit (A/C have 2 each), so report the constant-discrepancy hierarchical\n"
          f"model as primary and this trend as the caveat backing it.\n{'='*80}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-param-json", default=None,
                    help="JSON {pid: 2x2 decision cov} from a correlated refit (fills the correlated column)")
    ap.add_argument("--report-loading-trend", action="store_true",
                    help="B3: regress in-domain decision-LOEO residuals on loading (constant- vs "
                         "linear-in-loading discrepancy check) and exit; honors --folds-json.")
    ap.add_argument("--label", default="iid-committed")
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--sigma-meas-purity", type=float, default=SIGMA_MEAS[0])
    ap.add_argument("--sigma-meas-yield", type=float, default=SIGMA_MEAS[1])
    ap.add_argument("--folds-json", default=None,
                    help="template for per-product OU-consistent decision-LOEO folds, with '{product}' "
                         "(e.g. results/bayes/{product}_correlated_loeo.json from bayes_correlated_refit.py "
                         "--decision-loeo). Default: the committed i.i.d. {product}_validation.json folds.")
    ap.add_argument("--out", default=os.path.join(RES, "decision_discrepancy.json"))
    a = ap.parse_args()
    if a.report_loading_trend:
        trend = report_loading_trend(folds_template=a.folds_json)
        with open(a.out, "w") as f:
            json.dump(trend, f, indent=2)
        print(f"wrote {a.out}")
        return
    override = json.load(open(a.c_param_json)) if a.c_param_json else None
    # when a correlated C_param is supplied, also adopt the correlated refit's g_map
    # (it moves under the re-fit) from {pid}_correlated.json if present
    gov = None
    if a.c_param_json:
        gov = {}
        for _, pid, _, _ in PRODUCTS:
            cp = os.path.join(RES, f"{pid}_correlated.json")
            if os.path.exists(cp):
                rep = json.load(open(cp)).get("decision_report", {})
                if isinstance(rep, dict) and isinstance(rep.get("g_map"), dict):
                    gov[pid] = [rep["g_map"]["pool_purity"], rep["g_map"]["pool_yield"]]
    sm = np.array([a.sigma_meas_purity, a.sigma_meas_yield])
    out = run(c_param_override=override, sigma_meas=sm, shrink=a.shrink, label=a.label, g_override=gov,
              folds_template=a.folds_json)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
