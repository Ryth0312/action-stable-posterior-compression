"""Posterior-predictive decision-adequacy OPERATING WINDOW + value-of-information stopping rule.

Turns the §2.9 decision-adequacy verdict into the actionable engineering deliverable a process referee
wants -- from the committed posteriors, NO new wet-lab data:

  1. a probabilistic operating-window map -- P(purity>=spec AND yield>=spec | data) + expected regret +
     worst_dec over a Sobol pool of candidate operating conditions (plus the historical decision OP), and
     the NAMED decision-adequate window (widest determinate, highest P(meet));
  2. a VoI / EVSI stopping number -- the preposterior drop in worst_dec from each candidate next
     experiment, which is ~0 for sigma-informative runs and concentrates on the keq<->nu-resolving run, so
     a determined decision yields "more data is not worth taking for it".

    OMP_NUM_THREADS=4 python scripts/bayes_decision_window.py --products HLXSYN HLXSYN HLXSYN
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cex_model.bayes.adequacy_gate import DEFAULT_LOADING_CAP_G_PER_L
from cex_model.bayes.decision import decision_covariance_mc
from cex_model.bayes.decision_window import (
    candidate_ops_for,
    decision_voi,
    meet_prob_and_regret,
    operating_window_map,
    synthesize_recommendation,
)
from cex_model.bayes.loading_sweep import _load_product_setup

# Blinded labels for the double-blind figures (mAb A-E), used with --blind.
BLIND_LABELS = {'HLXSYN': 'mAb A'}


def _plot(product, owm, voi, path, *, loading_cap: float = DEFAULT_LOADING_CAP_G_PER_L, label=None) -> None:
    """Gated operating-window figure: candidates above the supported loading domain
    (``loading_cap``, the adequacy gate of :mod:`cex_model.bayes.adequacy_gate`) are marked
    ‘abstain’ (not action-eligible) in both panels."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    name = label or product
    rows, cand = owm["rows"], voi["candidates"]
    load = np.array([r["op"][0] for r in rows])
    wd = np.array([r["worst_dec"] for r in rows])
    pm = np.array([r["p_meet"] for r in rows])
    reg = np.array([r["expected_regret"] for r in rows])
    elig = load <= loading_cap + 1e-9
    wa_i = int(np.where(elig)[0][np.argmax(pm[elig])]) if elig.any() else None

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    # --- LEFT: probabilistic operating-window map, gated ---
    a = ax[0]
    sc = a.scatter(wd[elig], pm[elig], c=reg[elig], cmap="viridis_r", s=70, edgecolor="k",
                   linewidth=0.4, zorder=3, label=f"eligible (≤{loading_cap:g} g/L)")
    if (~elig).any():
        a.scatter(wd[~elig], pm[~elig], marker="x", s=70, c="0.6", linewidth=1.6, zorder=2,
                  label=f"abstain (>{loading_cap:g} g/L, out of domain)")
    a.scatter([wd[0]], [pm[0]], marker="*", s=300, c="#d62728", edgecolor="k", linewidth=0.5,
              zorder=5, label=f"historical OP ({load[0]:.0f} g/L)")
    if wa_i is not None:
        a.scatter([wd[wa_i]], [pm[wa_i]], marker="s", s=120, facecolor="none", edgecolor="#1f77b4",
                  linewidth=2.0, zorder=4, label=f"widest adequate window ({load[wa_i]:.0f} g/L)")
    a.axvline(1.0, ls="--", c="k", lw=0.9)
    a.text(1.02, 0.03, "determinable\n(worst_dec<1)", fontsize=7.5, color="0.3")
    a.set_xlabel("worst_dec  (decision determinability)")
    a.set_ylabel("P(meet spec | data)")
    a.set_ylim(-0.05, 1.08)
    a.set_title(f"{name}: probabilistic operating-window map (gated)", fontsize=10)
    a.legend(fontsize=7.0, loc="lower left", framealpha=0.92)
    fig.colorbar(sc, ax=a, label="expected regret (eligible)")
    # --- RIGHT: value of information, gated ---
    b = ax[1]
    red = np.array([c["worst_dec_reduction"] for c in cand])
    vload = np.array([c["op"][0] for c in cand])
    vlab = [c["most_informs"] for c in cand]
    order = np.argsort(red)[::-1][:12]
    y = np.arange(len(order))[::-1]
    cols = ["#4c78a8" if vload[i] <= loading_cap else "0.72" for i in order]
    b.barh(y, red[order], color=cols, edgecolor="k", linewidth=0.3)
    for yi, i in zip(y, order):
        tag = vlab[i] + ("" if vload[i] <= loading_cap else " (abstain)")
        b.text(max(red[order].max(), 1e-9) * 0.02, yi, f"{tag}  [{vload[i]:.0f} g/L]", va="center",
               fontsize=6.8, color="k" if vload[i] <= loading_cap else "0.5")
    b.set_yticks([])
    b.set_xlabel("value of information: worst_dec reduction from one experiment")
    b.set_title(f"{name}: value of information (top 12), gated", fontsize=10)
    b.text(0.98, 0.02, f"blue = in-domain\ngrey = abstain (>{loading_cap:g} g/L)",
           transform=b.transAxes, ha="right", va="bottom", fontsize=7, color="0.4")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def replot_product(product, *, in_dir, out_dir, loading_cap, blind) -> None:
    """Regenerate the gated figure from a committed ``{product}_decision_window.json`` --
    no solver, so it reruns anywhere in seconds."""
    rec = json.loads(Path(f"{in_dir}/{product}_decision_window.json").read_text(encoding="utf-8"))
    label = BLIND_LABELS.get(product, product) if blind else product
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{product}_decision_window.png"
    _plot(product, rec["operating_window_map"], rec["voi"], path, loading_cap=loading_cap, label=label)
    print(f"{product}: replotted gated (cap {loading_cap:g} g/L, label {label!r}) -> {path}")


def run_product(product, *, in_dir, out_dir, n_candidates, n_steps, n_samples, spec, seed,
                gate: str = "determinability", predictive: bool = False, disc_json=None,
                sigma_meas=(0.005, 0.008), loading_cap: float = DEFAULT_LOADING_CAP_G_PER_L) -> dict:
    post, base_op, tol, n, prior, bundle = _load_product_setup(product, in_dir)
    extra_cov, suffix = None, ""
    if predictive:
        # PREDICTIVE scan (main text §6.7 / SI §S17): swap in the correlated posterior Sigma_corr and add
        # the decision-level discrepancy + measurement covariance at every op, so P(meet) is the predictive
        # C_g^tot = G Sigma_corr G^T + C_delta + C_meas.  Answers abstain-vs-move-operating-point for mAb C.
        from cex_model.bayes.posterior import Posterior
        post = Posterior.load(f"{in_dir}/{product}_correlated_posterior.npz")
        disc = json.loads(Path(disc_json).read_text(encoding="utf-8"))["products"][product]
        Cd = np.asarray(disc["disc_own"]["C_delta"], float)
        extra_cov = Cd + np.diag(np.asarray(sigma_meas, float) ** 2)
        gate, suffix = "posterior_action", "_predictive"
        print(f"[{product}] PREDICTIVE: Sigma_corr + C_delta(own,n={disc['disc_own']['n_folds']}) + C_meas; "
              f"extra_cov diag={np.diag(extra_cov).tolist()}")
    ops = [list(o) for o in candidate_ops_for(bundle, n_candidates=n_candidates, seed=seed)]
    ops = [base_op] + ops                                       # include the historical decision OP
    # map via the fast linearized Gaussian (one Jacobian per op -> n_steps can match the committed
    # decision pipeline); the MC pushforward is the nonlinearity cross-check at the historical op.
    owm = operating_window_map(post, bundle, ops, spec=spec, tol=tol, method="gauss",
                               n_steps=n_steps, seed=seed, extra_cov=extra_cov, verbose=True)
    voi = decision_voi(post, bundle, base_op, ops[1:], spec=spec, tol=tol, n_steps=n_steps,
                       sigma_obs=post.sigma_obs, verbose=True)
    mc = decision_covariance_mc(post, bundle, base_op, n_samples=n_samples, seed=seed,
                                n_steps=n_steps, return_gs=True)
    mc_check = meet_prob_and_regret(mc.get("gs", []), spec=spec, tol=tol)
    recommendation = synthesize_recommendation(owm, voi, gate=gate)
    # predictive verdict: does ANY in-domain op reach a decisive predictive P(meet)? -> move-op else abstain
    predictive_scan = None
    if predictive:
        indom = [r for r in owm["rows"] if r["op"][0] <= loading_cap + 1e-9]
        best = max(indom, key=lambda r: r["p_meet"]) if indom else None
        decisive = bool(best is not None and best["p_meet"] >= 0.95)
        predictive_scan = {"loading_cap": loading_cap, "n_in_domain": len(indom),
                           "max_in_domain_p_meet": (best["p_meet"] if best else None),
                           "argmax_op": (best["op"] if best else None),
                           "verdict": "move-operating-point" if decisive else "abstain"}
    rec = {"product": product, "decision_op": base_op, "n_steps": n_steps, "spec": list(spec), "gate": gate,
           "predictive": predictive, "predictive_scan": predictive_scan,
           "recommendation": recommendation, "operating_window_map": owm, "voi": voi,
           "mc_crosscheck_at_decision_op": {"p_meet_mc": mc_check["p_meet"], "p_meet_gauss": owm["rows"][0]["p_meet"],
                                            "std_g_mc": mc["std_g"], "n_used": mc_check["n_used"]}}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{product}_decision_window{suffix}.json").write_text(json.dumps(rec, indent=2))
    _plot(product, owm, voi, out / f"{product}_decision_window{suffix}.png", loading_cap=loading_cap)

    print(f"\n=== {product} decision-adequacy deliverable ===")
    if predictive_scan is not None:
        print(f"  [PREDICTIVE] max in-domain P(meet)={predictive_scan['max_in_domain_p_meet']:.3f} "
              f"at op={predictive_scan['argmax_op']} ({predictive_scan['n_in_domain']} in-domain OPs) "
              f"-> {predictive_scan['verdict'].upper()}")
    print(f"  recommended operating window (widest determinate, max P(meet)): {owm['recommended_op']}")
    print(f"    P(meet spec) there = {owm['recommended_p_meet']:.3f}   "
          f"({owm['n_determinate']}/{owm['n_ops']} candidate OPs determinate)")
    print(f"  decision at historical OP: worst_dec={voi['worst_dec_current']:.3f}  "
          f"P(meet)={voi['p_meet_current']:.3f}  state={voi['decision_state']}")
    mcc = rec["mc_crosscheck_at_decision_op"]
    print(f"  MC cross-check at historical OP: P(meet) gauss={mcc['p_meet_gauss']:.3f} vs "
          f"MC={mcc['p_meet_mc']:.3f} (n={mcc['n_used']})")
    print(f"  VoI: best candidate cuts worst_dec by {voi['max_voi_worst_dec']:.4f} -> "
          f"more_data_worthwhile={voi['more_data_worthwhile']}")
    print(f"  RECOMMENDATION [{recommendation['action']}]: {recommendation['why']}")
    print(f"  json -> {out / f'{product}_decision_window.json'}")
    return rec


def run_product_hier_scan(product, *, in_dir, out_dir, n_candidates, n_steps, n_samples, spec, seed,
                          draws_path, sigma_meas=(0.005, 0.008),
                          loading_cap: float = DEFAULT_LOADING_CAP_G_PER_L) -> dict:
    """B2 re-run: predictive operating-window scan INTEGRATED over the hierarchy posterior draws.

    Uses the canonical rho<=0.9 correlated posterior, the Gibbs (b_p, Sigma_delta) draws (from
    bayes_decision_discrepancy_hier.py --dump-draws), nonlinear MC for mAb C (rel-Frob>0.5), and
    FIXES the best in-domain candidate before re-scoring its marginal P(meet), credible interval,
    action frequency, and Pr(move|D) -- no per-draw argmax oracle.
    """
    from cex_model.bayes.posterior import Posterior
    from cex_model.bayes.decision_window import hier_predictive_window_map

    post, base_op, tol, n, prior, bundle = _load_product_setup(product, in_dir)
    post = Posterior.load(f"{in_dir}/{product}_correlated_posterior.npz")
    dec = json.loads(Path(f"{in_dir}/{product}_decision.json").read_text(encoding="utf-8"))
    # An absent crosscheck means the screen was never run for this product, not that the discrepancy is zero.
    # Defaulting it to 0.0 both selects the Gaussian law silently and writes a fabricated diagnostic into the
    # artifact, which is how a stale 0.06 once reached the paper as though it were current.
    _cc = dec.get("crosscheck")
    relf = float(_cc["rel_frobenius"]) if _cc and "rel_frobenius" in _cc else None
    if relf is None:
        print(f"[{product}] WARNING: no linearisation crosscheck in {product}_decision.json; "
              f"defaulting to the Gaussian law and recording rel_frobenius=null")
    method = "mc" if (relf or 0.0) > 0.5 else "gauss"             # nonlinear MC fallback for mAb C
    dr = np.load(draws_path, allow_pickle=True)
    bias_draws = dr[f"b_{product}"]
    sd_draws = dr["Sd_draws"]
    c_meas = np.diag(np.asarray(sigma_meas, float) ** 2)
    ops = [list(o) for o in candidate_ops_for(bundle, n_candidates=n_candidates, seed=seed)]
    print(f"[{product}] HIER-SCAN: rho<=0.9 posterior, {len(sd_draws)} hierarchy draws, "
          f"method={method} (rel-Frob {'n/a' if relf is None else f'{relf:.2f}'}); "
          f"{len(ops)} candidate OPs + historical {base_op}")
    res = hier_predictive_window_map(post, bundle, ops, bias_draws=bias_draws, sd_draws=sd_draws,
                                     decision_op=base_op, spec=spec, tol=tol, c_meas=c_meas,
                                     method=method, n_samples=n_samples, seed=seed, n_steps=n_steps,
                                     load_cap=loading_cap, verbose=True)
    res["product"] = product
    res["n_steps"] = n_steps
    # Provenance: which hierarchy draws, pool and seed produced this scan. Without it a reader cannot tell one
    # discrepancy prior from another after the fact, and two of these files were once silently produced from the
    # low-discrepancy draws while a third used the primary ones.
    res["provenance"] = {"hier_draws": str(draws_path), "n_candidates": int(n_candidates),
                         "seed": int(seed), "n_samples": int(n_samples),
                         "sigma_meas": [float(x) for x in sigma_meas],
                         "loading_cap": float(loading_cap), "spec": [float(x) for x in spec]}
    res["rel_frobenius"] = relf
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{product}_decision_window_predictive_hier.json").write_text(json.dumps(res, indent=2))
    h = res["historical"]; cand = res.get("recommended_candidate")
    print(f"\n=== {product} hierarchy-integrated predictive scan ===")
    print(f"  historical OP {res['decision_op']}: P(meet)={h['p_meet']:.3f} "
          f"[{h['p_meet_ci'][0]:.2f},{h['p_meet_ci'][1]:.2f}]  Pr(decisive|D)={h['p_action_decisive']:.3f}")
    if cand:
        print(f"  FIXED best in-domain candidate {[round(x,1) for x in cand['op']]} ({cand['loading']:.1f} g/L): "
              f"P(meet)={cand['p_meet_marginal']:.3f} [{cand['p_meet_ci'][0]:.2f},{cand['p_meet_ci'][1]:.2f}]  "
              f"Pr(decisive|D)={cand['p_action_decisive']:.3f}  Pr(move|D)={cand['p_move_given_D']:.3f}")
    print(f"  ACTION: {res['action']}  -> {out / f'{product}_decision_window_predictive_hier.json'}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+", default=['HLXSYN'])
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--out-dir", default="results/bayes")
    ap.add_argument("--n-candidates", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=300,
                    help="match the committed decision pipeline (300) so g(MAP) is paper-consistent")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--spec", nargs=2, type=float, default=[0.70, 0.50],
                    help="purity_min yield_min (pipeline defaults, not product GMP values)")
    ap.add_argument("--gate", choices=("determinability", "posterior_action"), default="determinability",
                    help="four-state action gate: 'determinability' (worst_dec<tau, default/unchanged) or "
                         "'posterior_action' (paper §4: driven by P(meet)+regret, worst_dec a diagnostic)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loading-cap", type=float, default=DEFAULT_LOADING_CAP_G_PER_L,
                    help="supported-loading gate for the figure; candidates above it are marked abstain")
    ap.add_argument("--replot", action="store_true",
                    help="regenerate the gated figure(s) from committed {product}_decision_window.json "
                         "(no solver); use to refresh fig_decision_window.png without a Colab run")
    ap.add_argument("--blind", action="store_true",
                    help="title figures with the blinded mAb A-E labels (double-blind submission)")
    ap.add_argument("--predictive", action="store_true",
                    help="PREDICTIVE scan (SI §S17): use the correlated posterior {product}_correlated_"
                         "posterior.npz and add C_delta+C_meas at every op, so P(meet) is C_g^tot; reports "
                         "the abstain-vs-move-operating-point verdict. Requires the OU refit + discrepancy first.")
    ap.add_argument("--disc-json", default="results/bayes/decision_discrepancy_correlated.json",
                    help="discrepancy JSON supplying disc_own.C_delta per product (for --predictive)")
    ap.add_argument("--sigma-meas", nargs=2, type=float, default=[0.005, 0.008],
                    help="observed-QoI measurement std (purity yield) added as C_meas (for --predictive)")
    ap.add_argument("--hier-draws", default=None,
                    help="B2 re-run: path to the Gibbs (b_p, Sigma_delta) draws .npz "
                         "(bayes_decision_discrepancy_hier.py --dump-draws). Runs the hierarchy-INTEGRATED "
                         "predictive scan (nonlinear MC for mAb C; fixed-candidate marginal P(meet), CI, "
                         "action frequency and Pr(move|D)) instead of the posterior-mean plug-in.")
    args = ap.parse_args()
    for product in args.products:
        if args.replot:
            replot_product(product, in_dir=args.in_dir, out_dir=args.out_dir,
                           loading_cap=args.loading_cap, blind=args.blind)
        elif args.hier_draws:
            run_product_hier_scan(product, in_dir=args.in_dir, out_dir=args.out_dir,
                                  n_candidates=args.n_candidates, n_steps=args.n_steps,
                                  n_samples=args.n_samples, spec=tuple(args.spec), seed=args.seed,
                                  draws_path=args.hier_draws, sigma_meas=tuple(args.sigma_meas),
                                  loading_cap=args.loading_cap)
        else:
            run_product(product, in_dir=args.in_dir, out_dir=args.out_dir, n_candidates=args.n_candidates,
                        n_steps=args.n_steps, n_samples=args.n_samples, spec=tuple(args.spec), seed=args.seed,
                        gate=args.gate, predictive=args.predictive, disc_json=args.disc_json,
                        sigma_meas=tuple(args.sigma_meas), loading_cap=args.loading_cap)


if __name__ == "__main__":
    main()
