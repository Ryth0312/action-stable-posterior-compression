"""Posterior-predictive decision-adequacy OPERATING WINDOW + value-of-information (VoI) stopping rule.

This turns the §2.9 decision-adequacy *verdict* (``worst_dec < 1``: "the pooling decision is determined")
into the actionable engineering objects a process referee wants -- computed entirely from the committed
posteriors, **no new wet-lab data**:

  1. A **probabilistic operating-window map** (:func:`operating_window_map`): over a grid of candidate
     operating conditions, push the posterior through the (fast, detached) solver to the pooled-window
     purity/yield decision and report ``P(purity >= spec AND yield >= spec | data)`` + expected regret +
     the linearized ``worst_dec`` -- a GSK-style probabilistic operating space that NAMES the
     decision-adequate window, rather than just certifying one historical window.

  2. A **value-of-information / EVSI stopping number** (:func:`decision_voi`): for each candidate next
     experiment, the preposterior (local-Gaussian) reduction in decision uncertainty -- the expected gain
     in ``P(meet-spec)`` and the drop in ``worst_dec`` -- so the design loop can answer "is another run
     worth it FOR THIS DECISION?".  Because sigma is decision-null and keq<->nu is the only
     decision-relevant ridge, the VoI is ~0 for sigma-informative experiments and concentrates on the
     keq<->nu-resolving one; when even the best candidate's VoI is below threshold the honest, citable
     conclusion is "more data is not worth taking for this decision" (the positive read of the §3.7 null).

This is the local-Gaussian / delta-method EVSI (preposterior decision value under the Laplace posterior),
the same discipline as §2.9's linearized ``worst_dec``; the MC pushforward of (1) is the nonlinearity
cross-check (the rel-Frobenius guard of §2.9 flags where it cannot be trusted).  Reuses ``decision``'s
Jacobian / MC pushforward and ``design``'s candidate pool + EIG ``H``/``J_c`` convention; no new estimator.
"""

from __future__ import annotations

import numpy as np

from cex_model.bayes.decision import (
    DECISION_NAMES,
    decision_covariance,
    decision_covariance_mc,
    decision_jacobian,
)
from cex_model.bayes.design import _param_reductions, candidate_predict_fn, candidate_pool_for
from cex_model.diffpeak.design import OP_FEATURES

__all__ = [
    "meet_prob_and_regret",
    "gaussian_meet_prob",
    "worst_dec_from_cov",
    "operating_window_map",
    "hier_predictive_window_map",
    "decision_voi",
    "synthesize_recommendation",
]

# Project-canonical pooling spec defaults (NOT product GMP values; see §3.7 -- the verdict's *level*
# moves with a product's true spec, the determinability does not). purity >= 0.70 main-group fraction;
# yield spec is user-facing and defaults low so the AND is purity-driven unless a real target is set.
DEFAULT_SPEC: tuple[float, float] = (0.70, 0.50)
DEFAULT_TOL: tuple[float, float] = (0.02, 0.05)

# --- posterior-action gate thresholds (used only when gate="posterior_action" in
# synthesize_recommendation) ---.  Under this gate the action is driven by the posterior meet-probability
# and the tolerance-weighted expected regret, with worst_dec kept as a reported determinability DIAGNOSTIC
# (not the action gate).  The thresholds are pre-registered; the verdicts are insensitive here because
# P(meet) is bimodal (~0/~1) on the study products (the posterior mean sits many std from the spec).
P_MEET_HI: float = 0.90   # P(meet) at/above this = the action is decisively "meets spec"
P_MEET_LO: float = 0.10   # best in-pool P(meet) at/below this = decisively "misses spec" (a level issue)
PMEET_VOI_FLOOR: float = 0.20   # a candidate must move |P(meet)| by at least this for take_data to be worthwhile


def worst_dec_from_cov(C, tol) -> float:
    """``worst_dec = sqrt(lambda_max(T^-1 C T^-1))`` from a decision covariance ``C`` (the §2.9 scalar)."""
    C = np.asarray(C, float)
    dinv = 1.0 / np.asarray(tol, float)
    Ctil = C * dinv[:, None] * dinv[None, :]
    return float(np.sqrt(max(float(np.linalg.eigvalsh(Ctil).max()), 0.0)))


def meet_prob_and_regret(gs, spec=DEFAULT_SPEC, tol=None) -> dict:
    """From decision draws ``gs`` (n x 2 = [purity, yield]) and a ``spec`` (purity_min, yield_min):
    ``P(meet both)`` + expected (tolerance-weighted) shortfall regret + the marginal meet probabilities.

    ``regret`` is the mean over draws of the spec shortfall ``w_p*max(0, p_spec - purity) +
    w_y*max(0, y_spec - yield)`` with ``w = 1/tol`` -- 0 when the draw meets spec, growing as it misses
    (the "redesign the pool" signal for a product whose achievable purity *level* is below spec).
    """
    gs = np.atleast_2d(np.asarray(gs, float))
    pur, yld = gs[:, 0], gs[:, 1]
    ok = np.isfinite(pur) & np.isfinite(yld)
    pur, yld = pur[ok], yld[ok]
    if pur.size == 0:
        return {"p_meet": float("nan"), "expected_regret": float("nan"),
                "p_purity": float("nan"), "p_yield": float("nan"), "n_used": 0}
    wp, wy = (1.0, 1.0) if tol is None else (1.0 / float(tol[0]), 1.0 / float(tol[1]))
    short = wp * np.clip(spec[0] - pur, 0.0, None) + wy * np.clip(spec[1] - yld, 0.0, None)
    return {
        "p_meet": float(np.mean((pur >= spec[0]) & (yld >= spec[1]))),
        "expected_regret": float(np.mean(short)),
        "p_purity": float(np.mean(pur >= spec[0])),
        "p_yield": float(np.mean(yld >= spec[1])),
        "n_used": int(pur.size),
    }


def gaussian_meet_prob(g_map, C, spec=DEFAULT_SPEC) -> float:
    """``P(purity >= spec0 AND yield >= spec1)`` for ``g ~ N(g_map, C)`` (bivariate normal upper orthant).

    Uses ``-g ~ N(-g_map, C)`` so the upper orthant is a standard CDF; a tiny jitter guards a
    near-singular ``C`` (e.g. a degenerate decision direction).  The analytic companion to
    :func:`meet_prob_and_regret` for the preposterior EVSI of :func:`decision_voi`.
    """
    from scipy.stats import multivariate_normal

    g_map = np.asarray(g_map, float)
    C = np.asarray(C, float)
    spec = np.asarray(spec, float)
    C = C + 1e-12 * np.eye(C.shape[0])
    return float(multivariate_normal(mean=-g_map, cov=C, allow_singular=True).cdf(-spec))


def _gauss_regret(g_map, C, spec, tol) -> float:
    """Analytic tolerance-weighted expected shortfall ``sum_q (1/tol_q) E[(spec_q - g_q)_+]`` for
    ``g ~ N(g_map, C)`` -- the Gaussian companion of :func:`meet_prob_and_regret`'s MC regret.
    ``E[(s-g)_+] = sigma*phi(z) + (s-mu)*Phi(z)``, ``z = (s-mu)/sigma``."""
    from scipy.stats import norm

    mu = np.asarray(g_map, float)
    sig = np.sqrt(np.clip(np.diag(np.asarray(C, float)), 1e-18, None))
    spec = np.asarray(spec, float)
    tol = np.asarray(tol, float)
    z = (spec - mu) / sig
    es = sig * norm.pdf(z) + (spec - mu) * norm.cdf(z)
    return float(np.sum(es / tol))


def operating_window_map(posterior, bundle, ops, *, spec=DEFAULT_SPEC, tol=DEFAULT_TOL,
                         method: str = "gauss", n_samples: int = 200, seed: int = 0,
                         n_steps: int = 120, extra_cov=None, verbose: bool = False) -> dict:
    """Probabilistic decision-adequacy over a grid of candidate operating conditions.

    For each ``op`` in ``ops`` (each = ``[loading, gradient_start, gradient_end, elution_cv]``): select the
    pooling window on that op's MAP curve, push the posterior to the decision ``g``, and report
    ``P(meet-spec)`` / expected regret / mean+std purity+yield and the linearized ``worst_dec``.
    ``method="gauss"`` (default) uses the linearized decision Gaussian ``g ~ N(g_map, G Sigma G^T)`` --
    ONE reverse-mode Jacobian per op, so a paper-consistent ``n_steps`` is affordable over a whole pool;
    ``method="mc"`` uses the ``n_samples``-draw nonlinear pushforward (``decision_covariance_mc``), the
    honest cross-check where the linearization is suspect (the §2.9 rel-Frobenius regime).  Returns the
    per-op rows and the **recommended** op = the determinate (``worst_dec < 1``) op with the highest
    ``P(meet-spec)`` (regret tie-break), i.e. the named decision-adequate operating window.

    ``extra_cov`` (optional 2x2): a decision-space covariance ADDED to ``G Sigma G^T`` at every op, turning
    the conditional map into the **predictive** map ``C_g^tot = G Sigma_corr G^T + C_delta + C_meas`` (pass
    ``posterior`` = the correlated posterior and ``extra_cov`` = ``C_delta + C_meas``).  Used to answer
    abstain-vs-move-operating-point under the predictive covariance (main text §6.7 / SI §S17).
    """
    if method not in ("gauss", "mc"):
        raise ValueError(f"method must be 'gauss' or 'mc', got {method!r}")
    extra = None if extra_cov is None else np.asarray(extra_cov, float)
    names = list(DECISION_NAMES)
    rows = []
    for op in ops:
        op = [float(x) for x in op]
        G, g_map, _ = decision_jacobian(bundle, op, posterior.u_map, n_steps=n_steps, return_extra=True)
        C = decision_covariance(G, posterior.cov)
        if extra is not None:                       # predictive: add C_delta + C_meas at every op
            C = C + extra
        wd = worst_dec_from_cov(C, tol)
        if method == "gauss":
            from scipy.stats import norm
            p_meet = gaussian_meet_prob(g_map, C, spec)
            regret = _gauss_regret(g_map, C, spec, tol)
            std = np.sqrt(np.clip(np.diag(C), 0.0, None))
            mean_g = {names[i]: float(g_map[i]) for i in range(len(names))}
            std_g = {names[i]: float(std[i]) for i in range(len(names))}
            p_pur = float(norm.sf((spec[0] - g_map[0]) / max(std[0], 1e-18)))
            p_yld = float(norm.sf((spec[1] - g_map[1]) / max(std[1], 1e-18)))
            n_used = None
        else:
            mc = decision_covariance_mc(posterior, bundle, op, n_samples=n_samples, seed=seed,
                                        n_steps=n_steps, return_gs=True)
            gs = np.asarray(mc.get("gs", []), float)
            if extra is not None and gs.ndim == 2 and gs.shape[0] >= 2:  # inject discrepancy+meas draws
                rng = np.random.default_rng(seed + 1)
                gs = gs + rng.multivariate_normal(np.zeros(gs.shape[1]), extra, size=gs.shape[0])
            mr = meet_prob_and_regret(gs, spec=spec, tol=tol)
            p_meet, regret, p_pur, p_yld = mr["p_meet"], mr["expected_regret"], mr["p_purity"], mr["p_yield"]
            mean_g, std_g, n_used = mc["mean_g"], mc["std_g"], mr["n_used"]
        rows.append({
            "op": op, "p_meet": p_meet, "expected_regret": regret, "p_purity": p_pur, "p_yield": p_yld,
            "worst_dec": wd, "mean_g": mean_g, "std_g": std_g, "n_used": n_used,
            "determinate": bool(np.isfinite(wd) and wd < 1.0),
        })
        if verbose:
            print(f"  op={op}  P(meet)={p_meet:.3f}  regret={regret:.3f}  worst_dec={wd:.3f}  "
                  f"purity={mean_g['pool_purity']:.3f}±{std_g['pool_purity']:.3f}")

    feasible = [r for r in rows if r["determinate"] and np.isfinite(r["p_meet"])]
    pick = max(feasible, key=lambda r: (r["p_meet"], -r["expected_regret"])) if feasible else None
    return {
        "spec": list(spec), "tol": list(tol), "method": method, "n_samples": n_samples,
        "n_ops": len(rows), "rows": rows, "recommended_op": (pick["op"] if pick else None),
        "recommended_p_meet": (pick["p_meet"] if pick else float("nan")),
        "n_determinate": int(sum(r["determinate"] for r in rows)),
    }


def hier_predictive_window_map(posterior, bundle, ops, *, bias_draws, sd_draws, decision_op,
                               spec=DEFAULT_SPEC, tol=DEFAULT_TOL, c_meas=None, method: str = "gauss",
                               n_samples: int = 200, seed: int = 0, n_steps: int = 300,
                               load_cap: float = 35.0, decisive: float = 0.95,
                               verbose: bool = False) -> dict:
    """Predictive operating-window scan that INTEGRATES over the hierarchy posterior draws
    (reviewer B2 re-run), rather than plugging in the posterior-mean ``Sigma_delta``.

    ``bias_draws`` (K x 2) and ``sd_draws`` (K x 2x2) are the Gibbs draws of the product bias ``b_p``
    and the shared discrepancy covariance ``Sigma_delta`` (from ``bayes_decision_discrepancy_hier.py
    --dump-draws``).  Per op we compute the correlated parameter-epistemic base once (``method="gauss"``:
    ``G Sigma_corr G^T``; ``method="mc"``: the nonlinear pushforward covariance from
    ``decision_covariance_mc`` -- the deployed nonlinear fallback for mAb C), then the per-draw conditional
    meet-probability ``P(meet | h) = Phi_2(g_map + b_p^{(k)}; C_par + Sigma_delta^{(k)} + C_meas)`` and the
    **marginal** ``P(meet) = mean_k P(meet | h^{(k)})`` -- matching ``bayes_decision_discrepancy_hier``'s
    integration exactly.

    Selection is on the MARGINAL P(meet) over the IN-DOMAIN pool (loading <= ``load_cap``).  Crucially the
    recommended candidate is FIXED once, then re-scored: we report the marginal P(meet), its per-draw
    credible interval, the action frequency ``Pr(P(meet | h) >= decisive | D)`` at the fixed candidate, and
    the honest move frequency ``Pr(move | D)`` = fraction of draws in which the historical op is NOT decisive
    while the FIXED candidate IS (no per-draw argmax oracle).
    """
    if method not in ("gauss", "mc"):
        raise ValueError(f"method must be 'gauss' or 'mc', got {method!r}")
    bias_draws = np.asarray(bias_draws, float)          # (K,2)
    sd_draws = np.asarray(sd_draws, float)              # (K,2,2)
    K = len(sd_draws)
    Cm = np.zeros((2, 2)) if c_meas is None else np.asarray(c_meas, float)
    names = list(DECISION_NAMES)

    def base_cov_and_gmap(op):
        """(C_par, g_map): correlated parameter-epistemic decision covariance + MAP decision at op."""
        G, g_map, _ = decision_jacobian(bundle, op, posterior.u_map, n_steps=n_steps, return_extra=True)
        if method == "gauss":
            return decision_covariance(G, posterior.cov), np.asarray(g_map, float)
        mc = decision_covariance_mc(posterior, bundle, op, n_samples=n_samples, seed=seed,
                                    n_steps=n_steps, return_gs=True)
        gs = np.asarray(mc.get("gs", []), float)
        C_par = np.cov(gs.T) if gs.ndim == 2 and gs.shape[0] > 2 else decision_covariance(G, posterior.cov)
        return np.asarray(C_par, float), np.asarray(g_map, float)

    def pmeet_draws(op):
        """Per-draw conditional P(meet|h) integrated over the K hierarchy draws at a FIXED op."""
        C_par, g_map = base_cov_and_gmap(op)
        out = np.empty(K)
        for k in range(K):
            out[k] = gaussian_meet_prob(g_map + bias_draws[k], C_par + sd_draws[k] + Cm, spec)
        wd = worst_dec_from_cov(C_par + sd_draws.mean(axis=0) + Cm, tol)
        return out, float(g_map[0]), float(g_map[1]), wd

    rows = []
    for op in ops:
        op = [float(x) for x in op]
        pd, gp, gy, wd = pmeet_draws(op)
        rows.append({"op": op, "loading": op[0], "in_domain": bool(op[0] <= load_cap),
                     "p_meet": float(pd.mean()),
                     "p_meet_ci": [float(np.percentile(pd, 5)), float(np.percentile(pd, 95))],
                     "p_action_decisive": float(np.mean(pd >= decisive)),
                     "worst_dec": wd, "mean_g": {"pool_purity": gp, "pool_yield": gy},
                     "_pd": pd})
        if verbose:
            print(f"  op={op}  P(meet)_marg={pd.mean():.3f} [{np.percentile(pd,5):.2f},{np.percentile(pd,95):.2f}]"
                  f"  Pr(dec|D)={np.mean(pd>=decisive):.2f}  wdec={wd:.3f}  in_domain={op[0]<=load_cap}")

    # historical op per-draw meet
    hist_pd, hgp, hgy, hwd = pmeet_draws([float(x) for x in decision_op])
    # in-domain candidate selection on the MARGINAL, then FIX it
    ind = [r for r in rows if r["in_domain"]]
    pick = max(ind, key=lambda r: r["p_meet"]) if ind else None
    result = {
        "spec": list(spec), "tol": list(tol), "method": method, "n_draws": int(K),
        "load_cap": load_cap, "decisive": decisive, "decision_op": list(decision_op),
        "historical": {"p_meet": float(hist_pd.mean()),
                       "p_meet_ci": [float(np.percentile(hist_pd, 5)), float(np.percentile(hist_pd, 95))],
                       "p_action_decisive": float(np.mean(hist_pd >= decisive)),
                       "mean_g": {"pool_purity": hgp, "pool_yield": hgy}, "worst_dec": hwd},
        "rows": [{k: v for k, v in r.items() if k != "_pd"} for r in rows],
    }
    if pick is not None:
        cand_pd = pick["_pd"]
        move = np.mean((hist_pd < decisive) & (cand_pd >= decisive))
        result["recommended_candidate"] = {
            "op": pick["op"], "loading": pick["loading"],
            "p_meet_marginal": float(cand_pd.mean()),
            "p_meet_ci": [float(np.percentile(cand_pd, 5)), float(np.percentile(cand_pd, 95))],
            "p_action_decisive": float(np.mean(cand_pd >= decisive)),
            "p_move_given_D": float(move),
            "mean_g": pick["mean_g"], "note": "FIXED candidate re-scored (not per-draw argmax oracle)"}
        result["scan_flag"] = ("move-operating-point" if move >= 0.5 else
                            ("operate-as-is" if result["historical"]["p_action_decisive"] >= 0.5 else
                             "prospective-or-abstain"))
    else:
        result["recommended_candidate"] = None
        result["scan_flag"] = "redesign-pool" if result["historical"]["p_meet"] < 0.5 else "abstain"
    return result


def decision_voi(posterior, bundle, decision_op, candidate_ops, *, spec=DEFAULT_SPEC, tol=DEFAULT_TOL,
                 tau_dec: float = 1.0, p_meet_floor: float = 0.8, wd_red_floor: float = 0.05,
                 n_steps: int = 120, n_points: int = 12, sigma_obs: float | None = None,
                 verbose: bool = False) -> dict:
    """Preposterior (local-Gaussian) value of information of each candidate next experiment FOR THE
    DECISION at ``decision_op``.

    The decision lives at ``g ~ N(g_map, C)`` with ``C = G Sigma G^T`` and precision ``H = inv(Sigma)``.
    A candidate experiment ``c`` updates ``H -> H + J_c^T J_c / sigma^2`` (the §2.6 EIG convention), so the
    decision covariance shrinks to ``C_c = G (H + J_c^T J_c/sigma^2)^-1 G^T`` (mean held at ``g_map`` -- the
    preposterior expectation).  The robust per-candidate VoI is the **drop in ``worst_dec``** (the
    determinability gain, always >=0); we also report the preposterior ``P(meet-spec)`` change and which
    parameter the candidate most informs (``design._param_reductions``) -- so "VoI ~ 0 for sigma-informative
    runs, concentrated on the keq<->nu-resolving run" is read directly.

    The verdict separates the two orthogonal questions of §3.7 -- **determinability** (``worst_dec``) vs
    **level/feasibility** (``g_map`` vs spec):
      * ``undetermined`` (``worst_dec >= tau_dec``): a next experiment is worth it iff it materially cuts
        ``worst_dec`` (``more_data_worthwhile``);
      * ``determined_and_met`` (``worst_dec < tau_dec`` and ``P(meet) high``): more data NOT worth it;
      * ``determined_misses_spec`` (``worst_dec < tau_dec`` but ``g_map`` below spec): redesign the pool,
        more data will NOT help -- the "precisely known to miss spec" case (HLXSYN), a *level* issue.
    """
    sigma_obs = float(sigma_obs if sigma_obs is not None else posterior.sigma_obs)
    u_map = np.asarray(posterior.u_map, float)
    names = list(posterior.names)
    G, g_map, _ = decision_jacobian(bundle, decision_op, u_map, n_steps=n_steps, return_extra=True)
    Sigma = np.asarray(posterior.cov, float)
    H = np.linalg.inv(Sigma)
    C0 = G @ Sigma @ G.T
    wd0 = worst_dec_from_cov(C0, tol)
    pmeet0 = gaussian_meet_prob(g_map, C0, spec)

    import torch

    from cex_model.diffsolver.torch_solver import DTYPE
    rows = []
    for op in candidate_ops:
        op = [float(x) for x in op]
        J = torch.autograd.functional.jacobian(
            candidate_predict_fn(bundle, op, n_steps=n_steps, n_points=n_points, u_map=u_map),
            torch.tensor(u_map, dtype=DTYPE)).detach().numpy()
        M = (J.T @ J) / (sigma_obs ** 2)
        Sig_c = np.linalg.inv(H + M)
        C_c = G @ Sig_c @ G.T
        wd_c = worst_dec_from_cov(C_c, tol)
        pmeet_c = gaussian_meet_prob(g_map, C_c, spec)
        top = _param_reductions(H, M, names)[0]                 # (param, fractional std reduction)
        rows.append({
            "op": op, "worst_dec_after": wd_c, "worst_dec_reduction": float(wd0 - wd_c),
            "p_meet_after": pmeet_c, "p_meet_change": float(pmeet_c - pmeet0),
            "most_informs": top[0], "most_informs_reduction": float(top[1]),
        })
        if verbose:
            print(f"  op={op}  Δworst_dec={wd0-wd_c:+.4f}  ΔP(meet)={pmeet_c-pmeet0:+.4f}  "
                  f"informs {top[0]}({top[1]:.0%})")

    max_voi_wd = max((r["worst_dec_reduction"] for r in rows), default=float("nan"))
    max_voi_pmeet = max((abs(r["p_meet_change"]) for r in rows), default=float("nan"))
    determined = bool(np.isfinite(wd0) and wd0 < tau_dec)
    feasible = bool(np.isfinite(pmeet0) and pmeet0 >= p_meet_floor)
    if not determined:
        state = "undetermined"
        worthwhile = bool(np.isfinite(max_voi_wd) and max_voi_wd > wd_red_floor)
        verdict = ("decision not yet determined; the keq<->nu-resolving experiment has positive VoI"
                   if worthwhile else "decision undetermined but no candidate materially cuts worst_dec")
    elif feasible:
        state, worthwhile = "determined_and_met", False
        verdict = "decision determined and met to tolerance; more data is not worth taking for it"
    else:
        state, worthwhile = "determined_misses_spec", False
        verdict = ("decision precisely determined to MISS spec (level issue); redesign the pool -- "
                   "more data will not help")
    return {
        "decision_op": [float(x) for x in decision_op], "spec": list(spec), "tol": list(tol),
        "g_map": {DECISION_NAMES[i]: float(g_map[i]) for i in range(len(DECISION_NAMES))},
        "worst_dec_current": wd0, "p_meet_current": pmeet0, "decision_state": state,
        "max_voi_worst_dec": float(max_voi_wd), "max_voi_pmeet": float(max_voi_pmeet),
        "more_data_worthwhile": worthwhile, "verdict": verdict,
        "candidates": rows, "op_features": list(OP_FEATURES),
    }


def synthesize_recommendation(owm, voi, *, p_meet_floor: float = 0.8,
                              gate: str = "determinability",
                              p_meet_hi: float = P_MEET_HI, p_meet_lo: float = P_MEET_LO,
                              pmeet_voi_floor: float = PMEET_VOI_FLOOR) -> dict:
    """Combine the operating-window map (is there a feasible OP?) with the VoI (would data help?) into one
    actionable recommendation.

    ``gate="determinability"`` (default, behaviour UNCHANGED): the action is gated on the determinability
    verdict ``worst_dec < tau_dec`` carried in ``voi["decision_state"]`` --

      * ``operate_as_is``        -- determined and met at the historical OP;
      * ``move_operating_point`` -- historical OP misses spec but a feasible OP exists in the pool;
      * ``redesign_pool``        -- determined and NO OP in the pool meets spec (a level issue -- the §3.7
                                    precisely-known-to-miss-spec case);
      * ``take_data`` / ``take_data_low_value`` -- the decision is not yet determined.

    ``gate="posterior_action"``: the action is driven by the posterior meet-probability and the
    tolerance-weighted expected regret, with ``worst_dec`` demoted to a reported determinability DIAGNOSTIC.
    ``take_data`` fires only when the action is genuinely *ambiguous* -- ``P(meet)`` between the
    decisive-miss / decisive-met thresholds AND a candidate would move it materially -- rather than merely
    because ``worst_dec >= 1``.  This resolves the over-conservatism of a pure determinability gate when the
    posterior mean sits far from the action boundary (§4): a decision can fail the fixed precision tolerance
    yet have an effectively certain optimal action.  The three main products land identically to the
    determinability gate, but the label is now robust to a lost determinability certificate.
    """
    rec_pmeet = owm.get("recommended_p_meet", float("nan"))
    rec_op = owm.get("recommended_op")
    has_feasible_op = bool(rec_op is not None and np.isfinite(rec_pmeet) and rec_pmeet >= p_meet_floor)
    wd0 = voi.get("worst_dec_current", float("nan"))

    if gate == "posterior_action":
        pmeet0 = voi.get("p_meet_current", float("nan"))
        max_voi_pmeet = voi.get("max_voi_pmeet", float("nan"))
        decisive_feasible = bool(rec_op is not None and np.isfinite(rec_pmeet) and rec_pmeet >= p_meet_hi)
        if np.isfinite(pmeet0) and pmeet0 >= p_meet_hi:
            state, worthwhile, action = "met", False, "operate_as_is"
            why = (f"posterior action decisive (P(meet)={pmeet0:.2f} at the historical OP); meets spec, "
                   f"more data is not worth taking (worst_dec={wd0:.2f}, diagnostic)")
        elif decisive_feasible:
            state, worthwhile, action = "feasible_elsewhere", False, "move_operating_point"
            why = (f"historical OP misses spec (P(meet)={pmeet0:.2f}) but a decisive feasible window exists "
                   f"(P(meet)={rec_pmeet:.2f} at {[round(x, 1) for x in rec_op]}); move the operating point")
        elif not (np.isfinite(rec_pmeet) and rec_pmeet > p_meet_lo):
            state, worthwhile, action = "misses_all", False, "redesign_pool"
            why = ("posterior action decisive: NO operating condition in the pool meets spec with "
                   "meaningful probability; redesign the pool / process (a level issue, not a data issue)")
        else:
            state = "ambiguous"
            worthwhile = bool(np.isfinite(max_voi_pmeet) and max_voi_pmeet > pmeet_voi_floor)
            action = "take_data" if worthwhile else "take_data_low_value"
            why = ("posterior action ambiguous; a candidate experiment materially moves P(meet) -- take data"
                   if worthwhile else
                   "posterior action ambiguous but no candidate materially moves P(meet)")
        return {"action": action, "why": why, "decision_state": state, "gate": gate,
                "recommended_op": rec_op, "recommended_p_meet": rec_pmeet,
                "worst_dec_diagnostic": wd0, "more_data_worthwhile": worthwhile}

    # --- determinability gate (default; behaviour unchanged) ---
    state = voi["decision_state"]
    if state == "undetermined":
        action = "take_data" if voi["more_data_worthwhile"] else "take_data_low_value"
        why = ("decision not yet determined; run the highest-VoI (keq<->nu-resolving) experiment"
               if voi["more_data_worthwhile"] else
               "decision undetermined but no single candidate materially cuts worst_dec")
    elif state == "determined_and_met":
        action, why = "operate_as_is", "decision determined and met at the historical OP; no more data needed"
    elif has_feasible_op:
        action = "move_operating_point"
        why = (f"historical OP misses spec but the decision is determined; a feasible operating window "
               f"exists (P(meet)={rec_pmeet:.2f} at {[round(x, 1) for x in rec_op]}) -- move the operating "
               f"point, more data is not needed")
    else:
        action = "redesign_pool"
        why = ("the decision is determined and NO operating condition in the pool meets spec; redesign "
               "the pool / process -- more data will not help (a level issue, not a data issue)")
    return {"action": action, "why": why, "decision_state": state,
            "recommended_op": rec_op, "recommended_p_meet": rec_pmeet,
            "more_data_worthwhile": voi["more_data_worthwhile"]}


def candidate_ops_for(bundle, *, n_candidates: int = 24, seed: int = 0):
    """The shared safe-bounds Sobol operating-condition pool (thin re-export of ``candidate_pool_for``)."""
    return candidate_pool_for(bundle, n_candidates=n_candidates, seed=seed)
