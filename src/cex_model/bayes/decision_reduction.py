"""Decision-design invariance under the σ common-mode reduction (Arm D).

§3.11 (``sigma_ablation``) establishes that compressing σ to its single data-identified common-capacity
mode (Arm D) preserves the decision **outcome** *at the historical operating point*. This module asks the
next question the operating-window/VoI deliverable (§2.12) raises — **does the experiment-design layer
itself move under the reduction?** It reruns the two §2.12 objects — the probabilistic operating-window
map and the value-of-information stopping rule — under Arm D's reduced posterior and checks whether the
**recommended operating window, the four-state recommendation, the decision state, and the decision
covariance** are invariant vs the full model (Arm A).

The honest, real-data-supportable claim is *invariance*: because the pooling decision is already
determined from the existing sparse data (§3.9, ``worst_dec`` 0.29–0.52, every candidate's VoI ≤ 0.07),
the operative statement is not "reallocate experiments from σ to keq↔ν" — that σ-vs-keq VoI *contrast* is
null on real products, since no candidate is worth running at all — but "the design does not change under
the reduction": one can field a smaller, ~34–122× better-conditioned model (Arm D) *without re-planning
experiments*. The four discrete design outputs are the **gated** invariants; the per-candidate VoI ranking
(all near-zero on a determined decision) is reported as a soft diagnostic, never gated — the same
discrete-outcome-vs-continuous-magnitude discipline as ``sigma_ablation.verdict``.

Reuses ``sigma_ablation``'s affine-reduction algebra (``build_arm`` / ``_at_map_cov``: the reduced
posterior is a pure block of the committed Hessian, no solver) and the §2.12 ``decision_window``
deliverable unchanged; no new estimator, no change to any existing module.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from cex_model.bayes.decision import decision_covariance, decision_jacobian
from cex_model.bayes.decision_window import (
    candidate_ops_for,
    decision_voi,
    operating_window_map,
    synthesize_recommendation,
)
from cex_model.bayes.likelihood import AKTA_NOISE_FLOOR_G_L, mechanistic_model
from cex_model.bayes.posterior import Posterior, map_fit
from cex_model.bayes.prior import components_to_u, physical_prior
from cex_model.bayes.sigma_ablation import _at_map_cov, _jacobian_full, _psd_regularize, build_arm
from cex_model.diffsolver.calibrate_diff import targets_from_bundle

__all__ = [
    "DEFAULT_GATE",
    "design_layer",
    "compare_invariance",
    "run_reduction",
    "run_product",
    "run_synthetic",
]

# The decision-covariance equivalence gate — the §3.11 relative-Frobenius threshold, so a design that
# passes here is invariant on the same axis Arm D already clears at the historical op.
DEFAULT_GATE: float = 0.25

# The at-MAP reductions this module compares against the full model. Both are pure block algebra on the
# committed Hessian (no solver); D is the paper's clean lever, B the blunt freeze kept for contrast.
_REDUCTIONS: tuple[str, ...] = ("D_sigma_commonmode",)


def _op_match(a, b, *, atol: float = 1e-6) -> bool:
    """Two operating conditions are the same window (both ``None`` counts as invariant)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=atol))


def design_layer(post, bundle, op, ops, *, tol, spec, n_steps, engine="?") -> dict:
    """Run the §2.12 deliverable (operating-window map + VoI stopping + four-state recommendation) under
    one posterior and return the design-layer summary.

    A reduced arm's full-dim ``Σ = A·cov_free·Aᵀ`` is rank-deficient (frozen σ directions have zero
    variance), so ``decision_voi``'s ``inv(Σ)`` needs the same ``_psd_regularize`` floor the ablation's
    ``--recommend`` path uses (a frozen direction is *known* ⇒ near-infinite precision ⇒ no experiment
    informs it; the decision quantities ``C = GΣGᵀ`` are unchanged to the 1e-12 floor).
    """
    post = replace(post, cov=_psd_regularize(post.cov))
    owm = operating_window_map(post, bundle, ops, spec=tuple(spec), tol=tuple(tol),
                               method="gauss", n_steps=n_steps)
    voi = decision_voi(post, bundle, list(op), ops[1:], spec=tuple(spec), tol=tuple(tol), n_steps=n_steps)
    rec = synthesize_recommendation(owm, voi)
    cand = voi["candidates"]
    top = max(cand, key=lambda r: r["worst_dec_reduction"]) if cand else None
    return {
        "engine": engine,
        "action": rec["action"],
        "decision_state": voi["decision_state"],
        "worst_dec_current": voi["worst_dec_current"],
        "p_meet_current": voi["p_meet_current"],
        "recommended_op": owm["recommended_op"],
        "recommended_p_meet": owm["recommended_p_meet"],
        "n_determinate": owm["n_determinate"],
        "max_voi_worst_dec": voi["max_voi_worst_dec"],
        "more_data_worthwhile": voi["more_data_worthwhile"],
        "top_candidate_op": (top["op"] if top else None),
        "top_candidate_reduction": (top["worst_dec_reduction"] if top else float("nan")),
        "top_candidate_informs": (top["most_informs"] if top else None),
    }


def compare_invariance(base: dict, red: dict, rel_cov_frobenius: float, *, gate: float = DEFAULT_GATE) -> dict:
    """Is the reduced arm's design layer invariant vs the full model?

    **Gated** (the actionable design outputs): the four-state recommendation, the decision state, the
    decision-covariance relative Frobenius (< ``gate``), and the recommended operating window **only when
    it is operative** — i.e. when the action is ``move_operating_point``, the one state where you would
    actually move to ``recommended_op``. For ``operate_as_is`` / ``redesign_pool`` / ``take_data`` the
    operative locus is the historical op / pool-level, so the widest-window argmax is a **non-operative**
    output: under a saturated ``P(meet)`` plateau (many determinate ops at ``P(meet) ≈ 1``, and both arms
    share the same MAP so ``g_map(op)`` is identical — only the ≤``gate`` covariance differs) its argmax
    is a near-arbitrary tie-break, and gating on it would report a design change where the *recommendation*
    is unchanged. So a window move under a non-operative action is recorded as ``window_argmax_flip``, not a
    hard fail. **Soft** (reported, never gated): whether the top-VoI candidate op matches — on a determined
    decision every VoI is ≈ 0, so the "top" candidate is near-arbitrary too.
    """
    window_match = _op_match(red["recommended_op"], base["recommended_op"])
    window_operative = "move_operating_point" in (base["action"], red["action"])
    reasons = []
    if red["action"] != base["action"]:
        reasons.append("action_changed")
    if red["decision_state"] != base["decision_state"]:
        reasons.append("decision_state_changed")
    if window_operative and not window_match:
        reasons.append("window_changed")
    if not (np.isfinite(rel_cov_frobenius) and rel_cov_frobenius < gate):
        reasons.append("cov_frobenius")
    return {
        "verdict": "DESIGN_INVARIANT" if not reasons else "DESIGN_CHANGED",
        "reasons": reasons,
        "rel_cov_frobenius": float(rel_cov_frobenius),
        "gate": float(gate),
        "action_invariant": red["action"] == base["action"],
        "decision_state_invariant": red["decision_state"] == base["decision_state"],
        "window_invariant": window_match,
        "window_operative": window_operative,
        # non-operative widest-window argmax reshuffle (saturated P(meet) plateau) — reported, not gated:
        "window_argmax_flip": bool(not window_match and not window_operative),
        "top_candidate_op_match": _op_match(red["top_candidate_op"], base["top_candidate_op"]),
    }


def run_reduction(bundle, prior, n, *, op, tol=(0.02, 0.05), spec=(0.70, 0.50),
                  reductions=_REDUCTIONS, n_steps=120, n_candidates=12, seed=0, gate=DEFAULT_GATE,
                  map_iters=150, lr=0.05, sigma_obs=AKTA_NOISE_FLOOR_G_L, product="?",
                  targets=None, post_A=None, verbose=False) -> dict:
    """Design-invariance of each at-MAP σ reduction vs the full model, on one product / synthetic bundle.

    Arm A (full) is reused from a committed ``post_A`` (a ``{product}_posterior.npz``; arm A then
    byte-matches the paper) or fit fresh. Each reduction's posterior is a block of the committed Hessian
    ``H_A`` (no solver, via ``sigma_ablation._at_map_cov``). The §2.12 design layer is rerun under each and
    compared to arm A. Returns a JSON-ready dict.
    """
    tmpl = bundle.components
    P = prior.precision()

    t0 = time.time()
    if post_A is not None:                            # reuse committed posterior -> no fit, no forward model
        u_map_A = np.asarray(post_A.u_map, float)
        cov_A = np.asarray(post_A.cov, float)
        reused = True
    else:                                             # fresh fit needs the forward model
        if targets is None:
            targets = targets_from_bundle(bundle, n_steps=n_steps)
        predict_fn, obs = mechanistic_model(targets, n)
        u_map_A, _ = map_fit(components_to_u(tmpl), predict_fn, obs, prior,
                             sigma_obs=sigma_obs, iters=map_iters, lr=lr, progress=verbose)
        J_A = _jacobian_full(predict_fn, u_map_A)
        H = (J_A.T @ J_A) / sigma_obs ** 2 + P
        cov_A = np.linalg.inv(0.5 * (H + H.T) + 1e-9 * np.eye(H.shape[0]))
        reused = False
    H_A = 0.5 * (np.linalg.inv(cov_A) + np.linalg.inv(cov_A).T)
    F_A = H_A - P                                     # data GGN/Fisher (for the σ-prior collapse, if used)
    rt_A = time.time() - t0

    ops = [list(op)] + list(candidate_ops_for(bundle, n_candidates=n_candidates, seed=seed))
    G, _g_map, _window = decision_jacobian(bundle, op, u_map_A, n_steps=n_steps, return_extra=True)
    C_A = decision_covariance(G, cov_A)

    post_A_obj = Posterior.from_template(mean=u_map_A, cov=cov_A, u_map=u_map_A, prior=prior,
                                         template=tmpl, sigma_obs=sigma_obs, engine="reduction:A_full")
    design_A = design_layer(post_A_obj, bundle, op, ops, tol=tol, spec=spec, n_steps=n_steps,
                            engine="A_full")
    if verbose:
        print(f"  A_full              action={design_A['action']}  state={design_A['decision_state']}  "
              f"window={_fmt_op(design_A['recommended_op'])}  maxVoI={design_A['max_voi_worst_dec']:.3f}")

    arms = {"A_full": {"design": design_A, "reused_posterior": reused, "runtime_s": float(rt_A)}}
    for name in reductions:
        t0 = time.time()
        arm = build_arm(name, n, u_map_A, prior)
        Sigma_R, _H_free = _at_map_cov(name, arm, H_A, F_A, prior, n)
        post_R = Posterior.from_template(mean=u_map_A, cov=Sigma_R, u_map=u_map_A, prior=prior,
                                         template=tmpl, sigma_obs=sigma_obs, engine=f"reduction:{name}")
        design_R = design_layer(post_R, bundle, op, ops, tol=tol, spec=spec, n_steps=n_steps, engine=name)
        C_R = decision_covariance(G, Sigma_R)
        rel_frob = float(np.linalg.norm(C_R - C_A) / max(np.linalg.norm(C_A), 1e-30))
        inv = compare_invariance(design_A, design_R, rel_frob, gate=gate)
        arms[name] = {"design": design_R, "invariance": inv, "runtime_s": float(time.time() - t0)}
        if verbose:
            print(f"  {name:18s} action={design_R['action']}  state={design_R['decision_state']}  "
                  f"window={_fmt_op(design_R['recommended_op'])}  relFrob={rel_frob:.3f}  -> {inv['verdict']}")

    return {
        "product": product, "n_protein": n, "n_steps": n_steps, "gate": float(gate),
        "op": [float(x) for x in op], "tol": [float(x) for x in tol], "spec": [float(x) for x in spec],
        "n_candidates": int(n_candidates), "reused_posterior": bool(post_A is not None), "arms": arms,
        "design_invariant": [name for name in reductions
                             if arms[name]["invariance"]["verdict"] == "DESIGN_INVARIANT"],
    }


def _fmt_op(op):
    return None if op is None else [round(float(x), 1) for x in op]


def run_product(product, *, in_dir="results/bayes", n_steps=120, reductions=_REDUCTIONS,
                n_candidates=12, seed=0, gate=DEFAULT_GATE, reuse_posterior=True, verbose=False) -> dict:
    """Design-invariance for a real product at its committed decision OP + tol (``{product}_decision.json``).

    Loads the committed ``{product}_posterior.npz`` for arm A (byte-matches the paper); use ``--n-steps``
    matching the committed fit (production posteriors are n_steps=300). Mirrors ``sigma_ablation.run_product``.
    """
    import json
    from pathlib import Path

    import cex_model.app_support as A
    from cex_model.bayes.compare import filter_experiments
    from cex_model.bayes.groenwall import _PRODUCT_MAP

    in_dir = Path(in_dir)
    dec = json.loads((in_dir / f"{product}_decision.json").read_text())
    op = [float(x) for x in dec["decision_op"]]
    tol = tuple(dec.get("tol", (0.02, 0.05)))
    prod, drop = _PRODUCT_MAP.get(product, (product, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
    n = bundle.components.n_protein
    prior = physical_prior(n)
    post_A = None
    if reuse_posterior:
        npz = in_dir / f"{product}_posterior.npz"
        if npz.exists():
            post_A = Posterior.load(npz)
        elif verbose:
            print(f"  [warn] {npz} absent -> fitting arm A fresh")
    return run_reduction(bundle, prior, n, op=op, tol=tol, reductions=reductions, n_steps=n_steps,
                         n_candidates=n_candidates, seed=seed, gate=gate, product=product,
                         post_A=post_A, verbose=verbose)


def run_synthetic(*, n_comp=3, fractions=(18.0, 70.0, 12.0), keq_ladder=True, loading=10.0, n_steps=40,
                  n_points=60, n_exp=3, map_iters=120, reductions=_REDUCTIONS, n_candidates=8, seed=0,
                  gate=DEFAULT_GATE, verbose=False) -> dict:
    """Design-invariance on a synthetic SMA bundle (CI-friendly). Mirrors ``sigma_ablation.run_synthetic``."""
    from cex_model.bayes.active import simulate_target
    from cex_model.bayes.synthetic import synthetic_sma_bundle

    bundle, comps, u_true = synthetic_sma_bundle(n_comp, fractions=list(fractions), keq_ladder=keq_ladder,
                                                 loading=loading)
    n = comps.n_protein
    e0 = bundle.experiments[0]
    base = [e0.loading_g_l, e0.gradient_start_pct, e0.gradient_end_pct, e0.elution_cv]
    deltas = [(0.0, 0.0, 0.0), (4.0, 5.0, 5.0), (-3.0, -5.0, -5.0), (6.0, 10.0, 0.0)][:max(1, n_exp)]
    ops = [[base[0] + dl, base[1] + dg0, base[2] + dg1, base[3]] for (dl, dg0, dg1) in deltas]
    targets = [simulate_target(bundle, op, u_true, n_steps=n_steps, n_points=n_points,
                               sigma_obs=AKTA_NOISE_FLOOR_G_L, seed=seed + i) for i, op in enumerate(ops)]
    prior = physical_prior(n)
    return run_reduction(bundle, prior, n, op=ops[0], tol=(0.02, 0.05), reductions=reductions,
                         n_steps=n_steps, n_candidates=n_candidates, seed=seed, gate=gate,
                         map_iters=map_iters, product=f"SYN{n_comp}", targets=targets, verbose=verbose)
