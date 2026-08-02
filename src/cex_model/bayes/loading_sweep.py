"""Loading-sweep test of the SMA σ decision-null THEOREM (docs/decision_null_theorem.md §5).

Theorem 1 (Common-Mode Capacity duality): σ enters the dynamics only through the shared
available-charge term ``Λ̄`` (``torch_solver.rhs``), so its decision sensitivity is the pure
``O(ε)`` shared-capacity channel with the single structural small parameter

    ε = max_j γ_j Q_j / Λ̄          (per-molecule loading fraction; ``bayes_loading_fraction``)

whereas ν, keq carry ``O(1)`` DIRECT (selectivity) channels.  Solver-only (no wet-lab)
falsifiable prediction, tested here by sweeping the DECISION-window loading on a real product
(raising loading ⇒ raising ε) and recomputing ε(L) and the decision Jacobian G(L):

    * the tolerance+prior-whitened σ decision-sensitivity ``‖B_σ‖`` is ~LINEAR in ε through the
      origin; ν, keq sensitivities stay ~FLAT (``O(1)``) ⇒ σ-decision-share ~ ``ε²``.

Scope: this is the DECISION-null half (Theorem 1.2/1.4).  ``worst_dir`` is a property of the
committed posterior ``Σ`` (the fitting data), hence CONSTANT across a decision-loading sweep
and reported only for reference; the Fisher-null half (Theorem 1.1, ``worst_dir → 1`` as the
FITTING ε → 0) needs a refit-per-loading synthetic study (the natural extension).

Pure helpers (``loading_grid`` / ``linear_fit`` / ``scaling_verdict``) are solver-free and
unit-tested; ``sweep_product`` is the heavy driver (reuses ``decision_jacobian`` +
``decompose_decision`` + the ε state-integration of ``bayes_loading_fraction``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import cex_model.app_support as A
from cex_model.bayes.active import _sim_for_op
from cex_model.bayes.compare import filter_experiments
from cex_model.bayes.decision import decision_forward, decision_jacobian
from cex_model.bayes.likelihood import unpack_u
from cex_model.bayes.posterior import Posterior
from cex_model.bayes.prior import physical_prior
from cex_model.bayes.spectral_transfer import decompose_decision
from cex_model.bayes.synthetic import synthetic_sma_bundle
from cex_model.diffsolver.torch_solver import DTYPE

__all__ = [
    "loading_grid",
    "linear_fit",
    "scaling_verdict",
    "row_sensitivities",
    "decision_null_ratio",
    "epsilon_at_op",
    "sweep_product",
    "sweep_capacity",
    "sweep_sigma_channel",
]

# pseudo-product -> (real id for load_product, experiment-name substrings to drop). Mirrors
# bayes_loading_fraction.py / bayes_spectral_transfer.py so HLXSYN uses the excl-DT cohort.
_PRODUCT_MAP = {"HLXSYN": ("HLXSYN", ["DT"])}
_ROWS = ("keq", "kkin", "nu", "sigma")
DEFAULT_FACTORS = (0.5, 0.7, 1.0, 1.4, 2.0, 2.8)
DEFAULT_CAPACITY_SCALES = (0.85, 0.925, 1.0, 1.075, 1.15)   # GENTLE Λ₀ multipliers (±15%): beyond
# this the Λ̄^ν (ν≈10) coupling reshapes/degenerates the chromatogram, so this is the usable window.
DEFAULT_SIGMA_SCALES = (0.25, 0.5, 1.0, 1.5, 2.0)   # σ-channel coupling multipliers (state frozen)


# --------------------------------------------------------------------------- pure helpers
# (solver-free, unit-tested in tests/test_bayes_loading_sweep.py)

def loading_grid(base_loading: float, factors, lo: float, hi: float) -> list[float]:
    """Multiplicative loading grid around ``base_loading``, clamped to [lo, hi], de-duplicated."""
    return sorted({round(float(np.clip(base_loading * f, lo, hi)), 6) for f in factors})


def linear_fit(x, y) -> dict:
    """Least-squares ``y ≈ slope·x + intercept`` with R²; NaNs if degenerate (<2 pts / no x spread)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2 or np.ptp(x) == 0.0:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": int(x.size)}
    coef = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
    slope, intercept = float(coef[0]), float(coef[1])
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": slope, "intercept": intercept, "r2": r2, "n": int(x.size)}


def _rel_span(y) -> float:
    """(max−min)/|mean| of a series — the fractional range over the sweep."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size < 2 or y.mean() == 0.0:
        return float("nan")
    return float((y.max() - y.min()) / abs(y.mean()))


def scaling_verdict(eps, sigma_sens, nu_sens, keq_sens, *, r2_min: float = 0.9,
                    span_ratio_min: float = 2.0, intercept_frac_max: float = 0.25) -> dict:
    """Classify the swept sensitivities against Theorem 1's STRICT ``O(ε)`` isolation signature.

    This is the *strict, unconfounded* signature only: STRICT iff (a) ``‖B_σ‖`` is linear in ε
    (R² ≥ ``r2_min``, positive slope, near-origin intercept) AND (b) σ varies far more
    (relatively) over the sweep than ν and keq (rel-span(σ) ≥ ``span_ratio_min`` · max(ν, keq)).
    PARTIAL if only one holds, NONE if neither.  NOTE: on a REAL loading sweep (b) is expected
    to be confounded — raising loading also overloads the column and suppresses ν/keq, so loading
    is not a clean ε-knob.  The headline decision-null claim is judged separately by the σ-share
    being small across the sweep (see ``sweep_product``'s ``decision_null_robust``).  Thresholds
    are guidance on noisy real data, not a hard gate.
    """
    fit = linear_fit(eps, sigma_sens)
    sig = np.asarray(sigma_sens, float)
    smax = float(np.nanmax(np.abs(sig))) if sig.size else float("nan")
    intercept_ok = bool(np.isfinite(fit["intercept"]) and smax > 0
                        and abs(fit["intercept"]) <= intercept_frac_max * smax)
    linear_ok = bool(np.isfinite(fit["r2"]) and fit["r2"] >= r2_min and fit["slope"] > 0 and intercept_ok)

    rs_sig, rs_nu, rs_keq = _rel_span(sigma_sens), _rel_span(nu_sens), _rel_span(keq_sens)
    other = float(np.nanmax([rs_nu, rs_keq]))
    span_ok = bool(np.isfinite(rs_sig) and np.isfinite(other) and rs_sig >= span_ratio_min * max(other, 1e-9))

    verdict = "STRICT" if (linear_ok and span_ok) else ("PARTIAL" if (linear_ok or span_ok) else "NONE")
    return {"verdict": verdict, "linear_ok": linear_ok, "span_ok": span_ok,
            "sigma_linear_fit": fit, "rel_span": {"sigma": rs_sig, "nu": rs_nu, "keq": rs_keq}}


def decision_null_ratio(row_sens: dict, epsilon: float) -> dict:
    """Propagation-constant-free MEASURABLE witness of the O(ε) decision-null (Theorem 1(ii), §2.11):
    the σ decision channel relative to the O(1) selectivity (ν) channel, ``r = ‖B_σ‖/‖B_ν‖``.

    By Theorem 1, ``‖B_σ‖ = O(ε)`` (capacity-only forcing) while ``‖B_ν‖`` carries an O(1) direct channel,
    so ``r = O(ε)`` — and the σ-independent variational-flow gain (the open ``κ_S``/``κ_eff``) **cancels in
    the ratio**, so ``r`` is a ratio of two committed-``Σ``+``G`` observables with **no propagation
    constant**.  This is the right measurable form of the decision-null, in contrast to the operator-norm
    bound ``‖B e_σ‖ ≤ ε·‖B‖`` which is vacuous because ``‖B‖`` is dominated by the decades-wide log-keq
    directions (the same reason the submultiplicative ``worst_dec ≤ ‖B‖·worst_dir`` bound of §2.10 is
    vacuous).  ``r²`` predicts the σ decision-share (the O(ε²) of §3.7).  ``row_sens`` = the ``row_sens``
    field of :func:`row_sensitivities`.
    """
    bsig, bnu = float(row_sens["sigma"]), float(row_sens["nu"])
    r = bsig / bnu if bnu > 0 else float("nan")
    eps = float(epsilon)
    return {"ratio_sigma_over_nu": r, "ratio_over_eps": (r / eps if eps > 0 else float("nan")),
            "sigma_share_predicted": r * r, "epsilon": eps,
            "B_sigma": bsig, "B_nu": bnu}


def row_sensitivities(G, prior_std, tol, n: int) -> dict:
    """Tolerance+prior-whitened per-SMA-row decision sensitivity ``‖B_row‖`` and σ-share —
    identical convention to ``bayes_spectral_transfer`` (``B = (1/tol)·G·σ_prior``)."""
    G = np.atleast_2d(np.asarray(G, float))
    prior_std = np.asarray(prior_std, float)
    B = (1.0 / np.asarray(tol, float))[:, None] * G * prior_std[None, :]
    row_sens = {r: float(np.linalg.norm(B[:, i * n:(i + 1) * n])) for i, r in enumerate(_ROWS)}
    tot2 = sum(v * v for v in row_sens.values()) or 1.0
    raw = {r: float(np.linalg.norm(G[:, i * n:(i + 1) * n])) for i, r in enumerate(_ROWS)}
    return {"row_sens": row_sens, "sigma_share": row_sens["sigma"] ** 2 / tot2, "raw_G_row_norm": raw}


# ----------------------------------------------------------------------------- solver path

def _run_full_state(sim, keq, kkin, nu, sig):
    """Full state (all species, all nodes) at every step — mirror of
    ``bayes_loading_fraction._run_full_state`` (BDF1 then BDF2, no_grad)."""
    ys = [sim.y0]
    y_prev, y_pprev = sim.y0, None
    for k in range(1, sim.t.shape[0]):
        h = sim.t[k] - sim.t[k - 1]
        ic = sim.inlet_c[k]
        a0 = 1.0 if y_pprev is None else 1.5
        hist = y_prev if y_pprev is None else (2.0 * y_prev - 0.5 * y_pprev)
        y = sim._newton(y_prev, ic, a0, hist.detach(), h, keq, kkin, nu, sig, None)
        y_pprev, y_prev = y_prev, y
        ys.append(y)
    return torch.stack(ys)


def epsilon_at_op(sim, u_map, n: int) -> tuple[float, float]:
    """``(max_j γ_j Q_j/Λ̄ , f_sat)`` at the MAP on this OP's simulator — ε and steric occupancy
    of ``bayes_loading_fraction`` recomputed at the swept loading."""
    keq, kkin, nu, sig = unpack_u(torch.tensor(np.asarray(u_map, float), dtype=DTYPE), n)
    nusig = nu + sig
    with torch.no_grad():
        Y = _run_full_state(sim, keq, kkin, nu, sig)
    nc, gs = sim.nc, sim.gs
    Qp = Y.reshape(Y.shape[0], 2 * nc - 1, gs)[:, nc:, :]               # (n_t, npr, gs)
    bound = torch.einsum("j,njg->ng", sim.gamma_p * nusig, Qp)          # Σ_j (ν_j+σ_j) γ_j Q_j
    f_sat = float((bound / sim.inocap).max())
    Lam = (sim.inocap - bound).clamp(min=1e-12)
    gQ = (sim.gamma_p[None, :, None] * Qp) / Lam[:, None, :]
    return float(gQ.amax()), f_sat


def _load_product_setup(product, in_dir):
    """Load Σ/u_map (posterior), the base decision OP + tol, the prior, and the bundle."""
    in_dir = Path(in_dir)
    post = Posterior.load(in_dir / f"{product}_posterior.npz")
    dec = json.loads((in_dir / f"{product}_decision.json").read_text())
    base_op = [float(x) for x in dec["decision_op"]]
    tol = tuple(dec.get("tol", (0.02, 0.05)))
    n = post.n_protein
    prior = physical_prior(n)
    prod, drop = _PRODUCT_MAP.get(product, (product, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
    return post, base_op, tol, n, prior, bundle


def _eval_point(bundle, post, prior, tol, n, op, n_steps, *, inocap_scale: float = 1.0) -> dict:
    """One sweep point: ε(op, Λ₀-scale), G(op, Λ₀-scale), whitened row sensitivities, decomposition."""
    sim = _sim_for_op(bundle, op, n_steps)
    if inocap_scale != 1.0:
        sim.inocap = sim.inocap * inocap_scale
    eps, f_sat = epsilon_at_op(sim, post.u_map, n)
    G, g_map, _ = decision_jacobian(bundle, op, post.u_map, n_steps=n_steps,
                                    inocap_scale=inocap_scale, return_extra=True)
    g_norm = float(np.linalg.norm(np.atleast_2d(np.asarray(G, float))))
    rs = row_sensitivities(G, prior.std, tol, n)
    dd = decompose_decision(post.cov, G, prior.std, tol, names=post.names)
    return {
        "epsilon": eps, "f_sat": f_sat,
        "sigma_sens": rs["row_sens"]["sigma"], "nu_sens": rs["row_sens"]["nu"],
        "keq_sens": rs["row_sens"]["keq"], "kkin_sens": rs["row_sens"]["kkin"],
        "sigma_share": rs["sigma_share"], "raw_G_row_norm": rs["raw_G_row_norm"], "G_norm": g_norm,
        "worst_dir": dd["worst_dir"], "worst_dec": dd["worst_dec"],
        "sloppy_share": (dd["worst_dec_sloppy"] / dd["worst_dec"]) if dd["worst_dec"] > 0 else float("nan"),
        "g_map": {"pool_purity": float(g_map[0]), "pool_yield": float(g_map[1])},
        # G collapses to ~0 when the OP pushes the chromatogram out of a feasible pooling window
        # (e.g. Λ̄^ν over/under-binding under a big capacity change) -> the decision is undefined here.
        "degenerate": bool(g_norm < 1e-8), "ok": True,
    }


def _summarize(product, n, n_steps, tol, base_op, points, share_max, knob) -> dict:
    """Build the record: strict-isolation verdict + headline (decision-null robustness).

    Degenerate points (``G ≈ 0`` -- the OP pushed the chromatogram out of a feasible pooling
    window) are excluded from the fits/verdict and counted in ``n_degenerate``: a knob that
    produces them is, by that fact, not a clean ε-knob.
    """
    good = [p for p in points if p.get("ok") and not p.get("degenerate")]
    n_degenerate = sum(1 for p in points if p.get("ok") and p.get("degenerate"))
    eps = [p["epsilon"] for p in good]
    shares = [p["sigma_share"] for p in good]
    verdict = scaling_verdict(eps, [p["sigma_sens"] for p in good],
                              [p["nu_sens"] for p in good], [p["keq_sens"] for p in good])
    share_fit = linear_fit([e * e for e in eps], shares)  # σ-share ~ ε²

    # Headline = the THEOREM's actual claim (σ is decision-null): the σ-share stays small across the
    # whole swept ε.  The strict O(ε)-isolation signature (σ∝ε, ν/keq flat) is reported separately --
    # NEITHER a loading sweep (overload) NOR a Λ₀ sweep (the Λ̄^ν coupling reshapes/degenerates the
    # chromatogram) isolates ε cleanly; the O(ε) law itself is the exact analytic result of L1.
    sigma_share_max = float(np.nanmax(shares)) if shares else float("nan")
    decision_null_robust = bool(np.isfinite(sigma_share_max) and sigma_share_max <= share_max)
    sigma_rises_with_eps = bool(np.isfinite(share_fit["slope"]) and share_fit["slope"] > 0)
    if not good:
        headline = "INCONCLUSIVE (no non-degenerate points -- knob reshaped the chromatogram)"
    elif decision_null_robust and verdict["verdict"] == "STRICT":
        headline = "CONFIRMED-STRICT (σ decision-null + clean O(ε) isolation)"
    elif decision_null_robust and sigma_rises_with_eps:
        headline = f"CONFIRMED (σ decision-null robust; σ rises with ε; strict-isolation={verdict['verdict']})"
    elif decision_null_robust:
        headline = "CONFIRMED-NULL (σ decision-null robust; no clear ε-trend)"
    else:
        headline = "REFUTED (σ decision-null violated)"

    return {
        "product": product, "n_protein": n, "n_steps": n_steps, "tol": list(tol),
        "knob": knob, "base_decision_op": base_op, "n_degenerate": n_degenerate,
        "worst_dir_constant": good[0]["worst_dir"] if good else float("nan"),
        "headline": headline, "decision_null_robust": decision_null_robust,
        "sigma_share_max": sigma_share_max, "sigma_rises_with_eps": sigma_rises_with_eps,
        "points": points, "scaling": verdict, "sigma_share_vs_eps2_fit": share_fit,
    }


def sweep_product(product, *, in_dir="results/bayes", n_steps: int = 120, factors=DEFAULT_FACTORS,
                  load_min: float = 5.0, load_max: float = 80.0, loadings=None, share_max: float = 0.01,
                  verbose: bool = True) -> dict:
    """LOADING sweep (vary decision-window loading): tests decision-null robustness; strict
    O(ε)-isolation is expected to be confounded by overload (use ``sweep_capacity`` for the clean test)."""
    post, base_op, tol, n, prior, bundle = _load_product_setup(product, in_dir)
    grid = [float(x) for x in loadings] if loadings else loading_grid(base_op[0], factors, load_min, load_max)
    points = []
    for L in grid:
        op = [L, base_op[1], base_op[2], base_op[3]]
        try:
            pt = _eval_point(bundle, post, prior, tol, n, op, n_steps)
            pt["loading_g_l"] = L
            points.append(pt)
            if verbose:
                print(f"  L={L:7.2f}  ε={pt['epsilon']*100:6.3f}%  ‖B_σ‖={pt['sigma_sens']:8.3f}  "
                      f"‖B_ν‖={pt['nu_sens']:8.2f}  ‖B_keq‖={pt['keq_sens']:8.2f}  "
                      f"σ_share={pt['sigma_share']*100:6.3f}%  worst_dec={pt['worst_dec']:.3f}"
                      + ("  [DEGENERATE G≈0]" if pt.get("degenerate") else ""))
        except Exception as exc:  # blown-up loading (e.g. csalt**nu overflow): record + continue
            points.append({"loading_g_l": L, "ok": False, "error": repr(exc)})
            if verbose:
                print(f"  L={L:7.2f}  FAILED: {exc!r}")
    return _summarize(product, n, n_steps, tol, base_op, points, share_max, "loading")


def sweep_capacity(product, *, in_dir="results/bayes", n_steps: int = 120,
                   capacity_scales=DEFAULT_CAPACITY_SCALES, share_max: float = 0.01,
                   verbose: bool = True) -> dict:
    """CAPACITY (Λ₀) sweep at FIXED loading: the CLEAN ε-knob for Theorem 1.

    Scales the column ionic capacity Λ₀ (``inocap_scale``) at the committed decision OP, so
    ε = γQ/Λ̄ varies WITHOUT changing keq/ν selectivity or forcing overload (unlike a loading
    sweep).  This is the isolated test of ``‖B_σ‖ ∝ ε`` (strict O(ε)-isolation)."""
    post, base_op, tol, n, prior, bundle = _load_product_setup(product, in_dir)
    points = []
    for s in sorted({round(float(x), 6) for x in capacity_scales}):
        try:
            pt = _eval_point(bundle, post, prior, tol, n, base_op, n_steps, inocap_scale=s)
            pt["inocap_scale"] = s
            points.append(pt)
            if verbose:
                print(f"  Λ₀×{s:5.2f}  ε={pt['epsilon']*100:6.3f}%  ‖B_σ‖={pt['sigma_sens']:8.3f}  "
                      f"‖B_ν‖={pt['nu_sens']:8.2f}  ‖B_keq‖={pt['keq_sens']:8.2f}  "
                      f"σ_share={pt['sigma_share']*100:6.3f}%  worst_dec={pt['worst_dec']:.3f}"
                      + ("  [DEGENERATE G≈0]" if pt.get("degenerate") else ""))
        except Exception as exc:
            points.append({"inocap_scale": s, "ok": False, "error": repr(exc)})
            if verbose:
                print(f"  Λ₀×{s:5.2f}  FAILED: {exc!r}")
    return _summarize(product, n, n_steps, tol, base_op, points, share_max, "capacity")


def sweep_sigma_channel(*, n_comp: int = 2, nu=4.0, sigma=30.0, loading: float = 8.0,
                        fractions=(70.0, 30.0), keq_ladder=True, scales=DEFAULT_SIGMA_SCALES,
                        n_steps: int = 80, tol=(0.02, 0.05), verbose: bool = True) -> dict:
    """ISOLATED σ-channel test of Theorem 1 on a SYNTHETIC bundle -- the clean STRICT check the
    physical (loading/Λ₀) knobs cannot give.

    σ's only channel is the shared Λ̄, which reshapes the chromatogram via ``Λ̄^ν`` -- so no physical
    knob can move ε while holding the state (hence keq/ν selectivity + window) fixed.  Here we instead
    scale ONLY the σ-channel COUPLING by ``s`` with the forward state FROZEN at the operating point,
    via the reparametrisation ``σ_eff = σ_nom + s·δ`` differentiated at ``δ = 0``: the forward at
    δ=0 uses σ_nom for every ``s`` (state, keq/ν, window all byte-identical), while the propagated
    decision sensitivity ``∂g/∂δ = s·∂g/∂σ``.  Since the variational map is linear in the forcing this
    is EXACT, so ``‖B_σ‖ ∝ ε`` (ε_eff = s·ε_nom) through the origin with keq/ν invariant -- the STRICT
    signature.  Each ``s`` is a real reverse-mode autograd pass through the synthetic BDF solver (the
    deviation from ``s·∂g/∂σ`` is reported as ``max_rel_dev_from_exact``: the implementation/no-hidden-
    nonlinearity check), so this is a genuine computation, not multiplication by ``s``.
    """
    bundle, comps, u_true = synthetic_sma_bundle(n_comp, fractions=list(fractions), keq_ladder=keq_ladder,
                                                 nu=nu, sigma=sigma, loading=loading)
    n = comps.n_protein
    prior = physical_prior(n)
    e0 = bundle.experiments[0]
    op = [e0.loading_g_l, e0.gradient_start_pct, e0.gradient_end_pct, e0.elution_cv]

    # nominal: ε at the operating point + the full decision Jacobian (for the frozen keq/ν rows).
    sim = _sim_for_op(bundle, op, n_steps)
    eps_nom, f_sat = epsilon_at_op(sim, u_true, n)
    G_nom = np.atleast_2d(np.asarray(decision_jacobian(bundle, op, u_true, n_steps=n_steps), float))
    rs_nom = row_sensitivities(G_nom, prior.std, tol, n)
    G_sigma_nom = G_nom[:, 3 * n:4 * n]                       # ∂g/∂σ
    sig_prior = prior.std[3 * n:4 * n]
    tinv = 1.0 / np.asarray(tol, float)

    # reparametrised forward: differentiate w.r.t. δ with σ_eff = σ_nom + s·δ, frozen at δ=0.
    g_fn, _, _ = decision_forward(bundle, op, u_true, n_steps=n_steps)
    u_t = torch.tensor(np.asarray(u_true, float), dtype=DTYPE)
    head, sigma_nom = u_t[:3 * n], u_t[3 * n:4 * n].clone()

    def _G_sigma_at(s: float) -> np.ndarray:
        def g_of_delta(delta):
            return g_fn(torch.cat([head, sigma_nom + s * delta]))
        return torch.autograd.functional.jacobian(
            g_of_delta, torch.zeros(n, dtype=DTYPE)).detach().numpy()   # (k, n) = s·∂g/∂σ

    points = []
    for s in sorted({round(float(x), 6) for x in scales}):
        Gd = np.atleast_2d(_G_sigma_at(s))
        Bsig = tinv[:, None] * Gd * sig_prior[None, :]
        sigma_sens = float(np.linalg.norm(Bsig))
        denom = s * float(np.linalg.norm(G_sigma_nom)) + 1e-30
        dev = float(np.linalg.norm(Gd - s * G_sigma_nom)) / denom        # exactness (impl) check
        points.append({"sigma_scale": s, "epsilon": s * eps_nom, "sigma_sens": sigma_sens,
                       "nu_sens": rs_nom["row_sens"]["nu"], "keq_sens": rs_nom["row_sens"]["keq"],
                       "rel_dev_from_exact": dev, "ok": True})
        if verbose:
            print(f"  σ-coupling×{s:5.2f}  ε_eff={s*eps_nom*100:6.3f}%  ‖B_σ‖={sigma_sens:8.3f}  "
                  f"‖B_ν‖={rs_nom['row_sens']['nu']:8.2f}  ‖B_keq‖={rs_nom['row_sens']['keq']:8.2f}  "
                  f"rel_dev={dev:.1e}")

    eps = [p["epsilon"] for p in points]
    verdict = scaling_verdict(eps, [p["sigma_sens"] for p in points],
                              [p["nu_sens"] for p in points], [p["keq_sens"] for p in points])
    max_dev = max((p["rel_dev_from_exact"] for p in points), default=float("nan"))
    headline = ("STRICT-CONFIRMED (isolated σ-channel: ‖B_σ‖∝ε exact through origin, keq/ν frozen)"
                if verdict["verdict"] == "STRICT" else f"UNEXPECTED ({verdict['verdict']})")
    return {
        "knob": "sigma_channel", "n_comp": n_comp, "nu": nu, "sigma": sigma, "loading": loading,
        "n_steps": n_steps, "tol": list(tol), "eps_nominal": eps_nom, "f_sat": f_sat,
        "sigma_sens_nominal": rs_nom["row_sens"]["sigma"], "nu_sens": rs_nom["row_sens"]["nu"],
        "keq_sens": rs_nom["row_sens"]["keq"], "headline": headline,
        "max_rel_dev_from_exact": max_dev, "points": points, "scaling": verdict,
        "sigma_linear_fit": verdict["sigma_linear_fit"],
    }
