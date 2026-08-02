"""Held-out validation + automatic outlier/OOD diagnostics.

Two reviewer-facing concerns:
- an experiment should be flagged as protocol-mismatch / out-of-distribution by a
  diagnostic, NOT silently dropped to improve coverage (``flag_outlier_experiments``);
- coverage on the SAME experiments used for the fit is a posterior-predictive *check*,
  not predictive validation -- ``leave_one_experiment_out`` does the real thing.

The LOEO driver is heavy (n refits on the real solver) -> Colab; the outlier flag is
solver-free and unit-tested here.
"""

from __future__ import annotations

import numpy as np

__all__ = ["flag_outlier_experiments", "leave_one_experiment_out", "retrospective_eig",
           "retrospective_design", "empirical_sigma_obs", "sigma_obs_sensitivity",
           "n_steps_sensitivity", "prior_sensitivity", "fisher_ablation", "keq_nu_ridge_width",
           "real_design_loop", "design_metrics"]


def flag_outlier_experiments(op_matrix, per_exp_coverage, *, cov_drop: float = 0.30,
                             leverage_z: float = 3.5) -> list[dict]:
    """Diagnose each experiment as OOD / protocol-mismatch (so it is flagged, not hidden).

    Flags experiment k if EITHER (a) its posterior-predictive coverage is far below the
    others (``coverage < median - cov_drop``), OR (b) its operating condition is an
    OP-space outlier -- the robust modified z-score (Iglewicz-Hoaglin) of ANY OP feature
    exceeds ``leverage_z``.  The modified z is ``(x - median) / scale`` with
    ``scale = 1.4826 * MAD``; when ``MAD == 0`` (a feature that is near-constant across a
    small cohort, e.g. a shared gradient program) it falls back to ``1.2533 * MeanAD``
    (the Iglewicz-Hoaglin recommendation), and a truly constant feature contributes 0 --
    so a lone differing experiment gets a *finite, interpretable* z instead of blowing up.
    The per-feature modified z is used rather than a sample-covariance Mahalanobis distance
    because the latter is "masked" by the very outlier it should flag (with few
    experiments the leverage is capped at ``(n-1)²/n``).  Both signals are reported per
    experiment so the with/without analysis can be presented transparently.
    """
    op = np.asarray(op_matrix, float)
    cov = np.asarray(per_exp_coverage, float)
    med = np.median(op, axis=0)
    absdev = np.abs(op - med)
    mad = np.median(absdev, axis=0)
    meanad = absdev.mean(axis=0)
    # robust per-feature scale = 1.4826*MAD; fall back to 1.2533*MeanAD when MAD==0;
    # inf for a constant feature so its modified z is 0 (no spurious blow-up).
    scale = np.where(mad > 0, mad / 0.6745, np.where(meanad > 0, meanad * 1.253314, np.inf))
    lev_z = np.abs((op - med) / scale).max(axis=1)  # worst-feature modified z-score
    med_cov = float(np.median(cov))
    out = []
    for k in range(len(op)):
        cov_out = bool(cov[k] < med_cov - cov_drop)
        op_out = bool(lev_z[k] > leverage_z)
        out.append({"index": int(k), "coverage": float(cov[k]), "op_leverage_z": float(lev_z[k]),
                    "coverage_outlier": cov_out, "op_outlier": op_out, "flagged": cov_out or op_out})
    return out


def _subset_bundle(bundle, idx):
    """A shallow copy of ``bundle`` whose ``.experiments`` is the given index subset."""
    import copy
    b = copy.copy(bundle)
    b.experiments = [bundle.experiments[i] for i in idx]
    return b


def leave_one_experiment_out(bundle, prior, *, n_steps: int = 300, map_iters: int = 150,
                             lr: float = 0.05, predict_samples: int = 100,
                             sigma_obs: float | None = None, u_init=None,
                             seed: int = 0, progress: bool = False):
    """LOEO: fit on n-1 experiments, predict the held-out one on RK23; rotate.

    Returns ``{folds: [...], aggregate: {...}}`` with per-fold held-out coverage / RMSE
    and main-peak errors -- genuine out-of-sample validation. Heavy (n refits) -> Colab.
    ``u_init`` warm-starts every fold's MAP from a committed full-data ``u_map`` (LOEO
    removes one of n experiments, so the n-1 optimum is close): far fewer ``map_iters``
    reach convergence than a from-scratch generic init.
    """
    from cex_model.bayes.likelihood import AKTA_NOISE_FLOOR_G_L, mechanistic_model
    from cex_model.bayes.posterior import laplace_posterior, map_fit
    from cex_model.bayes.predictive import posterior_predictive_rk23
    from cex_model.bayes.prior import components_to_u
    from cex_model.diffsolver.calibrate_diff import targets_from_bundle

    sigma_obs = float(sigma_obs if sigma_obs is not None else AKTA_NOISE_FLOOR_G_L)
    n = bundle.components.n_protein
    u0 = components_to_u(bundle.components) if u_init is None else np.asarray(u_init, float)
    exps = list(range(len(bundle.experiments)))
    folds = []
    for held in exps:
        fit_idx = [i for i in exps if i != held]
        fit_b, test_b = _subset_bundle(bundle, fit_idx), _subset_bundle(bundle, [held])
        predict_fn, obs = mechanistic_model(targets_from_bundle(fit_b, n_steps=n_steps), n)
        u_map, _ = map_fit(u0, predict_fn, obs, prior, sigma_obs=sigma_obs, iters=map_iters, lr=lr)
        post = laplace_posterior(u_map, predict_fn, obs, prior, template=bundle.components,
                                 sigma_obs=sigma_obs)
        res = posterior_predictive_rk23(post, test_b, n_samples=predict_samples, seed=seed)
        e = res["per_experiment"][0]
        folds.append({"held_out": bundle.experiments[held].name, "coverage": e["coverage"],
                      "rmse_total": e["rmse_total"], "rt_err_s": e["rt_err_s"],
                      "height_err_g_l": e["height_err_g_l"], "purity_abs_err": e["purity_abs_err"]})
        if progress:
            print(f"  LOEO held={bundle.experiments[held].name[:30]:30s} "
                  f"coverage={e['coverage']:.3f} rmse={e['rmse_total']:.3f}")
    cov = float(np.mean([f["coverage"] for f in folds]))
    rmse = float(np.mean([f["rmse_total"] for f in folds]))
    return {"folds": folds, "aggregate": {"coverage": cov, "rmse_total": rmse, "n_folds": len(folds)}}


# numpy>=2.0 renamed trapz -> trapezoid; support both (matches predictive.py / decision.py).
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _observed_pool_decision(curve, window_s, main_cols):
    """Pooled main-group purity and window recovery read from a MEASURED curve.

    ``curve`` is the experiment's observed array ``[time, obs1, obs2, ...]`` (g/L) and
    ``main_cols`` the observed-column indices of the MAIN component type.  Integrates over
    the physical-time window ``window_s = (start_s, end_s)`` -- the numpy mirror of
    :func:`cex_model.bayes.decision._pool_quantities`, on the observed ``[time, obs...]``
    layout (no salt column) rather than the simulator's ``[time, salt, comp...]``.  Returns
    ``(purity, recovery)`` or ``(nan, nan)`` if the window holds fewer than two sample points
    (the measured curves are sparsely sampled, so a too-narrow window is not integrable).
    """
    curve = np.asarray(curve, float)
    t = curve[:, 0]
    conc = np.clip(curve[:, 1:], 0.0, None)
    t0, t1 = float(window_s[0]), float(window_s[1])
    mask = (t >= t0) & (t <= t1)
    if int(mask.sum()) < 2:
        return float("nan"), float("nan")
    area = _trapz(conc[mask], t[mask], axis=0)
    total = float(area.sum())
    if total <= 0.0:
        return 0.0, 0.0
    purity = float(area[list(main_cols)].sum()) / total if main_cols else 0.0
    whole = float(_trapz(conc.sum(axis=1), t))
    recovery = total / whole if whole > 0.0 else 0.0
    return purity, recovery


def _rel_frob(C_lin, C_mc) -> float:
    """Relative Frobenius ``||C_mc - C_lin|| / ||C_mc||`` (the linearization validity flag);
    ``nan`` if either matrix is missing or non-finite."""
    if C_mc is None:
        return float("nan")
    A, B = np.asarray(C_lin, float), np.asarray(C_mc, float)
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
        return float("nan")
    denom = float(np.linalg.norm(B))
    return float(np.linalg.norm(B - A) / denom) if denom > 0.0 else float("nan")


def decision_leave_one_experiment_out(bundle, prior, *, tol=(0.02, 0.05), tau_dec: float = 1.0,
                                      n_steps: int = 300, map_iters: int = 150, lr: float = 0.05,
                                      sigma_obs: float | None = None, u_init=None,
                                      mc_samples: int = 200, level: float = 0.9,
                                      seed: int = 0, progress: bool = False):
    """Decision-level LOEO: is the pooled DECISION trustworthy out-of-sample?

    The decision-space analogue of :func:`leave_one_experiment_out` (which validates the
    curve).  For each experiment ``held``: refit on the other ``n-1`` runs -- warm-started
    from ``u_init`` (e.g. the committed full-data ``u_map``), so a slow-to-fit product
    converges in few ``map_iters`` instead of thousands -- then, at the held-out
    experiment's OWN operating condition, compute

      (i)  the ``n-1`` posterior's predicted pooled purity/yield with a Gaussian band
           (``g_map +/- z*std``) and a Monte-Carlo posterior-predictive band
           (:func:`cex_model.bayes.decision.decision_covariance_mc` draws), plus the
           determinability ``worst_dec`` at that OP; and
      (ii) the OBSERVED pooled purity/yield read from the held-out MEASURED curve over the
           SAME fixed window (:func:`_observed_pool_decision`),

    then flag whether the observed decision falls inside the predicted band (MC band is the
    primary criterion; the Gaussian band is a fallback when the MC draws are unusable) and
    whether the held-out decision is determinable (``worst_dec < tau_dec``).  Heavy (``n``
    refits on the real solver) -> Colab.  Returns ``{folds: [...], aggregate: {...}}``.
    """
    from cex_model.app_support import group_indices
    from cex_model.bayes.decision import decision_covariance_mc, decision_report
    from cex_model.bayes.likelihood import AKTA_NOISE_FLOOR_G_L, mechanistic_model
    from cex_model.bayes.posterior import laplace_posterior, map_fit
    from cex_model.bayes.prior import components_to_u
    from cex_model.diffsolver.calibrate_diff import targets_from_bundle
    from scipy.stats import norm

    sigma_obs = float(sigma_obs if sigma_obs is not None else AKTA_NOISE_FLOOR_G_L)
    n = bundle.components.n_protein
    u0 = components_to_u(bundle.components) if u_init is None else np.asarray(u_init, float)
    gi = group_indices(bundle.components)
    main_set = set(gi["main"])
    og = bundle.observation_groups
    obs_groups = list(og) if og else [[j] for j in range(n)]
    main_cols = [k for k, grp in enumerate(obs_groups) if set(grp) <= main_set]
    z = float(norm.ppf(0.5 * (1.0 + level)))
    q_lo, q_hi = 0.5 * (1.0 - level), 0.5 * (1.0 + level)
    names = ("pool_purity", "pool_yield")

    exps = list(range(len(bundle.experiments)))
    folds = []
    for held in exps:
        fit_b = _subset_bundle(bundle, [i for i in exps if i != held])
        he = bundle.experiments[held]
        op = [float(he.loading_g_l), float(he.gradient_start_pct),
              float(he.gradient_end_pct), float(he.elution_cv)]
        predict_fn, obs = mechanistic_model(targets_from_bundle(fit_b, n_steps=n_steps), n)
        u_map, _ = map_fit(u0, predict_fn, obs, prior, sigma_obs=sigma_obs, iters=map_iters, lr=lr)
        post = laplace_posterior(u_map, predict_fn, obs, prior, template=bundle.components,
                                 sigma_obs=sigma_obs)
        # predict the decision at the held-out OP from the n-1 posterior (product constants
        # come from the full bundle; the parameters/covariance come from the n-1 posterior)
        rep = decision_report(post, bundle, op, tol=tol, tau_dec=tau_dec, n_steps=n_steps)
        mc = decision_covariance_mc(post, bundle, op, n_samples=mc_samples, n_steps=n_steps,
                                    seed=seed, return_gs=True)
        gs = np.asarray(mc.get("gs", []), float)
        mc_ok = gs.ndim == 2 and gs.shape[0] >= 2
        pur_obs, yld_obs = _observed_pool_decision(he.curve, rep["window_s"], main_cols)
        observed = {"pool_purity": pur_obs, "pool_yield": yld_obs}

        gauss_band, mc_band, in_gauss, in_mc = {}, {}, {}, {}
        for i, nm in enumerate(names):
            g, s, o = float(rep["g_map"][nm]), float(rep["decision_std"][nm]), observed[nm]
            gauss_band[nm] = [g - z * s, g + z * s]
            in_gauss[nm] = bool(gauss_band[nm][0] <= o <= gauss_band[nm][1]) if np.isfinite(o) else False
            if mc_ok:
                mc_band[nm] = [float(np.quantile(gs[:, i], q_lo)), float(np.quantile(gs[:, i], q_hi))]
                in_mc[nm] = bool(mc_band[nm][0] <= o <= mc_band[nm][1]) if np.isfinite(o) else False
            else:
                mc_band[nm] = [float("nan"), float("nan")]
                in_mc[nm] = False
        in_primary = all(in_mc.values()) if mc_ok else all(in_gauss.values())

        fold = {
            "held_out": he.name, "op": op,
            "worst_dec": float(rep["worst_dec"]), "determined": bool(rep["met"]),
            "g_pred": {nm: float(rep["g_map"][nm]) for nm in names},
            "decision_std": {nm: float(rep["decision_std"][nm]) for nm in names},
            "observed": observed, "gauss_band": gauss_band, "mc_band": mc_band,
            "in_gauss_band": in_gauss, "in_mc_band": in_mc,
            "in_band": bool(in_primary), "band_used": "mc" if mc_ok else "gauss",
            "window_s": [float(rep["window_s"][0]), float(rep["window_s"][1])],
            "mc_n_used": int(mc.get("n_used", 0)),
            "rel_frobenius": _rel_frob(rep.get("C"), mc.get("C_mc")),
        }
        folds.append(fold)
        if progress:
            print(f"  dLOEO held={str(he.name)[:24]:24s} worst_dec={fold['worst_dec']:.3f} "
                  f"obs_pur={pur_obs:.3f} pred_pur={fold['g_pred']['pool_purity']:.3f} "
                  f"in_band={fold['in_band']} ({fold['band_used']})")

    finite = [f for f in folds if np.isfinite(f["observed"]["pool_purity"])]
    agg = {
        "n_folds": len(folds),
        "n_observable": len(finite),
        "frac_in_band": float(np.mean([f["in_band"] for f in finite])) if finite else float("nan"),
        "worst_dec_range": ([min(f["worst_dec"] for f in folds), max(f["worst_dec"] for f in folds)]
                            if folds else []),
        "all_determined": bool(all(f["determined"] for f in folds)) if folds else False,
    }
    return {"folds": folds, "aggregate": agg}


def retrospective_eig(per_exp_jacobians, prior_precision, sigma_obs) -> list[dict]:
    """Retrospective BOED: credit each EXISTING experiment by its information content.

    With per-experiment Jacobians ``J_i = d(curve_i)/du`` at the MAP, the full
    posterior precision is ``H = prior_precision + sum_i J_iᵀJ_i/sigma²``; the
    leave-one-out logdet drop ``logdet(H) - logdet(H - M_i)`` is the nats of
    information experiment ``i`` adds (``>= 0``; ``H - M_i`` stays positive-definite
    because the prior does).  Ranking the *real* experiments this way shows the BOED
    criterion credits the knobs that actually matter (gradient slope, high loading)
    on data we already have -- a real-data complement to the synthetic adaptive loop,
    answering the "only a single-component toy" concern.  ``0.5x`` matches the EIG
    convention of :func:`cex_model.bayes.design.expected_info_gain`.
    """
    Ms = [(np.asarray(J, float).T @ np.asarray(J, float)) / (sigma_obs ** 2) for J in per_exp_jacobians]
    H = np.asarray(prior_precision, float) + sum(Ms)
    _, ld_full = np.linalg.slogdet(H)
    out = [{"index": i, "eig": 0.5 * float(ld_full - np.linalg.slogdet(H - M)[1])}
           for i, M in enumerate(Ms)]
    out.sort(key=lambda r: r["eig"], reverse=True)
    for rank, r in enumerate(out):
        r["rank"] = rank
    return out


def retrospective_design(bundle, prior, u_map, *, n_steps: int = 300, sigma_obs: float | None = None,
                         progress: bool = False) -> list[dict]:
    """Build each experiment's Jacobian on the real solver, then rank by retrospective EIG.

    Heavy (one reverse-mode Jacobian per experiment) -> Colab; the ranking math is
    the unit-tested :func:`retrospective_eig`.
    """
    import torch

    from cex_model.bayes.likelihood import AKTA_NOISE_FLOOR_G_L, mechanistic_model
    from cex_model.diffsolver.calibrate_diff import targets_from_bundle
    from cex_model.diffsolver.torch_solver import DTYPE

    sigma_obs = float(sigma_obs if sigma_obs is not None else AKTA_NOISE_FLOOR_G_L)
    n = bundle.components.n_protein
    u = torch.tensor(np.asarray(u_map, float), dtype=DTYPE)
    feats = ("loading_g_l", "gradient_start_pct", "gradient_end_pct", "elution_cv")
    jacs = []
    for i in range(len(bundle.experiments)):
        predict_fn, _ = mechanistic_model(targets_from_bundle(_subset_bundle(bundle, [i]), n_steps=n_steps), n)
        Ji = torch.autograd.functional.jacobian(predict_fn, u).detach().numpy()
        jacs.append(Ji)
        if progress:
            print(f"  retrospective J exp={bundle.experiments[i].name[:30]:30s} rows={Ji.shape[0]}")
    ranked = retrospective_eig(jacs, prior.precision(), sigma_obs)
    for r in ranked:
        e = bundle.experiments[r["index"]]
        r["name"], r["op"] = e.name, {f: float(getattr(e, f)) for f in feats}
    return ranked


def _keq_nu_block_dir(n_protein: int, j: int, dim: int) -> np.ndarray:
    """Unit vector along ``(e_keq_j - e_nu_j)/sqrt(2)`` in u-space (the keq<->nu ridge)."""
    v = np.zeros(dim)
    v[j] = 1.0 / np.sqrt(2.0)            # keq block is [0:n]
    v[2 * n_protein + j] = -1.0 / np.sqrt(2.0)   # nu block is [2n:3n]
    return v


def keq_nu_ridge_width(posterior) -> float:
    """Worst per-component posterior std along the keq<->nu ridge direction (u-space).

    A scalar summary of how wide the structural keq<->nu degeneracy still is -- used as
    one column of the design-baseline matrix.  Larger = the ridge is less resolved.
    """
    cov = np.asarray(posterior.cov, float)
    n = posterior.n_protein
    widths = []
    for j in range(n):
        v = _keq_nu_block_dir(n, j, cov.shape[0])
        widths.append(np.sqrt(max(float(v @ cov @ v), 0.0)))
    return float(np.max(widths)) if widths else 0.0


def design_metrics(posterior, prior) -> dict:
    """Identifiability / uncertainty columns for the design-baseline matrix (solver-free).

    All read off one posterior + the prior: the worst prior-whitened direction, the data
    precision floor, the posterior log-volume (``logdet(cov)``; larger = more uncertain),
    the mean sigma marginal width, and the keq<->nu ridge width.
    """
    from cex_model.bayes.active import identifiability_report

    rep = identifiability_report(posterior, prior)
    cov = np.asarray(posterior.cov, float)
    n = posterior.n_protein
    sigma_std = np.asarray(posterior.std_u(), float)[3 * n:4 * n]
    return {"worst_dir_shrinkage": float(rep["worst_dir_shrinkage"]),
            "min_eig_precision": float(rep["min_eig_precision"]),
            "posterior_volume": float(np.linalg.slogdet(cov)[1]),
            "sigma_uncertainty": float(np.mean(sigma_std)),
            "keq_nu_ridge_width": keq_nu_ridge_width(posterior)}


def fisher_ablation(posterior, prior) -> dict:
    """Fisher (no-prior) ablation: is the identifiability data-driven or prior-driven?

    The Laplace posterior precision is ``H = prior.precision() + JᵀJ/sigma²``; subtracting
    the (diagonal) prior precision leaves the pure data Fisher information
    ``H_fisher = inv(cov) - prior.precision()``.  Its smallest eigenvalue is ~0 along the
    structural keq<->nu ridge (and sigma) -- the degeneracy is in the DATA, not the prior;
    with the prior, that direction's precision is held up to ``prior_min_eig``.  The min-eig
    is reported RAW (a tiny negative value from the Laplace jitter + inverse round-trip is
    expected; the ~0 is the headline), and clipped to PSD only where a downstream logdet
    needs it.  The per-component keq<->nu Fisher curvature along ``(e_keq - e_nu)/sqrt(2)``
    makes the "ridge is a data limitation" claim concrete.
    """
    cov = np.asarray(posterior.cov, float)
    H = np.linalg.inv(cov)
    prec = np.asarray(prior.precision(), float)          # diagonal MATRIX, not a vector
    H_fisher = 0.5 * (H - prec + (H - prec).T)
    eigvals, eigvecs = np.linalg.eigh(H_fisher)
    names = list(posterior.names)
    worst = eigvecs[:, 0]
    worst_dir_params = [names[i] for i in np.argsort(np.abs(worst))[::-1][:3]]
    n = posterior.n_protein
    keq_nu = {}
    for j in range(n):
        v = _keq_nu_block_dir(n, j, cov.shape[0])
        keq_nu[f"c{j + 1}"] = float(v @ H_fisher @ v)    # Fisher curvature across the ridge
    prior_min_eig = float(np.min(np.diag(prec)))
    return {"min_eig_fisher": float(eigvals[0]), "min_eig_posterior": float(np.linalg.eigvalsh(H).min()),
            "prior_min_eig": prior_min_eig, "worst_dir_params": worst_dir_params,
            "keq_nu_curvature": keq_nu,
            # data alone leaves the worst direction ~unconstrained vs the prior floor:
            "fisher_vs_prior_floor_ratio": float(eigvals[0] / prior_min_eig)}


def _prospective_eig(H_seed, jacobians, sigma_obs) -> list[dict]:
    """Prospective EIG of each candidate at the SEED posterior precision ``H_seed``.

    Sibling of :func:`retrospective_eig`: for each candidate Jacobian ``J_i`` at the seed
    MAP, ``EIG_i = 0.5*(logdet(H_seed + J_iᵀJ_i/sigma²) - logdet(H_seed)) >= 0`` -- how
    much running that experiment would shrink the current posterior.  Ranks candidates;
    ``H_seed`` keeps the (PD) prior term so the logdets stay finite even from a 1-2
    experiment seed.
    """
    H = np.asarray(H_seed, float)
    _, ld0 = np.linalg.slogdet(H)
    out = []
    for i, J in enumerate(jacobians):
        J = np.asarray(J, float)
        _, ld1 = np.linalg.slogdet(H + (J.T @ J) / (sigma_obs ** 2))
        out.append({"index": int(i), "eig": 0.5 * float(ld1 - ld0)})
    out.sort(key=lambda r: r["eig"], reverse=True)
    for rank, r in enumerate(out):
        r["rank"] = rank
    return out


def real_design_loop(bundle, prior, *, seed_idx, n_steps: int = 300, map_iters: int = 150,
                     lr: float = 0.05, sigma_obs: float | None = None, predict_held: bool = True,
                     predict_samples: int = 60, tau: float = 0.5, u_init=None, seed: int = 0,
                     progress: bool = False) -> dict:
    """Real-data discrete prospective-design loop (P1 5B) -- the real-data BOED closure.

    Seed the posterior on a small REAL subset, score each REAL held-out experiment by its
    prospective EIG at the seed MAP (the discrete candidate set IS the real held-out OPs,
    so "recommendation ~ a real remaining experiment" holds by construction), then INGEST
    the top pick's real curve (via ``targets_from_bundle`` -- NOT a ``simulate_target``
    fabrication), refit, and report the worst-direction shrinkage / posterior-volume drop.
    A CONTROL arm ingests the LOWEST-EIG held-out from the same seed, so the gain is
    attributable to the design, not just to adding any experiment
    (``design_beats_arbitrary``).  Heavy (refits on the real solver) -> Colab.
    """
    import torch

    from cex_model.bayes.active import identifiability_report
    from cex_model.bayes.likelihood import AKTA_NOISE_FLOOR_G_L, mechanistic_model
    from cex_model.bayes.posterior import laplace_posterior, map_fit
    from cex_model.bayes.predictive import posterior_predictive_rk23
    from cex_model.bayes.prior import components_to_u
    from cex_model.diffsolver.calibrate_diff import targets_from_bundle
    from cex_model.diffsolver.torch_solver import DTYPE

    sigma_obs = float(sigma_obs if sigma_obs is not None else AKTA_NOISE_FLOOR_G_L)
    n = bundle.components.n_protein
    seed_idx = list(seed_idx)
    held = [i for i in range(len(bundle.experiments)) if i not in seed_idx]
    if not held:
        raise ValueError("need at least one held-out experiment beyond the seed")
    # warm-start every seed/ingest refit from a committed full-data u_map when given: the
    # seed/ingest subsets are perturbations of the full fit, so this converges in far fewer
    # map_iters than the generic-init default (which needs thousands for HLXSYN).
    u0 = components_to_u(bundle.components) if u_init is None else np.asarray(u_init, float)

    def _fit(idx):
        predict_fn, obs = mechanistic_model(targets_from_bundle(_subset_bundle(bundle, idx), n_steps=n_steps), n)
        u_map, _ = map_fit(u0, predict_fn, obs, prior, sigma_obs=sigma_obs, iters=map_iters, lr=lr)
        post = laplace_posterior(u_map, predict_fn, obs, prior, template=bundle.components, sigma_obs=sigma_obs)
        return u_map, post

    u_seed, post_seed = _fit(seed_idx)
    rep_seed = identifiability_report(post_seed, prior, tau=tau)
    seed_logdet = float(np.linalg.slogdet(np.asarray(post_seed.cov, float))[1])
    H_seed = np.linalg.inv(np.asarray(post_seed.cov, float))

    # prospective EIG over the REAL held-out experiments at the seed MAP
    u_t = torch.tensor(np.asarray(u_seed, float), dtype=DTYPE)
    jacs = []
    for k in held:
        predict_fn, _ = mechanistic_model(targets_from_bundle(_subset_bundle(bundle, [k]), n_steps=n_steps), n)
        jacs.append(torch.autograd.functional.jacobian(predict_fn, u_t).detach().numpy())
        if progress:
            print(f"  prospective J held={bundle.experiments[k].name[:28]:28s} rows={jacs[-1].shape[0]}")
    ranked = _prospective_eig(H_seed, jacs, sigma_obs)
    for r in ranked:
        r["exp_index"], r["name"] = held[r["index"]], bundle.experiments[held[r["index"]]].name
    top_k, bottom_k = ranked[0]["exp_index"], ranked[-1]["exp_index"]

    def _ingest(extra_k):
        _, post = _fit(seed_idx + [extra_k])
        rep = identifiability_report(post, prior, tau=tau)
        row = {"ingested": bundle.experiments[extra_k].name, "exp_index": int(extra_k),
               "worst_dir_shrinkage": rep["worst_dir_shrinkage"], "met": rep["met"],
               "logdet_cov": float(np.linalg.slogdet(np.asarray(post.cov, float))[1])}
        if predict_held:
            rest = [i for i in held if i != extra_k]
            if rest:
                pr = posterior_predictive_rk23(post, _subset_bundle(bundle, [rest[0]]),
                                               n_samples=predict_samples, seed=seed)
                row["held_coverage"] = pr["per_experiment"][0]["coverage"]
        return row

    top, bottom = _ingest(top_k), _ingest(bottom_k)
    d_worst_top = rep_seed["worst_dir_shrinkage"] - top["worst_dir_shrinkage"]
    d_worst_bottom = rep_seed["worst_dir_shrinkage"] - bottom["worst_dir_shrinkage"]
    d_logdet_top = seed_logdet - top["logdet_cov"]        # > 0 means the posterior volume shrank
    d_logdet_bottom = seed_logdet - bottom["logdet_cov"]
    return {
        "seed_idx": seed_idx, "held_idx": held, "ranked": ranked,
        "seed": {"worst_dir_shrinkage": rep_seed["worst_dir_shrinkage"], "logdet_cov": seed_logdet,
                 "met": rep_seed["met"]},
        "top_pick": int(top_k), "bottom_pick": int(bottom_k), "top": top, "bottom": bottom,
        "delta_worst_top": float(d_worst_top), "delta_worst_bottom": float(d_worst_bottom),
        "delta_logdet_top": float(d_logdet_top), "delta_logdet_bottom": float(d_logdet_bottom),
        "design_beats_arbitrary": bool(d_worst_top >= d_worst_bottom and d_logdet_top >= d_logdet_bottom),
    }


def empirical_sigma_obs(residuals) -> float:
    """Empirical-Bayes sigma_obs: RMS of the MAP residuals (model - measured), g/L.

    Justifies the fixed AKTA-noise-floor likelihood: if this data-driven estimate
    lands near 0.148 g/L, the assumed observation noise is consistent with the fit
    (a much smaller value would mean the model is over-fitting the curve).  Accepts a
    flat array or a list of per-experiment residual arrays.
    """
    if isinstance(residuals, (list, tuple)):
        r = np.concatenate([np.asarray(x, float).reshape(-1) for x in residuals])
    else:
        r = np.asarray(residuals, float).reshape(-1)
    return float(np.sqrt(np.mean(r ** 2)))


def sigma_obs_sensitivity(bundle, prior, *, sigma_grid=None, n_steps: int = 300, map_iters: int = 150,
                          lr: float = 0.05, predict_samples: int = 60, coverage: bool = True,
                          u_init=None, seed: int = 0, progress: bool = False) -> dict:
    """Refit MAP+Laplace across a grid of ``sigma_obs``; report worst-dir shrinkage + coverage.

    Checks the calibration is not knife-edge sensitive to the assumed observation
    noise.  Default grid is ``[0.5, 1, 2] x`` the AKTA noise floor.  Heavy (one refit
    per sigma) -> Colab. ``u_init`` warm-starts each refit from a committed ``u_map``.
    """
    from cex_model.bayes.active import identifiability_report
    from cex_model.bayes.likelihood import AKTA_NOISE_FLOOR_G_L, mechanistic_model
    from cex_model.bayes.posterior import laplace_posterior, map_fit
    from cex_model.bayes.predictive import posterior_predictive_rk23
    from cex_model.bayes.prior import components_to_u
    from cex_model.diffsolver.calibrate_diff import targets_from_bundle

    floor = AKTA_NOISE_FLOOR_G_L
    grid = list(sigma_grid) if sigma_grid is not None else [0.5 * floor, floor, 2.0 * floor]
    n = bundle.components.n_protein
    u0 = components_to_u(bundle.components) if u_init is None else np.asarray(u_init, float)
    predict_fn, obs = mechanistic_model(targets_from_bundle(bundle, n_steps=n_steps), n)
    rows = []
    for s in grid:
        u_map, _ = map_fit(u0, predict_fn, obs, prior, sigma_obs=s, iters=map_iters, lr=lr)
        post = laplace_posterior(u_map, predict_fn, obs, prior, template=bundle.components, sigma_obs=s)
        rep = identifiability_report(post, prior)
        row = {"sigma_obs": float(s), "worst_dir_shrinkage": rep["worst_dir_shrinkage"], "met": rep["met"]}
        if coverage:
            row["coverage"] = posterior_predictive_rk23(
                post, bundle, n_samples=predict_samples, seed=seed)["aggregate"]["coverage"]
        rows.append(row)
        if progress:
            print(f"  sigma={s:.4f} worst_dir={row['worst_dir_shrinkage']:.3f} "
                  f"coverage={row.get('coverage', float('nan')):.3f}")
    return {"grid": rows, "noise_floor": floor}


def n_steps_sensitivity(bundle, prior, *, n_steps_grid=(150, 300, 600), sigma_obs: float | None = None,
                        map_iters: int = 150, lr: float = 0.05, predict_samples: int = 60,
                        coverage: bool = True, u_init=None, seed: int = 0, progress: bool = False) -> dict:
    """Refit MAP+Laplace at several BDF time-grid resolutions; report MAP drift + coverage.

    Backs the (discretized-solver) claim that the posterior is not pathologically
    grid-dependent: the MAP drift across ``n_steps`` and the coverage should be small
    /stable.  Heavy (one refit per resolution) -> Colab. ``u_init`` warm-starts each
    refit from a committed ``u_map`` (params transfer across grids).
    """
    from cex_model.bayes.active import identifiability_report
    from cex_model.bayes.likelihood import AKTA_NOISE_FLOOR_G_L, mechanistic_model
    from cex_model.bayes.posterior import laplace_posterior, map_fit
    from cex_model.bayes.predictive import posterior_predictive_rk23
    from cex_model.bayes.prior import components_to_u
    from cex_model.diffsolver.calibrate_diff import targets_from_bundle

    sigma_obs = float(sigma_obs if sigma_obs is not None else AKTA_NOISE_FLOOR_G_L)
    n = bundle.components.n_protein
    u0 = components_to_u(bundle.components) if u_init is None else np.asarray(u_init, float)
    rows, u_ref = [], None
    for ns in n_steps_grid:
        predict_fn, obs = mechanistic_model(targets_from_bundle(bundle, n_steps=int(ns)), n)
        u_map, _ = map_fit(u0, predict_fn, obs, prior, sigma_obs=sigma_obs, iters=map_iters, lr=lr)
        post = laplace_posterior(u_map, predict_fn, obs, prior, template=bundle.components, sigma_obs=sigma_obs)
        rep = identifiability_report(post, prior)
        u_ref = u_map if u_ref is None else u_ref       # drift measured vs the coarsest grid
        row = {"n_steps": int(ns), "worst_dir_shrinkage": rep["worst_dir_shrinkage"],
               "map_drift_vs_coarsest": float(np.linalg.norm(u_map - u_ref))}
        if coverage:
            row["coverage"] = posterior_predictive_rk23(
                post, bundle, n_samples=predict_samples, seed=seed)["aggregate"]["coverage"]
        rows.append(row)
        if progress:
            print(f"  n_steps={ns} worst_dir={row['worst_dir_shrinkage']:.3f} "
                  f"drift={row['map_drift_vs_coarsest']:.3f} coverage={row.get('coverage', float('nan')):.3f}")
    return {"grid": rows}


def prior_sensitivity(bundle, prior, *, scale_grid=(0.5, 0.7, 1.0, 1.5, 2.0), shift_grid=(0.0,),
                      op=None, tol=(0.02, 0.05), tau_dec: float = 1.0, n_steps: int = 300,
                      map_iters: int = 150, lr: float = 0.05, sigma_obs: float | None = None,
                      decision: bool = True, u_init=None, seed: int = 0, progress: bool = False) -> dict:
    """Refit MAP+Laplace under widened / narrowed / shifted priors; report ``worst_dir`` + ``worst_dec``.

    Answers the reviewer question "is the conclusion prior-driven?".  For each prior
    variant -- std scaled by ``scale`` and mean shifted by ``shift`` prior-std -- we report
    the criteria an analyst *using that prior* would read: the prior-whitened parameter
    ``worst_dir`` (whitened by the variant's own prior) AND the tolerance-whitened decision
    ``worst_dec`` at the historical window (default: the first experiment's operating
    condition).  If the not-identified / determined verdicts hold across the grid, the
    conclusion is not an artifact of the specific weak prior.

    Heavy (one refit per variant) -> Colab.  ``u_init`` warm-starts each refit from a
    committed ``u_map`` (e.g. ``Posterior.load({product}_posterior.npz).u_map``), so the
    perturbed-prior fits start near the optimum and stay cheap.  ``decision=False`` skips
    the decision report (parameter-only, no operating window needed).
    """
    from cex_model.bayes.active import identifiability_report
    from cex_model.bayes.decision import decision_report
    from cex_model.bayes.likelihood import AKTA_NOISE_FLOOR_G_L, mechanistic_model
    from cex_model.bayes.posterior import laplace_posterior, map_fit
    from cex_model.bayes.prior import PhysicalPrior, components_to_u
    from cex_model.diffsolver.calibrate_diff import targets_from_bundle

    sigma_obs = float(sigma_obs if sigma_obs is not None else AKTA_NOISE_FLOOR_G_L)
    n = bundle.components.n_protein
    u0 = components_to_u(bundle.components) if u_init is None else np.asarray(u_init, float)
    if op is None:
        e0 = bundle.experiments[0]
        op = [e0.loading_g_l, e0.gradient_start_pct, e0.gradient_end_pct, e0.elution_cv]
    predict_fn, obs = mechanistic_model(targets_from_bundle(bundle, n_steps=n_steps), n)
    rows = []
    for scale in scale_grid:
        for shift in shift_grid:
            # The variant prior an analyst might instead have chosen: same physical center
            # (+ optional shift), broader/narrower width.  Used for BOTH the fit (via the
            # Hessian's prior precision) and the prior-whitening, so each row is the verdict
            # that variant would report -- not a mix of two priors.
            prior_v = PhysicalPrior(mean=prior.mean + float(shift) * prior.std,
                                    std=prior.std * float(scale), names=list(prior.names),
                                    n_protein=prior.n_protein)
            u_map, _ = map_fit(u0, predict_fn, obs, prior_v, sigma_obs=sigma_obs, iters=map_iters, lr=lr)
            post = laplace_posterior(u_map, predict_fn, obs, prior_v, template=bundle.components,
                                     sigma_obs=sigma_obs)
            rep = identifiability_report(post, prior_v)
            row = {"prior_scale": float(scale), "prior_shift": float(shift),
                   "worst_dir": rep["worst_dir_shrinkage"], "identified": rep["met"]}
            if decision:
                dec = decision_report(post, bundle, op, tol=tol, tau_dec=tau_dec, n_steps=n_steps)
                row.update(worst_dec=dec["worst_dec"], decision_met=dec["met"],
                           pool_purity=dec["g_map"]["pool_purity"], pool_yield=dec["g_map"]["pool_yield"])
            rows.append(row)
            if progress:
                msg = f"  scale={scale:.2f} shift={shift:+.1f} worst_dir={row['worst_dir']:.3f}"
                if decision:
                    msg += f" worst_dec={row['worst_dec']:.3f} (met={row['decision_met']})"
                print(msg)
    return {"grid": rows, "op": list(op), "tol": list(tol), "tau_dec": float(tau_dec),
            "baseline": {"prior_scale": 1.0, "prior_shift": 0.0}}
