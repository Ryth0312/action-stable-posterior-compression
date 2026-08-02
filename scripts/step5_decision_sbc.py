"""R5b (Colab / torch): decision calibration --- SBC for g, reliability for P(meet). Cost-minimised.

Two different questions, deliberately answered with two different tools (per the plan):

  * for the DECISION VECTOR g = (pool purity, pool yield) at a fixed action: standard SBC, i.e. the
    rank of the truth among posterior draws must be uniform (per quantity, plus a 2-D Mahalanobis PIT);
  * for P(meet): NOT rank-SBC (it is a probability, not a parameter). For replicate r we record
        p_hat_r = P{ g(theta, a) in A_meet | D_r },      Y_r = 1{ g(theta_r, a) in A_meet },
    and test the reliability property  E(Y | p_hat = p) = p --- reliability curve, Brier, log score,
    calibration intercept/slope (logistic fit of Y on logit p_hat), and the realised hit frequency in
    the decision-relevant tails p_hat >= 0.95 and p_hat <= 0.05.

Fixed vs data-selected actions are reported SEPARATELY (the selected arm carries selection optimism).

WHY THIS IS CHEAP (``--engine frozen``, the default). The paper's Laplace posterior *is* the
Gauss--Newton / linear-Gaussian posterior at the MAP: ``H = J^T J / sigma^2 + Lambda_0``. Freezing ``J``
at the reference MAP is therefore not a different method, only its fixed-Jacobian version, and it
collapses the per-replicate cost:

  * ``GGN = J^T J / sigma^2`` is recovered from the COMMITTED posterior for free, with no solver call:
    ``GGN = inv(post.cov) - prior.precision()`` (exactly how ``laplace_posterior`` built it);
  * the pre-posterior covariance ``Sigma = (GGN + Lambda_0)^{-1}`` is then data-INDEPENDENT: one
    matrix inverse for the whole run, no per-replicate Laplace;
  * the MAP is closed form, ``u_map_r = u0 + Sigma s_r`` with ``s_r = J^T(obs_r - y0)/sigma^2``: no Adam;
  * for i.i.d. noise ``J^T e / sigma^2 ~ N(0, GGN)`` exactly, so ``s_r`` needs NO solver call at all;
    for OU noise it is one vector-Jacobian product against a graph built ONCE at ``u0`` (1 backward).

Per-replicate cost drops from ~600 solve-equivalents (300 Adam iterations of forward+backward, plus a
GGN Hessian and a decision Jacobian per candidate) to ~1-2: a nonlinear forward for the ground-truth
decision g(theta_r) --- kept nonlinear on purpose, it is the outcome Y_r is read from --- plus at most
one backward. That is a ~200x reduction; a 100-replicate pilot runs in minutes rather than hours.

``--engine refit`` restores the exact per-replicate MAP+Laplace, but a full agreement study that way is
prohibitive (~900 solve-equivalents per replicate). ``--engine gradcheck`` validates the frozen engine far
more cheaply and more informatively: the frozen MAP is by construction the exact minimiser of the LINEARISED
objective, so the TRUE objective's gradient there measures exactly the nonlinearity the frozen engine drops,
and one Newton step with the frozen precision turns it into a first-order estimate of
``u_map_exact - u_map_frozen``. Those offsets are propagated analytically into ``g_map`` and ``P(meet)`` --
the quantities the calibration study consumes -- giving CONFIDENCE INTERVALS on the induced errors from ~2
forwards + 2 backwards per replicate. ``--ggn-checks k`` adds the second half of the validation: an exact GGN
at k of the frozen MAPs, testing directly whether ``Sigma`` is as data-independent as the frozen engine
assumes. Two or three ``--engine refit`` replicates remain a useful end-to-end anchor.

Generative layers (``--layer``): ``iid`` (data from the same residual model the inference assumes ---
calibration should PASS, validating the pipeline) and ``ou`` (OU-correlated residuals + nugget while
inference still assumes i.i.d. --- the conditional law should be over-confident and the discrepancy-floor
predictive law should recover it; that contrast is the headline R5b result).

SBC validity: the truth is drawn from the SAME prior the inference uses (diagonal, centred at the real
fitted MAP, scale = fitted marginal std x --prior-scale), so the SBC contract holds while the regime
stays realistic. The collection window is fixed at the reference MAP curve, matching the fixed-window
estimand of the linearised decision covariance.

Pilot (validate the pipeline first, ~minutes):
  OMP_NUM_THREADS=4 python scripts/step5_decision_sbc.py --product HLXSYN --n-datasets 100 --layer iid
then the contrast:
  OMP_NUM_THREADS=4 python scripts/step5_decision_sbc.py --product HLXSYN --n-datasets 100 --layer ou
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

import cex_model.app_support as A
from cex_model.bayes.compare import filter_experiments
from cex_model.bayes.decision import decision_forward, decision_jacobian
from cex_model.bayes.decision_window import gaussian_meet_prob
from cex_model.bayes.design import candidate_pool_for
from cex_model.bayes.likelihood import AKTA_NOISE_FLOOR_G_L, gaussian_loglik, mechanistic_model
from cex_model.bayes.posterior import Posterior, laplace_posterior, map_fit
from cex_model.bayes.prior import PhysicalPrior, physical_prior
from cex_model.diffsolver.calibrate_diff import targets_from_bundle
from cex_model.diffsolver.torch_solver import DTYPE

_PRODUCT_MAP = {"HLXSYN": ("HLXSYN", ["DT"])}


# --------------------------------------------------------------------------- cheap linear-Gaussian core
def _ggn_from_posterior(cov, prior_precision):
    """``J^T J / sigma^2`` recovered from a committed Laplace posterior --- no solver call.

    ``laplace_posterior`` built ``H = GGN + prior.precision()`` and stored ``cov = H^{-1}``.
    """
    G = np.linalg.inv(np.asarray(cov, float)) - np.asarray(prior_precision, float)
    G = 0.5 * (G + G.T)
    w, V = np.linalg.eigh(G)
    if w.min() < 0:                                   # numerical dust from the round trip
        w = np.clip(w, 0.0, None)
        G = V @ np.diag(w) @ V.T
    return G, float(w.min()), float(w.max())


def _ou_cov_blocks(targets, sigma_obs, rho, ell):
    """Per-(experiment, group) OU+nugget covariance factors, as Cholesky blocks over the time axis."""
    out = []
    for tg in targets:
        blk = int(tg.values.reshape(-1).numel())
        n_t = len(tg.times_s)
        n_g = max(1, blk // n_t) if n_t else 1
        t = np.asarray(tg.times_s, float).reshape(-1, 1)
        R = np.exp(-np.abs(t - t.T) / max(float(ell), 1e-9))
        Sig = sigma_obs**2 * ((1.0 - rho) * np.eye(n_t) + rho * R)
        out.append((np.linalg.cholesky(Sig + 1e-12 * np.eye(n_t)), n_t, n_g, blk))
    return out


def _draw_noise(layer, n_resid, sigma_obs, ou_blocks, rng):
    if layer == "iid":
        return sigma_obs * rng.standard_normal(n_resid)
    e = np.empty(n_resid)
    off = 0
    for L, n_t, n_g, blk in ou_blocks:
        e[off:off + blk] = (L @ rng.standard_normal((n_t, n_g))).reshape(-1)[:blk]
        off += blk
    return e


# calibration metrics live in _calibration_metrics.py (torch-free, shared with the phase-2 study)
_M = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("_calmetrics", Path(__file__).resolve().parent / "_calibration_metrics.py"))
_M.__loader__.exec_module(_M)
_reliability, _cal_intercept_slope = _M._reliability, _M._cal_intercept_slope
_ece, _score_block, _rank_stats = _M._ece, _M._score_block, _M._rank_stats


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="HLXSYN")
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--n-datasets", type=int, default=100)
    ap.add_argument("--n-post", type=int, default=200)
    ap.add_argument("--n-steps", type=int, default=120)
    ap.add_argument("--engine", choices=["frozen", "refit", "gradcheck"], default="frozen",
                    help="frozen: closed-form linear-Gaussian (default); refit: exact MAP+Laplace (expensive); "
                         "gradcheck: validate the frozen engine directly (~2 fwd + 2 bwd per replicate) and "
                         "propagate the measured offsets into the decision quantities")
    ap.add_argument("--ggn-checks", type=int, default=3,
                    help="gradcheck only: how many replicates also get an EXACT GGN at their frozen MAP "
                         "(tier 2; ~n_resid backward passes each). 0 disables.")
    ap.add_argument("--layer", choices=["iid", "ou"], default="iid")
    ap.add_argument("--laws", nargs="+", default=["conditional", "predictive"])
    ap.add_argument("--spec", type=float, nargs=2, default=[0.70, 0.50])
    ap.add_argument("--prior-cov", choices=["full", "diag"], default="full",
                    help="full: SBC prior cov = scale^2 x posterior cov (keeps truths ON the sloppy ridge; "
                         "diag uses only the marginals and throws truths off-ridge into the nonlinear regime)")
    ap.add_argument("--prior-scale", type=float, default=1.0)
    ap.add_argument("--scale-sweep", type=float, nargs="+", default=None,
                    help="run the whole study at several prior scales and print the calibration-vs-regime table")
    ap.add_argument("--floor-wdec", type=float, default=1.271,
                    help="isotropic whitened discrepancy floor (default matches the committed predictive read)")
    ap.add_argument("--pool-size", type=int, default=6, help="extra candidates for the selected-action arm (0 = off)")
    ap.add_argument("--disc-wdec", type=float, default=0.0,
                    help="generative DECISION discrepancy: per-replicate product-level bias with whitened "
                         "worst-direction size (0 = off). Set it near --floor-wdec to make the predictive "
                         "law correctly specified and the conditional law genuinely over-confident.")
    ap.add_argument("--ou-rho", type=float, default=0.8)
    ap.add_argument("--ou-ell", type=float, default=200.0)
    ap.add_argument("--fit-iters", type=int, default=300, help="refit engine only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = np.asarray(args.spec, float)
    out_path = args.out or f"results/bayes/r5b_decision_sbc_{args.product}_{args.layer}_{args.engine}.json"
    sigma_obs = AKTA_NOISE_FLOOR_G_L

    # ---- one-time setup -----------------------------------------------------
    post0 = Posterior.load(Path(args.in_dir) / f"{args.product}_posterior.npz")
    dj = json.loads((Path(args.in_dir) / f"{args.product}_decision.json").read_text())
    a0 = np.asarray(dj["decision_op"], float)
    tol = np.asarray(dj.get("tol", [0.02, 0.05]), float)
    prod, drop = _PRODUCT_MAP.get(args.product, (args.product, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
    n = bundle.components.n_protein
    u0 = np.asarray(post0.u_map, float)
    base_prior = physical_prior(n)

    GGN, gmin, gmax = _ggn_from_posterior(post0.cov, base_prior.precision())
    print(f"[setup] GGN recovered from the committed posterior with no solver call "
          f"(eig {gmin:.2e}..{gmax:.2e})")

    ops = [a0] + ([] if args.pool_size <= 0 else
                  [np.asarray(o, float) for o in candidate_pool_for(bundle, n_candidates=args.pool_size,
                                                                    seed=args.seed)])
    g_fns, G_a, g0_a = [], [], []
    for op in ops:                                        # decision Jacobians frozen at the reference MAP
        fn, _, _ = decision_forward(bundle, list(op), u0, n_steps=args.n_steps)
        Gj, gj, _ = decision_jacobian(bundle, list(op), u0, n_steps=args.n_steps, return_extra=True)
        g_fns.append(fn)
        G_a.append(np.atleast_2d(Gj))
        g0_a.append(np.asarray(gj, float))
    Sig_floor = (args.floor_wdec**2) * np.diag(tol**2)
    Sig_disc = (args.disc_wdec**2) * np.diag(tol**2)       # generative decision discrepancy (0 = off)

    targets = targets_from_bundle(bundle, n_steps=args.n_steps)
    predict_fn, obs_real = mechanistic_model(targets, n)
    n_resid = int(obs_real.numel())
    ou_blocks = _ou_cov_blocks(targets, sigma_obs, args.ou_rho, args.ou_ell) if args.layer == "ou" else None

    # graph built ONCE at u0; each replicate reuses it for a single backward (VJP) when needed
    need_vjp = (args.layer == "ou") and args.engine == "frozen"
    if need_vjp:
        u_t = torch.tensor(u0, dtype=DTYPE, requires_grad=True)
        y0_t = predict_fn(u_t)
        print(f"[setup] built the forward graph once at u0 for reusable VJPs (n_resid={n_resid})")
    L_GGN = np.linalg.cholesky(GGN + 1e-12 * np.eye(GGN.shape[0]))   # for the exact i.i.d. J^T e sampler
    dim = len(u0)

    if args.engine == "gradcheck":
        # ------------------------------------------------------------------ frozen-engine validation
        # The frozen MAP is by construction the exact minimiser of the LINEARISED objective, so the TRUE
        # objective's gradient there measures precisely the nonlinearity the frozen engine ignores. One
        # Newton step with the frozen precision converts it into a first-order estimate of
        # (u_map_exact - u_map_frozen) -- about 2 forwards + 2 backwards per replicate (data simulation, the
        # VJP for the closed-form MAP, and the true gradient) against ~900 solve-equivalents for a full refit.
        # Those offsets are then propagated analytically into g_map, P(meet) and wdec, which is what the
        # calibration study actually consumes.
        Sig_prior = ((args.prior_scale**2) * np.asarray(post0.cov, float) if args.prior_cov == "full"
                     else np.diag((np.asarray(post0.std_u(), float) * args.prior_scale) ** 2))
        Sig_prior = 0.5 * (Sig_prior + Sig_prior.T)
        L_prior = np.linalg.cholesky(Sig_prior + 1e-14 * np.eye(dim))
        Lam0 = np.linalg.inv(Sig_prior)
        H = GGN + Lam0
        Sigma = np.linalg.inv(H)
        Sigma = 0.5 * (Sigma + Sigma.T)
        C0 = G_a[0] @ Sigma @ G_a[0].T
        from cex_model.bayes.posterior import _ggn_hessian
        u_graph = torch.tensor(u0, dtype=DTYPE, requires_grad=True)
        y0_graph = predict_fn(u_graph)                       # ONE forward, reused for every VJP
        rng = np.random.default_rng(args.seed)
        rows, ggn_rows = [], []
        for r in range(args.n_datasets):
            theta = u0 + L_prior @ rng.standard_normal(dim)
            try:
                with torch.no_grad():
                    mu_pred = predict_fn(torch.tensor(theta, dtype=DTYPE)).numpy()
                e = _draw_noise(args.layer, n_resid, sigma_obs, ou_blocks, rng)
                obs_np = mu_pred + e
                # frozen closed-form MAP from the ACTUAL data (VJP against the graph at u0)
                ct = torch.tensor((obs_np - y0_graph.detach().numpy()) / sigma_obs**2, dtype=DTYPE)
                s_r = torch.autograd.grad(y0_graph, u_graph, grad_outputs=ct, retain_graph=True)[0].numpy()
                u_frozen = u0 + Sigma @ s_r
                # TRUE gradient of the negative log posterior at the frozen MAP (1 fwd + 1 bwd)
                ut = torch.tensor(u_frozen, dtype=DTYPE, requires_grad=True)
                nll = -gaussian_loglik(ut, predict_fn, torch.tensor(obs_np, dtype=DTYPE), sigma_obs)
                gnll = torch.autograd.grad(nll, ut)[0].numpy()
                grad_true = gnll + Lam0 @ (u_frozen - u0)     # prior term of the local reference Gaussian
                du = -Sigma @ grad_true                       # one Newton step = first-order MAP offset
                lam2 = float(grad_true @ Sigma @ grad_true)   # Newton decrement^2 (scale-free)
                dg = G_a[0] @ du
                g_fr = g0_a[0] + G_a[0] @ (u_frozen - u0)
                dp_cond = (gaussian_meet_prob(g_fr + dg, C0, spec) - gaussian_meet_prob(g_fr, C0, spec))
                Cpred = C0 + Sig_floor
                dp_pred = (gaussian_meet_prob(g_fr + dg, Cpred, spec) - gaussian_meet_prob(g_fr, Cpred, spec))
                rows.append({"du_norm": float(np.linalg.norm(du)),
                             "du_rel_post_sd": float(np.linalg.norm(du / np.sqrt(np.diag(Sigma)))) / np.sqrt(dim),
                             "newton_decrement": float(np.sqrt(max(lam2, 0.0))),
                             "dg_whitened": float(np.linalg.norm(dg / tol)),
                             "dp_conditional": float(dp_cond), "dp_predictive": float(dp_pred)})
                if len(ggn_rows) < args.ggn_checks:           # tier 2: is Sigma really data-independent?
                    Hx = _ggn_hessian(predict_fn, u_frozen, sigma_obs) + Lam0
                    Sx = np.linalg.inv(0.5 * (Hx + Hx.T))
                    Cx = G_a[0] @ Sx @ G_a[0].T
                    wd = lambda C: float(np.sqrt(np.linalg.eigvalsh(C / tol[:, None] / tol[None, :]).max()))
                    ggn_rows.append({"rel_dSigma": float(np.linalg.norm(Sx - Sigma) / np.linalg.norm(Sigma)),
                                     "rel_dC": float(np.linalg.norm(Cx - C0) / np.linalg.norm(C0)),
                                     "wdec_frozen": wd(C0), "wdec_exact": wd(Cx),
                                     "dp_from_C": float(gaussian_meet_prob(g_fr, Cx, spec)
                                                        - gaussian_meet_prob(g_fr, C0, spec))})
            except Exception as ex:
                print(f"  replicate {r}: FAILED {type(ex).__name__}: {ex}")
            if (r + 1) % 20 == 0:
                print(f"  ... {r + 1}/{args.n_datasets}")

        def _ci(v):
            v = np.asarray(v, float)
            return [float(np.mean(v)), float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

        out = {"product": args.product, "engine": "gradcheck", "layer": args.layer,
               "n_datasets": args.n_datasets, "n_ok": len(rows), "prior_scale": args.prior_scale,
               "map_offset": {k: _ci([x[k] for x in rows]) for k in
                              ("du_norm", "du_rel_post_sd", "newton_decrement", "dg_whitened")},
               "propagated_dP": {k: _ci([x[k] for x in rows]) for k in ("dp_conditional", "dp_predictive")},
               "sigma_checks": ggn_rows}
        print(f"\n=== frozen-engine gradcheck: {args.product} layer={args.layer} "
              f"scale={args.prior_scale:g} ({len(rows)}/{args.n_datasets} ok) ===")
        print("first-order offset of the frozen MAP from the exact MAP (mean [95% interval]):")
        for k, lab in [("du_rel_post_sd", "|du| / posterior sd"), ("newton_decrement", "Newton decrement"),
                       ("dg_whitened", "|dg| in tolerance units")]:
            m, lo, hi = out["map_offset"][k]
            print(f"  {lab:26} {m:9.2e}  [{lo:.2e}, {hi:.2e}]")
        print("propagated effect on the quantity the calibration study consumes:")
        for k, lab in [("dp_conditional", "dP(meet) conditional"), ("dp_predictive", "dP(meet) predictive")]:
            m, lo, hi = out["propagated_dP"][k]
            print(f"  {lab:26} {m:+9.2e}  [{lo:+.2e}, {hi:+.2e}]")
        if ggn_rows:
            print(f"exact-GGN spot checks ({len(ggn_rows)}): is Sigma data-independent?")
            for i, g in enumerate(ggn_rows):
                print(f"  [{i}] rel dSigma={g['rel_dSigma']:.3e} rel dC={g['rel_dC']:.3e} "
                      f"wdec {g['wdec_frozen']:.4f}->{g['wdec_exact']:.4f} dP={g['dp_from_C']:+.2e}")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {out_path}")
        print("Read: the frozen engine is adequate for this study when |dg| stays well inside tolerance and")
        print("  the propagated dP(meet) is small next to the calibration effects being measured.")
        return

    def _run_one(scale):
      """The whole study at one SBC prior scale. All per-scale algebra is closed form."""
      # SBC prior: FULL posterior covariance (truths stay on the sloppy ridge) or diagonal marginals.
      if args.prior_cov == "full":
        Sig_prior = (scale**2) * np.asarray(post0.cov, float)
      else:
        Sig_prior = np.diag((np.asarray(post0.std_u(), float) * scale) ** 2)
      Sig_prior = 0.5 * (Sig_prior + Sig_prior.T)
      L_prior = np.linalg.cholesky(Sig_prior + 1e-14 * np.eye(dim))
      Lam0 = np.linalg.inv(Sig_prior)
      Sigma = np.linalg.inv(GGN + Lam0)
      Sigma = 0.5 * (Sigma + Sigma.T)                      # data-INDEPENDENT under the frozen engine
      sbc_prior = PhysicalPrior(mean=u0.copy(), std=np.sqrt(np.diag(Sig_prior)),
                                names=list(base_prior.names), n_protein=n)
      C_a = {law: [Ga @ Sigma @ Ga.T + (Sig_floor if law == "predictive" else 0.0) for Ga in G_a]
             for law in args.laws}                        # data-independent -> computed once per scale
      wdec_prior = float(np.sqrt(np.linalg.eigvalsh((G_a[0] @ Sig_prior @ G_a[0].T)
                                                    / tol[:, None] / tol[None, :]).max()))
      wdec_post = float(np.sqrt(np.linalg.eigvalsh((G_a[0] @ Sigma @ G_a[0].T)
                                                   / tol[:, None] / tol[None, :]).max()))
      print(f"[scale {scale:g}] prior cov={args.prior_cov}; decision spread (tolerance units): "
            f"truth {wdec_prior:.3f} vs posterior {wdec_post:.3f}"
            + (f"; generative discrepancy {args.disc_wdec:.3f} vs floor {args.floor_wdec:.3f}"
               if args.disc_wdec > 0 else "; no generative discrepancy"))

      rng = np.random.default_rng(args.seed)
      rec = {law: {"fixed": {"p": [], "y": []}, "selected": {"p": [], "y": []}} for law in args.laws}
      # SBC ranks are recorded PER LAW: with a generative decision discrepancy the conditional law should
      # fail rank uniformity while the predictive law (which carries the floor) should pass.
      ranks = {law: {"purity": [], "yield": [], "pit": []} for law in args.laws}
      L_disc = np.linalg.cholesky(Sig_disc) if args.disc_wdec > 0 else None
      n_ok = 0

      for r in range(args.n_datasets):
        theta = u0 + L_prior @ rng.standard_normal(dim)
        try:
            if args.engine == "frozen":
                # s_r = J^T (obs_r - y0)/sigma^2 = GGN (theta-u0) + J^T e_r/sigma^2   (frozen J)
                if args.layer == "iid":
                    xi = L_GGN @ rng.standard_normal(GGN.shape[0])       # ~ N(0, GGN): EXACT, no solve
                else:
                    e = _draw_noise("ou", n_resid, sigma_obs, ou_blocks, rng)
                    ct = torch.tensor(e / sigma_obs**2, dtype=DTYPE)
                    xi = torch.autograd.grad(y0_t, u_t, grad_outputs=ct, retain_graph=True)[0].numpy()
                u_map_r = u0 + Sigma @ (GGN @ (theta - u0) + xi)
                Sig_r = Sigma
            else:                                                        # exact per-replicate MAP+Laplace
                with torch.no_grad():
                    mu = predict_fn(torch.tensor(theta, dtype=DTYPE)).numpy()
                e = _draw_noise(args.layer, n_resid, sigma_obs, ou_blocks, rng)
                obs_r = torch.tensor(mu + e, dtype=DTYPE)
                u_map_r, _ = map_fit(u0, predict_fn, obs_r, sbc_prior, sigma_obs=sigma_obs,
                                     iters=args.fit_iters)
                Sig_r = laplace_posterior(u_map_r, predict_fn, obs_r, sbc_prior, template=bundle.components,
                                          sigma_obs=sigma_obs).cov

            du = np.asarray(u_map_r, float) - u0
            g_map = [g0_a[k] + G_a[k] @ du for k in range(len(ops))]
            Cr = C_a if args.engine == "frozen" else \
                {law: [G_a[k] @ Sig_r @ G_a[k].T + (Sig_floor if law == "predictive" else 0.0)
                       for k in range(len(ops))] for law in args.laws}

            # ---- ground-truth decision at the fixed action (NONLINEAR), plus the generative
            #      product-level decision discrepancy the predictive floor is meant to cover
            delta = (L_disc @ rng.standard_normal(2)) if L_disc is not None else np.zeros(2)
            with torch.no_grad():
                g_obs0 = g_fns[0](torch.tensor(theta, dtype=DTYPE)).numpy() + delta
            y_fixed = float(np.all(g_obs0 >= spec))

            # ---- SBC on g at the fixed action, scored against EACH law's decision draws
            z = rng.standard_normal((args.n_post, 2))
            for law in args.laws:
                Cl = Cr[law][0]
                gd = g_map[0] + z @ np.linalg.cholesky(Cl + 1e-18 * np.eye(2)).T
                ranks[law]["purity"].append(int((gd[:, 0] < g_obs0[0]).sum()))
                ranks[law]["yield"].append(int((gd[:, 1] < g_obs0[1]).sum()))
                d0 = g_obs0 - g_map[0]
                q = float(d0 @ np.linalg.solve(Cl + 1e-18 * np.eye(2), d0))
                if np.isfinite(q):                            # guard: non-finite PIT would NaN the KS test
                    ranks[law]["pit"].append(float(stats.chi2(2).cdf(q)))
                rec[law]["fixed"]["p"].append(gaussian_meet_prob(g_map[0], Cl, spec))
                rec[law]["fixed"]["y"].append(y_fixed)

            # ---- selected action (separate arm; only the SELECTED op needs a nonlinear solve)
            if args.pool_size > 0:
                truth_cache = {0: g_obs0}
                for law in args.laws:
                    ps = [gaussian_meet_prob(g_map[k], Cr[law][k], spec) for k in range(len(ops))]
                    k = int(np.argmax(ps))
                    if k not in truth_cache:
                        with torch.no_grad():
                            truth_cache[k] = g_fns[k](torch.tensor(theta, dtype=DTYPE)).numpy() + delta
                    rec[law]["selected"]["p"].append(ps[k])
                    rec[law]["selected"]["y"].append(float(np.all(truth_cache[k] >= spec)))
            n_ok += 1
        except Exception as ex:
            print(f"  replicate {r}: FAILED {type(ex).__name__}: {ex}")
            continue
        if (r + 1) % 20 == 0:
            print(f"  ... {r + 1}/{args.n_datasets} ({n_ok} ok)")

      return {"product": args.product, "layer": args.layer, "engine": args.engine,
              "prior_cov": args.prior_cov, "prior_scale": float(scale),
              "wdec_truth_spread": wdec_prior, "wdec_posterior_spread": wdec_post,
              "n_datasets": args.n_datasets, "n_ok": n_ok, "n_post": args.n_post, "spec": spec.tolist(),
              "floor_wdec": args.floor_wdec, "disc_wdec": args.disc_wdec,
              "ou": {"rho": args.ou_rho, "ell": args.ou_ell},
              "sbc_g": {law: [_rank_stats(ranks[law]["purity"], args.n_post, "pool_purity"),
                              _rank_stats(ranks[law]["yield"], args.n_post, "pool_yield"),
                              {"label": "2d_mahalanobis_pit", "n": len(ranks[law]["pit"]),
                               "ks_pvalue": float(stats.kstest(ranks[law]["pit"], "uniform").pvalue)
                               if len(ranks[law]["pit"]) > 2 else float("nan")}]
                        for law in args.laws},
              "pmeet": {law: {arm: _score_block(rec[law][arm]["p"], rec[law][arm]["y"], f"{law}:{arm}")
                              for arm in ("fixed", "selected") if rec[law][arm]["p"]}
                        for law in args.laws}}

    scales = args.scale_sweep if args.scale_sweep else [args.prior_scale]
    runs = [_run_one(s) for s in scales]

    for out in runs:
        print(f"\n=== R5b {args.product} layer={args.layer} engine={args.engine} "
              f"prior={args.prior_cov} scale={out['prior_scale']:g} ({out['n_ok']}/{args.n_datasets} ok) ===")
        print("SBC on g per law (rank uniformity; PASS if p > 0.017 = 0.05/3 Bonferroni):")
        for law, ss in out["sbc_g"].items():
            for s in ss:
                extra = f"  mean rank frac={s['mean_rank_frac']:.3f}" if "mean_rank_frac" in s else ""
                print(f"  {law:12} {s['label']:20} KS p={s['ks_pvalue']:.3f}{extra}")
        print("P(meet) reliability (calibrated iff intercept~0, slope~1, tail hit ~ nominal):")
        for law, arms in out["pmeet"].items():
            for arm, b in arms.items():
                note = b.get("cal_note", "")
                print(f"  {law:12} {arm:8} mean_p={b['mean_p']:.3f} base={b['base_rate']:.3f} "
                      f"ECE={b['ece']:.3f} Brier={b['brier']:.4f} log={b['log_score']:.4f} "
                      f"int={b['intercept']:.2f} slope={b['slope']:.2f} "
                      f"| p>=.95: n={b['tail_hi_n']} hit={b['tail_hi_hit_freq']:.3f}" +
                      (f"  [{note}]" if note else ""))

    if len(runs) > 1:
        print(f"\n=== calibration vs regime width ({args.product}, layer={args.layer}) ===")
        print(f"{'scale':>6} {'truth wdec':>11} {'post wdec':>10} {'KS(cond)':>9} {'KS(pred)':>9} "
              f"{'Brier cond':>11} {'Brier pred':>11} {'base rate':>10}")
        for out in runs:
            bc = out["pmeet"].get("conditional", {}).get("fixed", {})
            bp = out["pmeet"].get("predictive", {}).get("fixed", {})
            ksc = out["sbc_g"].get("conditional", [{}])[0].get("ks_pvalue", float("nan"))
            ksp = out["sbc_g"].get("predictive", [{}])[0].get("ks_pvalue", float("nan"))
            print(f"{out['prior_scale']:6g} {out['wdec_truth_spread']:11.3f} {out['wdec_posterior_spread']:10.3f} "
                  f"{ksc:9.3f} {ksp:9.3f} {bc.get('brier', float('nan')):11.4f} "
                  f"{bp.get('brier', float('nan')):11.4f} {bc.get('base_rate', float('nan')):10.3f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(runs if len(runs) > 1 else runs[0], indent=2))
    print(f"\nwrote {out_path}")
    print("Read: layer=iid should be CALIBRATED (pipeline validation; the residual deviation is the")
    print("  decision-map linearisation only, and it grows with the prior scale). layer=ou should show the")
    print("  conditional law over-confident with the predictive law materially better.")


if __name__ == "__main__":
    main()
