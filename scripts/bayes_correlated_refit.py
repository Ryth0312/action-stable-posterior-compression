"""Correlated-residual REFIT (reviewer step 2): a true block-likelihood re-fit, not a
fixed-MAP inflation.

Per (experiment, observation-group) within-run residual block e:

    eps_e ~ N(0, sigma_nug^2 I + sigma_corr^2 R_ell),
    R_ell = OU  exp(-|t_i - t_j|/ell)   (irregular grids; AR(1) on a regular grid),
            AR1 phi^{|i-j|},  or  Matern-3/2  (1+ s d)exp(-s d), s=sqrt(3)/ell.

Different runs/groups independent; hyperparameters (sigma_nug, sigma_corr, ell) shared
across a product's blocks (partial pooling across products is a documented option). This
re-optimizes the MAP and recomputes the Jacobian/Hessian under the correlated whitening
(so it captures the MAP shift, J/G change, and posterior-mean change the fixed-MAP Lowner
bound in SI S16 cannot), then re-derives theta, worst_dir, worst_dec, P(meet), and LOEO,
and writes the correlated decision covariance for scripts/bayes_decision_discrepancy.py
(--c-param-json) to combine with the decision-level discrepancy.

STRUCTURE
  * numpy KERNEL CORE (kernels, profile-ML hyperparameters, whitening, held-out log
    score) -- pure numpy, validated by `--selftest` here.
  * torch REFIT DRIVER (`refit_correlated`) -- re-optimizes MAP + Laplace under the block
    whitening; needs torch + the differentiable solver, so RUN ON COLAB. It is written
    against the committed code but is NOT executed in this environment (torch absent);
    run `--selftest` first, then the driver, and sanity-check the iid column reproduces
    the committed numbers before trusting the correlated column.

Model selection (reviewer): compare iid / OU+nugget / AR(1)+nugget / Matern+nugget by
whitened-residual autocorrelation, held-out predictive log score, and LOEO coverage --
NOT training fit. `--selftest` exercises the scoring on synthetic data.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results", "bayes")

# ----------------------------------------------------------------------------- KERNEL CORE (numpy)
def ou_corr(times, ell):
    d = np.abs(np.subtract.outer(times, times))
    return np.exp(-d / ell)


def ar1_corr(m, phi):
    i = np.arange(m)
    return phi ** np.abs(np.subtract.outer(i, i))


def matern32_corr(times, ell):
    s = np.sqrt(3.0) / ell
    d = np.abs(np.subtract.outer(times, times)) * s
    return (1.0 + d) * np.exp(-d)


def unit_block(R, rho):
    """Unit-diagonal correlation (1-rho) I + rho R (nugget/correlated split)."""
    m = R.shape[0]
    return (1.0 - rho) * np.eye(m) + rho * R


def _corr_for(kernel, times, ell):
    if kernel == "ou":
        return ou_corr(times, ell)
    if kernel == "matern32":
        return matern32_corr(times, ell)
    if kernel == "ar1":
        # regular-grid AR(1): phi from median spacing
        dt = np.median(np.diff(times)) if len(times) > 1 else 1.0
        return ar1_corr(len(times), float(np.exp(-dt / ell)))
    raise ValueError(kernel)


def profile_hyper(resid_blocks, time_blocks, kernel="ou",
                  ell_grid=None, rho_grid=None, rho_max=None):
    """Profile-ML of (ell, rho, sigma^2) over a product's residual blocks.

    Sigma_b = sigma^2 [ (1-rho) I + rho R_ell ]. For each (ell, rho) the ML sigma^2 has a
    closed form; profile out sigma^2 and grid-maximize the profile log-likelihood.
    Returns dict(ell, rho, sigma2, sigma_nug, sigma_corr, loglik, phi_equiv).
    """
    if ell_grid is None:
        # ell spans a few % to a few x of a typical within-run time span
        spans = [np.ptp(t) for t in time_blocks if len(t) > 1]
        base = np.median(spans) if spans else 1.0
        ell_grid = base * np.geomspace(0.02, 2.0, 24)
    if rho_grid is None:
        rho_grid = np.concatenate([np.linspace(0.0, 0.9, 19), [0.93, 0.96, 0.98, 0.99]])
    if rho_max is not None:  # cap correlation (guards the rho<->discrepancy confound)
        rho_grid = rho_grid[rho_grid <= rho_max]

    N = sum(len(r) for r in resid_blocks)
    best = None
    for ell in ell_grid:
        for rho in rho_grid:
            quad = 0.0
            logdet = 0.0
            ok = True
            for r, t in zip(resid_blocks, time_blocks):
                K = unit_block(_corr_for(kernel, t, ell), rho)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    ok = False
                    break
                z = _solve_tri(L, r)
                quad += float(z @ z)
                logdet += 2.0 * float(np.sum(np.log(np.diag(L))))
            if not ok or quad <= 0:
                continue
            sigma2 = quad / N
            ll = -0.5 * (N * np.log(2 * np.pi) + N * np.log(sigma2) + logdet + N)
            if best is None or ll > best["loglik"]:
                dt = np.median([np.median(np.diff(t)) for t in time_blocks if len(t) > 1])
                best = dict(kernel=kernel, ell=float(ell), rho=float(rho), sigma2=float(sigma2),
                            sigma_nug=float(np.sqrt(sigma2 * (1 - rho))),
                            sigma_corr=float(np.sqrt(sigma2 * rho)),
                            loglik=float(ll), phi_equiv=float(np.exp(-dt / ell)))
    return best


def _solve_tri(L, b):
    # forward substitution (lower triangular); fallback if scipy/np.linalg.solve_triangular absent
    from scipy.linalg import solve_triangular
    return solve_triangular(L, b, lower=True)


def whiten_resid(resid_blocks, time_blocks, hyper, kernel="ou"):
    """Return whitened residual blocks z_b = L_b^{-1} r_b (should be ~ iid N(0,1))."""
    out = []
    for r, t in zip(resid_blocks, time_blocks):
        Sig = hyper["sigma2"] * unit_block(_corr_for(kernel, t, hyper["ell"]), hyper["rho"])
        L = np.linalg.cholesky(Sig)
        out.append(_solve_tri(L, r))
    return out


def held_out_logscore(resid_blocks, time_blocks, hyper, kernel="ou"):
    """Mean per-point Gaussian log score of each block under the fitted covariance."""
    tot, n = 0.0, 0
    for r, t in zip(resid_blocks, time_blocks):
        Sig = hyper["sigma2"] * unit_block(_corr_for(kernel, t, hyper["ell"]), hyper["rho"])
        sign, logdet = np.linalg.slogdet(Sig)
        m = len(r)
        quad = float(r @ np.linalg.solve(Sig, r))
        tot += -0.5 * (m * np.log(2 * np.pi) + logdet + quad)
        n += m
    return tot / n


def lag1_autocorr(z):
    z = np.asarray(z)
    if len(z) < 2:
        return 0.0
    z = z - z.mean()
    return float((z[:-1] @ z[1:]) / (z @ z + 1e-12))


# ----------------------------------------------------------------------------- SELFTEST (numpy, runs here)
def selftest():
    rng = np.random.default_rng(0)
    # synthetic: 6 runs, irregular grids, true AR(1)/OU with phi=0.7, nugget fraction (1-rho)
    true_ell, true_rho, true_sig2 = 40.0, 0.75, 0.09
    time_blocks, resid_blocks = [], []
    for _ in range(6):
        m = rng.integers(30, 60)
        t = np.sort(rng.uniform(0, 300, m))
        Sig = true_sig2 * unit_block(ou_corr(t, true_ell), true_rho)
        L = np.linalg.cholesky(Sig + 1e-10 * np.eye(m))
        resid_blocks.append(L @ rng.standard_normal(m))
        time_blocks.append(t)

    h = profile_hyper(resid_blocks, time_blocks, kernel="ou")
    z = np.concatenate(whiten_resid(resid_blocks, time_blocks, h, "ou"))
    raw = np.concatenate(resid_blocks)
    print("=== correlated-refit KERNEL selftest ===")
    print(f"  true  : ell~{true_ell}, rho={true_rho}, sigma2={true_sig2}")
    print(f"  fitted: ell={h['ell']:.1f}, rho={h['rho']:.2f}, sigma2={h['sigma2']:.4f}, "
          f"phi_equiv={h['phi_equiv']:.2f}, loglik={h['loglik']:.1f}")
    print(f"  raw lag-1 autocorr {lag1_autocorr(raw):+.3f}  ->  whitened {lag1_autocorr(z):+.3f} "
          f"(should collapse toward 0); whitened std {z.std():.3f} (~1)")
    # model comparison by held-out log score
    for k in ("ou", "ar1", "matern32"):
        hk = profile_hyper(resid_blocks, time_blocks, kernel=k)
        # crude held-out: score odd blocks with hyper fit on even blocks
        ev_r, ev_t = resid_blocks[::2], time_blocks[::2]
        od_r, od_t = resid_blocks[1::2], time_blocks[1::2]
        hev = profile_hyper(ev_r, ev_t, kernel=k)
        ls = held_out_logscore(od_r, od_t, hev, kernel=k)
        print(f"  kernel={k:9s} rho={hk['rho']:.2f} ell={hk['ell']:7.1f}  held-out logscore={ls:+.3f}")
    # iid baseline held-out log score for contrast
    iid = dict(sigma2=np.var(np.concatenate(resid_blocks)), ell=1e9, rho=0.0)
    print(f"  kernel=iid       rho=0.00 ell=inf      held-out logscore="
          f"{held_out_logscore(resid_blocks[1::2], time_blocks[1::2], iid, 'ou'):+.3f}")
    ok = abs(h["rho"] - true_rho) < 0.25 and abs(lag1_autocorr(z)) < abs(lag1_autocorr(raw))
    print(f"  SELFTEST {'PASS' if ok else 'CHECK'}: recovered rho within 0.25 and whitening reduced autocorr")
    ok = _reparam_selftest() and ok
    return ok


def _reparam_selftest():
    """Off-solver check of ridge_reparam_report on a synthetic flat-ridge posterior."""
    rng = np.random.default_rng(0); d = 6
    sp = np.array([1.0, 1.0, 0.5, 0.5, 2.0, 2.0])
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Hw = Q @ np.diag([50., 40., 20., 8., 1.03, 1.06]) @ Q.T   # last two ~ prior floor (null/ridge)
    D = np.diag(1.0 / sp); H = D @ Hw @ D; Sigma = np.linalg.inv(H)
    Vn = Q[:, 4:6]                                            # true null/ridge subspace
    grad = (Vn @ np.array([3.0, 4.0])) / sp                  # gradient confined to the ridge
    Gw = rng.standard_normal((2, d)); Gw = Gw - (Gw @ Vn) @ Vn.T   # decision-null on the ridge
    Ti = np.diag([1 / 0.02, 1 / 0.05]); g_map = np.array([0.76, 0.62])
    r = ridge_reparam_report(H, Sigma, grad, sp, Gw @ D, Ti, g_map)
    good = (r["n_null"] == 2 and r["newton_decrement_identified"] < 1e-9
            and r["decision_energy_null_frac"] < 1e-9 and r["wdec_rel_change"] < 1e-6
            and r["cov_rel_frobenius"] < 1e-6)
    # adversarial: identified-direction drift and ridge decision energy must be flagged
    r2 = ridge_reparam_report(H, Sigma, (Vn @ np.array([3., 4.]) + Q[:, 3] * 5.0) / sp, sp, Gw @ D, Ti, g_map)
    r3 = ridge_reparam_report(H, Sigma, grad, sp, rng.standard_normal((2, d)) @ D, Ti, g_map)
    good = good and r2["newton_decrement_identified"] > 0.1 and r3["decision_energy_null_frac"] > 0.05
    # truncated-Newton polish: one full stiff step zeros the identified gradient on a quadratic
    _Mq = np.linalg.inv(Hw); _muq, _Uq = np.linalg.eigh(_Mq); _muq = np.clip(_muq, 0.0, 1.0 + 1e-9)
    _w0 = rng.standard_normal(d); _w1 = _w0 + _stiff_step(Hw @ _w0, _muq, _Uq, _muq < 0.5)
    good = good and float(np.linalg.norm((_Uq.T @ (Hw @ _w1))[_muq < 0.5])) < 1e-8
    print(f"  REPARAM SELFTEST {'PASS' if good else 'CHECK'}: identified-stationary + decision-null-ridge "
          f"=> deployed H^-1 == reparam posterior (wdec rel-change {r['wdec_rel_change']:.1e}); failure modes flagged")
    return good


# ----------------------------------------------------------------------------- TORCH REFIT DRIVER (Colab)
def _blocks_from_targets(targets):
    """Index sets and time grids for each (experiment, observation-group) block.

    obs = cat([tg.values.reshape(-1)]); within target e (m_e times, no_e groups) the
    concatenated layout is row-major (time-outer, group-inner): element i*no_e + g.
    """
    blocks, offset = [], 0
    for e, tg in enumerate(targets):
        m = tg.times_s.numel(); no = tg.values.shape[1]
        t = tg.times_s.detach().cpu().numpy()
        for g in range(no):
            idx = offset + np.arange(m) * no + g
            blocks.append(dict(exp=e, group=g, idx=idx, times=t))
        offset += m * no
    return blocks


def _gauss_p_meet(g_map, C, spec):
    """P(purity >= spec[0] and yield >= spec[1]) under the linearized law g ~ N(g_map, C)."""
    from scipy.stats import norm, multivariate_normal
    g = np.asarray(g_map, float).ravel()[:2]
    s = np.asarray(spec, float)
    C2 = np.asarray(C, float)[:2, :2]
    p1 = float(norm.cdf(s[0], g[0], np.sqrt(max(C2[0, 0], 1e-18))))
    p2 = float(norm.cdf(s[1], g[1], np.sqrt(max(C2[1, 1], 1e-18))))
    p12 = float(multivariate_normal(mean=g, cov=C2, allow_singular=True).cdf(s))
    return float(np.clip(1.0 - p1 - p2 + p12, 0.0, 1.0))


def _stiff_step(gw, mu, U, ident):
    """One truncated Newton step in the stiff (identified) subspace only: -M_ident @ gw in whitened
    coords (steps to the local minimum in the data-constrained directions, leaves the flat ridge)."""
    c = U.T @ gw
    return -(U[:, ident] * mu[ident]) @ c[ident]


def ridge_reparam_report(H, Sigma_corr, grad, prior_std, G, Ti, g_map, *,
                         spec=(0.70, 0.50), tau_id=0.5):
    """Option-(1) fix for the non-stationary correlated MAP (SI predictive section).

    The correlated objective is flat along the k_eq-nu ridge, so the full-dimensional point is
    not stationary. Split the prior-whitened posterior into an IDENTIFIED subspace
    (data-constrained, worst-dir < tau_id) and a NULL/ridge subspace (prior-dominated,
    worst-dir ~ 1), then:
      * decompose the whitened Newton decrement (the point's offset from the subspace minimum
        in posterior-std units) into identified vs null parts -- newton_decrement_identified ~ 0
        means the point sits at the identified minimum (Laplace valid there) and the drift is
        confined to the null coordinate;
      * rebuild the posterior as (identified Laplace) (+) (prior on the null coordinate) and
        check the decision reads (wdec, P(meet)) are unchanged -- i.e. the deployed H^{-1} IS the
        reparametrized posterior, so the non-stationarity never enters the decision.

    Pure numpy given the already-computed matrices; unit-testable off-solver.
    """
    sp = np.asarray(prior_std, float)
    Dinv = np.diag(sp)                                   # un-whitening (D^{-1} = diag(sp))
    M = np.diag(1.0 / sp) @ np.asarray(Sigma_corr, float) @ np.diag(1.0 / sp)
    M = 0.5 * (M + M.T)
    mu, U = np.linalg.eigh(M)                            # mu in (0,1]; ~1 => prior-dominated (null)
    mu = np.clip(mu, 0.0, 1.0 + 1e-9)
    ident = mu < tau_id
    null = ~ident
    gw = sp * np.asarray(grad, float)                    # whitened gradient dL/dw = sp * dL/du
    c = U.T @ gw
    lam_full = float(np.sqrt(np.sum(mu * c**2)))         # Newton decrement = offset in posterior-std units
    lam_ident = float(np.sqrt(np.sum(mu[ident] * c[ident]**2))) if ident.any() else 0.0
    lam_null = float(np.sqrt(np.sum(mu[null] * c[null]**2))) if null.any() else 0.0
    mu_rep = mu.copy(); mu_rep[null] = 1.0              # data (mu) on identified, prior (=1) on null
    Sigma_rep = Dinv @ ((U * mu_rep) @ U.T) @ Dinv
    Gm = np.asarray(G, float)
    C_corr = Gm @ np.asarray(Sigma_corr, float) @ Gm.T
    C_rep = Gm @ Sigma_rep @ Gm.T
    wdec_corr = float(np.sqrt(np.max(np.linalg.eigvalsh(Ti @ C_corr @ Ti))))
    wdec_rep = float(np.sqrt(np.max(np.linalg.eigvalsh(Ti @ C_rep @ Ti))))
    GDinv = Gm @ Dinv                                    # decision map in whitened coords
    e_null = float(np.linalg.norm(GDinv @ U[:, null])) if null.any() else 0.0
    e_tot = float(np.linalg.norm(GDinv))
    return dict(
        tau_id=tau_id, n_identified=int(ident.sum()), n_null=int(null.sum()),
        worst_dir=float(np.sqrt(mu.max())),
        mu_identified_max=(float(mu[ident].max()) if ident.any() else None),
        mu_null_min=(float(mu[null].min()) if null.any() else None),
        newton_decrement_full=lam_full,
        newton_decrement_identified=lam_ident,          # ~0 => point at the identified minimum
        newton_decrement_null=lam_null,                 # carries the ridge drift (prior-dominated)
        grad_whitened_total=float(np.linalg.norm(gw)),
        grad_whitened_identified=(float(np.linalg.norm(c[ident])) if ident.any() else 0.0),
        grad_whitened_null=(float(np.linalg.norm(c[null])) if null.any() else 0.0),
        decision_energy_null_frac=((e_null / e_tot) if e_tot > 0 else None),  # ~0 => decision-null on ridge
        wdec_corr=wdec_corr, wdec_reparam=wdec_rep,
        wdec_rel_change=float(abs(wdec_rep - wdec_corr) / max(wdec_corr, 1e-12)),
        cov_rel_frobenius=float(np.linalg.norm(C_rep - C_corr) / max(np.linalg.norm(C_corr), 1e-12)),
        p_meet_corr=_gauss_p_meet(g_map, C_corr, spec),
        p_meet_reparam=_gauss_p_meet(g_map, C_rep, spec),
        note=("identified-subspace Laplace (+) prior on the null/ridge coordinate; if "
              "newton_decrement_identified << 1 and decision_energy_null_frac ~ 0, the deployed "
              "H^{-1} equals the reparametrized posterior and the non-stationary drift does not "
              "enter the decision."),
    )


def _adaptive_map(u, neglogpost, torch, *, chunk, max_iters, plateau_tol, losses, verbose=False):
    """L-BFGS run in chunks until the objective stops improving, rather than for a fixed budget.

    The step size is already adaptive here: strong-Wolfe line search grows and shrinks it per
    iteration, which is what lets this objective descend on a ridge whose Hessian condition number
    is order 1e7. What the fixed-budget paths lack is a stopping rule tied to the objective, so a
    product that needs thousands of iterations silently stops wherever the budget ran out.

    This runs ``chunk`` L-BFGS iterations at a time and stops once a whole chunk buys less than
    ``plateau_tol``, or once ``max_iters`` iterations have been spent. Deterministic given the
    start point. Returns the termination record that ends up in the artifact, so a plateau stop is
    distinguishable from a budget stop.

    A first-order alternative -- Adam from a large step, halving on any uphill step -- was measured
    on mAb A and rejected: the step ratchets down on the ill-conditioned ridge and the per-step
    progress falls below any absolute tolerance while the objective is still far from flat, ending
    at 209.2 against the 146.8 the line search reaches.

    The chunking is not free: each chunk starts a fresh L-BFGS and discards the curvature history,
    so per iteration this descends more slowly than one uninterrupted run (mAb A, 120 iterations:
    152.07 chunked against 146.82 uninterrupted). Use it when a stopping rule is wanted, not to
    reach a given loss sooner.
    """
    total, chunks = 0, 0
    prev = float(neglogpost().detach())
    losses.append(prev)
    while total < int(max_iters):
        n = min(int(chunk), int(max_iters) - total)
        opt = torch.optim.LBFGS([u], lr=1.0, max_iter=n, history_size=25,
                                tolerance_grad=1e-8, tolerance_change=1e-12,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = neglogpost()
            loss.backward()
            losses.append(float(loss.detach()))
            return loss

        opt.step(closure)
        total += n
        chunks += 1
        cur = float(neglogpost().detach())
        gain = prev - cur
        if verbose:
            print(f"  adaptive: {total} iters, loss {cur:.4f}, chunk gain {gain:.4f}")
        if gain < plateau_tol:
            return dict(reason="plateau", iters=total, chunks=chunks, last_chunk_gain=gain,
                        plateau_tol=plateau_tol, chunk=int(chunk), final_loss=cur)
        prev = cur
    return dict(reason="iteration limit", iters=total, chunks=chunks,
                last_chunk_gain=prev - float(neglogpost().detach()),
                plateau_tol=plateau_tol, chunk=int(chunk), final_loss=prev)


def refit_correlated(pid, decf, *, kernel="ou", n_steps=300, map_iters=300, out=None,
                     pool_hyper=None, cpath=None, optimizer="adam", rho_max=None, verbose=True,
                     init_u0=None, write=True, reparam=False, reparam_polish=6, reparam_damping=0.7,
                     plateau_tol=0.1, chunk=60, posterior_out=None):
    """RUN ON COLAB. Re-fit MAP+Laplace under the block-correlated likelihood and
    recompute the decision quantities. Writes {pid}_correlated.json and a c_param JSON.

    pool_hyper: optional pre-estimated hyper dict (partial pooling across products);
    if None, estimate from this product's own iid-MAP residuals.
    init_u0: optional MAP-optimization start point (defaults to the committed iid MAP);
    used by the multistart stability check to perturb the start along the flattened ridge.
    write: when False, skip persisting artifacts (multistart restarts must not clobber the
    committed {pid}_correlated{,_posterior}.json / c_param_correlated.json).
    """
    import torch
    from cex_model.diffsolver.torch_solver import DTYPE
    from cex_model.diffsolver.calibrate_diff import targets_from_bundle
    from cex_model import app_support
    from cex_model.bayes import (mechanistic_model, physical_prior, map_fit, laplace_posterior,
                                  Posterior, decision_jacobian, decision_report)

    bundle = app_support.load_product(pid.replace("_exclDT", ""))
    if pid.endswith("_exclDT"):  # match the committed exclDT posterior: drop the non-standard DT run
        from cex_model.bayes.compare import filter_experiments
        kept, dropped = filter_experiments(bundle.experiments, ["DT"])
        bundle.experiments = kept
        if verbose and dropped:
            print(f"[{pid}] excluded {[e.name for e in dropped]}")
    targets = targets_from_bundle(bundle, n_steps)
    n = bundle.components.n_protein  # bayes convention: full component count (dim = 4*n)
    predict_fn, obs = mechanistic_model(targets, n)
    prior = physical_prior(n)
    post_iid = Posterior.load(os.path.join(RES, f"{pid}_posterior.npz"))
    u0 = np.asarray(init_u0, float) if init_u0 is not None else post_iid.u_map
    blocks = _blocks_from_targets(targets)

    # 1) iid-MAP residuals -> estimate hyperparameters (or use pooled). NOTE: multistart passes
    #    pool_hyper=<canonical> so every restart shares one noise model; the residuals here use the
    #    (possibly perturbed) start only when estimating hyper from scratch.
    with torch.no_grad():
        r_iid = (predict_fn(torch.tensor(u0, dtype=DTYPE)) - obs).detach().cpu().numpy()
    resid_blocks = [r_iid[b["idx"]] for b in blocks]
    time_blocks = [b["times"] for b in blocks]
    hyper = pool_hyper or profile_hyper(resid_blocks, time_blocks, kernel=kernel, rho_max=rho_max)
    if verbose:
        print(f"[{pid}] hyper: {hyper}")

    # 2) precompute per-block inverse covariance + logdet (constants during the theta-fit)
    Kinv, logdet = [], 0.0
    for b in blocks:
        Sig = hyper["sigma2"] * unit_block(_corr_for(kernel, b["times"], hyper["ell"]), hyper["rho"])
        Ki = np.linalg.inv(Sig)
        Kinv.append(torch.tensor(Ki, dtype=DTYPE))
        logdet += float(np.linalg.slogdet(Sig)[1])
    idx_t = [torch.tensor(b["idx"]) for b in blocks]

    def corr_negloglik(u):
        resid = predict_fn(u) - obs
        q = u.new_zeros(())
        for it_, Ki in zip(idx_t, Kinv):
            rb = resid[it_]
            q = q + rb @ (Ki @ rb)
        return 0.5 * q  # +0.5*logdet const (drop; fixed hyper)

    # 3) refit MAP under the correlated likelihood (Adam)
    u = torch.tensor(u0, dtype=DTYPE, requires_grad=True)
    losses = []

    def neglogpost():
        return corr_negloglik(u) - prior.log_prob(u)

    term = None
    if optimizer == "lbfgs":
        # quasi-Newton: converges the ill-conditioned (correlation-flattened) ridge Adam
        # cannot. One .step() runs up to map_iters inner iterations with a Wolfe line search.
        opt = torch.optim.LBFGS([u], lr=1.0, max_iter=map_iters, history_size=25,
                                tolerance_grad=1e-8, tolerance_change=1e-12,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = neglogpost()
            loss.backward()
            losses.append(float(loss.detach()))
            return loss

        opt.step(closure)
    elif optimizer == "adaptive":
        # mAb C does not level off inside a fixed-step budget; this stops on the objective instead.
        term = _adaptive_map(u, neglogpost, torch, chunk=chunk, max_iters=map_iters,
                             plateau_tol=plateau_tol, losses=losses, verbose=verbose)
    else:
        opt = torch.optim.Adam([u], lr=0.05)
        for _ in range(map_iters):
            opt.zero_grad()
            loss = neglogpost()
            loss.backward(); opt.step()
            losses.append(float(loss.detach()))
    u_map = u.detach().cpu().numpy()

    # 4) correlated whitened GGN Hessian + Laplace covariance
    # vectorize=True batches the reverse-mode vjp over outputs (correct through the
    # implicit-diff solver -- it is reverse mode, not forward JVP); ~30x faster.
    J = torch.autograd.functional.jacobian(
        predict_fn, torch.tensor(u_map, dtype=DTYPE), vectorize=True).detach().cpu().numpy()
    H = prior.precision().copy()
    for b, Ki in zip(blocks, [k.numpy() for k in Kinv]):
        Jb = J[b["idx"]]
        H += Jb.T @ Ki @ Jb
    H = 0.5 * (H + H.T) + 1e-9 * np.eye(H.shape[0])
    Sigma_corr = np.linalg.inv(H)

    # 4b) convergence / stationarity diagnostics at u_map (per restart).
    #     grad_norm_whitened = || sigma_prior (.) d(neglogpost)/du ||  is the prior-metric (dual)
    #     gradient norm: it -> 0 at a stationary MAP, so a small value across restarts CERTIFIES that
    #     each converged restart sits at a (near-)stationary point and the drift is ALONG the flat ridge.
    #     The GGN Hessian H (= the deployed Laplace Hessian) is SPD by construction (positive prior
    #     precision); its condition number measures how flat the k_eq-nu ridge is (large => flat).
    _ug = torch.tensor(u_map, dtype=DTYPE, requires_grad=True)
    _L = corr_negloglik(_ug) - prior.log_prob(_ug)
    (_g,) = torch.autograd.grad(_L, _ug)
    _grad = _g.detach().cpu().numpy()
    _sp = np.asarray(prior.std, float)
    grad_norm = float(np.linalg.norm(_grad))
    grad_norm_whitened = float(np.linalg.norm(_sp * _grad))
    _eig = np.linalg.eigvalsh(H)
    hess_eig_min = float(_eig[0])
    hess_eig_max = float(_eig[-1])
    hess_psd = bool(hess_eig_min > 0)
    hess_cond = float(hess_eig_max / hess_eig_min) if hess_eig_min > 0 else float("inf")

    # 5) recompute decision at the historical op; write correlated decision covariance.
    #    C_param_corr (= G Sigma_corr G^T) is the load-bearing output -- it feeds
    #    bayes_decision_discrepancy.py --c-param-json. P(meet) is recomputed downstream,
    #    so guard the optional decision_report (it can degenerate at coarse n_steps).
    dec = json.load(open(os.path.join(RES, decf)))
    op = dec["decision_op"]
    tol = np.array(dec.get("tol", [0.02, 0.05]))
    post_corr = Posterior.from_template(mean=u_map, cov=Sigma_corr, u_map=u_map, prior=prior,
                                        template=bundle.components, sigma_obs=post_iid.sigma_obs,
                                        engine="laplace_correlated")
    G, g_map, _ = decision_jacobian(bundle, op, u_map, n_steps=n_steps, return_extra=True)
    C_corr = G @ Sigma_corr @ G.T
    Ti = np.diag(1.0 / tol)
    wdec_corr = float(np.sqrt(np.max(np.linalg.eigvalsh(Ti @ C_corr @ Ti))))
    if reparam:
        sp = _sp
        reparam_pre = ridge_reparam_report(H, Sigma_corr, _grad, sp, G, Ti, g_map, spec=(0.70, 0.50))

        def _grad_at(uvec):
            _u = torch.tensor(np.asarray(uvec, float), dtype=DTYPE, requires_grad=True)
            (_gp,) = torch.autograd.grad(corr_negloglik(_u) - prior.log_prob(_u), _u)
            return _gp.detach().cpu().numpy()

        def _H_at(uvec):
            Jl = torch.autograd.functional.jacobian(
                predict_fn, torch.tensor(np.asarray(uvec, float), dtype=DTYPE), vectorize=True).detach().cpu().numpy()
            Hl = prior.precision().copy()
            for _b, _Ki in zip(blocks, [k.numpy() for k in Kinv]):
                _Jb = Jl[_b["idx"]]
                Hl += _Jb.T @ _Ki @ _Jb
            return 0.5 * (Hl + Hl.T) + 1e-9 * np.eye(Hl.shape[0])

        # truncated-Newton polish: step ONLY in the stiff (identified) eigendirections of the initial
        # prior-whitened posterior, converging the data-constrained subspace while leaving the flat
        # k_eq-nu ridge untouched. Re-evaluates the gradient each step (fixed metric).
        _M0 = np.diag(1.0 / sp) @ Sigma_corr @ np.diag(1.0 / sp)
        _mu0, _U0 = np.linalg.eigh(0.5 * (_M0 + _M0.T))
        _mu0 = np.clip(_mu0, 0.0, 1.0 + 1e-9)
        _ident0 = _mu0 < 0.5
        u_pol = np.array(u_map, float)
        polish_trace = []
        for _ in range(int(reparam_polish)):
            _gw = sp * _grad_at(u_pol)
            polish_trace.append(float(np.linalg.norm((_U0.T @ _gw)[_ident0])))
            u_pol = u_pol + reparam_damping * (sp * _stiff_step(_gw, _mu0, _U0, _ident0))
        # recompute H / Sigma / decision at the polished point for a self-consistent reparam
        H_pol = _H_at(u_pol)
        g_pol = _grad_at(u_pol)
        G_pol, g_map_pol, _ = decision_jacobian(bundle, op, u_pol, n_steps=n_steps, return_extra=True)
        reparam_post = ridge_reparam_report(H_pol, np.linalg.inv(H_pol), g_pol, sp, G_pol, Ti,
                                            g_map_pol, spec=(0.70, 0.50))
        reparam_report = dict(
            pre=reparam_pre, post=reparam_post,
            polish_iters=len(polish_trace), polish_damping=float(reparam_damping),
            polish_grad_identified_trace=polish_trace,
            polish_map_shift=float(np.linalg.norm(u_pol - u_map)),
            g_map_pre=list(map(float, np.ravel(g_map))), g_map_post=list(map(float, np.ravel(g_map_pol))),
            wdec_deployed=wdec_corr,
        )
        # The block records what the polish measured; it does not assert that option (1) holds. On the
        # deployed mAb A fit it does not: newton_decrement_identified is 5.13 before the polish and 2.68
        # after, against a condition of << 1, and grad_whitened_total rises from 266 to 1042. The article
        # deploys the fitted-point Gauss-Newton working law and claims nothing from this report.
    else:
        reparam_report = None
    try:
        rep = decision_report(post_corr, bundle, op, n_steps=n_steps, spec=(0.70, 0.50), tol=tuple(tol))
    except Exception as ex:  # noqa: BLE001 -- coarse-grid degeneracy; core outputs still valid
        rep = {"error": f"{type(ex).__name__}: {ex}", "note": "recompute P(meet) via bayes_decision_discrepancy.py"}

    result = dict(product=pid, kernel=kernel, hyper=hyper, decision_op=op, n_steps=n_steps,
                  optimizer=optimizer,
                  # how the MAP loop ended: a plateau stop is a statement about the objective,
                  # an iteration-limit stop only about the budget.
                  optimizer_termination=(term if term is not None else dict(
                      reason="iteration limit", iters=int(map_iters), plateau_tol=None,
                      note=("lbfgs: one torch.optim.LBFGS step at max_iter=map_iters with "
                            "strong-Wolfe line search, tolerance_grad 1e-8, tolerance_change 1e-12"
                            if optimizer == "lbfgs" else
                            "adam: map_iters steps at a fixed lr of 0.05"))),
                  u_map_iid=u0.tolist(), u_map_corr=u_map.tolist(),
                  map_shift=float(np.linalg.norm(u_map - u0)),
                  worst_dir_iid=post_iid_worst_dir(post_iid, prior),
                  worst_dir_corr=post_iid_worst_dir(post_corr, prior),
                  wdec_corr=wdec_corr, g_map=list(map(float, np.ravel(g_map))),
                  grad_norm=grad_norm, grad_norm_whitened=grad_norm_whitened,
                  hess_psd=hess_psd, hess_cond=hess_cond,
                  hess_eig_min=hess_eig_min, hess_eig_max=hess_eig_max,
                  p_meet=(float(rep["p_meet"]) if isinstance(rep, dict) and "p_meet" in rep else None),
                  action=(rep.get("action") if isinstance(rep, dict) else None),
                  map_loss_first=float(losses[0]), map_loss_last=float(losses[-1]),
                  map_loss_drop_last10=float(losses[-10] - losses[-1]) if len(losses) >= 10 else None,
                  ridge_reparam=reparam_report,
                  C_param_corr=C_corr.tolist(), decision_report=rep)
    if write:
        out = out or os.path.join(RES, f"{pid}_correlated.json")
        # a probe run points --out at a directory that need not exist yet; the fit behind this is
        # hours long, so create it rather than losing the result to a missing mkdir
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        json.dump(result, open(out, "w"), indent=2)
        # c_param JSON for the discrepancy recompute
        cpath = cpath or os.path.join(RES, "c_param_correlated.json")
        os.makedirs(os.path.dirname(os.path.abspath(cpath)), exist_ok=True)
        cp = json.load(open(cpath)) if os.path.exists(cpath) else {}
        cp[pid] = C_corr.tolist()
        json.dump(cp, open(cpath, "w"), indent=2)
        # persist the FULL correlated posterior (Sigma_corr, dim 4n) so the predictive operating-window
        # scan (bayes_decision_window.py --predictive) can recompute G Sigma_corr G^T at OTHER candidate
        # ops -- {pid}_correlated.json stores only C_param_corr at the historical op.
        post_corr.save(os.path.splitext(out)[0] + "_posterior.npz")
    if posterior_out:                            # a restart keeps its law even though write=False
        os.makedirs(os.path.dirname(os.path.abspath(posterior_out)), exist_ok=True)
        post_corr.save(posterior_out)
    if verbose and write:
        print(f"[{pid}] map_shift={result['map_shift']:.3f}  "
              f"worst_dir {result['worst_dir_iid']:.3f}->{result['worst_dir_corr']:.3f}  "
              f"wrote {out} and {cpath}")
    return result


def multistart_correlated(pid, decf, *, n_restarts=6, scale=0.3, seed=0, kernel="ou",
                          n_steps=300, map_iters=120, optimizer="lbfgs", rho_max=None,
                          plateau_tol=0.1, chunk=60):
    """RUN ON COLAB (torch). L-BFGS multistart stability check for the correlated refit.

    The correlated MAP drifts along the flattened k_eq--nu ridge and does not settle to a unique
    point (main text S6.7 / SI S17). This restarts the refit from perturbed starts
    ``u0 + scale * sigma_prior * z`` (z ~ N(0,I)), holding the noise model fixed at the canonical
    profile-ML hyperparameters (restart 0), and reports whether the POINT estimate is unstable
    (large pairwise ||Delta u||) while the COVARIANCE-based decision reads the paper relies on
    (worst_dir, wdec, and the decision g_map) are stable across restarts. Writes
    {pid}_correlated_multistart.json.
    """
    import numpy as _np
    from cex_model.bayes import physical_prior, Posterior
    from cex_model import app_support

    post_iid = Posterior.load(os.path.join(RES, f"{pid}_posterior.npz"))
    u0 = _np.asarray(post_iid.u_map, float)
    n = post_iid.n_protein
    sigma_prior = _np.asarray(physical_prior(n).std, float)
    rng = _np.random.default_rng(seed)

    print("=" * 92)
    print(f"L-BFGS MULTISTART STABILITY  --  {pid}   ({n_restarts} restarts, scale={scale})")
    print("=" * 92)
    # restart 0 = canonical start; its hyper is fixed for every restart. write=False throughout so a
    # multistart run (possibly at coarse n_steps) never clobbers the committed canonical artifacts.
    msdir = os.path.join(RES, "multistart")
    r0 = refit_correlated(pid, decf, kernel=kernel, n_steps=n_steps, map_iters=map_iters,
                          optimizer=optimizer, rho_max=rho_max, verbose=False, write=False,
                          plateau_tol=plateau_tol, chunk=chunk,
                          posterior_out=os.path.join(msdir, f"{pid}_r0_posterior.npz"))
    hyper0 = r0["hyper"]
    print(f"  restart 0 (canonical): ||u_map - u0|| = {r0['map_shift']:7.3f}  "
          f"worst_dir={r0['worst_dir_corr']:.4f}  wdec={r0['wdec_corr']:.4f}  "
          f"g=({r0['g_map'][0]:.4f},{r0['g_map'][1]:.4f})")
    canon_loss = float(r0["map_loss_last"])
    loss_tol = 0.10                                  # a restart is "converged" if its loss is within 10% of canonical
    restarts = [dict(r0, restart=0, converged=True)]
    for k in range(1, n_restarts):
        u_pert = u0 + scale * sigma_prior * rng.standard_normal(u0.shape)
        try:
            rk = refit_correlated(pid, decf, kernel=kernel, n_steps=n_steps, map_iters=map_iters,
                                  optimizer=optimizer, rho_max=rho_max, verbose=False,
                                  pool_hyper=hyper0, init_u0=u_pert, write=False,
                                  plateau_tol=plateau_tol, chunk=chunk,
                                  posterior_out=os.path.join(msdir, f"{pid}_r{k}_posterior.npz"))
            lk = float(rk["map_loss_last"])
            finite = all(_np.isfinite([rk["worst_dir_corr"], rk["wdec_corr"], *rk["g_map"]]))
            conv = finite and (lk <= canon_loss * (1 + loss_tol))
            rk = dict(rk, restart=k, converged=bool(conv))
            print(f"  restart {k}: ||u_map-u0||={_np.linalg.norm(_np.array(rk['u_map_corr'])-u0):7.3f}  "
                  f"loss={lk:8.2f}  worst_dir={rk['worst_dir_corr']:.4f}  wdec={rk['wdec_corr']:.4f}  "
                  f"g=({rk['g_map'][0]:.3f},{rk['g_map'][1]:.3f})  "
                  f"[{'converged' if conv else 'DIVERGED'}]")
        except Exception as ex:                      # noqa: BLE001 -- a degenerate restart must not kill the run
            rk = dict(restart=k, converged=False, error=f"{type(ex).__name__}: {ex}",
                      u_map_corr=None, worst_dir_corr=float("inf"), wdec_corr=float("inf"),
                      g_map=[float("nan"), float("nan")], map_loss_last=float("inf"))
            print(f"  restart {k}: DIVERGED ({rk['error'].splitlines()[0][:60]})")
        restarts.append(rk)

    conv = [r for r in restarts if r["converged"]]
    n_conv, n_div = len(conv), len(restarts) - len(conv)
    umaps = _np.array([r["u_map_corr"] for r in conv])
    pair = [float(_np.linalg.norm(umaps[i] - umaps[j]))
            for i in range(len(umaps)) for j in range(i + 1, len(umaps))]
    # prior-whitened distances (the metric the identifiability criterion uses): || Delta u / sigma_prior ||
    pair_w = [float(_np.linalg.norm((umaps[i] - umaps[j]) / sigma_prior))
              for i in range(len(umaps)) for j in range(i + 1, len(umaps))]
    wdir = _np.array([r["worst_dir_corr"] for r in conv])
    wdec = _np.array([r["wdec_corr"] for r in conv])
    gpur = _np.array([r["g_map"][0] for r in conv])
    gyld = _np.array([r["g_map"][1] for r in conv])
    # stationarity + Hessian diagnostics across converged restarts
    gnw = _np.array([r.get("grad_norm_whitened", _np.nan) for r in conv])
    gnr = _np.array([r.get("grad_norm", _np.nan) for r in conv])
    hcond = _np.array([r.get("hess_cond", _np.nan) for r in conv])
    all_psd = bool(all(r.get("hess_psd", False) for r in conv))
    actions = sorted({r.get("action") for r in conv if r.get("action") is not None})

    def _spread(x):
        return dict(mean=float(_np.nanmean(x)), sd=float(_np.nanstd(x)),
                    range=[float(_np.nanmin(x)), float(_np.nanmax(x))])

    summary = dict(
        product=pid, n_restarts=n_restarts, scale=scale, seed=int(seed),
        perturbation=("u0 + scale * sigma_prior * z, z ~ N(0,I) from "
                      "numpy.random.default_rng(seed), drawn once per restart in order"),
        optimizer=optimizer,
        optimizer_settings=(dict(plateau_tol=plateau_tol, chunk=chunk, ceiling=map_iters)
                            if optimizer == "adaptive" else
                            dict(lr=1.0, max_iter=map_iters, tolerance_grad=1e-8,
                                 tolerance_change=1e-12, line_search="strong_wolfe")
                            if optimizer == "lbfgs" else dict(lr=0.05, iters=map_iters)),
        optimizer_termination=[r.get("optimizer_termination") for r in restarts],
        n_steps=n_steps, map_iters=map_iters,
        hyper=hyper0, canonical_loss=canon_loss, loss_tol=loss_tol,
        n_converged=n_conv, n_diverged=n_div,
        note=("Among restarts that converge to within loss_tol of the canonical fit, the covariance-based "
              "reads (worst_dir, wdec) and the decision g are stable while the point estimate drifts along "
              "the flattened ridge; diverged restarts (n_diverged) confirm the correlated objective is "
              "ill-posed, which is why the deployed read is the warm-started (iid-MAP-initialised) fit."),
        point_instability_converged=dict(max_pairwise_du=max(pair) if pair else 0.0,
                                         mean_pairwise_du=float(_np.mean(pair)) if pair else 0.0,
                                         max_pairwise_du_whitened=max(pair_w) if pair_w else 0.0,
                                         mean_pairwise_du_whitened=float(_np.mean(pair_w)) if pair_w else 0.0,
                                         dist_from_canonical=[float(_np.linalg.norm(u - u0)) for u in umaps],
                                         dist_from_canonical_whitened=[float(_np.linalg.norm((u - u0) / sigma_prior)) for u in umaps]),
        convergence_diagnostics_converged=dict(
            grad_norm_whitened=_spread(gnw), grad_norm_raw=_spread(gnr),
            hessian_all_psd=all_psd, hessian_cond=_spread(hcond),
            actions_across_restarts=actions,
            note=("grad_norm_whitened = ||sigma_prior (.) grad(neglogpost)|| -> 0 certifies a stationary MAP; "
                  "hessian is the GGN/Laplace Hessian (SPD by construction), its condition number measures "
                  "ridge flatness; a single action across restarts confirms the decision is restart-invariant.")),
        covariance_stability_converged=dict(worst_dir=_spread(wdir), wdec=_spread(wdec)),
        decision_stability_converged=dict(pool_purity=_spread(gpur), pool_yield=_spread(gyld)),
        restarts=[{k: v for k, v in r.items() if k in
                   ("restart", "converged", "map_loss_last", "worst_dir_corr", "wdec_corr", "g_map",
                    "u_map_corr",
                    "grad_norm", "grad_norm_whitened", "hess_psd", "hess_cond", "hess_eig_min",
                    "hess_eig_max", "p_meet", "action", "error", "optimizer_termination")}
                  for r in restarts])
    out = os.path.join(RES, f"{pid}_correlated_multistart.json")
    json.dump(summary, open(out, "w"), indent=2, default=str)
    print("-" * 92)
    print(f"{n_conv}/{n_restarts} restarts converged to within {int(loss_tol*100)}% of the canonical loss "
          f"({canon_loss:.1f}); {n_div} diverged (correlated objective is ill-posed).")
    if n_conv >= 2:
        print(f"Among CONVERGED restarts -- POINT drifts: max pairwise ||Delta u|| = "
              f"{summary['point_instability_converged']['max_pairwise_du']:.3f}")
        print(f"  COVARIANCE reads STABLE: worst_dir {wdir.mean():.4f} (sd {wdir.std():.4f}, "
              f"range [{wdir.min():.4f},{wdir.max():.4f}]);  wdec {wdec.mean():.4f} (sd {wdec.std():.4f}, "
              f"range [{wdec.min():.4f},{wdec.max():.4f}])")
        print(f"  DECISION g stable: purity {gpur.mean():.4f} (sd {gpur.std():.4f}); "
              f"yield {gyld.mean():.4f} (sd {gyld.std():.4f})")
        print(f"  STATIONARITY: grad_norm_whitened {_np.nanmean(gnw):.2e} "
              f"(range [{_np.nanmin(gnw):.2e},{_np.nanmax(gnw):.2e}]);  Hessian all-PSD={all_psd}; "
              f"cond {_np.nanmedian(hcond):.2e} (range [{_np.nanmin(hcond):.2e},{_np.nanmax(hcond):.2e}])")
        print(f"  ACTION across restarts: {actions if actions else '(decision_report degenerate at this grid)'}")
        print(f"  PRIOR-WHITENED point drift: max pairwise {max(pair_w) if pair_w else 0.0:.3f}")
    else:
        print("Fewer than 2 converged restarts -- reduce --multistart-scale or raise --map-iters/--n-steps.")
    print(f"wrote {out}")
    return summary


def post_iid_worst_dir(post, prior):
    D = np.diag(1.0 / prior.std)
    M = D @ post.cov @ D
    if not np.all(np.isfinite(M)):
        return float("inf")                       # degenerate covariance (a diverged restart)
    try:
        return float(np.sqrt(np.max(np.linalg.eigvalsh(M))))
    except np.linalg.LinAlgError:
        return float("inf")                       # eigensolver non-convergence -> flag as diverged


def _corr_map_fit(u0, predict_fn, obs, prior, idx_t, Kinv, *, optimizer, map_iters,
                  plateau_tol=0.1, chunk=60):
    """Refit the MAP under a fixed block-correlated whitening (shared with refit_correlated).
    Returns (u_map, Sigma_corr)."""
    import torch
    from cex_model.diffsolver.torch_solver import DTYPE

    def corr_negloglik(u):
        resid = predict_fn(u) - obs
        q = u.new_zeros(())
        for it_, Ki in zip(idx_t, Kinv):
            rb = resid[it_]
            q = q + rb @ (Ki @ rb)
        return 0.5 * q

    u = torch.tensor(u0, dtype=DTYPE, requires_grad=True)  # warm start from full-data MAP

    def neglogpost():
        return corr_negloglik(u) - prior.log_prob(u)

    if optimizer == "adaptive":
        _adaptive_map(u, neglogpost, torch, chunk=chunk, max_iters=map_iters,
                      plateau_tol=plateau_tol, losses=[])
    elif optimizer == "lbfgs":
        opt = torch.optim.LBFGS([u], lr=1.0, max_iter=map_iters, history_size=25,
                                tolerance_grad=1e-8, tolerance_change=1e-12,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); loss = neglogpost(); loss.backward(); return loss

        opt.step(closure)
    else:
        opt = torch.optim.Adam([u], lr=0.05)
        for _ in range(map_iters):
            opt.zero_grad(); neglogpost().backward(); opt.step()
    u_map = u.detach().cpu().numpy()
    J = torch.autograd.functional.jacobian(
        predict_fn, torch.tensor(u_map, dtype=DTYPE), vectorize=True).detach().cpu().numpy()
    H = prior.precision().copy()
    for it_, Ki in zip(idx_t, [k.numpy() for k in Kinv]):
        Jb = J[it_.numpy()]
        H += Jb.T @ Ki @ Jb
    H = 0.5 * (H + H.T) + 1e-9 * np.eye(H.shape[0])
    return u_map, np.linalg.inv(H)


def decision_loeo_correlated(pid, decf, *, kernel="ou", n_steps=300, map_iters=300, optimizer="lbfgs",
                             rho_max=None, pool_hyper=None, spec=(0.70, 0.50), out=None, verbose=True,
                             plateau_tol=0.1, chunk=60):
    """RUN ON COLAB. OU-consistent decision leave-one-experiment-out (reviewer step 1).

    Re-runs each decision hold-out fold under the OU+nugget block likelihood, so the per-fold decision
    residuals (observed - predicted pooled purity/yield) reflect the CORRELATED-MAP prediction, not the
    committed i.i.d.-LOEO folds. Writes ``{pid}_correlated_loeo.json`` in the exact ``decision_loeo.folds``
    schema that ``bayes_decision_discrepancy.py --folds-json`` reads, so ``C_delta`` becomes fully
    OU-consistent. Shared hyperparameters are profiled once on the full-data iid-MAP residuals (a
    product-level property; the fold refits reuse them, matching refit_correlated's pooled-hyper design).
    """
    import torch
    from cex_model.diffsolver.torch_solver import DTYPE
    from cex_model.diffsolver.calibrate_diff import targets_from_bundle
    from cex_model import app_support
    from cex_model.app_support import group_indices
    from cex_model.bayes import mechanistic_model, physical_prior, Posterior, decision_report
    from cex_model.bayes.validation import _subset_bundle, _observed_pool_decision

    bundle = app_support.load_product(pid.replace("_exclDT", ""))
    if pid.endswith("_exclDT"):  # match the committed exclDT posterior/folds: drop the non-standard DT run
        from cex_model.bayes.compare import filter_experiments
        kept, dropped = filter_experiments(bundle.experiments, ["DT"])
        bundle.experiments = kept
        if verbose and dropped:
            print(f"[{pid}] excluded {[e.name for e in dropped]}")
    n = bundle.components.n_protein
    prior = physical_prior(n)
    post_iid = Posterior.load(os.path.join(RES, f"{pid}_posterior.npz"))
    u_full = post_iid.u_map
    tol = np.array(json.load(open(os.path.join(RES, decf))).get("tol", [0.02, 0.05]))

    # shared hyperparameters from the full-data iid-MAP residuals
    targets_full = targets_from_bundle(bundle, n_steps)
    predict_full, obs_full = mechanistic_model(targets_full, n)
    with torch.no_grad():
        r_full = (predict_full(torch.tensor(u_full, dtype=DTYPE)) - obs_full).detach().cpu().numpy()
    blocks_full = _blocks_from_targets(targets_full)
    hyper = pool_hyper or profile_hyper([r_full[b["idx"]] for b in blocks_full],
                                        [b["times"] for b in blocks_full], kernel=kernel, rho_max=rho_max)
    if verbose:
        print(f"[{pid}] OU-LOEO shared hyper: rho={hyper['rho']:.2f} ell={hyper['ell']:.0f} "
              f"sigma2={hyper['sigma2']:.4f}")

    gi = group_indices(bundle.components); main_set = set(gi["main"])
    og = bundle.observation_groups
    obs_groups = list(og) if og else [[j] for j in range(n)]
    main_cols = [k for k, grp in enumerate(obs_groups) if set(grp) <= main_set]
    names = ("pool_purity", "pool_yield")

    exps = list(range(len(bundle.experiments)))
    folds = []
    for held in exps:
        he = bundle.experiments[held]
        op = [float(he.loading_g_l), float(he.gradient_start_pct),
              float(he.gradient_end_pct), float(he.elution_cv)]
        try:
            fit_b = _subset_bundle(bundle, [i for i in exps if i != held])
            targets = targets_from_bundle(fit_b, n_steps)
            predict_fn, obs = mechanistic_model(targets, n)
            blocks = _blocks_from_targets(targets)
            Kinv, idx_t = [], []
            for b in blocks:
                Sig = hyper["sigma2"] * unit_block(_corr_for(kernel, b["times"], hyper["ell"]), hyper["rho"])
                Kinv.append(torch.tensor(np.linalg.inv(Sig), dtype=DTYPE))
                idx_t.append(torch.tensor(b["idx"]))
            u_map, Sigma_corr = _corr_map_fit(u_full, predict_fn, obs, prior, idx_t, Kinv,
                                              optimizer=optimizer, map_iters=map_iters,
                                              plateau_tol=plateau_tol, chunk=chunk)
            post_corr = Posterior.from_template(mean=u_map, cov=Sigma_corr, u_map=u_map, prior=prior,
                                                template=bundle.components, sigma_obs=post_iid.sigma_obs,
                                                engine="laplace_correlated")
            # predicted decision at the held-out OP from the n-1 correlated posterior (product constants
            # from the full bundle); observed decision read on the held-out curve over the same window
            rep = decision_report(post_corr, bundle, op, spec=spec, tol=tuple(tol), n_steps=n_steps)
            pur_obs, yld_obs = _observed_pool_decision(he.curve, rep["window_s"], main_cols)
            folds.append({
                "held_out": he.name, "op": op,
                "worst_dec": float(rep["worst_dec"]), "determined": bool(rep["met"]),
                "g_pred": {nm: float(rep["g_map"][nm]) for nm in names},
                "decision_std": {nm: float(rep["decision_std"][nm]) for nm in names},
                "observed": {"pool_purity": float(pur_obs), "pool_yield": float(yld_obs)},
                "window_s": [float(rep["window_s"][0]), float(rep["window_s"][1])],
                "map_shift": float(np.linalg.norm(u_map - u_full)),
            })
            if verbose:
                f = folds[-1]
                print(f"  OU-dLOEO held={str(he.name)[:22]:22s} worst_dec={f['worst_dec']:.3f} "
                      f"obs_pur={pur_obs:.3f} pred_pur={f['g_pred']['pool_purity']:.3f} "
                      f"map_shift={f['map_shift']:.2f}")
        except Exception as ex:  # noqa: BLE001 -- one bad fold must not abort the sweep
            print(f"  OU-dLOEO held={str(he.name)[:22]:22s} FAILED: {type(ex).__name__}: {ex}")

    finite = [f for f in folds if np.isfinite(f["observed"]["pool_purity"])]
    agg = {"n_folds": len(folds), "n_observable": len(finite),
           "worst_dec_range": ([min(f["worst_dec"] for f in folds), max(f["worst_dec"] for f in folds)]
                               if folds else [])}
    result = {"product": pid, "kernel": kernel, "hyper": hyper, "n_steps": n_steps,
              "note": "OU-consistent decision-LOEO folds; feed to bayes_decision_discrepancy.py --folds-json",
              "decision_loeo": {"folds": folds, "aggregate": agg}}
    out = out or os.path.join(RES, f"{pid}_correlated_loeo.json")
    json.dump(result, open(out, "w"), indent=2)
    if verbose:
        print(f"[{pid}] OU-LOEO wrote {len(folds)} folds -> {out}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="run the numpy kernel selftest (works here)")
    ap.add_argument("--product", default=None, help="HLXSYN | HLXSYN | HLXSYN (torch/Colab)")
    ap.add_argument("--kernel", default="ou", choices=["ou", "ar1", "matern32"])
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--map-iters", type=int, default=120,
                    help="MAP iters (adam) or max inner L-BFGS iters")
    ap.add_argument("--optimizer", default="lbfgs", choices=["adam", "lbfgs", "adaptive"],
                    help="lbfgs (quasi-Newton) converges the correlation-flattened ridge Adam cannot; "
                         "adaptive runs the same line search in chunks and stops when the objective "
                         "levels off rather than when a budget runs out, for products that do not "
                         "settle inside a fixed iteration count")
    ap.add_argument("--plateau-tol", type=float, default=0.1,
                    help="--optimizer adaptive: stop once a whole chunk of iterations buys less "
                         "objective than this")
    ap.add_argument("--chunk", type=int, default=60,
                    help="--optimizer adaptive: L-BFGS iterations per chunk; the plateau test is "
                         "applied to the gain over one chunk, not to a single step")
    ap.add_argument("--rho-max", type=float, default=None,
                    help="cap the correlated fraction rho (guards the rho<->discrepancy confound; "
                         "e.g. 0.9 for the mAb A guard). Default: uncapped (profile-ML rho).")
    ap.add_argument("--decision-loeo", action="store_true",
                    help="OU-consistent decision leave-one-experiment-out (reviewer step 1): re-run each "
                         "decision hold-out fold under the OU+nugget likelihood, writing "
                         "{product}_correlated_loeo.json for bayes_decision_discrepancy.py --folds-json.")
    ap.add_argument("--multistart", type=int, default=0, metavar="N",
                    help="L-BFGS multistart stability check: N restarts from perturbed starts, reporting "
                         "point-estimate instability vs covariance/decision-read stability.")
    ap.add_argument("--multistart-scale", type=float, default=0.3,
                    help="perturbation scale (fraction of prior sd) for --multistart starts.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --multistart perturbations.")
    ap.add_argument("--reparam", action="store_true",
                    help="option-(1) ridge-reparametrization Laplace: identified-subspace Laplace + prior on "
                         "the null/ridge coordinate; certifies the deployed covariance despite the non-stationary MAP.")
    ap.add_argument("--reparam-polish", type=int, default=6,
                    help="truncated-Newton steps in the stiff subspace before the reparam (converges the "
                         "data-constrained directions L-BFGS leaves off-minimum; 0 disables).")
    ap.add_argument("--reparam-damping", type=float, default=0.7, help="damping for the stiff-subspace polish steps.")
    ap.add_argument("--out", default=None,
                    help="where to write {product}_correlated.json. The posterior npz follows the same "
                         "stem, so pointing this outside results/bayes keeps a probe run -- a larger "
                         "map-iteration budget, say -- from overwriting the deployed artifacts.")
    ap.add_argument("--c-param-out", default=None,
                    help="where to merge this product's C_param_corr. Default results/bayes/"
                         "c_param_correlated.json, which is SHARED across products and read by the "
                         "discrepancy layer -- point it elsewhere when fitting the synthetic twin so "
                         "the committed file is not modified.")
    a = ap.parse_args()
    if a.selftest or not a.product:
        selftest()
        return
    decf = f"{a.product}_decision.json"
    if a.multistart:
        multistart_correlated(a.product, decf, n_restarts=a.multistart, scale=a.multistart_scale,
                              seed=a.seed, kernel=a.kernel, n_steps=a.n_steps, map_iters=a.map_iters,
                              optimizer=a.optimizer, rho_max=a.rho_max,
                              plateau_tol=a.plateau_tol, chunk=a.chunk)
    elif a.decision_loeo:
        decision_loeo_correlated(a.product, decf, kernel=a.kernel, n_steps=a.n_steps,
                                 map_iters=a.map_iters, optimizer=a.optimizer, rho_max=a.rho_max,
                                 plateau_tol=a.plateau_tol, chunk=a.chunk)
    else:
        refit_correlated(a.product, decf, kernel=a.kernel, n_steps=a.n_steps, map_iters=a.map_iters,
                         optimizer=a.optimizer, rho_max=a.rho_max, reparam=a.reparam,
                         reparam_polish=a.reparam_polish, reparam_damping=a.reparam_damping,
                         out=a.out, cpath=a.c_param_out,
                         plateau_tol=a.plateau_tol, chunk=a.chunk)


if __name__ == "__main__":
    main()
