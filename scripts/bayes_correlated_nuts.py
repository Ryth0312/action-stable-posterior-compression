"""Correlated-posterior NUTS for the non-stationary correlated MAP (option 2).

The correlated (OU+nugget) objective has no clean stationary point-estimate MAP -- L-BFGS drifts along
the flattened k_eq-nu ridge and a truncated-Newton polish diverges on a cond ~ 2e7 objective -- so a
Laplace expansion point is ill-defined.  NUTS needs no MAP: it samples the correlated posterior directly.
We precondition with a correlated Laplace built at the iid-MAP (an exact change of variables; only
efficiency depends on it) and target the CORRELATED posterior under the block-OU likelihood.

Sampled subspaces (``--params``).  All three write ``u = reconstruct(u_sampled)`` into the full param
vector, freezing everything else at the iid MAP; the correlated posterior is targeted in the sampled
coordinates.  Because freezing the complement gives a CONDITIONAL read, each mode validates a different
conditional -- read the decision-covariance comparison, not just the parameter agreement:
  * ``main4`` (default): the main peak's 4 params [k_eq, k_kin, nu, sigma] -- the reduced anchor.
  * ``nosigma-full``: ALL components' [k_eq, k_kin, nu] (3n params), sigma frozen (decision-null).  In
    practice the 3n curved multi-ridge geometry is too stiff for fixed-metric NUTS (step size collapses)
    -- kept for the record, not recommended.
  * ``decision-active``: the r-dim DECISION-active subspace -- the prior-whitened row space of the
    decision Jacobian G (2 directions) plus the top ridge directions, built as a PRIOR-WHITENED
    orthonormal basis so the prior on the sampled coords is N(0,I) and the geometry is well-conditioned.
    This samples the correlated posterior in the directions that actually drive the pooled decision
    (mixing all components -- the cross-block content the 4-D conditional cannot give), and reports the
    decision covariance from the sampler versus the deployed correlated Laplace.  It validates a
    decision-relevant CONDITIONAL, which approximates the marginal to the extent the frozen complement is
    decision-null; the C_nuts-vs-C_laplace comparison is exactly that test.

Two ODE resolutions: ``--n-steps-lik`` (default 60) for the NUTS likelihood; ``--n-steps`` (default 300)
for the forward-only decision read (matches the deployed correlated refit).

Checkpointing: each run samples ONE chain -> ``{product}_correlated_nuts{tag}_chain{seed}.npz`` before
the decision read; pool >=2 chains for R-hat.

Heavy -> RUN ON COLAB.
    # --- one-session run (if per-iteration time is low enough) ---
    OMP_NUM_THREADS=4 python scripts/bayes_correlated_nuts.py --product HLXSYN --params decision-active \
        --active-dim 4 --warmup 200 --num-samples 200 --full-mass --max-tree-depth 6 --seed 0

    # --- two-session run (split warmup / sampling across Colab sessions) ---
    # session 1: warmup only -> saves adapted mass matrix + step size
    OMP_NUM_THREADS=4 python scripts/bayes_correlated_nuts.py --product HLXSYN --params decision-active \
        --active-dim 4 --warmup 200 --full-mass --max-tree-depth 6 --warmup-only --seed 0
    # session 2: sampling with frozen geometry (no further adaptation, no warmup)
    OMP_NUM_THREADS=4 python scripts/bayes_correlated_nuts.py --product HLXSYN --params decision-active \
        --active-dim 4 --full-mass --max-tree-depth 6 --num-samples 200 \
        --load-warmup results/bayes/HLXSYN_correlated_nuts_decactive_warmup_seed0.npz --seed 0
    # repeat for seed 1, then pool:
    OMP_NUM_THREADS=4 python scripts/bayes_correlated_nuts.py --product HLXSYN --params decision-active \
        --active-dim 4 --pool-only --chains 0,1
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import cex_model.app_support as A
from cex_model.bayes import mechanistic_model, physical_prior, Posterior
from cex_model.bayes.decision import decision_forward, decision_jacobian  # not exported at package top
from cex_model.diffsolver.calibrate_diff import targets_from_bundle
from cex_model.diffsolver.torch_solver import DTYPE

from bayes_correlated_refit import _blocks_from_targets, profile_hyper, unit_block, _corr_for
from bayes_nuts_real import _resolve_component, _worst_dir, _deviation, _float64, _nuts_diag

RES = "results/bayes"


class _Gauss:
    """Minimal mean/cov container exposing .mean/.cov/.std_u() (no Posterior/ComponentSet wrapper,
    whose dim would not match a partial-parameter or non-coordinate subspace)."""

    def __init__(self, mean, cov, samples=None):
        self.mean = np.asarray(mean, float)
        self.cov = np.atleast_2d(np.asarray(cov, float))
        self._samples = None if samples is None else np.asarray(samples, float)

    def std_u(self):
        if self._samples is not None:
            return self._samples.std(0)
        return np.sqrt(np.clip(np.diag(self.cov), 0.0, None))


def _corr_from_cov(cov):
    d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    return np.asarray(cov, float) / np.outer(d, d)


def _rel_frob(A_, B_):
    A_, B_ = np.asarray(A_, float), np.asarray(B_, float)
    return float(np.linalg.norm(A_ - B_) / max(np.linalg.norm(B_), 1e-30))


def _whitened_corr_nuts(reconstruct, predict_fn, obs, prior_mean_full, prior_std_full, lap, idx_blocks,
                        Kinv, *, num_samples, warmup, max_tree_depth, target_accept, seed, progress,
                        full_mass=False, resume_from=None, warmup_only=False):
    """NUTS in Laplace-whitened coords targeting the CORRELATED posterior.  z ~ N(0,I),
    u_s = m + L z, u_full = reconstruct(u_s); loglik = -0.5 sum_b r_b^T Kinv_b r_b; the prior is the
    full diagonal prior evaluated on u_full (the frozen complement contributes only a constant).

    resume_from: dict with 'inverse_mass_matrix' (np array) and 'step_size' (float) from a prior warmup
    checkpoint.  When provided, adaptation is disabled and warmup is skipped.
    warmup_only: run warmup (+ 1 sample to satisfy pyro), return None for U_s; the adapted state is in hist.

    Returns (U_s, hist) with U_s the (num_samples, d) draws in the sampled coordinates (None if warmup_only)."""
    import pyro
    import pyro.distributions as dist
    from pyro.infer import MCMC, NUTS

    d = len(lap.mean)
    L_np = np.linalg.cholesky(np.asarray(lap.cov, float) + 1e-12 * np.eye(d))
    m_np = np.asarray(lap.mean, float)

    adapt_step = True
    adapt_mass = True
    step_sz = 1.0
    if resume_from is not None:
        adapt_step = False
        adapt_mass = False
        step_sz = float(resume_from["step_size"])
        warmup = 0
        print(f"  resuming from checkpoint: step_size={step_sz:.4e}, "
              f"mass matrix shape={resume_from['inverse_mass_matrix'].shape}")

    actual_samples = 1 if warmup_only else num_samples

    with _float64():
        pyro.clear_param_store(); pyro.set_rng_seed(seed)
        m_t = torch.as_tensor(m_np, dtype=DTYPE)
        L_t = torch.as_tensor(L_np, dtype=DTYPE)
        pm = torch.as_tensor(np.asarray(prior_mean_full, float), dtype=DTYPE)
        ps = torch.as_tensor(np.asarray(prior_std_full, float), dtype=DTYPE)
        ob = obs.to(DTYPE) if torch.is_tensor(obs) else torch.as_tensor(np.asarray(obs, float), dtype=DTYPE)
        zeros, ones = torch.zeros(d, dtype=DTYPE), torch.ones(d, dtype=DTYPE)
        Kt = [k if torch.is_tensor(k) else torch.as_tensor(k, dtype=DTYPE) for k in Kinv]
        it = [i if torch.is_tensor(i) else torch.as_tensor(i) for i in idx_blocks]

        def wmodel():
            z = pyro.sample("z", dist.Normal(zeros, ones).to_event(1))
            u_s = m_t + L_t @ z
            u_full = reconstruct(u_s)
            resid = predict_fn(u_full) - ob
            q = u_full.new_zeros(())
            for idx, Ki in zip(it, Kt):
                rb = resid[idx]
                q = q + rb @ (Ki @ rb)
            loglik = -0.5 * q
            logprior = dist.Normal(pm, ps).log_prob(u_full).sum()
            logbase = dist.Normal(zeros, ones).log_prob(z).sum()
            pyro.factor("target", logprior + loglik - logbase)

        kernel = NUTS(wmodel, jit_compile=False, max_tree_depth=max_tree_depth,
                      target_accept_prob=target_accept, full_mass=full_mass,
                      adapt_step_size=adapt_step, adapt_mass_matrix=adapt_mass,
                      step_size=step_sz)
        mcmc = MCMC(kernel, num_samples=actual_samples, warmup_steps=warmup, num_chains=1,
                    initial_params={"z": torch.zeros(d, dtype=DTYPE)}, disable_progbar=not progress)

        if resume_from is not None:
            # _initialize_adapter() (called inside setup()) resets the mass matrix to identity;
            # monkey-patch setup to re-inject the saved matrix immediately after.
            _orig_setup = kernel.setup
            _saved_mass = torch.as_tensor(resume_from["inverse_mass_matrix"], dtype=DTYPE)
            def _patched_setup(*a, **kw):
                _orig_setup(*a, **kw)
                kernel.mass_matrix_adapter.inverse_mass_matrix = {("z",): _saved_mass}
            kernel.setup = _patched_setup

        mcmc.run()

        # extract adapted state (always, for potential checkpoint)
        inv_mass_dict = kernel.inverse_mass_matrix
        inv_mass_np = {str(k): v.detach().cpu().numpy() for k, v in inv_mass_dict.items()}
        adapted = {"step_size": float(kernel.step_size),
                   "inverse_mass_matrix": inv_mass_np.get("('z',)", next(iter(inv_mass_np.values())))}

        if warmup_only:
            return None, {"adapted": adapted, "diagnostics": mcmc.diagnostics()}

        Z = mcmc.get_samples()["z"].detach().numpy()
    U_s = m_np[None, :] + Z @ L_np.T
    return U_s, {"n_samples": int(actual_samples), "diagnostics": mcmc.diagnostics(), "adapted": adapted}


def _decision_active_basis(b, op, u_iid, prior_std, blocks, Kinv, predict_fn, n_steps, r):
    """r-dim decision-active subspace as a PRIOR-WHITENED orthonormal basis: the decision Jacobian's
    prior-whitened row space (<=2 directions), padded with the top prior-whitened correlated-Laplace
    covariance directions.  Returns (V, G, Sig): V (4n x r) columns in u-coords (u = u_iid + V s, so the
    prior on s is exactly N(0,I)); G (2 x 4n) the decision Jacobian at the iid MAP; Sig (4n x 4n) the FULL
    correlated-GGN covariance at the iid MAP (H = Pi_0 + sum_b J_b^T Kinv_b J_b)^-1, used for the
    full-marginal decision covariance in the Schur decomposition."""
    G = np.asarray(decision_jacobian(b, op, u_iid, n_steps=n_steps), float)   # (2, 4n)
    Gw = G * prior_std[None, :]                                               # prior-whiten columns
    _, Sg, Vg = np.linalg.svd(Gw, full_matrices=False)                        # Vg rows are whitened dirs
    basis = [Vg[k] for k in range(len(Sg)) if Sg[k] > 1e-8 * max(Sg.max(), 1e-12)]
    # full correlated-GGN covariance at the iid MAP (always; used for both padding and the marginal read)
    Ju = torch.autograd.functional.jacobian(
        predict_fn, torch.tensor(u_iid, dtype=DTYPE), vectorize=True).detach().cpu().numpy()
    H = np.diag(1.0 / prior_std ** 2)
    for bl, Ki in zip(blocks, [k.numpy() for k in Kinv]):
        Jb = Ju[bl["idx"]]
        H += Jb.T @ Ki @ Jb
    Sig = np.linalg.inv(0.5 * (H + H.T) + 1e-9 * np.eye(len(u_iid)))
    if r > len(basis):  # pad with the widest correlated-posterior directions (top Laplace-cov eigvecs)
        Sw = (Sig / prior_std[:, None]) / prior_std[None, :]                  # prior-whitened cov
        _, Ve = np.linalg.eigh(0.5 * (Sw + Sw.T))                             # ascending eigenvalues
        for k in range(Ve.shape[1] - 1, -1, -1):
            if len(basis) >= r:
                break
            v = Ve[:, k].copy()
            for bvec in basis:  # Gram-Schmidt against the current (whitened, orthonormal) basis
                v = v - (v @ bvec) * bvec
            nv = np.linalg.norm(v)
            if nv > 1e-8:
                basis.append(v / nv)
    Bw = np.array(basis[:r])                                                  # (r, 4n) whitened orthonormal
    V = (Bw * prior_std[None, :]).T                                           # (4n, r) un-whitened (u-coords)
    return V, G, Sig


def _build_setup(a):
    """Load product + iid posterior; correlated OU hyper/Kinv (at n_steps_lik); the sampled subspace
    (main4 / nosigma-full / decision-active), the reconstruct map, and the correlated Laplace on it."""
    base = a.product[:-len("_exclDT")] if a.product.endswith("_exclDT") else a.product
    b = A.load_product(base)
    if a.product.endswith("_exclDT"):
        from cex_model.bayes.compare import filter_experiments
        b.experiments, _ = filter_experiments(b.experiments, ["DT"])
    n = b.components.n_protein
    tgs = targets_from_bundle(b, n_steps=a.n_steps_lik)
    predict_fn, obs = mechanistic_model(tgs, n)
    post_iid = Posterior.load(os.path.join(a.out_dir, f"{a.product}_posterior.npz"))
    u_iid = np.asarray(post_iid.u_map, float)
    sig = float(post_iid.sigma_obs)
    full_prior = physical_prior(n)
    pm_full, ps_full = np.asarray(full_prior.mean, float), np.asarray(full_prior.std, float)

    blocks = _blocks_from_targets(tgs)
    with torch.no_grad():
        r_iid = (predict_fn(torch.tensor(u_iid, dtype=DTYPE)) - obs).detach().cpu().numpy()
    hyper = profile_hyper([r_iid[bl["idx"]] for bl in blocks], [bl["times"] for bl in blocks],
                          kernel=a.kernel, rho_max=a.rho_max)
    Kinv, idx_blocks = [], []
    for bl in blocks:
        Sig = hyper["sigma2"] * unit_block(_corr_for(a.kernel, bl["times"], hyper["ell"]), hyper["rho"])
        Kinv.append(torch.tensor(np.linalg.inv(Sig), dtype=DTYPE))
        idx_blocks.append(torch.tensor(bl["idx"]))

    u_iid_t = torch.as_tensor(u_iid, dtype=DTYPE)
    op = json.load(open(os.path.join(a.out_dir, f"{a.product}_decision.json")))["decision_op"]

    # ---- the sampled subspace: reconstruct(u_s) -> full u ; prior_prec_s ; label -----------------
    G = None
    Sig_full = None
    if a.params in ("main4", "nosigma-full"):
        if a.params == "main4":
            j = _resolve_component(a.only_component, b)
            idx_sample = np.array([j, n + j, 2 * n + j, 3 * n + j])
            keq_nu_pairs = [(0, 2, b.components.components[j].name)]
            label = f"main-4 conditional c{j+1}={b.components.components[j].name} (other {n-1} fixed)"
        else:
            idx_sample = np.arange(3 * n)
            keq_nu_pairs = [(c, 2 * n + c, b.components.components[c].name) for c in range(n)]
            label = f"nosigma-full: all {n} components' [k_eq,k_kin,nu] ({3*n}-D); sigma frozen"
        idx_t = torch.as_tensor(idx_sample, dtype=torch.long)

        def reconstruct(u_s, _idx=idx_t):
            return u_iid_t.index_copy(0, _idx, u_s)

        m0 = u_iid[idx_sample]                                    # Laplace mean = the iid-MAP values
        prior_prec_s = np.diag(1.0 / ps_full[idx_sample] ** 2)
        prior_std_s = ps_full[idx_sample]
        Vmat = None
    elif a.params == "decision-active":
        Vmat, G, Sig_full = _decision_active_basis(b, op, u_iid, ps_full, blocks, Kinv, predict_fn,
                                                   a.n_steps, a.active_dim)
        V_t = torch.as_tensor(Vmat, dtype=DTYPE)
        idx_sample, keq_nu_pairs = None, []
        label = f"decision-active {a.active_dim}-D (prior-whitened basis; u = iidMAP + V s)"

        def reconstruct(u_s, _V=V_t):
            return u_iid_t + _V @ u_s

        m0 = np.zeros(a.active_dim)                               # offset param, Laplace at s=0 (u_iid)
        prior_prec_s = Vmat.T @ np.diag(1.0 / ps_full ** 2) @ Vmat     # == I for a prior-whitened basis
        prior_std_s = np.sqrt(np.diag(np.linalg.inv(prior_prec_s)))
    else:  # pragma: no cover
        raise SystemExit(f"unknown --params {a.params}")

    dsub = len(m0)

    # correlated Laplace on the sampled coords: H = prior_prec_s + sum_b J_s_b^T Kinv_b J_s_b
    def predfn_s(u_s):
        return predict_fn(reconstruct(u_s))

    Js = torch.autograd.functional.jacobian(
        predfn_s, torch.tensor(m0, dtype=DTYPE), vectorize=True).detach().cpu().numpy()
    H = prior_prec_s.copy()
    for bl, Ki in zip(blocks, [k.numpy() for k in Kinv]):
        Jb = Js[bl["idx"]]
        H += Jb.T @ Ki @ Jb
    H = 0.5 * (H + H.T) + 1e-9 * np.eye(dsub)
    lap = _Gauss(mean=m0, cov=np.linalg.inv(H))

    return dict(b=b, n=n, obs=obs, predict_fn=predict_fn, reconstruct=reconstruct, pm_full=pm_full,
                ps_full=ps_full, idx_blocks=idx_blocks, Kinv=Kinv, u_iid=u_iid, sig=sig, hyper=hyper,
                lap=lap, prior_std_s=prior_std_s, keq_nu_pairs=keq_nu_pairs, idx_sample=idx_sample,
                Vmat=Vmat, G=G, Sig_full=Sig_full, op=op, label=label, dsub=dsub)


def _rhat(chains):
    if len(chains) < 2:
        return None
    m = len(chains)
    n = min(len(c) for c in chains)
    X = np.stack([np.asarray(c, float)[:n] for c in chains])
    means = X.mean(1)
    grand = means.mean(0)
    B = n / (m - 1) * ((means - grand) ** 2).sum(0)
    W = X.var(1, ddof=1).mean(0)
    var_hat = (n - 1) / n * W + B / n
    with np.errstate(divide="ignore", invalid="ignore"):
        rh = np.sqrt(var_hat / np.where(W > 0, W, np.nan))
    return float(np.nanmax(rh))


def _rhat_per_dim(chains):
    """Gelman--Rubin R-hat per coordinate (list), for the NUTS convergence report."""
    if len(chains) < 2:
        return None
    m = len(chains); n = min(len(c) for c in chains)
    X = np.stack([np.asarray(c, float)[:n] for c in chains])
    means = X.mean(1); grand = means.mean(0)
    B = n / (m - 1) * ((means - grand) ** 2).sum(0)
    W = X.var(1, ddof=1).mean(0)
    var_hat = (n - 1) / n * W + B / n
    with np.errstate(divide="ignore", invalid="ignore"):
        rh = np.sqrt(var_hat / np.where(W > 0, W, np.nan))
    return [float(x) for x in rh]


def _ess_bulk(chains):
    """Bulk effective sample size per coordinate (Geyer initial-positive-sequence estimator on the
    chain-averaged autocorrelation).  Rough but dependency-free; reported alongside R-hat."""
    X = [np.asarray(c, float) for c in chains]
    m = len(X); n = min(len(c) for c in X); d = X[0].shape[1]
    maxlag = min(n - 1, 200)
    out = []
    for j in range(d):
        acc = np.zeros(maxlag + 1); ok = True
        for c in X:
            x = c[:n, j] - c[:n, j].mean(); v = float(x @ x)
            if v <= 0:
                ok = False; break
            for t in range(maxlag + 1):
                acc[t] += float(x[:n - t] @ x[t:]) / v
        if not ok:
            out.append(float(m * n)); continue
        acc /= m
        s = acc[0]; t = 1
        while t + 1 <= maxlag:            # sum consecutive pairs until the pair turns negative (Geyer)
            pair = acc[t] + acc[t + 1]
            if pair < 0:
                break
            s += 2.0 * pair; t += 2
        out.append(float(min(m * n / max(s, 1e-6), m * n)))
    return out


def _reconstruct_np(u_s, setup):
    """numpy version of the reconstruct map (for the forward-only decision read)."""
    if setup["Vmat"] is not None:
        return setup["u_iid"] + setup["Vmat"] @ u_s
    uf = setup["u_iid"].copy(); uf[setup["idx_sample"]] = u_s
    return uf


def _decision_read(U_s, setup, a):
    """Forward-only correlated decision read at n_steps.  Reconstructs full u per draw (sampled coords
    written into the iid MAP; complement frozen); no Jacobian.  Returns (gs, stats)."""
    tol = np.array(json.load(open(os.path.join(a.out_dir, f"{a.product}_decision.json"))).get("tol", [0.02, 0.05]))
    Ti = np.diag(1.0 / tol)
    op, b = setup["op"], setup["b"]
    take = np.linspace(0, len(U_s) - 1, min(a.decision_max, len(U_s))).astype(int)
    gs = []
    for row in U_s[take]:
        uf = _reconstruct_np(row, setup)
        try:
            g_fn, _, _ = decision_forward(b, op, uf, n_steps=a.n_steps)
            g = g_fn(torch.tensor(uf, dtype=DTYPE)).detach().numpy()
            gs.append(np.ravel(g)[:2])
        except Exception:  # noqa: BLE001 -- rare degenerate draw; skip
            continue
    gs = np.asarray(gs, float)
    if gs.shape[0] < 2:
        raise ValueError(f"insufficient decision draws (n_used={gs.shape[0]}); most draws failed decision_forward")
    C = np.cov(gs.T)
    spec = (0.70, 0.50)
    stats = {"pool_purity": [float(gs[:, 0].mean()), float(gs[:, 0].std())],
             "pool_yield": [float(gs[:, 1].mean()), float(gs[:, 1].std())],
             "wdec": float(np.sqrt(np.max(np.linalg.eigvalsh(Ti @ C @ Ti)))),
             "p_meet": float(np.mean((gs[:, 0] >= spec[0]) & (gs[:, 1] >= spec[1]))),
             "C_decision_nuts": C.tolist(), "n_used": int(len(gs))}
    return gs, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--product", default="HLXSYN")
    ap.add_argument("--params", default="main4", choices=["main4", "nosigma-full", "decision-active"])
    ap.add_argument("--active-dim", type=int, default=4, help="(decision-active) subspace dim (>=2)")
    ap.add_argument("--only-component", default="main", help="(main4) component to sample; rest fixed at iid MAP")
    ap.add_argument("--kernel", default="ou", choices=["ou", "ar1", "matern32"])
    ap.add_argument("--rho-max", type=float, default=0.9)
    ap.add_argument("--n-steps-lik", type=int, default=60, help="ODE resolution for the NUTS likelihood")
    ap.add_argument("--n-steps", type=int, default=300, help="ODE resolution for the decision read / basis")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max-tree-depth", type=int, default=8)
    ap.add_argument("--target-accept", type=float, default=0.85)
    ap.add_argument("--full-mass", action="store_true",
                    help="adapt a DENSE mass matrix (captures correlated geometry at the displaced mode; "
                         "the diagonal default cannot). Recommended for decision-active.")
    ap.add_argument("--warmup-only", action="store_true",
                    help="run warmup (+ 1 sample), save adapted mass matrix + step size, then exit.  "
                         "Use --load-warmup in a second session to sample with the adapted geometry.")
    ap.add_argument("--load-warmup", default="",
                    help="path to a warmup checkpoint (.npz) saved by --warmup-only.  Disables adaptation, "
                         "sets warmup=0, and uses the saved mass matrix + step size for sampling.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chains", default="", help="comma-sep seeds to pool for the final JSON (default: just --seed)")
    ap.add_argument("--pool-only", action="store_true")
    ap.add_argument("--marginal-only", action="store_true",
                    help="(decision-active) build setup only -- no sampling, no chains -- and print the full "
                         "GGN marginal decision precision A = G Sig_full G^T and the conditional B at the "
                         "current --n-steps-lik/ell.  Run at several --n-steps-lik to see how much of the "
                         "decision-active marginal is ODE-resolution inflation.")
    ap.add_argument("--decision-max", type=int, default=300)
    ap.add_argument("--out-dir", default=RES)
    a = ap.parse_args()
    if a.params == "decision-active" and a.active_dim < 2:
        raise SystemExit("--active-dim must be >= 2")

    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    tag = {"main4": "", "nosigma-full": "_nosigma", "decision-active": "_decactive"}[a.params]
    def chain_path(s):
        return os.path.join(a.out_dir, f"{a.product}_correlated_nuts{tag}_chain{s}.npz")

    setup = _build_setup(a)
    print(f"[{a.product}] correlated hyper: {setup['hyper']}")
    print(f"sampling {setup['label']}; likelihood n_steps={a.n_steps_lik}, decision n_steps={a.n_steps}")

    if a.marginal_only:  # ODE-inflation probe: full GGN marginal A and conditional B, no sampling
        if setup["G"] is None or setup["Sig_full"] is None:
            raise SystemExit("--marginal-only requires --params decision-active")
        tol = np.array(json.load(open(os.path.join(a.out_dir, f"{a.product}_decision.json"))).get("tol", [0.02, 0.05]))
        Ti = np.diag(1.0 / tol)
        wd = lambda X: float(np.sqrt(np.max(np.linalg.eigvalsh(Ti @ X @ Ti))))
        Gfull = setup["G"]; Gs = Gfull @ setup["Vmat"]
        A_ = Gfull @ setup["Sig_full"] @ Gfull.T                  # full GGN marginal
        B_ = Gs @ setup["lap"].cov @ Gs.T                         # GGN conditional (complement frozen)
        mo = {"product": a.product, "n_steps_lik": a.n_steps_lik, "ell": setup["hyper"]["ell"],
              "wdec_marginal_A": wd(A_), "wdec_conditional_ggn_B": wd(B_),
              "C_decision_marginal": A_.tolist(), "C_decision_laplace": B_.tolist(),
              "deployed_reference_wdec": 0.543}
        out = os.path.join(a.out_dir, f"{a.product}_correlated_nuts_decactive_marginal_nlik{a.n_steps_lik}.json")
        json.dump(mo, open(out, "w"), indent=2)
        print(f"\n  MARGINAL-ONLY  n_steps_lik={a.n_steps_lik}  ell={setup['hyper']['ell']:.0f}")
        print(f"    A full GGN marginal  wdec {wd(A_):.3f}   (deployed finer-ODE marginal 0.543)")
        print(f"    B GGN conditional    wdec {wd(B_):.3f}")
        print(f"  wrote {out}")
        return

    resume_from = None
    if a.load_warmup:
        ckpt = np.load(a.load_warmup, allow_pickle=True)
        resume_from = {"inverse_mass_matrix": np.asarray(ckpt["inverse_mass_matrix"], float),
                       "step_size": float(ckpt["step_size"])}

    if not a.pool_only:
        U_c, hist = _whitened_corr_nuts(
            setup["reconstruct"], setup["predict_fn"], setup["obs"], setup["pm_full"], setup["ps_full"],
            setup["lap"], setup["idx_blocks"], setup["Kinv"], num_samples=a.num_samples, warmup=a.warmup,
            max_tree_depth=a.max_tree_depth, target_accept=a.target_accept, seed=a.seed, progress=True,
            full_mass=a.full_mass, resume_from=resume_from, warmup_only=a.warmup_only)

        if a.warmup_only:
            warmup_path = os.path.join(a.out_dir,
                                       f"{a.product}_correlated_nuts{tag}_warmup_seed{a.seed}.npz")
            ad = hist["adapted"]
            np.savez(warmup_path, inverse_mass_matrix=ad["inverse_mass_matrix"],
                     step_size=ad["step_size"], seed=a.seed, warmup_steps=a.warmup,
                     wall_s=round(time.time() - t0, 1))
            print(f"\n  warmup checkpoint saved -> {warmup_path}  ({round(time.time()-t0,1)}s)")
            print(f"  adapted step_size={ad['step_size']:.4e}")
            print(f"  mass matrix shape={ad['inverse_mass_matrix'].shape}")
            print(f"\n  To sample, run in a new session:")
            print(f"    python scripts/bayes_correlated_nuts.py --product {a.product} "
                  f"--params {a.params} --active-dim {a.active_dim} --full-mass "
                  f"--load-warmup {warmup_path} --num-samples {a.num_samples} "
                  f"--max-tree-depth {a.max_tree_depth} --seed {a.seed}")
            return

        np.savez(chain_path(a.seed), U=U_c, seed=a.seed, hyper=json.dumps(setup["hyper"]),
                 wall_s=round(time.time() - t0, 1))
        _nuts_diag(hist, a.num_samples)
        print(f"  chain {a.seed}: {U_c.shape[0]} draws -> {chain_path(a.seed)}  ({round(time.time()-t0,1)}s)")

    seeds = [int(s) for s in a.chains.split(",") if s.strip() != ""] or [a.seed]
    chains, missing = [], []
    for s in seeds:
        if os.path.exists(chain_path(s)):
            chains.append(np.asarray(np.load(chain_path(s))["U"], float))
        else:
            missing.append(s)
    if missing:
        raise SystemExit(f"missing chain files for seeds {missing}; run each (e.g. --seed {missing[0]}) "
                         f"then re-run with --pool-only --chains {a.chains or ','.join(map(str, seeds))}")
    U = np.concatenate(chains, 0)
    rhat = _rhat(chains)
    lap, prior_std_s = setup["lap"], setup["prior_std_s"]
    nut = _Gauss(mean=U.mean(0), cov=np.cov(U, rowvar=False), samples=U)
    nut_cov = nut.cov

    result = {
        "product": a.product, "params": a.params, "subspace_dim": setup["dsub"], "label": setup["label"],
        "kernel": a.kernel, "hyper": setup["hyper"], "n_steps_lik": a.n_steps_lik, "n_steps_decision": a.n_steps,
        "n_chains": len(chains), "draws_per_chain": [int(len(c)) for c in chains], "n_pooled": int(len(U)),
        "rhat_max": rhat, "wall_s": round(time.time() - t0, 1),
        "worst_dir_laplace": _worst_dir(lap.cov, prior_std_s),
        "worst_dir_nuts": _worst_dir(nut_cov, prior_std_s),
        "laplace_vs_nuts": _deviation(nut, lap, type("P", (), {"std": prior_std_s})()),
        "deployed_reference": {"wdec_corr": 0.543, "p_meet_corr": 0.889, "pool_purity": 0.720},
    }
    if setup["keq_nu_pairs"]:
        result["keq_nu_corr"] = [
            {"component": nm,
             "corr_laplace": float(lap.cov[kp, vp] / np.sqrt(lap.cov[kp, kp] * lap.cov[vp, vp])),
             "corr_nuts": float(nut_cov[kp, vp] / np.sqrt(nut_cov[kp, kp] * nut_cov[vp, vp]))}
            for (kp, vp, nm) in setup["keq_nu_pairs"]]
    if len(chains) >= 2:  # per-coordinate convergence report (cheap NUTS diagnostics)
        result["nuts_diag"] = {"rhat_per_dim": _rhat_per_dim(chains), "ess_bulk_per_dim": _ess_bulk(chains)}
    if setup["G"] is not None:  # decision-active: the three decision-covariance readouts (Schur decomp.)
        Gfull = setup["G"]                                       # (2, 4n) decision Jacobian at the iid MAP
        Gs = Gfull @ setup["Vmat"]                               # (2, r) decision Jacobian in s-coords
        result["C_decision_laplace"] = (Gs @ lap.cov @ Gs.T).tolist()          # B: GGN conditional (complement frozen)
        if setup.get("Sig_full") is not None:
            result["C_decision_marginal"] = (Gfull @ setup["Sig_full"] @ Gfull.T).tolist()  # A: full GGN marginal

    out = os.path.join(a.out_dir, f"{a.product}_correlated_nuts{tag}.json")
    json.dump(result, open(out, "w"), indent=2)

    gs = None
    try:
        gs, ds = _decision_read(U, setup, a)
        tol = np.array(json.load(open(os.path.join(a.out_dir, f"{a.product}_decision.json"))).get("tol", [0.02, 0.05]))
        Ti = np.diag(1.0 / tol)
        if "C_decision_laplace" in result:
            ds["rel_frob_C_nuts_vs_laplace"] = _rel_frob(np.array(ds["C_decision_nuts"]),
                                                         np.array(result["C_decision_laplace"]))
        # Schur decomposition in the SAME r-dim active basis: A full GGN marginal, B GGN conditional
        # (complement frozen), C NUTS conditional.  A-B isolates the conditional-vs-marginal (Schur) gap,
        # B-C the local-Gaussian error, A-C the total; all three worst_dec are reported side by side.
        if "C_decision_marginal" in result and "C_decision_laplace" in result:
            A_ = np.array(result["C_decision_marginal"]); B_ = np.array(result["C_decision_laplace"])
            C_ = np.array(ds["C_decision_nuts"])
            wd = lambda X: float(np.sqrt(np.max(np.linalg.eigvalsh(Ti @ X @ Ti))))
            nf = lambda X: float(np.linalg.norm(X))
            ds["schur_decomposition"] = {
                "wdec_marginal_A": wd(A_), "wdec_conditional_ggn_B": wd(B_), "wdec_conditional_nuts_C": wd(C_),
                "relfrob_schur_A_vs_B": nf(A_ - B_) / max(nf(A_), 1e-30),          # conditional vs marginal
                "relfrob_localgauss_B_vs_C": nf(B_ - C_) / max(nf(C_), 1e-30),     # GGN vs sampled (local Gaussian)
                "relfrob_total_A_vs_C": nf(A_ - C_) / max(nf(A_), 1e-30)}
        try:  # per-chain decision precision + batch-means MCSE (verdict convergence)
            ds["wdec_per_chain"] = [round(_decision_read(c, setup, a)[1]["wdec"], 4) for c in chains]
            ds["wdec_between_chain_sd"] = (float(np.std(ds["wdec_per_chain"], ddof=1))
                                           if len(ds["wdec_per_chain"]) > 1 else None)
            K, nb = 10, len(gs) // 10
            if nb >= 5:
                bw = [np.sqrt(np.max(np.linalg.eigvalsh(Ti @ np.cov(gs[i*nb:(i+1)*nb].T) @ Ti))) for i in range(K)]
                ds["wdec_mcse_batchmeans"] = float(np.std(bw, ddof=1) / np.sqrt(K))
        except Exception:  # noqa: BLE001
            pass
        result["decision_nuts"] = ds
        dec_line = (f"  DECISION (correlated NUTS): purity {ds['pool_purity'][0]:.3f}+/-{ds['pool_purity'][1]:.3f}  "
                    f"yield {ds['pool_yield'][0]:.3f}+/-{ds['pool_yield'][1]:.3f}  "
                    f"wdec {ds['wdec']:.3f}  P(meet) {ds['p_meet']:.3f}")
    except Exception as e:  # noqa: BLE001
        result["decision_nuts"] = {"error": repr(e)}
        dec_line = f"  DECISION read FAILED ({e!r}); parameter comparison still written"
    json.dump(result, open(out, "w"), indent=2)
    if gs is not None:
        try:
            np.savez(os.path.join(a.out_dir, f"{a.product}_correlated_nuts{tag}_samples.npz"), U=U, g=gs)
        except Exception as e:  # noqa: BLE001
            print(f"  (warning: samples npz not saved: {e!r})")

    print(f"\nwrote {out}  ({result['wall_s']}s; {len(chains)} chain(s), {len(U)} pooled draws, "
          f"R-hat max {rhat if rhat is None else round(rhat, 3)})")
    print(f"  worst_dir  Laplace {result['worst_dir_laplace']:.3f}  vs  NUTS {result['worst_dir_nuts']:.3f}")
    for r in result.get("keq_nu_corr", []):
        print(f"  keq-nu corr [{r['component']}]  Laplace {r['corr_laplace']:+.3f}  vs  NUTS {r['corr_nuts']:+.3f}")
    dn = result.get("decision_nuts") if isinstance(result.get("decision_nuts"), dict) else {}
    sd = dn.get("schur_decomposition")
    if sd:
        print(f"  SCHUR ({setup['dsub']}-D basis)  wdec: A marginal {sd['wdec_marginal_A']:.3f}  "
              f"B GGN-cond {sd['wdec_conditional_ggn_B']:.3f}  C NUTS-cond {sd['wdec_conditional_nuts_C']:.3f}")
        print(f"    rel-Frob  A-B (cond vs marg) {sd['relfrob_schur_A_vs_B']:.3f}  "
              f"B-C (local-Gauss) {sd['relfrob_localgauss_B_vs_C']:.3f}  A-C (total) {sd['relfrob_total_A_vs_C']:.3f}")
    if dn.get("wdec_per_chain"):
        print(f"  per-chain wdec {dn['wdec_per_chain']}  between-chain sd {dn.get('wdec_between_chain_sd')}  "
              f"batch-means MCSE {dn.get('wdec_mcse_batchmeans')}")
    if result.get("nuts_diag"):
        nd = result["nuts_diag"]
        print(f"  R-hat/dim {[round(x,3) for x in (nd['rhat_per_dim'] or [])]}  "
              f"ESS-bulk {[round(x) for x in (nd['ess_bulk_per_dim'] or [])]}")
    print(f"  whitened mean/std dev (NUTS vs Laplace): {result['laplace_vs_nuts']}")
    print(dec_line)
    print(f"  DEPLOYED reference: wdec 0.543  P(meet) 0.889  purity 0.720")


if __name__ == "__main__":
    main()
