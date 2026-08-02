#!/usr/bin/env python3
"""Nonlinear paired threshold certificate (Corollary 2.5) at the law the recommendation is taken under.

WHY THIS EXISTS. ``step2d_meet_margin_certificate.py`` certifies the deployed 0.95 gate with BOTH arms
linearised, by design: substituting the deployed Monte-Carlo pushforward into one arm only would make the flip
rates measure linearisation error on top of compression. For mAb C (HLXSYN) the linearisation screen fires
(rel-Frobenius 0.695 > 0.5) and the final recommendation rests on the NONLINEAR empirical convolution, under
which the best deployable candidate at 28.1 g/L reads 0.838 (not the screening 0.879) and the below-range
candidate at 19.7 g/L reads 0.906 (not 0.975). The certified law and the acted-on law therefore differ for that
one product. This script closes the gap by putting the NONLINEAR map on BOTH arms:

    X_a     = T^{-1}{ g^nl(theta)   + b + delta + e },     theta ~ correlated-residual posterior
    Y_{a,j} = T^{-1}{ g^nl(theta_j) + b + delta + e },     theta_j = pi_j(theta), same canonical coupling

with (b, delta, e) -- product bias, hierarchical discrepancy, measurement noise -- drawn ONCE per paired draw
and added to BOTH arms, so the predictive layer cancels from X - Y while still moving both draws relative to
the specification boundary. Nothing here is linearised; g^nl is the same solver pushforward
``decision_covariance_mc`` builds, on the same MAP-fixed collection window and with the same physical clip.

THE COUPLINGS ARE THE SAME ONES, WRITTEN IN PARAMETER SPACE. step2d builds them in DECISION space (it draws the
coupled Gaussian increments and multiplies by G). That construction cannot be reused verbatim when the map is
nonlinear, because there is no G to multiply by -- the reduced parameter vector itself has to reach the solver.
Both canonical couplings are, however, DETERMINISTIC maps of the full draw, which is what makes the rewrite
exact rather than merely equivalent in law. In the rotated coordinates of ``rotate_sigma_block`` (v = the sigma
differential block, u = everything else including the sigma common mode), writing z = theta - mu,

    Schur      pi_S(z) = ( z_u , K z_u ),                K = Sigma_vu Sigma_uu^+     [ v <- E[v|u] ]
    variant D  pi_D(z) = ( z_u - L z_v , 0 ),            L = Sigma_uv Sigma_vv^+     [ v <- mu_v, u | v ]

Then z - pi_S(z) = (0, z_v - E[v|u]) and z - pi_D(z) = (L z_v, z_v), so under the linearisation the induced
decision increments are exactly G_v (v - E[v|u]) and Gamma_v z_v with Gamma_v = G_v + G_u L = ridge_derivative,
i.e. exactly the two identities of Lemma 2.1. ``--selftest`` checks all of this numerically against
``decision_compression``: the marginal covariances of the three arms against C, C_S, C_D and the two paired
differences against C - C_S and C - C_D. pi_D's u-part has covariance Sigma_{u|v} = (H_uu)^{-1}, which is the
restricted local-Laplace posterior ``variant_d_cov`` uses. One caveat on ``--selftest``: it runs on the
UNCLIPPED draws, whereas the run clips to ``physical_u_bounds`` exactly as ``decision_covariance_mc`` does
(decision.py:249), and on HLXSYN that clip binds on ~86% of full-arm draws. The clip is applied identically to
all three arms, so each arm's marginal remains the clip-pushforward of its own law and every flip rate stays a
flip rate of the law actually deployed; but the exactness the selftest verifies is the exactness of the
pre-clip coupling. ``clip_frac_by_arm`` is recorded per candidate so the asymmetry is visible.

Sharing the full arm between the two reductions is the one deliberate departure from step2d, which draws an
independent full arm per reduction. Only each arm's MARGINAL law enters a flip rate, so a shared full arm is
equally valid, it saves a third of the solver calls, and it makes the two reductions' certificates read off one
common estimate of p_a instead of two.

WHY THE CERTIFICATE IS STATED ON TWO JOINT EVENTS, NOT ON p_a AND q^+- SEPARATELY. Corollary 2.5's population
conditions are

    p_a - q^-_{a,j} >= tau   =>   p_{a,j} >= tau        p_a + q^+_{a,j} < tau   =>   p_{a,j} < tau,

and on the canonical coupling p_a - q^- = P(X in Abar AND Y in Abar) and p_a + q^+ = P(X in Abar OR Y in Abar)
IDENTICALLY -- the two forms are the same population statement, not an approximation of one another. In the
linear certificate p_a is closed form (a mean of bivariate orthant probabilities over the committed hierarchy
draws) and only q^+- carries Monte-Carlo error, so a single limit per direction sufficed. Here p_a is itself
Monte Carlo, and the split form would need simultaneous limits on p_a AND on q^+-: four limits per cell instead
of two, combined by a union bound that throws away the perfect dependence between the counts (K_both = K_X -
K_minus is a single count, not a difference of two independent ones). The joint-event form is therefore
(i) exactly the same population condition, (ii) strictly tighter at the SAME budget -- at the 28.1 g/L cell,
n = 1200, p_a = 0.838, q^+ = 0.012, alpha = 0.05/126: U(P_either) = 0.8826 against
U(p_a) + U(q^+) = 0.8982; at the 19.65 g/L cell, n = 5000, p_a = 0.906, q^+ = 0.030, the joint form certifies
non-decisive (0.9470 < 0.95) and the split form does not (0.9583) -- and (iii) one sentence in the paper: the
compressed candidate is decisive whenever the two laws JOINTLY meet with probability at least tau, and
non-decisive whenever the probability that EITHER meets is below tau. It also comes with a free simultaneous
reading, since

    L(P_both) <= min(p_a, p_{a,j})  <=  max(p_a, p_{a,j}) <= U(P_either),

so one pair of limits bounds the deployed probability and the compressed one at once. q^- and q^+ are still
estimated and reported as diagnostics, and their limits ARE inside the familywise budget: ``_analyse`` emits
SEVEN Clopper-Pearson limits per cell -- L(K_both), U(K_either), U(K^-), U(K^+), U(K^- + K^+), L(K_X) and
U(K_upX), the last two being what the split-form verdict booleans consume -- and all seven are counted in
``alpha_per_test``, so nothing printed is uncovered.

CONFIDENCE. Every paired draw is an independent replicate of the whole joint -- one posterior draw, one
hierarchy index, one noise realisation -- so the two indicator counts are exactly Binomial and Clopper-Pearson
is exact. This is unlike the theta-CLUSTERED estimator in ``bayes_empirical_convolution.py``, which reuses a
fixed set of pushforwards with k_inner inner draws each and therefore needs batch-means. As in step2d the
statement covers the empirical mixture over the COMMITTED hierarchy Gibbs draws, not sampling error in the
hierarchy posterior itself.

SAMPLE-SIZE ALLOCATION IS PRE-REGISTERED, from committed artifacts only. Each paired draw costs THREE ODE
solves, so a common n over the pool is not affordable. n_a is chosen from a fixed ladder by the rule in
``_allocate``, driven by the screening probability already committed in
``empirical_convolution_window_{P}.json`` / ``empirical_convolution_{P}.json`` where a nonlinear read exists and
``meet_margin_certificate.json`` otherwise. The allocation therefore depends on no part of the new paired
sample, and Clopper-Pearson stays exact conditional on the fixed n_a. Nothing here is sequential.

Run (Colab, after `pip install -e ".[bayes]"`), THREE STEPS:

  # 1. measure the per-solve cost on this machine and print the projected wall clock; nothing is committed
  OMP_NUM_THREADS=1 python scripts/step2e_nonlinear_meet_certificate.py --product HLXSYN --probe --probe-n 5

  # 2. the run itself (resumable: raw pushforwards are cached per candidate)
  OMP_NUM_THREADS=1 python scripts/step2e_nonlinear_meet_certificate.py \
      --product HLXSYN --n-steps 300 --workers 8 \
      --posterior correlated_posterior --hier-draws results/bayes/hier_draws_capB.npz \
      --domain 25 35 --extra-loadings 19.65 --include-historical \
      --out results/bayes/nonlinear_meet_certificate_HLXSYN.json

  # 3. re-read the cache and redo only the (free) Clopper-Pearson analysis, e.g. at another delta
  ... same command with --analysis-only
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

TAU = 0.95
_PRODUCT_MAP = {"HLXSYN": ("HLXSYN", ["DT"])}
# pre-registered ladder and allocation band; see _allocate
LADDER = (150, 300, 600, 1200, 2500, 5000)
BAND = 0.04           # inflation of the screening probability before projecting a limit
MARGIN = 0.01         # required distance of the projected limit from tau
CRITICAL_P = 0.80     # a candidate any committed law puts above this gets the floor below
CRITICAL_FLOOR = 1200


# --------------------------------------------------------------------------- couplings (pure numpy)

def _rotation(dim: int, n: int, e_c=None) -> np.ndarray:
    """The orthogonal R with ``Sigma_rot = R^T Sigma R`` used by ``rotate_sigma_block``.

    Rebuilt here because that routine returns the rotated matrices but not the basis, and the nonlinear arms
    have to be mapped BACK to physical u before they reach the solver. ``_check_rotation`` asserts it agrees.
    """
    from cex_model.bayes.decision_compression import common_mode_direction

    if e_c is None:
        e_c = common_mode_direction(n)
    Q, _ = np.linalg.qr(np.column_stack([e_c, np.eye(n)]))
    if np.dot(Q[:, 0], e_c) < 0:
        Q[:, 0] *= -1.0
    R = np.eye(dim)
    R[3 * n:4 * n, 3 * n:4 * n] = Q
    return R


def _check_rotation(Sigma, Sr, R, tol=1e-10) -> float:
    err = float(np.max(np.abs(R.T @ Sigma @ R - Sr)))
    if err > tol:
        raise SystemExit(f"rebuilt rotation disagrees with rotate_sigma_block (max |diff| = {err:.3e})")
    return err


def _reduction_coefficients(Sr, ui, vi):
    pinv = np.linalg.pinv
    Suu = Sr[np.ix_(ui, ui)]
    Svv = Sr[np.ix_(vi, vi)]
    Svu = Sr[np.ix_(vi, ui)]
    return Svu @ pinv(Suu), Svu.T @ pinv(Svv)      # K = E[v|u] coeff, L = E[u|v] coeff


def paired_parameter_draws(mean, Sr, R, ui, vi, rng, n):
    """``(u_full, u_S, u_D)`` in PHYSICAL u coordinates, on the canonical couplings, unclipped.

    theta_S and theta_D are deterministic functions of theta, so the three arms are one draw, not three.
    """
    K, L = _reduction_coefficients(Sr, ui, vi)
    dim = Sr.shape[0]
    z = rng.multivariate_normal(np.zeros(dim), Sr, size=n)
    zu, zv = z[:, ui], z[:, vi]
    zS = z.copy()
    zS[:, vi] = zu @ K.T                                   # v <- E[v|u]
    zD = z.copy()
    zD[:, vi] = 0.0                                        # v <- mu_v
    zD[:, ui] = zu - zv @ L.T                              # u <- residual of u on v  (=> Cov = Sigma_{u|v})
    mean = np.asarray(mean, float)
    return mean + z @ R.T, mean + zS @ R.T, mean + zD @ R.T


# --------------------------------------------------------------------------- solver pushforward

_W: dict = {}


def _init_worker(product, op, n_steps, u_map, spec_kw):
    """Per-process solver state: one TorchSimulator and the ONE MAP-fixed collection window."""
    import torch

    torch.set_num_threads(1)
    import cex_model.app_support as A
    from cex_model.app_support import group_indices
    from cex_model.bayes.active import _sim_for_op
    from cex_model.bayes.compare import filter_experiments
    from cex_model.bayes.likelihood import unpack_u
    from cex_model.diffsolver.collection_objective import select_window_for_gradient
    from cex_model.diffsolver.torch_solver import DTYPE

    prod, drop = _PRODUCT_MAP.get(product, (product, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
    n = bundle.components.n_protein
    grp = group_indices(bundle.components)
    sim = _sim_for_op(bundle, op, n_steps)
    with torch.no_grad():
        curve_map = sim.elution_curve(
            *unpack_u(torch.tensor(np.asarray(u_map, float), dtype=DTYPE), n), differentiable=False).numpy()
    sel = select_window_for_gradient(curve_map, grid=spec_kw["grid"], acid_idx=grp["acid"],
                                     main_idx=grp["main"], basic_idx=grp["basic"],
                                     acid_max=spec_kw["acid_max"], main_min=spec_kw["main_min"],
                                     basic_max=spec_kw["basic_max"])
    _W.update(sim=sim, n=n, grp=grp, sel=sel, torch=torch, DTYPE=DTYPE, unpack_u=unpack_u)


def _eval_chunk(U):
    """``g^nl`` on a block of u-vectors: identical to ``decision_covariance_mc``'s inner loop (reselect=False)."""
    from cex_model.bayes.decision import _pool_quantities

    torch, DTYPE, unpack_u = _W["torch"], _W["DTYPE"], _W["unpack_u"]
    sim, n, grp, sel = _W["sim"], _W["n"], _W["grp"], _W["sel"]
    out = np.empty((len(U), 2), float)
    for i, u in enumerate(np.asarray(U, float)):
        try:
            with torch.no_grad():
                curve = sim.elution_curve(*unpack_u(torch.tensor(u, dtype=DTYPE), n),
                                          differentiable=False).numpy()
            out[i] = _pool_quantities(curve, sel.start_idx, sel.end_idx, grp["main"])
        except Exception:
            out[i] = np.nan
    return out


def _pushforward(U, *, product, op, n_steps, u_map, spec_kw, workers, chunk=25, progress=True):
    U = np.asarray(U, float)
    if workers <= 1:
        _init_worker(product, op, n_steps, u_map, spec_kw)
        chunks = [U[i:i + chunk] for i in range(0, len(U), chunk)]
        done, t0, out = 0, time.time(), []
        for c in chunks:
            out.append(_eval_chunk(c))
            done += len(c)
            if progress:
                el = time.time() - t0
                print(f"      {done}/{len(U)} solves  {el:7.1f}s  ({el / max(done, 1):.2f} s/solve)", end="\r")
        if progress:
            print()
        return np.vstack(out)

    import multiprocessing as mp

    # torch's OpenMP runtime is already initialised in this process (decision_jacobian ran above);
    # forking after that is a documented deadlock source unless OMP is single-threaded.
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("run with OMP_NUM_THREADS=1 when --workers > 1 "
                         "(torch autograd has already initialised OpenMP in this process)")
    ctx = mp.get_context("fork")
    chunks = [U[i:i + chunk] for i in range(0, len(U), chunk)]
    t0, done, out = time.time(), 0, []
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(product, op, n_steps, u_map, spec_kw)) as pool:
        for res in pool.imap(_eval_chunk, chunks):
            out.append(res)
            done += len(res)
            if progress:
                el = time.time() - t0
                print(f"      {done}/{len(U)} solves  {el:7.1f}s  ({el / max(done, 1) * workers:.2f} s/solve/core)",
                      end="\r")
    if progress:
        print()
    return np.vstack(out)


# --------------------------------------------------------------------------- Clopper-Pearson

def cp_upper(k, n, alpha):
    from scipy.stats import beta as _b
    return 1.0 if k >= n else float(_b.ppf(1.0 - alpha, k + 1, n - k))


def cp_lower(k, n, alpha):
    from scipy.stats import beta as _b
    return 0.0 if k <= 0 else float(_b.ppf(alpha, k, n - k + 1))


# --------------------------------------------------------------------------- allocation

def _screen_table(product, in_dir):
    """Committed screening probabilities per candidate loading, nonlinear first. No new computation."""
    in_dir = Path(in_dir)
    lin, dep, nonlin = {}, {}, {}
    p = in_dir / "meet_margin_certificate.json"
    if p.exists():
        for blk in json.loads(p.read_text()):
            if blk["product"] != product:
                continue
            for r in blk["rows"]:
                key = round(float(r["loading"]), 2)
                if "predictive_p_full" in r:
                    lin[key] = float(r["predictive_p_full"])
                if r.get("predictive_p_deployed") is not None:
                    dep[key] = float(r["predictive_p_deployed"])
    p = in_dir / f"empirical_convolution_window_{product}.json"
    if p.exists():
        for r in json.loads(p.read_text())["rows"]:
            nonlin[round(float(r["loading"]), 2)] = float(r["phat_meet"])
    p = in_dir / f"empirical_convolution_{product}.json"
    if p.exists():
        for r in json.loads(p.read_text())["ops"].values():
            nonlin.setdefault(round(float(r["op"][0]), 2), float(r["p_meet_empirical"]))
    return lin, dep, nonlin


def _allocate(p_screen, p_any_max, alpha):
    """Smallest ladder n whose PROJECTED limit resolves the cell, floored on near-threshold candidates.

    Depends only on committed artifacts, never on the new paired sample, so CP stays exact at the fixed n.
    """
    hi, lo = min(0.9995, p_screen + BAND), max(0.0005, p_screen - BAND)
    pick, target, proj = LADDER[-1], "unresolved", None
    for n in LADDER:
        u = cp_upper(int(round(hi * n)), n, alpha)
        if u < TAU - MARGIN:
            pick, target, proj = n, "non-decisive", u
            break
        low = cp_lower(int(round(lo * n)), n, alpha)
        if low >= TAU + MARGIN:
            pick, target, proj = n, "decisive", low
            break
    if p_any_max >= CRITICAL_P:
        pick = max(pick, CRITICAL_FLOOR)
    # recompute the projection AT the n finally used, so alloc_proj and n_paired always agree
    proj = (cp_lower(int(round(lo * pick)), pick, alpha) if target == "decisive"
            else cp_upper(int(round(hi * pick)), pick, alpha))
    return int(pick), target, float(proj)


# --------------------------------------------------------------------------- the run

def _load(product, in_dir, posterior):
    import cex_model.app_support as A
    from cex_model.bayes.compare import filter_experiments
    from cex_model.bayes.posterior import Posterior

    post = Posterior.load(Path(in_dir) / f"{product}_{posterior}.npz")
    dj = json.loads((Path(in_dir) / f"{product}_decision.json").read_text())
    tol = np.asarray(dj.get("tol", [0.02, 0.05]), float)
    base_op = [float(x) for x in dj["decision_op"]]
    prod, drop = _PRODUCT_MAP.get(product, (product, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
    return post, bundle, tol, base_op


def selftest(post, bundle, op, n_steps, n_draws=200_000, seed=0):
    """The parameter-space maps reproduce Lemma 2.1's two residuals under the linearisation."""
    from cex_model.bayes.decision import decision_jacobian
    from cex_model.bayes.decision_compression import (decision_cov, ridge_derivative, rotate_sigma_block,
                                                      schur_cov, variant_d_cov)

    n = post.n_protein
    G = np.atleast_2d(decision_jacobian(bundle, op, post.u_map, n_steps=n_steps))
    Sr, Gr, ui, vi = rotate_sigma_block(post.cov, G, n)
    R = _rotation(post.cov.shape[0], n)
    rot_err = _check_rotation(post.cov, Sr, R)
    _K, L = _reduction_coefficients(Sr, ui, vi)
    C = {"full": decision_cov(Gr, Sr), "S": decision_cov(Gr, schur_cov(Sr, ui, vi)),
         "D": variant_d_cov(Sr, Gr, ui, vi)}
    rng = np.random.default_rng(seed)
    uf, us, ud = paired_parameter_draws(post.mean, Sr, R, ui, vi, rng, n_draws)
    lin = lambda U: (U - post.mean) @ G.T                  # the linearised decision increment
    X, YS, YD = lin(uf), lin(us), lin(ud)
    rel = lambda A_, B_: float(np.linalg.norm(A_ - B_) / max(np.linalg.norm(B_), 1e-300))
    out = {
        "rotation_max_abs_err": rot_err,
        "gamma_v_identity": float(np.max(np.abs(ridge_derivative(Sr, Gr, ui, vi) - (Gr[:, vi] + Gr[:, ui] @ L)))),
        "cov_full_rel": rel(np.cov(X.T), C["full"]),
        "cov_S_rel": rel(np.cov(YS.T), C["S"]),
        "cov_D_rel": rel(np.cov(YD.T), C["D"]),
        "resid_S_rel": rel(np.cov((X - YS).T), C["full"] - C["S"]),
        "resid_D_rel": rel(np.cov((X - YD).T), C["full"] - C["D"]),
        "n_draws": int(n_draws),
    }
    out["ok"] = bool(max(out["cov_full_rel"], out["cov_S_rel"], out["cov_D_rel"],
                         out["resid_S_rel"], out["resid_D_rel"]) < 0.02 and rot_err < 1e-10)
    return out


def _analyse(g_full, g_red, spec, alpha, xi):
    """Counts and exact one-sided limits for one (candidate, reduction, law) cell.

    ``xi`` is the SHARED predictive draw (product bias + hierarchical discrepancy + measurement noise), one row
    per paired draw, added to BOTH arms; pass None for the conditional law.

    Non-finite pushforwards are never dropped -- dropping would condition the certificate on a data-dependent
    subset. They are resolved in the direction that keeps each one-sided limit conservative: IN the region for
    an upper limit, OUT of it for a lower one.
    """
    n = len(g_full)
    X, Y = np.array(g_full, float), np.array(g_red, float)
    if xi is not None:
        X, Y = X + xi, Y + xi
    fX, fY = np.isfinite(X).all(1), np.isfinite(Y).all(1)
    inX = fX & np.all(X >= spec, axis=1)
    inY = fY & np.all(Y >= spec, axis=1)
    upX, upY = (~fX) | inX, (~fY) | inY                     # conservative for upper limits
    k_both = int(np.sum(inX & inY))
    k_either = int(np.sum(upX | upY))
    k_minus = int(np.sum(inX & ~inY))                       # observed
    k_plus = int(np.sum(~inX & inY))
    k_minus_up = int(np.sum(upX & ~inY))                    # conservative for the q^- upper limit
    k_plus_up = int(np.sum(~inX & upY))
    lo_both = cp_lower(k_both, n, alpha)
    up_either = cp_upper(k_either, n, alpha)
    p_full = float(inX.mean())
    p_comp = float(inY.mean())
    return {
        "n_paired": n,
        "p_full": p_full, "p_comp": p_comp, "obs_dP": abs(p_full - p_comp),
        # --- the certificate (Corollary 2.5, joint-event form) ---
        "p_both": k_both / n, "p_either": k_either / n,
        "p_both_lower": lo_both, "p_either_upper": up_either,
        "certified_decisive": bool(lo_both >= TAU),
        "certified_not_decisive": bool(up_either < TAU),
        "lo": lo_both, "hi": up_either,                     # simultaneous bracket for p_a AND p_{a,j}
        # --- the split-form diagnostics, inside the same budget ---
        "q_minus": k_minus / n, "q_plus": k_plus / n,
        "q_minus_upper": cp_upper(k_minus_up, n, alpha), "q_plus_upper": cp_upper(k_plus_up, n, alpha),
        "eps": max(cp_upper(k_minus_up, n, alpha), cp_upper(k_plus_up, n, alpha)),
        "eps_paired": cp_upper(k_minus_up + k_plus_up, n, alpha),
        "certified_decisive_split": bool(cp_lower(int(np.sum(inX)), n, alpha)
                                         - cp_upper(k_minus_up, n, alpha) >= TAU),
        "certified_not_decisive_split": bool(cp_upper(int(np.sum(upX)), n, alpha)
                                             + cp_upper(k_plus_up, n, alpha) < TAU),
        "n_nonfinite_full": int(np.sum(~fX)), "n_nonfinite_comp": int(np.sum(~fY)),
        # present so table code written against meet_margin_certificate.json does not KeyError; the analytic
        # tube route is a Gaussian construction and has no counterpart at the nonlinear law.
        "eps_analytic": None, "t_star": None,
    }


def run(a):
    from cex_model.bayes.decision import decision_jacobian
    from cex_model.bayes.decision_compression import (decision_cov, rotate_sigma_block, schur_cov,
                                                      variant_d_cov, whiten)
    from cex_model.bayes.design import candidate_pool_for
    from cex_model.bayes.prior import physical_u_bounds

    post, bundle, tol, base_op = _load(a.product, a.in_dir, a.posterior)
    spec = np.asarray(a.spec, float)
    c_meas = np.diag(np.asarray(a.sigma_meas, float) ** 2)
    n = post.n_protein
    dim = post.cov.shape[0]
    spec_kw = dict(grid=40, acid_max=0.20, main_min=0.70, basic_max=0.10)
    lo_u, hi_u = physical_u_bounds(n)

    pool = [list(map(float, o)) for o in candidate_pool_for(bundle, n_candidates=a.n_candidates, seed=a.seed)]
    sel_idx = [i for i, o in enumerate(pool) if a.domain[0] <= o[0] <= a.domain[1]]
    for want in (a.extra_loadings or []):
        sel_idx.append(min(range(len(pool)), key=lambda i: abs(pool[i][0] - want)))
    if a.idx:
        sel_idx = list(a.idx)
    sel_idx = sorted(set(sel_idx))
    cands = ([{"idx": -1, "op": base_op, "tag": "historical"}] if a.include_historical else []) + \
            [{"idx": i, "op": pool[i], "tag": f"cand{i:02d}"} for i in sel_idx]
    for c in cands:
        c["loading"] = float(c["op"][0])
    cands.sort(key=lambda c: c["loading"])

    laws = list(a.laws)
    # Limits ACTUALLY emitted per (candidate, reduction, law) cell by _analyse:
    #   certificate (2): L(K_both), U(K_either)
    #   split diagnostics (+5): U(K^-), U(K^+), U(K^- + K^+), L(K_X), U(K_upX)
    # The last two are consumed by certified_*_split, so they are claims and must be paid for.
    n_limits = 7 if a.report_split_diagnostics else 2
    n_tests = len(cands) * 2 * n_limits * len(laws)
    alpha = a.delta / n_tests
    print(f"[{a.product}] familywise delta={a.delta} over {len(cands)} candidates x 2 reductions x "
          f"{n_limits} limits x {len(laws)} law(s) = {n_tests} tests -> alpha_per_test = {alpha:.6g}")

    lin, dep, nonlin = _screen_table(a.product, a.in_dir)
    for c in cands:
        key = round(c["loading"], 2)
        cand_p = [v.get(key) for v in (nonlin, dep, lin)]
        c["p_screen"] = next((v for v in cand_p if v is not None), 0.0)
        c["p_screen_source"] = ("nonlinear" if nonlin.get(key) is not None else
                                "deployed" if dep.get(key) is not None else
                                "linearised" if lin.get(key) is not None else "none")
        c["p_any_max"] = max([v for v in cand_p if v is not None] or [0.0])
        c["n_paired"], c["alloc_target"], c["alloc_proj"] = (
            _allocate(c["p_screen"], c["p_any_max"], alpha) if a.n_paired is None
            else (a.n_paired, "fixed", None))

    total = sum(c["n_paired"] for c in cands)
    print(f"[{a.product}] allocation (pre-registered, from committed artifacts only):")
    for c in cands:
        print(f"    {c['tag']:>10} load={c['loading']:7.3f}  p_screen={c['p_screen']:.3f}"
              f" ({c['p_screen_source']})  n={c['n_paired']:5d}  projected={c['alloc_target']}")
    print(f"[{a.product}] TOTAL {total} paired draws x 3 arms = {3 * total} solver calls")
    if a.probe:
        return _probe(a, cands, post, spec_kw, total)

    hd = np.load(a.hier_draws, allow_pickle=True) if a.hier_draws else None
    if hd is not None:
        b_draws, Sd_draws = hd[f"b_{a.product}"], hd["Sd_draws"]
        print(f"[{a.product}] predictive layer: {len(Sd_draws)} committed hierarchy draws from {a.hier_draws}")
    elif "predictive" in laws:
        raise SystemExit("--laws predictive needs --hier-draws")

    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    rows, t_solve = [], []
    for c in cands:
        op, nA = c["op"], c["n_paired"]
        rng = np.random.default_rng(a.seed * 1_000_003 + (c["idx"] + 2))

        # Cache key FIRST. The Jacobian is needed only for G_rot (-> Clin, B) and g_map, and both are cached
        # with the pushforwards, so a cached candidate costs no solver call at all -- which is what makes
        # --analysis-only genuinely free. Sigma_rot, u_idx and v_idx do NOT depend on G
        # (decision_compression.rotate_sigma_block), so the couplings and the rng stream are identical
        # whether or not the Jacobian was recomputed. n_candidates is in the key because `idx` indexes
        # candidate_pool_for(bundle, n_candidates=...), so a different pool size is a different op.
        tagk = (f"{a.product}_{a.posterior}_nc{a.n_candidates}_idx{c['idx']}"
                f"_n{nA}_s{a.seed}_st{a.n_steps}")
        cf = cache / f"gnl_{tagk}.npz"
        z = np.load(cf) if cf.exists() else None
        if z is not None and "G" in z.files:
            G, g_map = np.atleast_2d(z["G"]), z["g_map"]
        elif z is not None and a.analysis_only:
            raise SystemExit(f"  [{c['tag']}] cache {cf.name} predates the G/g_map fields; re-run "
                             f"without --analysis-only, or drop the B / mean_g diagnostics")
        else:
            G, g_map, _ = decision_jacobian(bundle, op, post.u_map, n_steps=a.n_steps, return_extra=True)
            G = np.atleast_2d(G)
        Sr, Gr, ui, vi = rotate_sigma_block(post.cov, G, n)
        R = _rotation(dim, n)
        _check_rotation(post.cov, Sr, R)
        Clin = {"full": decision_cov(Gr, Sr), "S": decision_cov(Gr, schur_cov(Sr, ui, vi)),
                "D": variant_d_cov(Sr, Gr, ui, vi)}

        uf, us, ud = paired_parameter_draws(post.mean, Sr, R, ui, vi, rng, nA)
        clip_frac = {k: float(np.mean(((x < lo_u) | (x > hi_u)).any(1)))
                     for k, x in (("full", uf), ("S", us), ("D", ud))}
        n_clip = int(round(clip_frac["full"] * nA))
        uf, us, ud = (np.clip(x, lo_u, hi_u) for x in (uf, us, ud))

        if z is not None:
            g_full, g_S, g_D = z["g_full"], z["g_S"], z["g_D"]
            print(f"  [{c['tag']}] load={c['loading']:.3f}  cached  ({3 * nA} solves reused)")
        elif a.analysis_only:
            print(f"  [{c['tag']}] load={c['loading']:.3f}  SKIPPED (--analysis-only, no cache)")
            continue
        else:
            print(f"  [{c['tag']}] load={c['loading']:.3f}  {3 * nA} solves on {a.workers} worker(s)")
            t0 = time.time()
            gs = _pushforward(np.vstack([uf, us, ud]), product=a.product, op=op,
                              n_steps=a.n_steps, u_map=post.u_map, spec_kw=spec_kw, workers=a.workers)
            t_solve.append((time.time() - t0) / (3 * nA))
            g_full, g_S, g_D = gs[:nA], gs[nA:2 * nA], gs[2 * nA:]
            np.savez_compressed(cf, g_full=g_full, g_S=g_S, g_D=g_D, op=np.asarray(op, float),
                                G=G, g_map=np.asarray(g_map, float))

        rec = {"idx": c["idx"], "tag": c["tag"], "op": [float(x) for x in op], "loading": c["loading"],
               "n_paired": nA, "p_screen": c["p_screen"], "p_screen_source": c["p_screen_source"],
               "alloc_target": c["alloc_target"], "alloc_proj": c["alloc_proj"],
               "n_clipped_full_arm": n_clip,
               "clip_frac_full_arm": clip_frac["full"], "clip_frac_by_arm": clip_frac,
               "mean_g_full": g_full[np.isfinite(g_full).all(1)].mean(0).tolist(),
               "g_map": [float(x) for x in g_map]}

        for law in laws:
            xi = None
            if law == "predictive":
                # ONE hierarchy index and ONE noise realisation per paired draw, shared by both arms and by
                # both reductions, so the predictive layer cancels in X - Y while moving both draws.
                jj = rng.integers(0, len(Sd_draws), size=nA)
                xi = np.empty((nA, 2))
                for h in np.unique(jj):
                    m = jj == h
                    xi[m] = rng.multivariate_normal(b_draws[h], Sd_draws[h] + c_meas, size=int(m.sum()))
                rec["hier_idx_unique"] = int(len(np.unique(jj)))
            for j, g_red in (("S", g_S), ("D", g_D)):
                cell = _analyse(g_full, g_red, spec, alpha, xi)
                Dj = float(np.linalg.norm(whiten(Clin["full"] - Clin[j], tol), "fro"))
                cj = float(np.linalg.eigvalsh(whiten(Clin[j], tol)).min())
                cell["B"] = Dj / (2.0 * np.sqrt(max(cj, 1e-12)))     # Theorem 2.2 constant, linearised
                cell["law"] = law
                rec[f"{law}_{j}"] = cell
            rec[f"{law}_p_full"] = rec[f"{law}_S"]["p_full"]
        rows.append(rec)
        for law in laws:
            print("        " + "  ".join(
                f"{law[:4]}/{j}: p={rec[f'{law}_{j}']['p_full']:.3f} "
                f"[{rec[f'{law}_{j}']['lo']:.3f},{rec[f'{law}_{j}']['hi']:.3f}] "
                f"{'DEC' if rec[f'{law}_{j}']['certified_decisive'] else ('NON' if rec[f'{law}_{j}']['certified_not_decisive'] else '???')}"
                for j in ("S", "D")))

    out = {"product": a.product, "law": "nonlinear-empirical-convolution", "tau": TAU,
           "spec": spec.tolist(), "tol": tol.tolist(), "posterior": a.posterior,
           "sigma_meas": list(map(float, a.sigma_meas)), "n_steps": a.n_steps,
           "hier_draws": str(a.hier_draws) if a.hier_draws else None, "laws": laws,
           "delta": a.delta, "n_tests": n_tests, "alpha_per_test": alpha,
           "certificate_form": "joint-event (P(both), P(either)); split form reported as diagnostic",
           "evaluated_scope": {"n_pool": int(a.n_candidates), "n_evaluated": len(rows),
                               "domain": [float(x) for x in a.domain],
                               "extra_loadings": [float(x) for x in (a.extra_loadings or [])],
                               "include_historical": bool(a.include_historical),
                               "note": ("certifies every pool candidate"
                                        if sum(r["tag"] != "historical" for r in rows) >= a.n_candidates else
                                        "certifies the evaluated subset only, NOT the whole pool")},
           "n_candidates_evaluated": len(rows), "n_paired_total": int(sum(r["n_paired"] for r in rows)),
           "solver_calls": int(3 * sum(r["n_paired"] for r in rows)),
           "sec_per_solve": float(np.mean(t_solve)) if t_solve else None,
           "allocation": {"ladder": list(LADDER), "band": BAND, "margin": MARGIN,
                          "critical_p": CRITICAL_P, "critical_floor": CRITICAL_FLOOR,
                          "rule": "pre-registered from committed artifacts; independent of the paired sample"},
           "rows": rows}
    if a.selftest and not a.analysis_only:
        # needs a fresh Jacobian, so it is skipped under --analysis-only to keep that mode solver-free
        out["selftest"] = selftest(post, bundle, cands[0]["op"], a.n_steps, seed=a.seed)
        print(f"[{a.product}] selftest: {out['selftest']}")

    for law in laws:
        for j in ("S", "D"):
            cells = [r[f"{law}_{j}"] for r in rows]
            if not cells:
                continue
            straddle = [c for c in cells if not (c["certified_decisive"] or c["certified_not_decisive"])]
            p = np.array([c["p_full"] for c in cells])
            aidx = int(np.argmax(p))
            out[f"summary_{law}_{j}"] = {
                "max_q_minus_upper": float(max(c["q_minus_upper"] for c in cells)),
                "max_q_plus_upper": float(max(c["q_plus_upper"] for c in cells)),
                "max_obs_dP": float(max(c["obs_dP"] for c in cells)),
                "n_certified_decisive": int(sum(c["certified_decisive"] for c in cells)),
                "n_certified_not_decisive": int(sum(c["certified_not_decisive"] for c in cells)),
                "n_straddling_tau": len(straddle),
                "straddling_loadings": [rows[i]["loading"] for i, c in enumerate(cells)
                                        if not (c["certified_decisive"] or c["certified_not_decisive"])],
                # scoped names: this run covers the deployment-domain subset (+ extras), NOT the whole pool
                "decisive_set_certified_on_evaluated_set": len(straddle) == 0,
                "no_decisive_candidate_in_evaluated_set": bool(all(c["certified_not_decisive"] for c in cells)),
                # the q-bar maxima above are dominated by the smallest-n cells: with ZERO observed flips
                # U(0/150) is already ~0.05 at this alpha, against 2.1e-5 at Table 4's 4e5 draws. They are
                # NOT comparable to Table 4's column; the *_critical entries are.
                "min_n_paired": int(min(c["n_paired"] for c in cells)),
                "q_upper_floor_at_min_n": cp_upper(0, int(min(c["n_paired"] for c in cells)), alpha),
                "max_q_minus_upper_critical": (float(max(c["q_minus_upper"] for c in cells
                                                        if c["n_paired"] >= CRITICAL_FLOOR))
                                               if any(c["n_paired"] >= CRITICAL_FLOOR for c in cells) else None),
                "max_q_plus_upper_critical": (float(max(c["q_plus_upper"] for c in cells
                                                       if c["n_paired"] >= CRITICAL_FLOOR))
                                              if any(c["n_paired"] >= CRITICAL_FLOOR for c in cells) else None),
                "argmax_certified": bool(
                    cells[aidx]["lo"] > max([c["hi"] for i, c in enumerate(cells) if i != aidx], default=-1.0)),
                "argmax_loading": rows[aidx]["loading"],
            }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # top level is a LIST of per-product blocks, exactly like meet_margin_certificate.json, so a reader
    # written as `for blk in json.load(f)` works against either file.
    Path(a.out).write_text(json.dumps([out], indent=2))
    print(f"\nwrote {a.out}")
    for k, v in out.items():
        if k.startswith("summary_"):
            print(f"  {k[8:]:22} dec={v['n_certified_decisive']} non={v['n_certified_not_decisive']} "
                  f"neither={v['n_straddling_tau']} {v['straddling_loadings']}  "
                  f"q-<={v['max_q_minus_upper']:.4f} q+<={v['max_q_plus_upper']:.4f} "
                  f"(floor at n={v['min_n_paired']}: {v['q_upper_floor_at_min_n']:.4f}; critical cells only: "
                  f"q-<={v['max_q_minus_upper_critical']} q+<={v['max_q_plus_upper_critical']})")
    return out


def _probe(a, cands, post, spec_kw, total):
    """Time a handful of real solves and project the wall clock. Commits nothing."""
    rng = np.random.default_rng(a.seed)
    U = rng.multivariate_normal(post.mean, post.cov, size=a.probe_n)
    from cex_model.bayes.prior import physical_u_bounds
    lo_u, hi_u = physical_u_bounds(post.n_protein)
    U = np.clip(U, lo_u, hi_u)
    t0 = time.time()
    g = _pushforward(U, product=a.product, op=cands[0]["op"], n_steps=a.n_steps,
                     u_map=post.u_map, spec_kw=spec_kw, workers=1, chunk=1, progress=False)
    ts = (time.time() - t0) / a.probe_n
    print(f"\n[probe] {a.probe_n} solves at n_steps={a.n_steps}: {ts:.2f} s/solve single-core "
          f"({int(np.sum(~np.isfinite(g).all(1)))} non-finite)")
    calls = 3 * total
    for w in (1, 2, 4, 8, 16):
        print(f"  {calls} solver calls on {w:2d} worker(s): {calls * ts / w / 3600:6.2f} h "
              f"(perfect scaling; expect 10-20% loss)")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="HLXSYN")
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--posterior", default="correlated_posterior")
    ap.add_argument("--n-candidates", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--spec", type=float, nargs=2, default=[0.70, 0.50])
    ap.add_argument("--sigma-meas", type=float, nargs=2, default=[0.005, 0.008])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--hier-draws", default="results/bayes/hier_draws_capB.npz")
    ap.add_argument("--laws", nargs="+", default=["predictive"], choices=["predictive", "conditional"],
                    help="every law reported enters the familywise budget; the deployed law is predictive")
    ap.add_argument("--domain", type=float, nargs=2, default=[25.0, 35.0],
                    help="deployment-domain loading window; every pool candidate inside it is certified")
    ap.add_argument("--extra-loadings", type=float, nargs="*", default=[19.65],
                    help="also certify the pool candidates nearest these loadings (the below-range one)")
    ap.add_argument("--idx", type=int, nargs="*", default=None, help="override the candidate selection")
    ap.add_argument("--include-historical", action="store_true", default=True)
    ap.add_argument("--no-include-historical", dest="include_historical", action="store_false")
    ap.add_argument("--n-paired", type=int, default=None,
                    help="fixed n for every candidate; omit to use the pre-registered allocation")
    ap.add_argument("--report-split-diagnostics", action="store_true", default=True,
                    help="also report q^- / q^+ limits, and pay for them in the budget (4 limits/cell)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--cache-dir", default="results/bayes/nonlinear_cert_cache")
    ap.add_argument("--analysis-only", action="store_true", help="re-run only the CP analysis from the cache")
    ap.add_argument("--probe", action="store_true", help="time the solver and project the wall clock, then exit")
    ap.add_argument("--probe-n", type=int, default=5)
    ap.add_argument("--selftest", action="store_true", default=True)
    ap.add_argument("--no-selftest", dest="selftest", action="store_false")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = f"results/bayes/nonlinear_meet_certificate_{a.product}.json"
    run(a)


if __name__ == "__main__":
    main()
