"""Real-data NUTS anchor for the Laplace posterior (the Laplace-credibility run).

All real-data UQ here rests on the MAP+Gauss-Newton **Laplace** posterior, and the keq<->nu
degeneracy (r ~= -0.99) is exactly the non-Gaussian geometry where a local-Gaussian
approximation can deviate. ``profile_likelihood`` certifies the *flat* directions Laplace-free;
this script closes the loop with the **NUTS gold standard on a REAL product** (default HLXSYN).

TWO THINGS MAKE REAL NUTS TRACTABLE HERE
----------------------------------------
1. **Laplace-whitened reparameterization (default).** The keq<->nu ridge (r ~= -0.999) is so
   anisotropic that vanilla NUTS with a diagonal mass matrix diverges (tiny step size, acc ~0.16,
   trees saturate ``max_tree_depth`` -> ~1000 s/draw -- measured). We instead sample
   ``z ~ N(0, I)`` and set ``u = m + L z`` with ``L = chol(Laplace cov)``: an EXACT change of
   variables (the target is unchanged) that makes the geometry NUTS sees ~isotropic near the mode,
   so it mixes in a few leapfrogs/draw. Laplace is only the preconditioner -- if it is a poor
   approximation, the samples are still exact, only the efficiency is affected. ``--no-whiten``
   reverts to the native-space sampler.
2. **One component by default.** The ridge / sloppy sigma are per-component, so we sample ONE
   component's 4 params (the main peak) with the others fixed at MAP -- a 4-D conditional posterior.
   The Laplace we compare against is recomputed in the same conditional (apples-to-apples).
   ``--full-dim`` samples all components jointly (with whitening, may be feasible; without, days).

Honest framing: Laplace may legitimately differ from NUTS on the non-Gaussian ridge; the full-cov
SVI guide should track NUTS more tightly. We report the deviation (raw + prior-whitened) and the NUTS
n_eff/r_hat so the "gold standard" is earned, not assumed.

    # smoke test (whitening should make this fast even at warmup 30 -- read it/s, Ctrl-C):
    OMP_NUM_THREADS=4 python scripts/bayes_nuts_real.py --product HLXSYN --n-steps 60 --warmup 30 --num-samples 30
    # real run:
    OMP_NUM_THREADS=4 python scripts/bayes_nuts_real.py --product HLXSYN --n-steps 60 --warmup 300 --num-samples 300

Outputs results/bayes/{product}_nuts_compare_<comp>.json (+ _nuts_posterior_<comp>.npz, + png).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import cex_model.app_support as A  # noqa: E402
from cex_model.bayes import (  # noqa: E402
    AKTA_NOISE_FLOOR_G_L,
    Posterior,
    laplace_posterior,
    map_fit,
    mechanistic_model,
    nuts_posterior,
    physical_prior,
    svi_posterior,
)
from cex_model.bayes.prior import components_to_u  # noqa: E402
from cex_model.components import ComponentSet  # noqa: E402
from cex_model.diffsolver.calibrate_diff import targets_from_bundle  # noqa: E402
from cex_model.diffsolver.torch_solver import DTYPE  # noqa: E402


@contextlib.contextmanager
def _float64():
    old = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def _whitened_nuts(predfn, obs, prior, lap, sigma_obs, template, *, num_samples, warmup,
                   max_tree_depth, target_accept, seed, progress):
    """NUTS in Laplace-whitened coordinates: sample z~N(0,I), u = m + L z, L=chol(lap.cov).

    Exact change of variables (target unchanged); the whitening only preconditions the geometry.
    """
    import pyro
    import pyro.distributions as dist
    from pyro.infer import MCMC, NUTS

    d = len(lap.mean)
    cov = np.asarray(lap.cov, float) + 1e-12 * np.eye(d)
    L_np = np.linalg.cholesky(cov)
    m_np = np.asarray(lap.mean, float)
    with _float64():
        pyro.clear_param_store(); pyro.set_rng_seed(seed)
        m_t = torch.as_tensor(m_np, dtype=DTYPE)
        L_t = torch.as_tensor(L_np, dtype=DTYPE)
        pm = torch.as_tensor(np.asarray(prior.mean, float), dtype=DTYPE)
        ps = torch.as_tensor(np.asarray(prior.std, float), dtype=DTYPE)
        ob = torch.as_tensor(obs, dtype=DTYPE)
        zeros, ones = torch.zeros(d, dtype=DTYPE), torch.ones(d, dtype=DTYPE)

        def wmodel():
            z = pyro.sample("z", dist.Normal(zeros, ones).to_event(1))
            u = m_t + L_t @ z
            pred = predfn(u)
            logprior = dist.Normal(pm, ps).log_prob(u).sum()
            loglik = dist.Normal(pred, sigma_obs).log_prob(ob).sum()
            logbase = dist.Normal(zeros, ones).log_prob(z).sum()
            pyro.factor("target", logprior + loglik - logbase)  # net log-density = logprior+loglik

        kernel = NUTS(wmodel, jit_compile=False, max_tree_depth=max_tree_depth,
                      target_accept_prob=target_accept)
        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup, num_chains=1,
                    initial_params={"z": torch.zeros(d, dtype=DTYPE)}, disable_progbar=not progress)
        mcmc.run()
        Z = mcmc.get_samples()["z"].detach().numpy()

    U = m_np[None, :] + Z @ L_np.T
    post = Posterior.from_template(mean=U.mean(0), cov=np.atleast_2d(np.cov(U, rowvar=False)),
                                   u_map=U.mean(0), prior=prior, template=template,
                                   sigma_obs=sigma_obs, engine="nuts", samples_u=U)
    return post, {"n_samples": int(num_samples), "diagnostics": mcmc.diagnostics()}


def _deviation(post, ref, prior):
    dm = np.abs(np.asarray(post.mean, float) - np.asarray(ref.mean, float))
    ds = np.abs(np.asarray(post.std_u(), float) - np.asarray(ref.std_u(), float))
    w = np.asarray(prior.std, float)
    return {"max_abs_mean_dev": float(dm.max()), "max_abs_std_dev": float(ds.max()),
            "max_whitened_mean_dev": float((dm / w).max()), "max_whitened_std_dev": float((ds / w).max())}


def _worst_dir(cov, prior_std):
    cov = np.asarray(cov, float); s = np.asarray(prior_std, float)
    return float(np.sqrt(max(float(np.linalg.eigvalsh(cov / np.outer(s, s)).max()), 0.0)))


def _nuts_diag(hist, num_samples):
    diag = (hist or {}).get("diagnostics", {})
    ud = (diag.get("z") or diag.get("u") or {}) if isinstance(diag, dict) else {}
    neff = np.asarray(ud.get("n_eff", []), float).ravel()
    rhat = np.asarray(ud.get("r_hat", []), float).ravel()
    out = {"n_eff_min": float(np.nanmin(neff)) if neff.size else None,
           "r_hat_max": float(np.nanmax(rhat)) if rhat.size and np.isfinite(rhat).any() else None}
    msg = f"  NUTS diag: n_eff min={out['n_eff_min']}/{num_samples}" if out["n_eff_min"] is not None else "  NUTS diag: (none)"
    if out["r_hat_max"] is not None:
        msg += f"  r_hat max={out['r_hat_max']:.3f}"
    if out["n_eff_min"] is not None and out["n_eff_min"] < 0.1 * num_samples:
        msg += "  [low n_eff -> poorly mixed; raise --num-samples/--max-tree-depth]"
    print(msg)
    return out


def _nuts_cov(nut):
    S = nut.samples_u if nut.samples_u is not None else nut.samples(2000, seed=0)
    return np.cov(np.asarray(S, float).T)


def _ridge_component(n, lap_cov):
    best_j, best_r = 0, 0.0
    for j in range(n):
        k, v = j, 2 * n + j
        r = lap_cov[k, v] / np.sqrt(lap_cov[k, k] * lap_cov[v, v])
        if abs(r) >= abs(best_r):
            best_j, best_r = j, r
    return best_j, float(best_r)


def _overlay_plot(lap, nut, n, j, out_path, title):
    k, v, s = j, 2 * n + j, 3 * n + j
    Sl = lap.samples(2000, seed=0)
    Sn = np.asarray(nut.samples_u if nut.samples_u is not None else nut.samples(2000, seed=0), float)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].scatter(Sl[:, k], Sl[:, v], s=4, alpha=0.12, color="tab:blue", edgecolors="none", label="Laplace")
    ax[0].scatter(Sn[:, k], Sn[:, v], s=4, alpha=0.20, color="tab:orange", edgecolors="none", label="NUTS")
    ax[0].set_xlabel("log10 keq"); ax[0].set_ylabel("nu"); ax[0].set_title("keq<->nu ridge")
    ax[0].legend(markerscale=3, fontsize=8)
    ax[1].hist(Sl[:, s], bins=40, density=True, color="tab:blue", alpha=0.5, label=f"Laplace (std {lap.std_u()[s]:.1f})")
    ax[1].hist(Sn[:, s], bins=40, density=True, color="tab:orange", alpha=0.5, label=f"NUTS (std {Sn[:, s].std():.1f})")
    ax[1].set_xlabel("sigma"); ax[1].set_yticks([]); ax[1].set_title("sigma marginal"); ax[1].legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path


def _resolve_component(spec, b):
    comps = b.components.components
    if spec == "main":
        for j, c in enumerate(comps):
            if c.component_type.value == "main":
                return j
        raise SystemExit("no 'main'-type component; pass --only-component cN")
    if spec.lower().startswith("c") and spec[1:].isdigit():
        return int(spec[1:]) - 1
    if spec.isdigit():
        return int(spec) - 1
    for j, c in enumerate(comps):
        if c.name == spec:
            return j
    raise SystemExit(f"component {spec!r} not found; options: {[c.name for c in comps]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--product", default="HLXSYN")
    ap.add_argument("--only-component", default="main",
                    help="component to sample with the rest at MAP ('main'/'cN'/name); --full-dim for joint")
    ap.add_argument("--full-dim", action="store_true", help="sample ALL components jointly")
    ap.add_argument("--no-whiten", action="store_true", help="disable Laplace-whitening (native-space NUTS)")
    ap.add_argument("--n-steps", type=int, default=60)
    ap.add_argument("--map-iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--svi-steps", type=int, default=1500)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--max-tree-depth", type=int, default=8)
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--no-svi", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/bayes")
    args = ap.parse_args()

    b = A.load_product(args.product)
    if args.exclude:
        from cex_model.bayes.compare import filter_experiments
        kept, dropped = filter_experiments(b.experiments, args.exclude)
        if dropped:
            print(f"excluded {len(dropped)} experiment(s): {[e.name for e in dropped]}")
            b.experiments = kept
    n_full = b.components.n_protein
    tgs = targets_from_bundle(b, n_steps=args.n_steps)
    predict_fn, obs = mechanistic_model(tgs, n_full)
    sig = AKTA_NOISE_FLOOR_G_L
    print(f"[{args.product}] n_protein={n_full}, full-dim={4*n_full}, n_exp={len(tgs)}, n_steps={args.n_steps}")

    t0 = time.time()
    u_map, _ = map_fit(components_to_u(b.components), predict_fn, obs, physical_prior(n_full),
                       sigma_obs=sig, iters=args.map_iters, lr=args.lr, progress=True)
    print(f"MAP (full) in {time.time()-t0:.1f}s")

    if args.full_dim:
        n, prior, template, predfn, u_init = n_full, physical_prior(n_full), b.components, predict_fn, u_map
        comp_label, label = "all", ""
    else:
        j = _resolve_component(args.only_component, b)
        comp = b.components.components[j]
        idx4 = torch.tensor([j, n_full + j, 2 * n_full + j, 3 * n_full + j], dtype=torch.long)
        u_map_t = torch.as_tensor(np.asarray(u_map, float), dtype=DTYPE)

        def predfn(u4):  # reduced forward: component j free, others fixed at MAP
            return predict_fn(u_map_t.index_copy(0, idx4, u4))

        n, prior = 1, physical_prior(1)
        template = ComponentSet(components=[comp], loading_correction=b.components.loading_correction)
        u_init = np.asarray(u_map, float)[[j, n_full + j, 2 * n_full + j, 3 * n_full + j]]
        comp_label, label = f"{comp.name} ({comp.component_type.value}; others at MAP)", f"_{comp.name}"
        print(f"sampling c{j+1}={comp.name} ({comp.component_type.value}); other {n_full-1} fixed at MAP (4-D conditional)")

    lap = laplace_posterior(u_init, predfn, obs, prior, template=template, sigma_obs=sig)
    svi = None
    if not args.no_svi:
        print("SVI (full-cov guide) ...")
        svi, _ = svi_posterior(predfn, obs, prior, template=template, sigma_obs=sig,
                               steps=args.svi_steps, lr=0.02, init_loc=lap.mean, seed=args.seed)

    whiten = not args.no_whiten
    print(f"NUTS ({'Laplace-whitened' if whiten else 'native'}) ... warmup={args.warmup} "
          f"draws={args.num_samples} max_tree_depth={args.max_tree_depth} (live ETA below)")
    t0 = time.time()
    if whiten:
        nut, nut_hist = _whitened_nuts(predfn, obs, prior, lap, sig, template,
                                       num_samples=args.num_samples, warmup=args.warmup,
                                       max_tree_depth=args.max_tree_depth, target_accept=args.target_accept,
                                       seed=args.seed, progress=True)
    else:
        nut, nut_hist = nuts_posterior(predfn, obs, prior, template=template, sigma_obs=sig,
                                       num_samples=args.num_samples, warmup=args.warmup, init_loc=lap.mean,
                                       seed=args.seed, progress=True, max_tree_depth=args.max_tree_depth,
                                       target_accept_prob=args.target_accept)
    print(f"NUTS in {time.time()-t0:.1f}s")
    diag = _nuts_diag(nut_hist, args.num_samples)

    nut_cov = _nuts_cov(nut)
    lap_std, nut_std = lap.std_u(), nut.std_u()
    per_comp = []
    for jj in range(n):
        k, v, s = jj, 2 * n + jj, 3 * n + jj
        per_comp.append({"corr_keq_nu_laplace": float(lap.cov[k, v] / np.sqrt(lap.cov[k, k] * lap.cov[v, v])),
                         "corr_keq_nu_nuts": float(nut_cov[k, v] / np.sqrt(nut_cov[k, k] * nut_cov[v, v])),
                         "std_keq": [float(lap_std[k]), float(nut_std[k])],
                         "std_nu": [float(lap_std[v]), float(nut_std[v])],
                         "std_sigma": [float(lap_std[s]), float(nut_std[s])]})
    j_ridge, r_ridge = _ridge_component(n, lap.cov)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    nut.save(out / f"{args.product}_nuts_posterior{label}.npz")
    png = _overlay_plot(lap, nut, n, j_ridge, out / f"{args.product}_nuts_vs_laplace{label}.png",
                        f"{args.product}: Laplace vs NUTS [{comp_label}]")

    result = {"product": args.product, "sampled": comp_label, "whitened": whiten, "dim": 4 * n,
              "n_steps": args.n_steps, "n_exp": len(tgs), "nuts_diag": diag,
              "worst_dir_laplace": _worst_dir(lap.cov, prior.std),
              "worst_dir_nuts": _worst_dir(nut_cov, prior.std),
              "laplace_vs_nuts": _deviation(lap, nut, prior),
              "ridge": {"corr_keq_nu_laplace": r_ridge, "corr_keq_nu_nuts": per_comp[j_ridge]["corr_keq_nu_nuts"]},
              "per_component": per_comp}
    if svi is not None:
        result["svi_vs_nuts"] = _deviation(svi, nut, prior)

    print(f"\n=== real-data engine agreement vs NUTS [{comp_label}] ===")
    d = result["laplace_vs_nuts"]
    print(f"  laplace_vs_nuts  max|Δmean|={d['max_abs_mean_dev']:.3f} (whitened {d['max_whitened_mean_dev']:.3f})"
          f"  max|Δstd|={d['max_abs_std_dev']:.3f} (whitened {d['max_whitened_std_dev']:.3f})")
    if svi is not None:
        d = result["svi_vs_nuts"]
        print(f"  svi_vs_nuts      max|Δmean|={d['max_abs_mean_dev']:.3f} (whitened {d['max_whitened_mean_dev']:.3f})"
              f"  max|Δstd|={d['max_abs_std_dev']:.3f} (whitened {d['max_whitened_std_dev']:.3f})")
    print(f"  worst_dir: Laplace={result['worst_dir_laplace']:.3f}  NUTS={result['worst_dir_nuts']:.3f}")
    print(f"  keq<->nu ridge: Laplace r={r_ridge:+.3f}  NUTS r={per_comp[j_ridge]['corr_keq_nu_nuts']:+.3f}")
    print(f"  sigma std: Laplace={per_comp[j_ridge]['std_sigma'][0]:.1f}  NUTS={per_comp[j_ridge]['std_sigma'][1]:.1f}")

    p = out / f"{args.product}_nuts_compare{label}.json"
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\njson -> {p}\nfigure -> {png}")


if __name__ == "__main__":
    main()
