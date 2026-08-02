"""Fisher-no-prior ablation (P1 #7): is the identifiability / design data-driven?

Two read-outs on a product's posterior:
  * identifiability arm (cheap): the pure data Fisher precision H_fisher = inv(cov) -
    prior.precision() has a ~0 smallest eigenvalue along the structural keq<->nu ridge,
    while WITH the prior that direction is held up to the prior floor -- so the degeneracy
    is in the DATA, not the prior (`bayes.validation.fisher_ablation`);
  * design arm (heavy, Jacobians): rank the candidate experiments by EIG using H (with
    prior) vs H_fisher (no prior) over the SAME Sobol pool; a high Spearman rho means the
    BOED recommendations are not an artefact of the prior either.

Loads the saved {product}_posterior.npz (or fits Laplace fresh).  The design arm is the
cost (one reverse-mode Jacobian per candidate) -> Colab; skip it with --no-design-arm.

    OMP_NUM_THREADS=4 python scripts/bayes_fisher_ablation.py --product HLXSYN
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import cex_model.app_support as A
from cex_model.bayes import (
    AKTA_NOISE_FLOOR_G_L,
    Posterior,
    components_to_u,
    fisher_ablation,
    laplace_posterior,
    map_fit,
    mechanistic_model,
    physical_prior,
)
from cex_model.bayes.design import candidate_pool_for, candidate_predict_fn, expected_info_gain
from cex_model.diffsolver.calibrate_diff import targets_from_bundle
from cex_model.diffsolver.torch_solver import DTYPE


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float((ra @ rb) / denom) if denom > 0 else float("nan")


def _psd_clip(H, floor=1e-9):
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    return V @ np.diag(np.clip(w, floor, None)) @ V.T


def _load_or_fit(bundle, args, out: Path) -> Posterior:
    path = Path(args.posterior) if args.posterior else out / f"{args.product}_posterior.npz"
    if path.exists():
        print(f"loaded posterior: {path}")
        return Posterior.load(path)
    print(f"no saved posterior at {path}; fitting Laplace fresh (n_steps={args.fit_n_steps}) ...")
    n = bundle.components.n_protein
    predict_fn, obs = mechanistic_model(targets_from_bundle(bundle, n_steps=args.fit_n_steps), n)
    prior = physical_prior(n)
    u_map, _ = map_fit(components_to_u(bundle.components), predict_fn, obs, prior,
                       sigma_obs=AKTA_NOISE_FLOOR_G_L, iters=args.map_iters, lr=0.05, progress=True)
    return laplace_posterior(u_map, predict_fn, obs, prior, template=bundle.components,
                             sigma_obs=AKTA_NOISE_FLOOR_G_L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--product", default="HLXSYN")
    ap.add_argument("--posterior", default=None)
    ap.add_argument("--n-candidates", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=120, help="solver grid for the design Jacobian")
    ap.add_argument("--n-points", type=int, default=12)
    ap.add_argument("--fit-n-steps", type=int, default=300)
    ap.add_argument("--map-iters", type=int, default=120)
    ap.add_argument("--no-design-arm", action="store_true", help="skip the heavy Jacobian Spearman arm")
    ap.add_argument("--out-dir", default="results/bayes")
    args = ap.parse_args()

    bundle = A.load_product(args.product)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    post = _load_or_fit(bundle, args, out)
    prior = physical_prior(post.n_protein)

    abl = fisher_ablation(post, prior)
    print("\n=== #7 Fisher-no-prior ablation (identifiability arm) ===")
    print(f"  min-eig(H_fisher, data only) = {abl['min_eig_fisher']:.3e}")
    print(f"  min-eig(H, with prior)       = {abl['min_eig_posterior']:.3e}")
    print(f"  prior precision floor        = {abl['prior_min_eig']:.3e}")
    print(f"  fisher / prior-floor ratio   = {abl['fisher_vs_prior_floor_ratio']:.3e}  "
          f"(~0 => worst direction is prior-held, i.e. unconstrained by data)")
    print(f"  worst data direction params  = {abl['worst_dir_params']}")
    print("  per-component keq<->nu Fisher curvature (≈0 = ridge is a data limitation):")
    for c, v in abl["keq_nu_curvature"].items():
        print(f"    {c}: {v:.3e}")

    result = {"product": args.product, "identifiability_arm": abl}

    if not args.no_design_arm:
        H = np.linalg.inv(np.asarray(post.cov, float))
        H_fisher = _psd_clip(H - np.asarray(prior.precision(), float))
        pool = candidate_pool_for(bundle, n_candidates=args.n_candidates, seed=0)
        u_map = np.asarray(post.u_map, float)
        eig_prior, eig_fisher = [], []
        t0 = time.time()
        for idx, op in enumerate(pool):
            J = torch.autograd.functional.jacobian(
                candidate_predict_fn(bundle, op, n_steps=args.n_steps, n_points=args.n_points, u_map=u_map),
                torch.tensor(u_map, dtype=DTYPE)).detach().numpy()
            eig_prior.append(expected_info_gain(H, J, post.sigma_obs))
            eig_fisher.append(expected_info_gain(H_fisher, J, post.sigma_obs))
            print(f"  candidate {idx + 1}/{len(pool)}: EIG_prior={eig_prior[-1]:.3f} "
                  f"EIG_fisher={eig_fisher[-1]:.3f}")
        rho = _spearman(eig_prior, eig_fisher)
        top_prior = int(np.argmax(eig_prior))
        top_fisher = int(np.argmax(eig_fisher))
        print(f"\n=== #7 design arm ({len(pool)} candidates in {time.time() - t0:.1f}s) ===")
        print(f"  Spearman rho(EIG_prior, EIG_fisher) = {rho:.3f}  (high => design not prior-driven)")
        print(f"  top candidate: with-prior #{top_prior}, no-prior #{top_fisher}, "
              f"{'SAME' if top_prior == top_fisher else 'differ'}")
        result["design_arm"] = {"spearman_rho": rho, "top_with_prior": top_prior,
                                "top_no_prior": top_fisher, "n_candidates": int(len(pool))}

    p = out / f"{args.product}_fisher_ablation.json"
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfisher ablation -> {p}")


if __name__ == "__main__":
    main()
