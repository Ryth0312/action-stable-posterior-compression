"""σ-block identifiability direction audit (paper Appendix B).

Pure-numpy read of the committed posteriors — no solver, no new fit. Answers "which σ *directions* are
sloppy?" and shows σ is **not uniformly null** at the real products: the common-capacity mode
``v_cm ∝ (1,…,1)`` (a uniform steric shift that moves all peaks) is comparatively well-determined, while the
unidentified content is the differential / minor-component σ directions. ``worst_dir_σ`` is therefore a
σ-dominated *mix*, not the σ block as a whole. This is the audit behind the honest scoping in §3.2/§4 and
the reason a directional sloppy-Fisher certificate along the common mode would (wrongly) certify the *stiff*
direction.

Quantities (prior-whitened, uniform σ prior ``σ_prior``; whitened std = posterior/prior std along a direction):
- per-component **marginal** shrinkage ``sqrt(Σ_σσ[j,j])/σ_prior``;
- common-mode **marginal** ``sqrt(v_cmᵀ Σ_σσ v_cm)/σ_prior`` and **conditional** ``1/√(1+σ_prior² v_cmᵀ F_σσ v_cm)``;
- minor-most-component **conditional** ``1/√(1+σ_prior² e_jᵀ F_σσ e_j)`` (j = argmin feed fraction);
- ``worst_dir_σ = sqrt(λmax(Σ_σσ))/σ_prior`` (the sloppiest whitened direction).

``F_σσ`` is the committed σ-block Fisher ``(inv(cov) − prior.precision())[3n:4n, 3n:4n]``.

    OMP_NUM_THREADS=4 python scripts/bayes_sigma_direction_audit.py --products HLXSYN HLXSYN HLXSYN HLXSYN HLXSYN
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cex_model.bayes.posterior import Posterior
from cex_model.bayes.prior import physical_prior


def audit_product(product: str, in_dir: str) -> dict:
    post = Posterior.load(f"{in_dir}/{product}_posterior.npz")
    n = post.n_protein
    prior = physical_prior(n)
    sp = float(prior.std[3 * n])                                   # uniform σ prior std
    cov = np.asarray(post.cov, float)
    Sig = cov[3 * n:4 * n, 3 * n:4 * n]                            # posterior σ-block covariance
    F = np.linalg.inv(cov) - prior.precision()
    Fss = 0.5 * (F[3 * n:4 * n, 3 * n:4 * n] + F[3 * n:4 * n, 3 * n:4 * n].T)   # committed σ-block Fisher

    def cond(v):                                                   # conditional whitened shrinkage along v
        v = np.asarray(v, float); v = v / np.linalg.norm(v)
        return 1.0 / np.sqrt(1.0 + sp ** 2 * float(v @ Fss @ v))

    def marg(v):                                                   # marginal whitened std along v
        v = np.asarray(v, float); v = v / np.linalg.norm(v)
        return np.sqrt(max(float(v @ Sig @ v), 0.0)) / sp

    frac = np.asarray(post.fractions, float)
    jmin = int(np.argmin(frac))
    ones = np.ones(n)
    marg_per_comp = [float(np.sqrt(max(Sig[j, j], 0.0)) / sp) for j in range(n)]
    worst_dir = float(np.sqrt(max(float(np.linalg.eigvalsh(Sig)[-1]), 0.0)) / sp)
    return {
        "n_protein": n, "sigma_prior_std": sp,
        "fractions_pct": [float(x) for x in frac],
        "sigma_marginal_shrinkage": marg_per_comp,
        "common_mode_marginal": float(marg(ones)),
        "common_mode_conditional": float(cond(ones)),
        "min_component_index": jmin,
        "min_component_conditional": float(cond(np.eye(n)[jmin])),
        "min_component_fraction_pct": float(frac[jmin]),
        "worst_dir_sigma": worst_dir,
        "fisher_eigs": [float(x) for x in np.linalg.eigvalsh(Fss)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--out-dir", default="results/bayes")
    ap.add_argument("--products", nargs="*",
                    default=['HLXSYN'])
    args = ap.parse_args()

    out = {}
    for p in args.products:
        try:
            out[p] = audit_product(p, args.in_dir)
        except FileNotFoundError:
            print(f"  [skip] {p}: no committed posterior")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "sigma_direction_audit.json").write_text(json.dumps(out, indent=2))

    print(f"{'product':14s} {'σ marginals (per-comp)':40s} {'common(marg/cond)':18s} "
          f"{'min-comp cond(frac%)':22s} {'worst_dir_σ':11s}")
    for p, r in out.items():
        mg = "[" + " ".join(f"{x:.2f}" for x in r["sigma_marginal_shrinkage"]) + "]"
        cm = f"{r['common_mode_marginal']:.2f}/{r['common_mode_conditional']:.3f}"
        mc = f"{r['min_component_conditional']:.2f} ({r['min_component_fraction_pct']:.1f})"
        print(f"{p:14s} {mg:40s} {cm:18s} {mc:22s} {r['worst_dir_sigma']:.3f}")
    print(f"\n  json -> {Path(args.out_dir) / 'sigma_direction_audit.json'}")


if __name__ == "__main__":
    main()
