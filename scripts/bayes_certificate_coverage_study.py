"""Action-level familywise coverage of the three compression certificates, on a synthetic pool.

WHAT THIS ANSWERS.  The deployed-law run certifies six product-by-reduction cells and an exact mixture-risk
calculation confirms all six, but one application cannot separate "the certificate is valid" from "the
application was easy".  This study puts the three certificates on the same synthetic decision problems --
24 candidates, two reductions, a predictive mixture layer -- across many replicates, in a regime where the
compressed minimiser genuinely moves and in one where it does not, and measures both halves of the question:

    POWER                 how often the certificate certifies;
    FALSE CERTIFICATION   how often it certifies while the compressed minimiser is in fact NOT the full one.

The truth is available because the risks are closed form under the empirical mixture, so every certificate is
scored against it rather than against another certificate.

FOUR CERTIFICATES.  Three are the article's and are deterministic given the covariances -- they cannot make a
sampling error, so only their power is at issue:

    bures      main text Corollary 3.5 carried to the mixture by the shared mixing index;
    eps        the distribution-free envelope of Corollary 3.3 at the closed-form coupling bound;
    pairwise   the signed pairwise regret certificate, a simultaneous Catoni limit at level delta.

The fourth removes the one oracle the pairwise route uses.  ``pairwise_split`` does not receive the full-law
minimiser: it estimates it on an independent pilot sample and then certifies, on the certification sample,
that its own pick is the compressed minimiser through simultaneous limits on

    E[ ell(a, Z_{a,j}) - ell(a_0, Z_{a_0,j}) ],

whose deterministic second-moment proxy is again a Minkowski bound, this time on the closed-form second
moments of the hinge rather than on an increment.  Its false-certification rate is the quantity that decides
whether the pairwise route survives without a closed-form full risk, which is the case a referee will ask
about; nothing in the application depends on it.

    python scripts/bayes_certificate_coverage_study.py --n-rep 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bayes_pairwise_regret_shift import catoni_score, draw_arms
from bayes_route_b_predictive_action import eps_analytic, mixture_bures, predictive_risk

from cex_model.bayes.decision_compression import (
    decision_cov,
    rotate_sigma_block,
    schur_cov,
    variant_d_cov,
)

SPEC = np.array([0.70, 0.50])
TOL = np.array([0.02, 0.05])
W_LOSS = np.array([1.0, 1.0])
C_MEAS = np.diag(np.array([0.005, 0.008]) ** 2)
DELTA = 0.05
N_POOL, N_PROTEIN, K_HIER = 24, 3, 8
SIGMA_V_SCALE = 0.06                                       # damps the compressible block to the deployed size
REGIMES = {"near-tie": 0.004, "separated": 0.030}          # the curvature of the pool's mean sweep


def hinge_l2(g, C, bias, sd_draws):
    """sum_q w_q sqrt( E[(sbar_q - Z_q)_+^2] ) under the mixture -- closed form, no sampling.

    For Z ~ N(mu, s^2) and t = (sbar - mu)/s, E[(sbar - Z)_+^2] = s^2{(1 + t^2)Phi(t) + t phi(t)}, and the
    mixture second moment is the mean of that over the committed draws.  Minkowski then bounds the second
    moment of the loss itself, which is what the split certificate needs before it sees its sample.
    """
    mu = (g + bias) / TOL
    var = np.diagonal(sd_draws + C + C_MEAS, axis1=1, axis2=2) / TOL ** 2
    sd = np.sqrt(np.maximum(var, 1e-300))
    t = (SPEC / TOL - mu) / sd
    m2 = (var * ((1.0 + t ** 2) * norm.cdf(t) + t * norm.pdf(t))).mean(0)
    return float((W_LOSS * np.sqrt(np.maximum(m2, 0.0))).sum())


def replicate(rng, curv):
    """One synthetic decision problem: a smooth 24-candidate pool over a random posterior."""
    dim = 4 * N_PROTEIN
    A = rng.standard_normal((dim, dim))
    Sigma = (A @ A.T) / dim + 0.05 * np.eye(dim)
    sig = slice(3 * N_PROTEIN, 4 * N_PROTEIN)                  # the sigma block the reduction compresses
    Sigma[sig, :] *= SIGMA_V_SCALE
    Sigma[:, sig] *= SIGMA_V_SCALE
    G0, G1 = 0.02 * rng.standard_normal((2, dim)), 0.02 * rng.standard_normal((2, dim))
    t = np.linspace(0.0, 1.0, N_POOL)
    t0 = rng.uniform(0.25, 0.75, 2)
    gmaps = SPEC + 0.5 * TOL + curv * (t[:, None] - t0[None, :]) ** 2 * np.array([1.0, -1.0])
    bias = 0.002 * rng.standard_normal((K_HIER, 2))
    L = 0.004 * rng.standard_normal((K_HIER, 2, 2))
    sd_draws = L @ np.transpose(L, (0, 2, 1)) + 1e-6 * np.eye(2)

    cov = {"full": [], "S": [], "F": []}
    for a in range(N_POOL):
        Sr, Gr, ui, vi = rotate_sigma_block(Sigma, G0 + t[a] * G1, N_PROTEIN)
        cov["full"].append(decision_cov(Gr, Sr))
        cov["S"].append(decision_cov(Gr, schur_cov(Sr, ui, vi)))
        cov["F"].append(variant_d_cov(Sr, Gr, ui, vi))
    return gmaps, bias, sd_draws, cov


def score(rng, curv, n_mc, n_pilot):
    gmaps, bias, sd_draws, cov = replicate(rng, curv)
    R = {k: np.array([predictive_risk(gmaps[a], cov[k][a], bias, sd_draws) for a in range(N_POOL)])
         for k in ("full", "S", "F")}
    a_star = int(np.argmin(R["full"]))
    r_pi = R["full"] - R["full"][a_star]
    m = 2 * (N_POOL - 1)
    alpha = DELTA / m
    out = {"Delta_act": float(np.sort(R["full"])[1] - R["full"][a_star]), "cells": {}}

    idx = {"k": rng.integers(0, K_HIER, n_mc), "z": rng.standard_normal((n_mc, 2)),
           "u": rng.standard_normal((n_mc, 2))}
    pilot = {"k": rng.integers(0, K_HIER, n_pilot), "z": rng.standard_normal((n_pilot, 2)),
             "u": rng.standard_normal((n_pilot, 2))}
    # the pilot estimates the FULL-law risk, which does not depend on the reduction: draw it once
    pil = np.array([draw_arms(gmaps[a], cov["full"][a], cov["S"][a], bias, sd_draws, pilot, None,
                              n_pilot)[0].mean() for a in range(N_POOL)])
    a0 = int(np.argmin(pil))

    for j in ("S", "F"):
        preserved = bool(int(np.argmin(R[j])) == a_star)
        b_mix = np.array([mixture_bures(cov["full"][a], cov[j][a], sd_draws) for a in range(N_POOL)])
        e_an = np.array([eps_analytic(cov["full"][a], cov[j][a]) for a in range(N_POOL)])
        env = lambda b: np.flatnonzero(r_pi <= b)
        cert = {k: bool(env(b).size == 1 and env(b)[0] == a_star) for k, b in (("bures", b_mix), ("eps", e_an))}

        lf0, lc0, l20 = draw_arms(gmaps[a_star], cov["full"][a_star], cov[j][a_star],
                                  bias, sd_draws, idx, None, n_mc)
        shift0 = lc0 - lf0
        ok = True
        for a in range(N_POOL):
            if a == a_star or not ok:
                continue
            lf, lc, l2a = draw_arms(gmaps[a], cov["full"][a], cov[j][a], bias, sd_draws, idx, None, n_mc)
            X = (lc - lf) - shift0
            ok &= catoni_score(X, -r_pi[a], alpha, (l2a + l20) ** 2) > 0
        cert["pairwise"] = bool(ok)

        # the split variant: the reference action came from the pilot sample, never from the truth
        l2 = {a: hinge_l2(gmaps[a], cov[j][a], bias, sd_draws) for a in range(N_POOL)}
        _, lc_ref, _ = draw_arms(gmaps[a0], cov["full"][a0], cov[j][a0], bias, sd_draws, idx, None, n_mc)
        ok_s = True
        for a in range(N_POOL):
            if a == a0 or not ok_s:
                continue
            _, lc_a, _ = draw_arms(gmaps[a], cov["full"][a], cov[j][a], bias, sd_draws, idx, None, n_mc)
            ok_s &= catoni_score(lc_a - lc_ref, 0.0, alpha, (l2[a] + l2[a0]) ** 2) > 0
        cert["pairwise_split"] = bool(ok_s)

        out["cells"][j] = {"preserved": preserved, "argmin_full": a_star,
                           "argmin_compressed": int(np.argmin(R[j])),
                           "a0_pilot": a0, "split_truth": bool(int(np.argmin(R[j])) == a0),
                           "certified": cert}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-rep", type=int, default=1000)
    ap.add_argument("--n-mc", type=int, default=10_000)
    ap.add_argument("--n-pilot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bayes/certificate_coverage_study.json")
    a = ap.parse_args()

    names = ("bures", "eps", "pairwise", "pairwise_split")
    res = {"n_rep": a.n_rep, "n_mc": a.n_mc, "n_pilot": a.n_pilot, "delta": DELTA,
           "n_pool": N_POOL, "k_hier": K_HIER, "regimes": {}}
    for reg, curv in REGIMES.items():
        rng = np.random.default_rng(a.seed)
        tally = {k: {"certified": 0, "false": 0, "truth": 0} for k in names}
        n_cells = n_pres = 0
        gaps = []
        for _ in range(a.n_rep):
            r = score(rng, curv, a.n_mc, a.n_pilot)
            gaps.append(r["Delta_act"])
            for c in r["cells"].values():
                n_cells += 1
                n_pres += c["preserved"]
                for k in names:
                    truth = c["split_truth"] if k == "pairwise_split" else c["preserved"]
                    tally[k]["truth"] += truth
                    tally[k]["certified"] += c["certified"][k]
                    tally[k]["false"] += c["certified"][k] and not truth
        res["regimes"][reg] = {"curv": curv, "n_cells": n_cells,
                               "preservation_rate": n_pres / n_cells,
                               "Delta_act_median": float(np.median(gaps)),
                               "certificates": {k: {"power": tally[k]["certified"] / n_cells,
                                                    "conditional_power": (tally[k]["certified"]
                                                                          - tally[k]["false"])
                                                    / max(tally[k]["truth"], 1),
                                                    "truth_rate": tally[k]["truth"] / n_cells,
                                                    "false_certification": tally[k]["false"] / n_cells,
                                                    "n_false": tally[k]["false"]} for k in names}}
    Path(a.out).write_text(json.dumps(res, indent=1))

    for reg, r in res["regimes"].items():
        print(f"\n{reg}: {r['n_cells']} cells, preservation {r['preservation_rate']:.3f}, "
              f"median Delta_act {r['Delta_act_median']:.2e}")
        print(f"  {'certificate':16}{'power':>9}{'truth':>8}{'cond pow':>10}{'false cert':>12}{'n false':>9}")
        for k, v in r["certificates"].items():
            print(f"  {k:16}{v['power']:9.3f}{v['truth_rate']:8.3f}{v['conditional_power']:10.3f}"
                  f"{v['false_certification']:12.4f}{v['n_false']:9d}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
