"""Step 3B: rebuild the experiment-level candidates in PRIOR-WHITENED parameter coordinates.

step3_voi_nullity.py builds each candidate's information square-root B_c as an orthonormal basis in RAW
u-coordinates scaled by one common amplitude, so "equal Fisher intensity per direction" is equal in the raw
metric. That metric is not parameterisation-invariant: log-rate rows and the nu/sigma rows carry very different
prior scales, so "the leading eigenvector of Sigma" means the leading eigenvector *in raw coordinates*, and a
referee is entitled to ask why that is the fair budget.

The rest of the paper measures practical identifiability in the PRIOR metric (worst_dir is the largest
prior-whitened posterior/prior std ratio), so this script redoes the same computation there. With
x = Sigma_0^{-1/2}(theta - mu_0) and Sigma_0 = diag(prior_std^2):

    Sigma_x = D Sigma D,    G_x = G D^{-1},    D = diag(1/prior_std),

candidate bases are orthonormal in x, and the D-optimal probe becomes the leading eigenvector of Sigma_x, i.e.
the worst-identified direction in the same sense as worst_dir. The decision-space whitening W = T^{-2} is
unchanged: only the parameter-space metric moves.

Torch-free: the prior is Gaussian in u with std = (hi - lo)/4 per row (log10 for the rate rows), mirrored here
from cex_model.bayes.prior so this stays a post-processing step.

Run:  python scripts/step3b_voi_prior_whitened.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

RES = pathlib.Path(__file__).resolve().parents[1] / "results/bayes"
PRODUCTS = ['HLXSYN']
N_PROTEIN = 5
_BOUNDS = {"keq": (1e-7, 1e-1), "kkin": (1e-11, 1e-3), "nu": (1.0, 18.0), "sigma": (1.0, 100.0)}
_ROWS, _LOG = ("keq", "kkin", "nu", "sigma"), {"keq", "kkin"}
SCALE = 50.0 / 0.02          # the same Fisher amplitude as step3; saturated, so the value is amplitude-free


def prior_std(n: int) -> np.ndarray:
    out = []
    for r in _ROWS:
        lo, hi = _BOUNDS[r]
        if r in _LOG:
            lo, hi = np.log10(lo), np.log10(hi)
        out += [(hi - lo) / 4.0] * n
    return np.asarray(out, float)


def _sigma_bases(dim: int, n: int):
    """Common mode (1/sqrt(n)) and its orthogonal complement inside the sigma block, embedded in R^dim."""
    e = np.ones(n) / np.sqrt(n)
    Q, _ = np.linalg.qr(np.column_stack([e, np.eye(n)]))
    common, diff = np.zeros((dim, 1)), np.zeros((dim, n - 1))
    common[3 * n:4 * n, :] = Q[:, :1]
    diff[3 * n:4 * n, :] = Q[:, 1:n]
    return common, diff


def voi_trace(Sigma, G, w, B):
    """tr{W (C - C_c)} with C_c = G (Sigma^-1 + B B^T)^-1 G^T; same estimand as step3."""
    H = np.linalg.inv(Sigma)
    A0 = np.linalg.inv(H)
    A1 = np.linalg.inv(H + B @ B.T)
    return float(sum(w[q] * (G[q] @ A0 @ G[q] - G[q] @ A1 @ G[q]) for q in range(G.shape[0])))


def voi_per_qoi(Sigma, G, w, B):
    """The per-QoI terms of ``voi_trace``, so the joint score can be read coordinate by coordinate.

    Main text Table 1's ``<=1.1% of the ceiling`` is a JOINT figure over (purity, yield) under W = T^-2; it
    is not a per-coordinate bound. Splitting it is what lets the supplement say so.
    """
    H = np.linalg.inv(Sigma)
    A0, A1 = np.linalg.inv(H), np.linalg.inv(H + B @ B.T)
    return [float(w[q] * (G[q] @ A0 @ G[q] - G[q] @ A1 @ G[q])) for q in range(G.shape[0])]


def analyse(product: str) -> dict:
    z = np.load(RES / f"{product}_posterior.npz", allow_pickle=False)
    Sigma = np.asarray(z["cov"], float)
    dj = json.loads((RES / f"{product}_decision.json").read_text())
    G = np.atleast_2d(np.asarray(dj["decision_jacobian"], float))
    tol = np.asarray(dj["tol"], float)
    dim = Sigma.shape[0]
    n = dim // 4
    s0 = prior_std(n)

    # prior whitening: x = D (theta - mu_0), D = diag(1/prior_std)
    D, Dinv = np.diag(1.0 / s0), np.diag(s0)
    Sig_x = D @ Sigma @ D
    G_x = G @ Dinv
    w = 1.0 / tol ** 2

    GS = G_x @ Sig_x
    _, s, Vt = np.linalg.svd(GS, full_matrices=True)
    r = int(np.sum(s > 1e-10 * s.max()))
    dec_dirs, null_dirs = Vt[:r].T, Vt[r:].T
    evals, evecs = np.linalg.eigh(Sig_x)
    top = evecs[:, -1:]
    common, diff = _sigma_bases(dim, n)

    cands = {"null": null_dirs[:, : dim - r], "dec": dec_dirs, "sigma_diff": diff,
             "sigma_cmn": common, "d_optimal": top}
    out = {"product": product, "dim": dim, "dec_rank": r, "metric": "prior-whitened",
           "theta_only_ceiling_trWC": float(sum(w[q] * (G_x @ Sig_x @ G_x.T)[q, q] for q in range(G.shape[0]))),
           "cand": {}}
    for k, B in cands.items():
        out["cand"][k] = {"k_columns": int(B.shape[1]),
                          "voi_trace": voi_trace(Sig_x, G_x, w, SCALE * B),
                          "voi_per_qoi": voi_per_qoi(Sig_x, G_x, w, SCALE * B)}
    Cx = G_x @ Sig_x @ G_x.T
    out["theta_only_ceiling_per_qoi"] = [float(w[q] * Cx[q, q]) for q in range(G.shape[0])]
    return out


def main() -> None:
    rows = [analyse(p) for p in PRODUCTS]
    raw = {r["product"]: r for r in json.loads((RES / "voi_nullity.json").read_text())}
    hdr = f"{'candidate':14} {'k':>2} | " + " | ".join(f"{p:>21}" for p in PRODUCTS)
    print("VoI tr{W(C-C_c)} in PRIOR-WHITENED coordinates (raw-metric value in brackets)")
    print(hdr)
    print("-" * len(hdr))
    for c in ["null", "sigma_cmn", "d_optimal", "sigma_diff", "dec"]:
        cells = []
        for r in rows:
            pw, rw = r["cand"][c]["voi_trace"], raw[r["product"]]["cand"][c]["voi_trace"]
            cells.append(f"{pw:9.3e} [{rw:8.2e}]")
        print(f"{c:14} {rows[0]['cand'][c]['k_columns']:2d} | " + " | ".join(cells))
    cells = [f"{r['theta_only_ceiling_trWC']:9.3e} [{raw[r['product']]['theta_only_ceiling_trWC']:8.2e}]"
             for r in rows]
    print(f"{'ceiling':14} {'--':>2} | " + " | ".join(cells))
    print("\nshare of the ceiling (%), prior-whitened:")
    for c in ["sigma_cmn", "d_optimal", "sigma_diff", "dec"]:
        sh = [100 * r["cand"][c]["voi_trace"] / r["theta_only_ceiling_trWC"] for r in rows]
        print(f"  {c:14} " + " ".join(f"{x:8.3f}%" for x in sh))
    (RES / "voi_nullity_prior_whitened.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {RES / 'voi_nullity_prior_whitened.json'}")


if __name__ == "__main__":
    main()
