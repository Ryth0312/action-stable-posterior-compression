"""Union-bound the eight candidate-level linearisation certificates to a joint 0.95.

Each artifact of ``step4_r2_paired_coupling.py`` is certified at delta=0.05 spread over its own
t-grid and support-gap bridge, so eight of them read together deliver only 1-8*delta. This
recomputes every Clopper-Pearson limit at delta/8 from the stored counts -- arithmetic only, no
solver -- so the eight conditions hold simultaneously at 0.95, and reports what that costs.

Run:
  python scripts/bayes_candidate_certificate_joint_level.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

N_BIG = 4_000_000                 # tube draws in step4_r2_paired_coupling._term1_tube
DELTA, M_GRID = 0.05, 50
TAU_HI, TAU_LO = 0.95, 0.05


def cp_upper(k, n, alpha):
    return 1.0 if k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))


def _bridge(k_meet, p_lin, alpha):
    return max(abs(cp_upper(k_meet, N_BIG, alpha) - p_lin),
               abs((1.0 - cp_upper(N_BIG - k_meet, N_BIG, alpha)) - p_lin))


def _solve_k_meet(p_lin, gap_up, alpha):
    """Recover the clipped-law meet count the stored bridge was computed from."""
    lo = max(0, int(p_lin * N_BIG) - 20_000)
    best, resid = lo, np.inf
    for k in range(lo, min(N_BIG, lo + 40_001)):
        d = abs(_bridge(k, p_lin, alpha) - gap_up)
        if d < resid:
            resid, best = d, k
        if d < 1e-12:
            break
    return best, resid


def rebound(d, alpha):
    n = d["n_draws"]
    u1 = np.array([cp_upper(int(round(h * N_BIG)), N_BIG, alpha) for h in d["term1_hat_curve"]])
    u2 = np.array([cp_upper(int(round(h * n)), n, alpha) for h in d["term2_hat_curve"]])
    k_meet, resid = _solve_k_meet(d["P_lin_untruncated"], d["support_gap_upper"],
                                  DELTA / (2 * M_GRID + 2))
    total = u1 + u2 + _bridge(k_meet, d["P_lin_untruncated"], alpha)
    j = int(np.argmin(total))
    b, p = float(total[j]), d["P_lin_untruncated"]
    lo, hi = max(0.0, p - b), min(1.0, p + b)
    safe = (lo >= TAU_HI) if p >= TAU_HI else ((hi <= TAU_LO) if p <= TAU_LO else False)
    return b, float(d["t_grid"][j]), bool(safe), float(resid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/bayes/r2_paired_coupling_*correlated*_op*.json")
    ap.add_argument("--out", default="results/bayes/candidate_certificate_joint_level.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    alpha = DELTA / len(files) / (2 * M_GRID + 2)
    rows = []
    for f in files:
        d = json.load(open(f))[0]
        b, t, safe, resid = rebound(d, alpha)
        rows.append({"product": d["product"], "loading": d["op"][0], "posterior": d["posterior"],
                     "bound_per_condition": d["certified_bound"], "bound_joint": b,
                     "t_star_joint": t, "safe_per_condition": d["classification_safe"],
                     "safe_joint": safe, "state": d["classification_state"],
                     "k_meet_residual": resid, "source": Path(f).name})
        print(f"{d['product']:14s} {d['op'][0]:8.4f}  b={d['certified_bound']:.6f} -> {b:.6f}  "
              f"safe {d['classification_safe']} -> {safe}")

    out = {"n_certificates": len(files), "delta": DELTA, "m_grid": M_GRID,
           "alpha_per_test_joint": alpha, "joint_level": 1.0 - DELTA,
           "max_bound_inflation": max(r["bound_joint"] - r["bound_per_condition"] for r in rows),
           "verdict_changes": sum(r["safe_joint"] != r["safe_per_condition"] for r in rows),
           "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nalpha/test {alpha:.3e}; max inflation {out['max_bound_inflation']:.4f}; "
          f"verdict changes {out['verdict_changes']}\nwrote {args.out}")


if __name__ == "__main__":
    main()
