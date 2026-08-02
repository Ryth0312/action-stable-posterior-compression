"""Decision-space gain witness for the open a-priori `kappa_S` gap (kappa closure brief, Deliverable A).

`kappa_S` (the full-state variational-flow Grönwall constant, `bayes/groenwall.py` /
`bayes/contraction.py`) is vacuous for SMA: Euclidean/log-norm and per-species-diagonal contraction
metrics fail, and the general off-diagonal/time-varying metric is still open (a magnitude-
certification gap, not a scaling gap; `docs/decision_null_theorem.md` §7). This module does **not**
attempt a full-state `kappa_S` certificate. Instead it packages the DECISION-SPACE gain that is
already exact and constant-free -- the tolerance-and-prior-whitened decision map

    B = T^-1 G diag(sigma_prior),   G = dg/du,   g = (pool_purity, pool_yield)

and the ratio law `r = ||B_sigma|| / ||B_nu|| = O(eps)` (`bayes/loading_sweep.py::decision_null_ratio`,
`docs/decision_null_theorem.md` Corollary 1R) -- into the brief's requested `DecisionGainConfig` /
`compute_decision_gain` / `mesh_stability` API, with genuine mesh-stability (`G` and `eps` recomputed
by a real solve at each `n_steps`, not read once from a committed snapshot).

Every numeric quantity here is reused, not reimplemented: `G` from `bayes.decision.decision_jacobian`,
`eps` from `bayes.loading_sweep.epsilon_at_op`, the whitened row norms + ratio law from
`bayes.loading_sweep.row_sensitivities` / `decision_null_ratio`, and the submultiplicative
`||B||_2` / widest-direction decision energy from `bayes.spectral_transfer.decompose_decision`.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cex_model.bayes.decision import decision_jacobian
from cex_model.bayes.loading_sweep import (
    _load_product_setup,
    decision_null_ratio,
    epsilon_at_op,
    row_sensitivities,
)
from cex_model.bayes.spectral_transfer import decompose_decision
from cex_model.bayes.active import _sim_for_op

__all__ = [
    "DecisionGainConfig",
    "load_decision_blocks",
    "compute_decision_gain",
    "mesh_stability",
    "run_decision_gain_report",
]

_DEFAULT_N_STEPS_SINGLE = 300   # matches the committed *_spectral.json "g_source" convention


@dataclass
class DecisionGainConfig:
    products: list[str]
    n_steps: list[int]
    qois: tuple[str, ...] = ("purity", "yield")
    spectral_dir: str = "results/bayes"
    loading_fraction_path: str = "results/bayes/loading_fraction.json"
    out_path: str = "results/bayes/decision_gain_certificate.json"
    # this repo's DT-excluded HLXSYN cohort is committed as "HLXSYN", not "HLXSYN"
    # (see results/bayes/kappa_closure_discovery.json "notes"); "HLXSYN" alone has no spectral.json.
    strict_main_products: tuple[str, ...] = field(default_factory=lambda: ("HLXSYN", "HLXSYN", "HLXSYN"))


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=Path(__file__).resolve().parent).stdout.strip()
    except Exception:
        return "unknown"


def load_decision_blocks(product: str, n_steps: int | None = None, *, in_dir: str = "results/bayes") -> dict:
    """Load posterior/prior/bundle for ``product`` and recompute ``G``, ``eps`` at ``n_steps`` via a
    real solve (default: 300, matching the committed ``*_spectral.json`` convention). Returns the raw
    blocks (``G``, ``cov``, ``prior_std``, ``tol``, ``names``, ``epsilon``, ``f_sat``, ``op``)."""
    post, base_op, tol, n, prior, bundle = _load_product_setup(product, in_dir)
    ns = int(n_steps) if n_steps is not None else _DEFAULT_N_STEPS_SINGLE
    sim = _sim_for_op(bundle, base_op, ns)
    eps, f_sat = epsilon_at_op(sim, post.u_map, n)
    G, g_map, _ = decision_jacobian(bundle, base_op, post.u_map, n_steps=ns, return_extra=True)
    return {
        "product": product, "n_steps": ns, "op": [float(x) for x in base_op], "tol": tol, "n": n,
        "cov": post.cov, "prior_std": prior.std, "names": post.names, "G": G,
        "epsilon": eps, "f_sat": f_sat, "g_map": [float(x) for x in g_map],
    }


def compute_decision_gain(product: str, n_steps: int | None = None, *, in_dir: str = "results/bayes") -> dict:
    """Norms, ratio-law quantities, sigma-share prediction, and metadata (Deliverable A §3.1) for one
    product at one mesh setting."""
    blk = load_decision_blocks(product, n_steps, in_dir=in_dir)
    rs = row_sensitivities(blk["G"], blk["prior_std"], blk["tol"], blk["n"])
    r = decision_null_ratio(rs["row_sens"], blk["epsilon"])
    dd = decompose_decision(blk["cov"], blk["G"], blk["prior_std"], blk["tol"], names=blk["names"])
    eps = blk["epsilon"]
    return {
        "product": product, "n_steps": blk["n_steps"], "epsilon": eps,
        "B_sigma_norm": r["B_sigma"], "B_nu_norm": r["B_nu"],
        "B_keq_norm": rs["row_sens"]["keq"], "B_kkin_norm": rs["row_sens"]["kkin"],
        "r_sigma_over_nu": r["ratio_sigma_over_nu"], "r_over_epsilon": r["ratio_over_eps"],
        "kappa_g_eff": (r["B_sigma"] / eps) if eps > 0 else float("nan"),
        "sigma_share_exact": rs["sigma_share"], "sigma_share_pred_from_r2": r["sigma_share_predicted"],
        "worst_dec": dd["worst_dec"], "worst_dir": dd["worst_dir"],
        "kappa_g_submult": dd["kappa_g_submult"], "residual_widest": dd["residual_widest"],
        "g_map": blk["g_map"],
    }


def _rel_spread(vals) -> float:
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size < 2 or np.mean(np.abs(v)) == 0.0:
        return float("nan")
    return float((v.max() - v.min()) / np.mean(np.abs(v)))


def mesh_stability(product: str, n_steps: list[int], *, in_dir: str = "results/bayes") -> dict:
    """Relative spread of ``B_sigma``, ``B_nu``, ``r/eps``, and sigma-share over ``n_steps`` -- a
    real discretization sweep (each point re-solves ``G``/``eps``), not a single committed snapshot."""
    rows = [compute_decision_gain(product, ns, in_dir=in_dir) for ns in n_steps]
    return {
        "n_steps": list(n_steps),
        "B_sigma_rel_spread": _rel_spread(r["B_sigma_norm"] for r in rows),
        "B_nu_rel_spread": _rel_spread(r["B_nu_norm"] for r in rows),
        "r_over_epsilon_rel_spread": _rel_spread(r["r_over_epsilon"] for r in rows),
        "sigma_share_rel_spread": _rel_spread(r["sigma_share_exact"] for r in rows),
        "rows": rows,
    }


def _product_report(product: str, config: DecisionGainConfig) -> dict:
    headline_ns = max(config.n_steps) if config.n_steps else _DEFAULT_N_STEPS_SINGLE
    headline = compute_decision_gain(product, headline_ns, in_dir=config.spectral_dir)
    mesh = mesh_stability(product, config.n_steps, in_dir=config.spectral_dir) if len(config.n_steps) > 1 else None

    warnings = []
    is_main = product in config.strict_main_products
    if is_main and not (0.1 < headline["r_over_epsilon"] < 10.0):
        warnings.append("WARN_R_OVER_EPS_NOT_O1")
    if mesh is not None:
        for key, label in (("B_sigma_rel_spread", "Bsigma"), ("r_over_epsilon_rel_spread", "r/eps")):
            spread = mesh[key]
            if is_main and np.isfinite(spread) and spread >= 0.15:
                warnings.append(f"WARN_MESH_UNSTABLE_{label}")
    share_ratio = (headline["sigma_share_pred_from_r2"] / headline["sigma_share_exact"]
                  if headline["sigma_share_exact"] > 0 else float("inf"))
    factor2_ok = np.isfinite(share_ratio) and (0.5 <= share_ratio <= 2.0)
    if not factor2_ok:
        warnings.append("WARN_R2_SHARE_DEVIATION_GT_FACTOR2")

    status = "PASS" if not warnings else "PASS_WITH_WARNINGS"
    out = dict(headline)
    out["sigma_share_pred_over_exact_ratio"] = share_ratio
    out["mesh"] = mesh
    out["warnings"] = warnings
    out["status"] = status
    return out


def run_decision_gain_report(config: DecisionGainConfig) -> dict:
    """Run all ``config.products`` and write the JSON report (Deliverable A §3.4)."""
    products_out = {}
    for p in config.products:
        try:
            products_out[p] = _product_report(p, config)
        except FileNotFoundError as exc:
            products_out[p] = {"product": p, "status": "SKIPPED_MISSING_DATA", "error": str(exc)}

    main = [products_out[p] for p in config.strict_main_products
           if p in products_out and products_out[p].get("status") != "SKIPPED_MISSING_DATA"]
    r_over_eps_main = [m["r_over_epsilon"] for m in main if np.isfinite(m["r_over_epsilon"])]
    sigma_shares = [v["sigma_share_exact"] for v in products_out.values()
                    if isinstance(v.get("sigma_share_exact"), float) and np.isfinite(v["sigma_share_exact"])]
    any_warn = any(v.get("status") == "PASS_WITH_WARNINGS" for v in products_out.values())
    any_skip = any(v.get("status") == "SKIPPED_MISSING_DATA" for v in products_out.values())
    overall_status = "SKIPPED_SOME_PRODUCTS" if any_skip else ("PASS_WITH_WARNINGS" if any_warn else "PASS")

    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _git_commit(),
            "status": overall_status,
            "claim_level": "magnitude_witness_not_full_state_kappa_S",
        },
        "global_summary": {
            "main_products_r_over_eps_mean": float(np.mean(r_over_eps_main)) if r_over_eps_main else float("nan"),
            "main_products_r_over_eps_cv": (float(np.std(r_over_eps_main) / np.mean(r_over_eps_main))
                                            if r_over_eps_main and np.mean(r_over_eps_main) != 0 else float("nan")),
            "max_sigma_share_main_products": (max(m["sigma_share_exact"] for m in main) if main else float("nan")),
            "max_sigma_share_all_products": max(sigma_shares) if sigma_shares else float("nan"),
            "full_state_kappa_S_claimed": False,
        },
        "products": products_out,
    }

    out_path = Path(config.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    csv_path = out_path.with_suffix(".csv")
    cols = ["product", "n_steps", "epsilon", "B_sigma_norm", "B_nu_norm", "r_sigma_over_nu",
           "r_over_epsilon", "kappa_g_eff", "sigma_share_exact", "sigma_share_pred_from_r2", "status"]
    lines = [",".join(cols)]
    for p, v in products_out.items():
        if v.get("status") == "SKIPPED_MISSING_DATA":
            lines.append(f"{p}," + ",".join([""] * (len(cols) - 2)) + ",SKIPPED_MISSING_DATA")
            continue
        lines.append(",".join(str(v.get(c, "")) for c in cols))
    csv_path.write_text("\n".join(lines) + "\n")

    _plot_ratio(products_out, config, out_path.with_name(out_path.stem + "_ratio.png"))
    return report


def _plot_ratio(products_out: dict, config: DecisionGainConfig, png_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    names, eps, r_over_eps, colors = [], [], [], []
    for p, v in products_out.items():
        if v.get("status") == "SKIPPED_MISSING_DATA" or not np.isfinite(v.get("r_over_epsilon", float("nan"))):
            continue
        names.append(p); eps.append(v["epsilon"]); r_over_eps.append(v["r_over_epsilon"])
        colors.append("tab:blue" if p in config.strict_main_products else "tab:orange")
    if not names:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(eps, r_over_eps, c=colors, s=60, zorder=3)
    for x, y, nm in zip(eps, r_over_eps, names):
        ax.annotate(nm, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(np.nanmean([v for v, c in zip(r_over_eps, colors) if c == "tab:blue"]) or 0.0,
              color="tab:blue", ls="--", lw=1, alpha=0.6, label="main-product mean")
    ax.set_xlabel(r"$\varepsilon = \max \gamma Q/\bar\Lambda$")
    ax.set_ylabel(r"$r/\varepsilon = \|B_\sigma\|/\|B_\nu\| / \varepsilon$")
    ax.set_title("Decision-null ratio law (Corollary 1R)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
