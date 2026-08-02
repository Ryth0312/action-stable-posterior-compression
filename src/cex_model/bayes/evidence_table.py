"""Manuscript-ready tiered evidence table (Task 4).

Assembles the committed ``results/bayes/*.json`` into one evidence table with product tiers and, crucially,
two VISUALLY-SEPARATED evidence families (acceptance rule #5):

  * Tier A -- main products (HLXSYN, HLXSYN, HLXSYN);
  * Tier B -- minimal-data stress tests (HLXSYN, HLXSYN);
  * Tier C -- synthetic / class-level tests (SYN2 sigma-channel, CMC second model class, the Level-3
              curvature-diagnostic validation).

For each product tier it emits a **Decision-null evidence** table (worst_dir / worst_dec / sigma-share /
sloppy-share / r-over-eps / recommendation) SEPARATELY from a **Coverage & linearization evidence** table
(delta_q / kappa_q status / rel-Frobenius / N=1000 MC coverage / HLXSYN yield triage verdict) -- so a reader
never conflates "the decision is determined" with "the coverage diagnostic is trustworthy".

Pure formatting over committed JSON (reuses ``bayes.compare._md_table``); no solver call. A forbidden-phrase
guard asserts the emitted table makes no full-state-kappa_S claim and never labels kappa_q a certificate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from cex_model.bayes.compare import _md_table

__all__ = ["build_evidence", "render_markdown", "run_evidence_table", "FORBIDDEN_PHRASES"]

TIER_A = ["HLXSYN", "HLXSYN", "HLXSYN"]
TIER_B = ["HLXSYN", "HLXSYN"]

# Positive claims that must never appear (disclaimers like "we do NOT claim a full-state kappa_S bound"
# are fine -- the guard matches these exact positive phrasings only).
FORBIDDEN_PHRASES = [
    "full_state_kappa_S claim",
    "full-state kappa_S bound is claimed",
    "kappa_q certificate",
    "kappa_q is certified as a coverage certificate",
    "certified coverage certificate",
]


def _load(in_dir: Path, name: str) -> dict | None:
    p = in_dir / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _dig(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return d if d is not None else default


def _f(x, nd=3):
    return "n/a" if x is None else f"{float(x):.{nd}f}"


def _pct(x, nd=2):
    return "n/a" if x is None else f"{100.0 * float(x):.{nd}f}%"


def _cov(x):
    return "n/a" if x is None else f"{float(x):.2f}"


def _cov_n(x, n):
    """Empirical coverage with its per-cell draw count (products can carry different N -- e.g. HLXSYN is
    refreshed at N=1000 while the others stay at the N=100 pilot until a Colab sweep)."""
    if x is None:
        return "n/a"
    return f"{float(x):.2f} (N={int(n)})" if n is not None else f"{float(x):.2f}"


# ------------------------------------------------------------------- per-product row builders

def _decision_null_row(p: str, in_dir: Path, dnr: dict | None, dg: dict | None) -> list:
    sp = _load(in_dir, f"{p}_spectral.json") or {}
    dec = _load(in_dir, f"{p}_decision.json") or {}
    win = _load(in_dir, f"{p}_decision_window.json") or {}
    worst_dir = sp.get("worst_dir", _dig(dg, "products", p, "worst_dir"))
    worst_dec = sp.get("worst_dec", _dig(dec, "decision", "worst_dec"))
    met = _dig(dec, "decision", "met")
    r_over_eps = _dig(dnr, "products", p, "ratio_over_eps", default=_dig(dg, "products", p, "r_over_epsilon"))
    action = _dig(win, "recommendation", "action")
    return [p,
            f"{_f(worst_dir, 2)} (No)" if worst_dir is not None else "n/a",
            f"{_f(worst_dec, 2)} ({'Yes' if met else 'No' if met is not None else '?'})",
            _pct(sp.get("sigma_decision_share")),
            _pct(sp.get("sloppy_share"), 1),
            _f(r_over_eps, 2),
            action or "n/a"]


def _coverage_row(p: str, in_dir: Path, wc: dict | None, mc: dict | None) -> list:
    dec = _load(in_dir, f"{p}_decision.json") or {}
    dq_pur = _dig(wc, "products", p, "purity", "delta_q")
    dq_yld = _dig(wc, "products", p, "yield", "delta_q")
    kq_pur = _dig(wc, "products", p, "purity", "status")
    kq_yld = _dig(wc, "products", p, "yield", "status")
    relfrob = _dig(dec, "crosscheck", "rel_frobenius")
    cov_pur = _dig(mc, "products", p, "purity", "empirical_coverage")
    cov_yld = _dig(mc, "products", p, "yield", "empirical_coverage")
    n_pur = _dig(mc, "products", p, "purity", "n_draws_used")
    n_yld = _dig(mc, "products", p, "yield", "n_draws_used")
    triage = _load(in_dir, f"{p}_yield_triage.json")
    verdict = triage.get("verdict") if triage else ("--" if p != "HLXSYN" else "n/a")
    return [p, _f(dq_pur, 3), _f(dq_yld, 3), _kq_short(kq_pur), _kq_short(kq_yld),
            _f(relfrob, 3), _cov_n(cov_pur, n_pur), _cov_n(cov_yld, n_yld), verdict]


def _kq_short(s: str | None) -> str:
    return {"PASS_CERTIFIED_CURVATURE": "certified", "WARN_DIAGNOSTIC_ONLY": "diagnostic",
            "FAIL_DIAGNOSTIC_ONLY": "fail-diag"}.get(s, s or "n/a")


def _tier_b_row(p: str, in_dir: Path, mc: dict | None) -> list:
    sp = _load(in_dir, f"{p}_spectral.json") or {}
    dec = _load(in_dir, f"{p}_decision.json") or {}
    relfrob = _dig(dec, "crosscheck", "rel_frobenius")
    cov_pur = _dig(mc, "products", p, "purity", "empirical_coverage")
    cov_yld = _dig(mc, "products", p, "yield", "empirical_coverage")
    n_pur = _dig(mc, "products", p, "purity", "n_draws_used")
    n_yld = _dig(mc, "products", p, "yield", "n_draws_used")
    return [p, _f(sp.get("worst_dir"), 2), _f(sp.get("worst_dec"), 2), _pct(sp.get("sloppy_share"), 1),
            _f(relfrob, 3), _cov_n(cov_pur, n_pur), _cov_n(cov_yld, n_yld)]


# ------------------------------------------------------------------------------ assembly

@dataclass
class EvidenceConfig:
    in_dir: str = "results/bayes"
    out_md: str = "docs/evidence_table.md"
    out_json: str = "results/bayes/evidence_table.json"
    tier_a: list = field(default_factory=lambda: list(TIER_A))
    tier_b: list = field(default_factory=lambda: list(TIER_B))


def build_evidence(config: EvidenceConfig | None = None) -> dict:
    config = config or EvidenceConfig()
    in_dir = Path(config.in_dir)
    wc = _load(in_dir, "whitened_curvature.json")
    mc = _load(in_dir, "coverage_mc_band.json")
    dnr = _load(in_dir, "decision_null_ratio.json")
    dg = _load(in_dir, "decision_gain_certificate.json")
    syn = _load(in_dir, "SYN2_sigma_channel.json") or {}
    cmc = _load(in_dir, "CMC_class_level.json") or {}
    mc_draws = _dig(mc, "metadata", "draws")

    tier_a_dn = [_decision_null_row(p, in_dir, dnr, dg) for p in config.tier_a]
    tier_a_cov = [_coverage_row(p, in_dir, wc, mc) for p in config.tier_a]
    tier_b = [_tier_b_row(p, in_dir, mc) for p in config.tier_b]

    v = _dig(wc, "validation_summary") or {}
    tier_c = [
        ["SYN2 sigma-channel (SMA, isolated)", "‖B_σ‖ ∝ ε (R², intercept through origin)",
         f"R²={_f(_dig(syn, 'sigma_linear_fit', 'r2'), 4)}, thru origin",
         _dig(syn, "scaling", "verdict") or "n/a"],
        ["CMC 2nd model class (frozen)", "chain-rule identity (decision-null R2)",
         f"R2={_f(_dig(cmc, 'decision_null_fit_vs_eps', 'r2'), 4)} (=1: identity)", cmc.get("verdict") or "n/a"],
        ["CMC 2nd model class (physical R0 sweep, moving state)", "decision-null O(eps) / Fisher-null O(eps2)",
         f"R2={_f(_dig(cmc, 'physical_eps_sweep', 'decision_null_fit_vs_eps', 'r2'), 4)} / "
         f"{_f(_dig(cmc, 'physical_eps_sweep', 'fisher_null_fit_vs_eps2', 'r2'), 4)}",
         _dig(cmc, "physical_eps_sweep", "verdict") or "n/a"],
        ["Level-3 kappa_q validation (CMC ground truth)", "whitened FD-HVP vs full-AD Hessian",
         f"level3={v.get('level3_synthetic_sma', 'n/a')}", v.get("level3_cmc_status") or "n/a"],
        ["Level-3 kappa_q validation (SMA self-consistency)", "FD-HVP self-consistency on real IFT-BDF",
         "diagnostic-only on real products", v.get("level3_sma_status") or "n/a"],
    ]

    return {
        "meta": {"mc_draws": mc_draws, "tier_a": config.tier_a, "tier_b": config.tier_b,
                 "sources": "results/bayes/*.json (committed)"},
        "tier_a_decision_null": tier_a_dn, "tier_a_coverage": tier_a_cov,
        "tier_b": tier_b, "tier_c": tier_c,
    }


def render_markdown(ev: dict) -> str:
    out = ["# Evidence table (manuscript-ready)", "",
           "*Generated from committed `results/bayes/*.json`. Decision-null evidence (is the process "
           "decision determined?) is kept separate from coverage/linearization evidence (is the cheap "
           "delta-method interval trustworthy?) -- the two are distinct questions.*", "",
           "## Tier A -- main products", "",
           "### A1. Decision-null evidence", "",
           _md_table(["product", "worst_dir (θ ident.)", "worst_dec (met)", "σ decision-share",
                      "sloppy-block share", "r/ε", "recommendation"], ev["tier_a_decision_null"]),
           "",
           "### A2. Coverage & linearization evidence", "",
           _md_table(["product", "δ_q purity", "δ_q yield", "κ_q purity", "κ_q yield", "rel-Frob (lin↔MC)",
                      "MC cov purity (N/cell)", "MC cov yield (N/cell)", "yield triage"],
                     ev["tier_a_coverage"]),
           "",
           "*κ_q is reported as a **diagnostic** (never a certificate): the Level-3 SMA self-consistency "
           "leg is inconsistent on the real IFT-BDF solver, so no real-product κ_q is certified (Tier C).*",
           "",
           "## Tier B -- minimal-data stress tests (2 experiments)", "",
           _md_table(["product", "worst_dir", "worst_dec", "sloppy-block share", "rel-Frob (lin↔MC)",
                      "MC cov purity (N/cell)", "MC cov yield (N/cell)"], ev["tier_b"]),
           "",
           "*At 2 experiments the delta-method breaks (large rel-Frob) and the MC decision is not met -- "
           "the honest data floor of the decision-vs-parameter decoupling.*", "",
           "## Tier C -- synthetic / class-level tests", "",
           _md_table(["test", "quantity", "result", "verdict"], ev["tier_c"]),
           ""]
    md = "\n".join(out)
    for phrase in FORBIDDEN_PHRASES:
        assert phrase.lower() not in md.lower(), f"forbidden phrase leaked into evidence table: {phrase!r}"
    return md


def run_evidence_table(config: EvidenceConfig | None = None) -> dict:
    config = config or EvidenceConfig()
    ev = build_evidence(config)
    md = render_markdown(ev)
    Path(config.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(config.out_md).write_text(md)
    Path(config.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(config.out_json).write_text(json.dumps(ev, indent=2))
    return ev
