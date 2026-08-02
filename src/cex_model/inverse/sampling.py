"""Wide parameter sampling for the inverse-network training set.

The inverse network learns ``curve -> par``, so the training set must vary the
SMA parameters (the forward surrogate datasets instead hold par fixed and vary
operating conditions).  Each plan samples a parameter set *and* an operating
condition; running it through the ODE yields a ``(curve, par)`` pair.

Sampling is **multiplicative on the product's calibrated par** so it generalizes
across products without hand-coding absolute physical ranges:

* ``keq``/``kkin`` are sampled log-uniformly (they span decades);
* ``nu``/``sigma`` linearly;
* a fraction ``anchor_frac`` of samples use a range shrunk toward 1.0
  (``factor**anchor_shrink``) to concentrate density near the real par, the rest
  use the full range for coverage / off-anchor robustness;
* feed fractions are held at the calibrated values (the mechanistic optimizer
  also fixes fractions).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from cex_model.components import ComponentSet

PARAM_ORDER: tuple[str, ...] = ("nu", "keq", "kkin", "sigma")
_LOG = {"keq", "kkin"}


@dataclass(frozen=True)
class ParamRange:
    factor_lo: float
    factor_hi: float
    transform: str  # "linear" or "log"


@dataclass
class SamplingSpec:
    """Per-parameter multiplicative factor ranges and anchored/global mix."""

    params: dict[str, ParamRange]
    anchor_frac: float = 0.6
    anchor_shrink: float = 0.5

    def shrunk(self, name: str) -> ParamRange:
        """Factor range narrowed toward 1.0 for the anchored block."""
        pr = self.params[name]
        s = self.anchor_shrink
        return ParamRange(pr.factor_lo**s, pr.factor_hi**s, pr.transform)


def default_sampling_spec() -> SamplingSpec:
    # Factor ranges kept wide enough to make the inverse problem non-trivial
    # (~25x span on keq/kkin, +/-25% on nu) but bounded away from the extreme
    # high-nu / extreme-keq corners where the stiff EDM+SMA ODE integrates very
    # slowly. Widen via --sampling-config if more coverage is needed.
    return SamplingSpec(
        params={
            "nu": ParamRange(0.8, 1.25, "linear"),
            "keq": ParamRange(0.2, 5.0, "log"),
            "kkin": ParamRange(0.2, 5.0, "log"),
            "sigma": ParamRange(0.5, 2.0, "linear"),
        },
        anchor_frac=0.6,
        anchor_shrink=0.5,
    )


def _factor(u: np.ndarray, pr: ParamRange) -> np.ndarray:
    """Map unit-cube coordinates to multiplicative factors."""
    if pr.transform == "log":
        lo, hi = np.log10(pr.factor_lo), np.log10(pr.factor_hi)
        return np.power(10.0, lo + u * (hi - lo))
    return pr.factor_lo + u * (pr.factor_hi - pr.factor_lo)


def _scale_operating(u: np.ndarray, ranges: list[tuple[float, float]]) -> np.ndarray:
    lows = np.array([r[0] for r in ranges])
    highs = np.array([r[1] for r in ranges])
    return lows + u * (highs - lows)


def _components_from_unit(
    base: ComponentSet, unit_row: np.ndarray, spec: SamplingSpec, anchored: bool
) -> ComponentSet:
    """Build a sampled ComponentSet from a (4*n_protein,) unit-cube row."""
    n = base.n_protein
    sampled = list(base.components)
    block = unit_row.reshape(len(PARAM_ORDER), n)
    factors: dict[str, np.ndarray] = {}
    for k, name in enumerate(PARAM_ORDER):
        pr = spec.shrunk(name) if anchored else spec.params[name]
        factors[name] = _factor(block[k], pr)
    out = []
    for j, comp in enumerate(sampled):
        out.append(
            replace(
                comp,
                nu=float(max(comp.nu * factors["nu"][j], 1e-9)),
                keq=float(max(comp.keq * factors["keq"][j], 1e-12)),
                kkin=float(max(comp.kkin * factors["kkin"][j], 1e-12)),
                sigma=float(max(comp.sigma * factors["sigma"][j], 1e-9)),
            )
        )
    return ComponentSet(out)


def plan_inverse_samples(
    *,
    base_components: ComponentSet,
    base_exp: dict,
    buffer_a: float,
    buffer_b: float,
    n_samples: int,
    spec: SamplingSpec,
    operating_ranges: dict[str, tuple[float, float]],
    seed: int = 1,
    include_base_case: bool = True,
) -> list[dict]:
    """Deterministic sample plans (par + operating condition) for the ODE sweep.

    ``operating_ranges`` keys: ``loading_g_L``, ``gradient_start_pct``,
    ``gradient_end_pct``, ``gradient_length_CV``.  Returns plan dicts consumable
    by ``generate_surrogate_dataset._generate_sample``.
    """
    from scipy.stats import qmc

    n = base_components.n_protein
    d = 4 * n + 4
    op_keys = ("loading_g_L", "gradient_start_pct", "gradient_end_pct", "gradient_length_CV")
    op_ranges = [operating_ranges[k] for k in op_keys]

    n_base = 1 if include_base_case else 0
    n_eff = max(n_samples - n_base, 0)
    n_anchor = int(round(spec.anchor_frac * n_eff))
    n_global = n_eff - n_anchor

    def _lhs(count: int, s: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0, d))
        return qmc.LatinHypercube(d=d, seed=s).random(n=count)

    blocks = [(_lhs(n_anchor, seed), True), (_lhs(n_global, seed + 1), False)]

    plans: list[dict] = []
    idx = 0
    if include_base_case:
        plans.append(
            {
                "sample_idx": idx,
                "sample_exp": _make_exp("base", base_exp, base_components),
                "buffer_a": float(buffer_a),
                "buffer_b": float(buffer_b),
                "components": base_components,
            }
        )
        idx += 1

    for unit_block, anchored in blocks:
        for row in unit_block:
            comps = _components_from_unit(base_components, row[: 4 * n], spec, anchored)
            op = _scale_operating(row[4 * n :], op_ranges)
            sample_exp = {
                "id": "inverse_anchor" if anchored else "inverse_global",
                "loading_g_L": float(max(op[0], 1e-6)),
                "gradient_start_pct": float(op[1]),
                "gradient_end_pct": float(op[2]),
                "gradient_length_CV": float(max(op[3], 1e-6)),
                "component_fractions_pct": comps.fraction_array().tolist(),
            }
            plans.append(
                {
                    "sample_idx": idx,
                    "sample_exp": sample_exp,
                    "buffer_a": float(buffer_a),
                    "buffer_b": float(buffer_b),
                    "components": comps,
                }
            )
            idx += 1
    return plans


def _make_exp(exp_id: str, base_exp: dict, components: ComponentSet) -> dict:
    return {
        "id": exp_id,
        "loading_g_L": float(base_exp["loading_g_L"]),
        "gradient_start_pct": float(base_exp["gradient_start_pct"]),
        "gradient_end_pct": float(base_exp["gradient_end_pct"]),
        "gradient_length_CV": float(base_exp["gradient_length_CV"]),
        "component_fractions_pct": components.fraction_array().tolist(),
    }
