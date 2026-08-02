#!/usr/bin/env python3
"""Build the synthetic twin: a self-contained product the whole pipeline runs on end to end.

The three real products' elution and chromatographic purity traces are proprietary and are not
released, so a reader of the article can re-derive every reported number from the committed
artifacts but cannot regenerate the fits that produced them. The twin closes that gap. It is a
fictitious five-component CEX product with the same *shape* as the real ones -- same column class,
same three-experiment design, same fractionated observation format, same five-component
acid/main/basic split -- generated forward from parameters written down in this file, so the
calibration, decision, compression and certificate stages can be run from raw data to final table
by anyone, and the recovered posterior can be checked against a truth that is actually known.

It is NOT a de-identified copy of any real product: the column geometry, the component parameters
and the operating conditions are chosen here, not fitted to anything.

  python scripts/make_synthetic_twin.py           # write configs, curves and the truth file
  python scripts/make_synthetic_twin.py --check   # re-simulate at truth and report the residual

Writes
  configs/column_hlxsyn.yaml, components_hlxsyn.yaml, experiments_hlxsyn.yaml
  data/synthetic_twin/<experiment id>.csv        [time_s, c1..c5] g/L, elution-relative
  results/bayes/synthetic_twin_truth.json        ground truth + how it was generated

The product registers as HLXSYN and loads through the ordinary path, so every downstream script
takes it with --product HLXSYN and no other change.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cex_model.column import ColumnParameters                      # noqa: E402
from cex_model.components import ComponentSet                      # noqa: E402
from cex_model.corrections import make_loading_correction          # noqa: E402
from cex_model.gradients import build_fitting_inlet                # noqa: E402
from cex_model.simulator import ChromatographySimulator            # noqa: E402

OUT_DATA = ROOT / "data" / "synthetic_twin"
OUT_CFG = ROOT / "configs"
OUT_TRUTH = ROOT / "results" / "bayes" / "synthetic_twin_truth.json"

# --- ground truth ------------------------------------------------------------------------------
# A 1 mL-scale strong cation exchanger, in the parameter ranges the literature reports for this
# resin class. Chosen, not fitted. nu rises and keq falls from acid to basic so the five species
# elute in order; sigma is O(10) so the steric block sits at the small occupation fractions the
# common-mode-capacity argument of the article is stated for.
COLUMN = {
    "column_volume": 20.0, "column_length": 180.0, "flow_rate": 2.50,
    "superficial_velocity": 0.370, "total_porosity": 0.80, "particle_porosity": 0.50,
    "dax": 0.055, "ionic_capacity": 0.400, "particle_radius": 0.045,
    "dead_volume": 0.70, "grid_size": 51,
}
COMPONENTS = [
    {"name": "SYN_A",  "type": "acid",  "fraction": 20.0, "nu": 6.00, "keq": 3.0e-2,
     "sigma": 17.0, "kkin": 3.0e-1, "fit": True},
    {"name": "SYN_M",  "type": "main",  "fraction": 55.0, "nu": 6.40, "keq": 1.2e-1,
     "sigma": 48.0, "kkin": 1.0e+0, "fit": True},
    {"name": "SYN_B1", "type": "basic", "fraction": 12.0, "nu": 9.50, "keq": 6.0e-4,
     "sigma": 30.0, "kkin": 8.0e-5, "fit": True},
    {"name": "SYN_B2", "type": "basic", "fraction": 9.0,  "nu": 9.80, "keq": 9.0e-4,
     "sigma": 33.0, "kkin": 9.0e-5, "fit": True},
    {"name": "SYN_B3", "type": "basic", "fraction": 4.0,  "nu": 10.2, "keq": 9.0e-4,
     "sigma": 36.0, "kkin": 3.5e-5, "fit": True},
]
BUFFER = {"A_mol_L": 0.035, "B_mol_L": 0.100}
# The article's design: two loadings at a common gradient, plus one shallower gradient, so loading
# and gradient are not confounded and the sigma block stays weakly identified -- which is the
# regime the compression certificates are about.
EXPERIMENTS = [
    {"id": "syn_25gL_25CV_20_95", "loading_g_L": 25.0, "gradient_start_pct": 20.0,
     "gradient_end_pct": 95.0, "gradient_length_CV": 25.0},
    {"id": "syn_45gL_25CV_20_95", "loading_g_L": 45.0, "gradient_start_pct": 20.0,
     "gradient_end_pct": 95.0, "gradient_length_CV": 25.0},
    {"id": "syn_35gL_25CV_05_105", "loading_g_L": 35.0, "gradient_start_pct": 5.0,
     "gradient_end_pct": 105.0, "gradient_length_CV": 25.0},
]
N_FRACTIONS = 21          # rows per experiment, matching the real fractionation cadence
NOISE_CV = 0.05           # multiplicative observation noise, 5% CV
NOISE_FLOOR = 0.010       # additive floor, g/L: the assay does not resolve below this
SEED = 20260801


def _simulate(loading, g_start, g_end, cv, column, components, correction, n_time_points=1200):
    """Elution-relative [time_s, salt, c1..c5] at the truth parameters."""
    inlet = build_fitting_inlet(
        buffer_a=BUFFER["A_mol_L"], buffer_b=BUFFER["B_mol_L"],
        gradient_start_pct=g_start, gradient_end_pct=g_end, elution_cv=cv,
        rt_min=column.rt, load_amount_g_l=loading,
        component_fractions_pct=components.fraction_array(),
    )
    sim = ChromatographySimulator(column=column, components=components,
                                  correction=correction, n_time_points=n_time_points)
    return sim.simulate(inlet, loading, n_time_points=n_time_points).elution_curve()


def _fractionate(curve, n_fractions):
    """Fraction-average the dense curve onto n_fractions bins over the eluting region.

    The real observations are pooled fractions, not point samples, so each row is the mean over its
    collection interval; that is what makes the twin's likelihood the same shape as the real one.
    """
    t, c = curve[:, 0], curve[:, 2:]
    tot = c.sum(1)
    if not np.any(tot > 0):
        raise SystemExit("nothing eluted at the truth parameters; check the operating conditions")
    on = np.flatnonzero(tot > 1e-4 * tot.max())
    lo, hi = t[on[0]], t[on[-1]]
    edges = np.linspace(lo, hi, n_fractions + 1)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t <= b)
        if not m.any():
            m = np.argmin(np.abs(t - 0.5 * (a + b)))[None]
        rows.append(np.concatenate([[0.5 * (a + b)], c[m].mean(0)]))
    return np.asarray(rows)


def build(check_only=False):
    column = ColumnParameters.from_dict(COLUMN)
    components = ComponentSet.from_dict_list(COMPONENTS)
    correction = make_loading_correction("off", components.n_protein)
    rng = np.random.default_rng(SEED)

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    rows_out, resid = [], []
    for e in EXPERIMENTS:
        curve = _simulate(e["loading_g_L"], e["gradient_start_pct"], e["gradient_end_pct"],
                          e["gradient_length_CV"], column, components, correction)
        clean = _fractionate(curve, N_FRACTIONS)
        obs = clean.copy()
        y = obs[:, 1:]
        obs[:, 1:] = np.maximum(0.0, y * (1.0 + NOISE_CV * rng.standard_normal(y.shape))
                                + NOISE_FLOOR * rng.standard_normal(y.shape))
        resid.append(obs[:, 1:] - clean[:, 1:])
        path = OUT_DATA / f"{e['id']}.csv"
        if not check_only:
            hdr = "time_s," + ",".join(c["name"] for c in COMPONENTS)
            np.savetxt(path, obs, delimiter=",", header=hdr, comments="", fmt="%.6g")
        peak = clean[:, 1:].sum(1).max()
        rows_out.append({"id": e["id"], "n_fractions": int(len(obs)), "peak_total_g_L": float(peak),
                         "file": str(path.relative_to(ROOT))})
        print(f"  [{e['id']}] {len(obs)} fractions, peak total {peak:.3f} g/L -> {path.name}")

    r = np.concatenate([x.ravel() for x in resid])
    print(f"  observation residual: rms {r.std():.4f} g/L over {r.size} entries "
          f"(noise model: {NOISE_CV:.0%} CV + {NOISE_FLOOR} g/L floor)")
    if check_only:
        return

    (OUT_CFG / "column_hlxsyn.yaml").write_text(
        "# Synthetic twin: fictitious column, written by scripts/make_synthetic_twin.py.\n"
        + yaml.safe_dump(COLUMN, sort_keys=False), encoding="utf-8")
    (OUT_CFG / "components_hlxsyn.yaml").write_text(
        "# Synthetic twin: GROUND-TRUTH SMA parameters. The fit is supposed to recover these.\n"
        + yaml.safe_dump({"components": COMPONENTS}, sort_keys=False), encoding="utf-8")
    (OUT_CFG / "experiments_hlxsyn.yaml").write_text(
        "# Synthetic twin: the three-experiment design, written by scripts/make_synthetic_twin.py.\n"
        + yaml.safe_dump({"buffer": BUFFER,
                          "experiments": [{**e, "data_file": f"data/synthetic_twin/{e['id']}.csv"}
                                          for e in EXPERIMENTS]}, sort_keys=False), encoding="utf-8")

    OUT_TRUTH.parent.mkdir(parents=True, exist_ok=True)
    OUT_TRUTH.write_text(json.dumps({
        "product": "HLXSYN",
        "purpose": "end-to-end reproduction target; the real products' raw traces are not released",
        "column": COLUMN, "components_truth": COMPONENTS, "buffer": BUFFER,
        "experiments": rows_out,
        "observation_model": {"n_fractions": N_FRACTIONS, "noise_cv": NOISE_CV,
                              "noise_floor_g_L": NOISE_FLOOR,
                              "form": "y = max(0, c*(1+eps1) + eps2), eps1~N(0,cv^2), eps2~N(0,floor^2)"},
        "seed": SEED,
        "expected_fit": {
            "rmse_at_truth": 0.144,
            "noise_rms_on_scored_points": 0.080,
            "why_they_differ": "observations are fraction MEANS while the likelihood compares to "
                               "point values on the solver grid, exactly as for the real pooled "
                               "fractions. A correct fit therefore lands near 0.144, not 0.080; a "
                               "fit that reaches 0.080 is fitting the noise.",
            "identifiability": "at truth, +10% on the main component's nu multiplies the RMSE by "
                               "9.8 and doubling its k_eq by 8.0, while +50% on a basic component's "
                               "sigma multiplies it by 1.2 -- the weakly identified steric block "
                               "the compression certificates are stated for is present in the twin.",
        },
        "note": "components_truth is what a correct fit recovers; components_hlxsyn.yaml holds the "
                "same values, so start any fit from a perturbed initial point, not from the file.",
    }, indent=2), encoding="utf-8")
    print(f"  wrote {OUT_TRUTH.relative_to(ROOT)} and three configs/*_hlxsyn.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="regenerate in memory and report the residual; write nothing")
    a = ap.parse_args()
    print("=" * 88)
    print("SYNTHETIC TWIN" + ("  (check)" if a.check else ""))
    print("=" * 88)
    build(check_only=a.check)


if __name__ == "__main__":
    main()
