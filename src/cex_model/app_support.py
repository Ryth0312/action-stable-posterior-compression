"""Support for the Streamlit pipeline app: parse uploads -> calibrate -> optimize.

Two upload formats (same product per session; 4- and 5-component data must not be
mixed):

* CSV / XLSX in the 实验1/2/3 grid layout — operating conditions in cells
  (rows 1-12, cols 0-1) and the chromatogram in columns O.. (col 14 = t in
  seconds from injection, cols 15.. = each component in g/L).
* MAT in the ``exp_results`` layout — ``data`` (col 0 = time, cols 1.. =
  components), ``com_per``, ``load``, ``bufferAB``, ``startendper``,
  ``elution_CV``, ``RT``, ``length``.
"""

from __future__ import annotations

import io as _io
from dataclasses import dataclass, replace
import json
import re
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from cex_model.collection import (
    apply_concentration_window,
    fixed_window_thresholds,
    optimize_collection_window,
)
from cex_model.column import ColumnParameters
from cex_model.components import ComponentSet, ComponentType
from cex_model.corrections import IdentityCorrection, LoadingCorrection, make_loading_correction
from cex_model.fitting import FittingConfig, fit_sma_parameters
from cex_model.gradients import build_fitting_inlet
from cex_model.io import (
    load_column_config,
    load_components_config,
    load_fractionated_experiment_curve,
    load_hlxsyn_column_from_excel,
    load_hlxsyn_hplc_curve,
    load_hlxsyn_raw_curve,
    load_project_config,
)
from cex_model.simulator import ChromatographySimulator, count_solves

ROOT = Path(__file__).resolve().parents[2]

# Per-component-count starting templates (real lab par of the right size, used as
# the initial guess + to assign acid/main/basic groups for purity constraints).
_TEMPLATES = {
    5: "configs/components_hlxsyn.yaml",
}
DEFAULT_RT_MIN = 8.0  # HLXSYN residence time (volume/flow = 23.04/2.88)


@dataclass
class Experiment:
    """One parsed experiment (curve already on the elution-relative time axis)."""

    name: str
    curve: np.ndarray            # [time_s, comp1..n] g/L, elution-relative
    loading_g_l: float
    buffer_a: float
    buffer_b: float
    gradient_start_pct: float
    gradient_end_pct: float
    elution_cv: float
    fractions_pct: list[float]
    rt_min: float
    n_components: int
    length_mm: float | None = None
    # Mapping from each observed experimental protein column to one or more
    # mechanistic component indices. None = one-to-one. Example: [[0, 1], [2]]
    # means observed peak 1 is simulated component 1+2.
    observation_groups: list[list[int]] | None = None

    def fit_dict(self) -> dict:
        """Shape for fit_sma_parameters / a single fitting case."""
        return {
            "buffer_a": self.buffer_a, "buffer_b": self.buffer_b,
            "gradient_start_pct": self.gradient_start_pct,
            "gradient_end_pct": self.gradient_end_pct,
            "elution_cv": self.elution_cv, "load_amount_g_l": self.loading_g_l,
            "data": self.curve, "observation_groups": self.observation_groups,
        }


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _clean_curve(time_s: np.ndarray, prot: np.ndarray, elution_start_s: float, *,
                 force_absolute: bool = False) -> np.ndarray:
    """Drop NaN rows, shift to elution-relative (if the times look absolute), sort."""
    t = pd.to_numeric(pd.Series(time_s), errors="coerce").to_numpy()
    cols = [pd.to_numeric(pd.Series(prot[:, j]), errors="coerce").to_numpy() for j in range(prot.shape[1])]
    P = np.column_stack(cols)
    ok = np.isfinite(t) & np.all(np.isfinite(P), axis=1)
    t, P = t[ok], P[ok]
    # Times far above the elution start are absolute-from-injection -> re-zero.
    # Some workbooks include negative pre-elution padding even when the positive
    # chromatogram block is absolute, so callers can force the subtraction after
    # inspecting the peak positions.
    if elution_start_s > 0 and (force_absolute or t.min() > 0.5 * elution_start_s):
        t = t - elution_start_s
    keep = t >= 0
    t, P = t[keep], P[keep]
    order = np.argsort(t)
    return np.column_stack([t[order], P[order]])


def _looks_like_absolute_grid_time(
    time_s: np.ndarray, prot: np.ndarray, fractions_pct: list[float], elution_start_min: float
) -> bool:
    """Detect grid files whose positive col-O times are absolute-from-injection.

    The HLXSYN 15CV/30g workbook is a known mixed-layout file: it has negative
    pre-elution padding but the positive chromatogram block is still absolute. If
    the apparent main peak occurs well after the recorded elution start, treat the
    block as absolute and subtract the start time.
    """
    if elution_start_min <= 0 or prot.size == 0:
        return False
    fr = _normalized_fractions(prot.shape[1], fractions_pct)
    main_idx = int(np.argmax(fr)) if fr.size else 0
    t = pd.to_numeric(pd.Series(time_s), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(prot[:, main_idx]), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(y) & (t >= 0.0)
    if not np.any(ok) or np.nanmax(y[ok]) <= 0.0:
        return False
    peak_min = float(t[ok][int(np.argmax(y[ok]))] / 60.0)
    return peak_min > elution_start_min + 5.0


def parse_grid(name: str, df: pd.DataFrame, rt_min: float = DEFAULT_RT_MIN) -> Experiment:
    """Parse a 实验-style grid (CSV or XLSX read with header=None)."""
    arr = df.to_numpy()
    n = arr.shape[1] - 15  # cols 14=time, 15.. = components
    if n < 1:
        raise ValueError(f"{name}: expected the chromatogram in columns O.. (>=16 columns)")

    def cell(r, c=1):
        return float(pd.to_numeric(pd.Series([arr[r, c]]), errors="coerce").iloc[0])

    loading = cell(1)
    fractions = [cell(2 + j) for j in range(n)]
    buffer_a, buffer_b = cell(7), cell(8)
    g_start, g_end = cell(9), cell(10)
    elution_cv = cell(11)
    elution_start_min = cell(12)
    prot = arr[1:, 15 : 15 + n].astype(object)
    force_absolute = _looks_like_absolute_grid_time(arr[1:, 14], prot, fractions, elution_start_min)
    curve = _clean_curve(arr[1:, 14], prot, elution_start_min * 60.0, force_absolute=force_absolute)
    return Experiment(name=name, curve=curve, loading_g_l=loading, buffer_a=buffer_a,
                      buffer_b=buffer_b, gradient_start_pct=g_start, gradient_end_pct=g_end,
                      elution_cv=elution_cv, fractions_pct=fractions, rt_min=rt_min, n_components=n)


def parse_mat(name: str, raw: bytes) -> Experiment:
    """Parse an ``exp_results`` MAT file."""
    import scipy.io as sio

    e = sio.loadmat(_io.BytesIO(raw), squeeze_me=True, struct_as_record=False)["exp_results"]
    data = np.asarray(e.data, dtype=float)
    n = data.shape[1] - 1
    t = data[:, 0].copy()
    # Auto units: a chromatogram in minutes is O(10-300); in seconds O(1e3-1e4).
    if np.nanmax(t) < 500:
        t = t * 60.0
    # exp_results 的时间是绝对运行时间(从进样起点)，与 simulate_elution 的进样原点对齐；
    # 早先误当作"洗脱相对"而减 t.min，会引入 ~(load+wash) 时长的恒定时间偏移，使 mat 源产品
    # (HLXSYN/HLXSYN) 的 compute_rmse / 后验预测虚高(~1.0 而非 ~0.14)。保留绝对时间即自动对齐。
    curve = np.column_stack([t, data[:, 1 : 1 + n]])
    curve = curve[curve[:, 0] >= 0]
    se = np.atleast_1d(e.startendper).astype(float)
    com = np.atleast_1d(e.com_per).astype(float)[:n]
    return Experiment(
        name=name, curve=curve, loading_g_l=float(e.load),
        buffer_a=float(np.atleast_1d(e.bufferAB)[0]), buffer_b=float(np.atleast_1d(e.bufferAB)[1]),
        gradient_start_pct=float(se[0]) * 100.0, gradient_end_pct=float(se[1]) * 100.0,
        elution_cv=float(e.elution_CV), fractions_pct=[float(x) for x in com],
        rt_min=float(np.atleast_1d(e.RT).ravel()[0]), n_components=n,
        length_mm=float(e.length) if hasattr(e, "length") else None,
    )


def parse_uploaded(name: str, raw: bytes, rt_min: float = DEFAULT_RT_MIN) -> Experiment:
    """Dispatch by extension. ``raw`` is the uploaded file's bytes."""
    suffix = Path(name).suffix.lower()
    if suffix == ".mat":
        return parse_mat(name, raw)
    if suffix in (".xlsx", ".xls"):
        return parse_grid(name, pd.read_excel(_io.BytesIO(raw), header=None), rt_min)
    if suffix == ".csv":
        return parse_grid(name, pd.read_csv(_io.BytesIO(raw), header=None), rt_min)
    raise ValueError(f"Unsupported file type: {suffix} (use .csv, .xlsx, or .mat)")


# --------------------------------------------------------------------------- #
# Model setup + calibration
# --------------------------------------------------------------------------- #
def _normalized_fractions(n_components: int, fractions_pct: list[float]) -> np.ndarray:
    fr = np.asarray(fractions_pct[:n_components], dtype=float)
    if fr.size < n_components:
        fr = np.pad(fr, (0, n_components - fr.size), constant_values=1.0)
    fr = np.nan_to_num(fr, nan=0.0, posinf=0.0, neginf=0.0)
    return (fr / fr.sum() * 100.0) if fr.sum() > 0 else np.full(n_components, 100.0 / n_components)


def infer_component_types(fractions_pct: list[float]) -> list[ComponentType]:
    """Infer acid/main/basic groups from elution order and feed fraction only.

    New-product Phase 1 intentionally avoids requiring a product YAML. The largest
    fraction is treated as the main peak; earlier peaks are acidic and later peaks
    are basic, matching the standard CEX fractionation ordering.
    """
    fr = _normalized_fractions(len(fractions_pct), fractions_pct)
    if fr.size == 0:
        return []
    main = int(np.argmax(fr))
    types: list[ComponentType] = []
    for j in range(fr.size):
        if j == main:
            types.append(ComponentType.MAIN)
        elif j < main:
            types.append(ComponentType.ACID)
        else:
            types.append(ComponentType.BASIC)
    return types


def generic_components(
    n_components: int, fractions_pct: list[float], *, split_first_peak: bool = False,
    first_split_fraction: float = 0.25
) -> ComponentSet:
    """Create a product-agnostic SMA initial guess from uploaded data only.

    Starting values are in a physical magnitude (keq~0.05, kkin~0.5 table units;
    real products span keq~0.0005-0.5) so proteins actually elute within the data
    window; calibration then refines from there (fast tunes nu/kkin; high precision
    refines all four SMA rows against all uploaded experiments). A keq that is orders
    of magnitude too large binds everything too strongly and nothing elutes.
    Loading correction is disabled for unknown products to avoid applying a
    product-specific correction (for example HLXSYN) to unrelated molecules.
    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    fr = _normalized_fractions(n_components, fractions_pct)
    if split_first_peak:
        if n_components < 1:
            raise ValueError("cannot split first peak without observed components")
        alpha = float(np.clip(first_split_fraction, 0.05, 0.95))
        fr = np.concatenate([[fr[0] * alpha, fr[0] * (1.0 - alpha)], fr[1:]])
        n_components = int(fr.size)
    types = infer_component_types(fr.tolist())
    if n_components == 1:
        nu_grid = np.array([7.0])
        sigma_grid = np.array([30.0])
    else:
        nu_grid = np.linspace(6.0, 9.5, n_components)
        sigma_grid = np.linspace(20.0, 55.0, n_components)
    comps = []
    from cex_model.components import SMAComponent

    for j in range(n_components):
        comps.append(
            SMAComponent(
                name=("C1a" if split_first_peak and j == 0 else
                      "C1b" if split_first_peak and j == 1 else
                      f"C{j if split_first_peak else j + 1}"),
                component_type=types[j],
                fraction=float(fr[j]),
                keq=0.05,
                kkin=0.5,
                nu=float(nu_grid[j]),
                sigma=float(sigma_grid[j]),
            )
        )
    return ComponentSet(comps, loading_correction=None)


def template_components(n_components: int, fractions_pct: list[float]) -> ComponentSet:
    """Initial-guess ComponentSet of the right size with the data's feed fractions.

    Kept for backwards-compatible tests and known-product scripts. The Streamlit
    Phase 1 path uses :func:`generic_components` so a new product can be calibrated
    from uploads without asking users to choose a YAML template.
    """
    path = _TEMPLATES.get(n_components)
    if path is None:
        raise ValueError(f"No template for {n_components} components (supported: 3/4/5)")
    base = load_components_config(ROOT / path)
    comps = base.components[:n_components]
    fr = _normalized_fractions(n_components, fractions_pct)
    cs = ComponentSet([replace(c, fraction=float(fr[i])) for i, c in enumerate(comps)],
                      loading_correction=base.loading_correction)
    return cs


def build_column(rt_min: float, length_mm: float | None = None, *, total_porosity: float = 0.84,
                 dax: float = 0.059, ionic_capacity: float = 0.398, grid_size: int = 51,
                 superficial_velocity: float = 0.385, column_volume: float | None = None,
                 flow_rate: float | None = None, dead_volume: float = 0.0) -> ColumnParameters:
    """Build a column from manual Phase-1 inputs.

    If a physical volume/flow pair is provided, RT is derived from it; otherwise the
    legacy convention uses ``column_volume=rt_min`` and ``flow_rate=1``.
    """
    volume = float(column_volume) if column_volume is not None else float(rt_min)
    flow = float(flow_rate) if flow_rate is not None else 1.0
    return ColumnParameters(
        column_volume=volume, flow_rate=flow, column_length=length_mm or 185.0,
        total_porosity=total_porosity, superficial_velocity=superficial_velocity,
        dax=dax, ionic_capacity=ionic_capacity, grid_size=grid_size, dead_volume=dead_volume,
    )


def column_from_base_params(path_or_bytes: str | Path | bytes) -> ColumnParameters:
    """Build ``ColumnParameters`` from a ``基本参数.xlsx``-style workbook.

    The Excel parsing itself is delegated to ``load_hlxsyn_column_from_excel``; bytes
    support lets Streamlit consume uploaded files without exposing YAML selection.
    """
    if isinstance(path_or_bytes, bytes):
        with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp:
            tmp.write(path_or_bytes)
            tmp.flush()
            return ColumnParameters.from_dict(load_hlxsyn_column_from_excel(tmp.name))
    return ColumnParameters.from_dict(load_hlxsyn_column_from_excel(path_or_bytes))


def make_correction(components: ComponentSet, *, unknown_product: bool = True) -> LoadingCorrection:
    """Resolve loading correction; unknown products default to identity."""
    if unknown_product and components.loading_correction is None:
        return IdentityCorrection()
    return make_loading_correction("auto", components.n_protein, coeffs=components.loading_correction)


def _restore_component_metadata(fitted: ComponentSet, init_components: ComponentSet) -> ComponentSet:
    return ComponentSet(
        [replace(c, name=init_components.components[i].name,
                 component_type=init_components.components[i].component_type)
         for i, c in enumerate(fitted.components)],
        loading_correction=init_components.loading_correction,
    )


def split_first_peak_groups(n_observed: int) -> list[list[int]]:
    """Observation mapping for a two-species split of the first observed peak."""
    if n_observed < 1:
        return []
    return [[0, 1]] + [[j + 1] for j in range(1, n_observed)]


def observed_component_names(exp: Experiment, components: ComponentSet) -> list[str]:
    groups = exp.observation_groups
    if not groups:
        return components.names[: exp.n_components]
    names = []
    for obs_idx, group in enumerate(groups):
        if len(group) == 1 and group[0] < len(components.names):
            names.append(components.names[group[0]])
        else:
            parts = [components.names[j] for j in group if j < len(components.names)]
            names.append("+".join(parts) if parts else f"Obs {obs_idx + 1}")
    return names


def apply_observation_mapping(curve: np.ndarray, observation_groups: list[list[int]] | None) -> np.ndarray:
    """Aggregate simulated protein columns to observed experimental peaks."""
    if not observation_groups:
        return curve
    cols = [curve[:, 2 + np.asarray(group, dtype=int)].sum(axis=1) for group in observation_groups]
    return np.column_stack([curve[:, 0], curve[:, 1], *cols])


def _peak_features(time_s: np.ndarray, conc: np.ndarray) -> dict[str, float]:
    time_s = np.asarray(time_s, dtype=float)
    conc = np.maximum(np.asarray(conc, dtype=float), 0.0)
    if time_s.size == 0 or conc.size == 0 or float(np.max(conc)) <= 0.0:
        return {"tr_min": np.nan, "height_g_l": 0.0, "area": 0.0, "width_min": np.nan}
    idx = int(np.argmax(conc))
    height = float(conc[idx])
    half = 0.5 * height
    above = np.where(conc >= half)[0]
    width = float((time_s[above[-1]] - time_s[above[0]]) / 60.0) if above.size >= 2 else 0.0
    return {
        "tr_min": float(time_s[idx] / 60.0),
        "height_g_l": height,
        "area": float(np.trapezoid(conc, time_s)),
        "width_min": width,
    }


def peak_shape_diagnostics(
    column: ColumnParameters, components: ComponentSet, exp: Experiment, correction: LoadingCorrection,
    *, method: str = "RK23"
) -> list[dict[str, float | str]]:
    """Per-observed-peak diagnostics: retention, height, area and half-height width."""
    sim = simulate_elution(
        column, components, buffer_a=exp.buffer_a, buffer_b=exp.buffer_b,
        gradient_start_pct=exp.gradient_start_pct, gradient_end_pct=exp.gradient_end_pct,
        elution_cv=exp.elution_cv, loading_g_l=exp.loading_g_l, correction=correction, method=method,
    )
    sim_obs = apply_observation_mapping(sim, exp.observation_groups)
    names = observed_component_names(exp, components)
    rows = []
    n_obs = exp.curve.shape[1] - 1
    for j in range(n_obs):
        exp_f = _peak_features(exp.curve[:, 0], exp.curve[:, 1 + j])
        sim_f = _peak_features(sim_obs[:, 0], sim_obs[:, 2 + j])
        rows.append({
            "component": names[j] if j < len(names) else f"Obs {j + 1}",
            "exp_tr_min": exp_f["tr_min"], "sim_tr_min": sim_f["tr_min"],
            "delta_tr_min": sim_f["tr_min"] - exp_f["tr_min"],
            "exp_height_g_l": exp_f["height_g_l"], "sim_height_g_l": sim_f["height_g_l"],
            "height_ratio": sim_f["height_g_l"] / max(exp_f["height_g_l"], 1e-12),
            "exp_area": exp_f["area"], "sim_area": sim_f["area"],
            "area_ratio": sim_f["area"] / max(exp_f["area"], 1e-12),
            "exp_width_min": exp_f["width_min"], "sim_width_min": sim_f["width_min"],
        })
    return rows


def export_calibration_package(
    column: ColumnParameters, components: ComponentSet, experiments: list[Experiment], rmse: dict[str, float],
    *, calibration_mode: str
) -> bytes:
    """Export calibrated product/column/session metadata as a JSON package."""
    payload = {
        "schema": "cex_calibration_package_v1",
        "calibration_mode": calibration_mode,
        "column": {
            "column_volume": column.column_volume, "column_length": column.column_length,
            "flow_rate": column.flow_rate, "total_porosity": column.total_porosity,
            "superficial_velocity": column.superficial_velocity, "dax": column.dax,
            "ionic_capacity": column.ionic_capacity, "particle_porosity": column.particle_porosity,
            "particle_radius": column.particle_radius, "dead_volume": column.dead_volume,
            "grid_size": column.grid_size,
        },
        "components": [
            {
                "name": c.name, "type": c.component_type.value, "fraction": c.fraction,
                "keq": c.keq, "kkin": c.kkin, "nu": c.nu, "sigma": c.sigma,
            }
            for c in components.components
        ],
        "loading_correction": components.loading_correction,
        "experiments": [
            {
                "name": e.name, "loading_g_l": e.loading_g_l, "gradient_start_pct": e.gradient_start_pct,
                "gradient_end_pct": e.gradient_end_pct, "elution_cv": e.elution_cv,
                "buffer_a": e.buffer_a, "buffer_b": e.buffer_b,
                "observation_groups": e.observation_groups,
                "rmse": rmse.get(e.name),
            }
            for e in experiments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


# --------------------------------------------------------------------------- #
# Known-product registry: load pre-fitted params from a dropdown (no calibration)
# --------------------------------------------------------------------------- #
USER_PRODUCTS_DIR = ROOT / "configs" / "user_products"
_MAT_ROOT = ROOT.parent / "program" / "Fitting" / "data"

# Built-in products with pre-fitted mechanistic params. ``data_source``: "grid"
# (xlsx via the experiments YAML ``data_file``) or "mat"
# (``program/Fitting/data/<mat_dir>/<id>_Results.mat``). ``observation_groups``
# bridges model columns to data columns (None = 1:1); HLXSYN splits the acid peak
# into AP1+AP2, HLXSYN folds its near-inert c5 into the last observed peak.
# ``correction``: loading-correction mode -- "auto" uses the product's own coeffs
# (or the hard-coded HLXSYN default for HLXSYN's 5-protein no-coeffs case); "off"
# forces identity so HLXSYN (also 5-protein, no coeffs) does NOT inherit HLXSYN's.
_BUILTIN_PRODUCTS: dict[str, dict] = {
    # Synthetic twin: the reproduction target released in place of the proprietary traces. Built by
    # scripts/make_synthetic_twin.py, which also writes the ground truth a fit should recover.
    "HLXSYN": {"label": "HLXSYN (synthetic twin·5组分)", "column": "configs/column_hlxsyn.yaml",
               "components": "configs/components_hlxsyn.yaml",
               "experiments": "configs/experiments_hlxsyn.yaml",
               "data_source": "synthetic", "observation_groups": None, "correction": "off"},
}


@dataclass
class ProductBundle:
    """A loaded product: column + pre-fitted components + experiments + correction."""

    product_id: str
    label: str
    column: ColumnParameters
    components: ComponentSet
    experiments: list[Experiment]   # may be [] / params-only if data files are absent
    correction: LoadingCorrection
    observation_groups: list[list[int]] | None
    source: str                     # "builtin" | "user"


def list_products() -> list[tuple[str, str]]:
    """``[(product_id, label), ...]`` for the dropdown: built-ins then user products.

    A user file whose id matches a built-in is a *shadow* (an "update current"
    save) and is folded into the built-in entry (marked 已调整), not listed twice.
    """
    builtin_ids = set(_BUILTIN_PRODUCTS)
    out: list[tuple[str, str]] = []
    for pid, spec in _BUILTIN_PRODUCTS.items():
        shadowed = (USER_PRODUCTS_DIR / f"{pid}.json").exists()
        out.append((pid, spec["label"] + (" · 已调整" if shadowed else "")))
    if USER_PRODUCTS_DIR.exists():
        for path in sorted(USER_PRODUCTS_DIR.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 -- skip malformed user files
                continue
            pid = payload.get("product_id") or path.stem
            if pid in builtin_ids:
                continue  # shadow of a built-in
            out.append((pid, (payload.get("label") or pid) + " · 用户"))
    return out


def _conditions_finite(e: Experiment) -> bool:
    """True iff the operating conditions the simulator needs are all finite."""
    vals = (e.loading_g_l, e.buffer_a, e.buffer_b, e.gradient_start_pct,
            e.gradient_end_pct, e.elution_cv, e.rt_min)
    return all(np.isfinite(v) for v in vals)


def _build_experiments_for_product(
    spec: dict, column: ColumnParameters, n_model: int
) -> list[Experiment]:
    """Load a product's experiments from disk (xlsx grid or .mat), with groups attached."""
    proj = load_project_config(ROOT / spec["experiments"])
    groups = spec.get("observation_groups")
    buf = proj.get("buffer", {})
    exps: list[Experiment] = []
    for ed in proj.get("experiments", []):
        try:
            if spec["data_source"] == "synthetic":
                # Synthetic twin: a plain CSV, [time_s, c1..cn] in g/L already on the
                # elution-relative axis, so there is no time convention to reconstruct.
                path = ROOT / ed["data_file"]
                if not path.exists():
                    continue
                cur = np.loadtxt(path, delimiter=",", skiprows=1)
                e = Experiment(
                    name=str(ed["id"]), curve=cur,
                    loading_g_l=float(ed["loading_g_L"]),
                    buffer_a=float(buf["A_mol_L"]), buffer_b=float(buf["B_mol_L"]),
                    gradient_start_pct=float(ed["gradient_start_pct"]),
                    gradient_end_pct=float(ed["gradient_end_pct"]),
                    elution_cv=float(ed["gradient_length_CV"]),
                    fractions_pct=list(ed.get("component_fractions_pct") or []),
                    rt_min=column.rt, n_components=cur.shape[1] - 1,
                )
        except Exception:  # noqa: BLE001 -- one bad file must not kill the bundle
            continue
        if not _conditions_finite(e):
            continue  # a still-NaN condition would crash the simulator downstream
        if groups is not None:
            # Align the data to the observed-group count: a grid workbook may carry
            # trailing zero-padding columns (HLXSYN's IEC files parse to 5 columns,
            # 4 real + 1 empty) -> keep the first len(groups) so the aggregated model
            # (e.g. AP1+AP2, M, B1, B2) lines up with the data columns.
            n_obs = len(groups)
            curve = e.curve[:, : 1 + n_obs] if e.curve.shape[1] - 1 > n_obs else e.curve
            e = replace(e, curve=curve, n_components=min(e.n_components, n_obs),
                        observation_groups=groups)
        exps.append(e)
    return exps


def load_product(product_id: str) -> ProductBundle:
    """Load a known/user product (column + pre-fitted components + experiments)."""
    shadow = USER_PRODUCTS_DIR / f"{product_id}.json"
    if product_id not in _BUILTIN_PRODUCTS or shadow.exists():
        if shadow.exists():
            bundle = import_product_package(shadow.read_bytes(), source="user")
            # A shadow over a BUILT-IN (e.g. a "C" joint recalibration) is params-only; re-attach
            # the built-in's experiments so tab1 still shows data points + RMSE for the new params.
            if product_id in _BUILTIN_PRODUCTS:
                spec = _BUILTIN_PRODUCTS[product_id]
                try:
                    exps = _build_experiments_for_product(spec, bundle.column, bundle.components.n_protein)
                    if exps:
                        bundle = replace(bundle, experiments=exps)
                except Exception:  # noqa: BLE001 -- keep the params-only bundle if data is absent
                    pass
            return bundle
        raise ValueError(f"未知产品: {product_id}")
    spec = _BUILTIN_PRODUCTS[product_id]
    column = load_column_config(ROOT / spec["column"])
    components = load_components_config(ROOT / spec["components"])
    correction = make_loading_correction(spec.get("correction", "auto"), components.n_protein,
                                         coeffs=components.loading_correction)
    exps = _build_experiments_for_product(spec, column, components.n_protein)
    return ProductBundle(product_id, spec["label"], column, components, exps,
                         correction, spec.get("observation_groups"), "builtin")


def _correction_coeffs(correction: LoadingCorrection) -> dict | None:
    """Serialize a loading correction's effective ``{a, b}`` coeffs (None for identity)."""
    a = getattr(correction, "a", None)
    b = getattr(correction, "b", None)
    if a is None or b is None:
        return None
    return {"a": np.asarray(a).tolist(), "b": np.asarray(b).tolist()}


def export_product_package(
    product_id: str, label: str, column: ColumnParameters, components: ComponentSet,
    experiments: list[Experiment], rmse: dict[str, float], observation_groups,
    *, correction: LoadingCorrection | None = None, calibration_mode: str = "",
) -> bytes:
    """Serialize a product (column + components + experiment conditions) as JSON bytes.

    The effective loading correction is recorded as ``{a, b}`` coeffs (not the
    components' possibly-empty block) so an imported/round-tripped product keeps
    HLXSYN's loading correction instead of silently reverting to identity.
    """
    pkg = json.loads(export_calibration_package(column, components, experiments, rmse,
                                                calibration_mode=calibration_mode))
    pkg["schema"] = "cex_product_package_v1"
    pkg["product_id"] = product_id
    pkg["label"] = label
    pkg["observation_groups"] = observation_groups
    if correction is not None:
        pkg["loading_correction"] = _correction_coeffs(correction)
    return json.dumps(pkg, ensure_ascii=False, indent=2).encode("utf-8")


def _experiment_from_pkg(e: dict, n_components: int, rt_min: float) -> Experiment:
    """A condition-only Experiment (placeholder curve) from a package entry.

    Product packages store experiment *conditions*, not curves, so an imported
    product is params-only: the 1-row zero curve keeps op-seeding / plotting from
    crashing while the tab-1 overlay shows the model line without data dots.
    """
    return Experiment(
        name=str(e.get("name", "exp")),
        curve=np.zeros((1, 1 + n_components)),
        loading_g_l=float(e.get("loading_g_l", 25.0)),
        buffer_a=float(e.get("buffer_a", 0.05)),
        buffer_b=float(e.get("buffer_b", 0.3)),
        gradient_start_pct=float(e.get("gradient_start_pct", 20.0)),
        gradient_end_pct=float(e.get("gradient_end_pct", 80.0)),
        elution_cv=float(e.get("elution_cv", 20.0)),
        fractions_pct=[],
        rt_min=float(rt_min),
        n_components=n_components,
        observation_groups=e.get("observation_groups"),
    )


def import_product_package(raw, *, source: str = "user") -> ProductBundle:
    """Reconstruct a :class:`ProductBundle` from JSON bytes (params-only).

    Accepts both ``cex_product_package_v1`` and the legacy
    ``cex_calibration_package_v1`` (synthesizing id/label).
    """
    pkg = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
    column = ColumnParameters.from_dict(pkg["column"])
    components = ComponentSet.from_dict_list(pkg["components"])
    coeffs = pkg.get("loading_correction")
    components.loading_correction = coeffs
    # Use the stored coeffs directly ("on"/identity) -- never "auto", so a 5-protein
    # product without coeffs cannot inherit HLXSYN's defaults.
    correction = make_loading_correction("on" if coeffs else "off", components.n_protein, coeffs=coeffs)
    groups = pkg.get("observation_groups")
    exps = [_experiment_from_pkg(e, components.n_protein, column.rt)
            for e in pkg.get("experiments", [])]
    pid = pkg.get("product_id") or "IMPORTED"
    label = pkg.get("label") or pid
    return ProductBundle(pid, label, column, components, exps, correction, groups, source)


def save_user_product(
    product_id: str, label: str, column: ColumnParameters, components: ComponentSet,
    experiments: list[Experiment], rmse: dict[str, float], observation_groups,
    *, correction: LoadingCorrection | None = None, calibration_mode: str = "",
) -> Path:
    """Persist a product package under ``configs/user_products/`` (never touches built-in YAMLs)."""
    USER_PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(product_id)).strip("_") or "product"
    path = USER_PRODUCTS_DIR / f"{safe}.json"
    path.write_bytes(export_product_package(safe, label, column, components, experiments, rmse,
                                            observation_groups, correction=correction,
                                            calibration_mode=calibration_mode))
    return path


def save_user_product_from_package(raw) -> Path:
    """Persist an uploaded product package JSON into ``configs/user_products/`` (for the dropdown)."""
    pkg = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
    pid = pkg.get("product_id") or "IMPORTED"
    USER_PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(pid)).strip("_") or "product"
    pkg["schema"] = pkg.get("schema") or "cex_product_package_v1"
    pkg["product_id"] = safe
    pkg.setdefault("label", safe)
    pkg.setdefault("observation_groups", None)
    path = USER_PRODUCTS_DIR / f"{safe}.json"
    path.write_text(json.dumps(pkg, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _mean_fit_rmse(column: ColumnParameters, components: ComponentSet, experiments: list[Experiment],
                   correction: LoadingCorrection, method: str) -> float:
    return float(np.mean([fit_rmse(column, components, e, correction, method) for e in experiments]))


def estimate_high_precision_ode_count(
    n_components: int, n_experiments: int, *, de_maxiter: int = 12, de_popsize: int = 5
) -> int:
    """Approximate ODE solves for the high-precision DE objective.

    Differential evolution evaluates roughly ``(maxiter + 1) * popsize * dim``
    candidate parameter vectors, where ``dim = 4 * n_components`` for
    keq/kkin/nu/sigma. Each candidate is scored against every uploaded experiment.
    This intentionally excludes the fast warm-start and final validation solves, so
    the UI can present the dominant cost with a simple, explainable formula.
    """
    if n_components < 1 or n_experiments < 1:
        return 0
    dim = 4 * int(n_components)
    return int((int(de_maxiter) + 1) * int(de_popsize) * dim * int(n_experiments))


def _bounds_around(table: np.ndarray, *, wide: bool = False) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    factors = (0.1, 10.0) if wide else (0.3, 3.0)
    floors = np.array([0.01, 0.001, 1.0, 0.0])[:, None]
    caps = np.array([20.0, 20.0, 15.0, 150.0])[:, None]
    vals = table[1:5, :]  # rows: keq, kkin, nu, sigma
    lo = np.maximum(vals * factors[0], floors)
    hi = np.minimum(np.maximum(vals * factors[1], floors * 2.0), caps)
    # keq/kkin are unknown for a new product and span orders of magnitude across
    # components, so a relative box around the (rough) warm-start cannot reach e.g. a
    # basic component's keq~6e-4 / kkin~3e-5. Search the full physical range (absolute);
    # nu/sigma keep the relative box (they start close enough).
    lo[0, :], hi[0, :] = 1e-4, 1.0   # keq (table units): real products ~5e-4..0.5
    lo[1, :], hi[1, :] = 1e-6, 5.0   # kkin (table units): real products ~3e-5..1.1
    # sigma may legitimately start at zero; keep a positive search interval.
    hi = np.maximum(hi, lo + 1e-6)
    for v_lo, v_hi in zip(lo.ravel(), hi.ravel()):
        bounds.append((float(v_lo), float(v_hi)))
    return bounds


def _is_one_to_one_observation(experiments: list[Experiment], components: ComponentSet) -> bool:
    for e in experiments:
        if e.observation_groups and (len(e.observation_groups) != components.n_protein or
                                     any(group != [idx] for idx, group in enumerate(e.observation_groups))):
            return False
    return True


def calibrate(experiments: list[Experiment], column: ColumnParameters,
              init_components: ComponentSet, correction: LoadingCorrection, *,
              method: str = "RK23", sim_cycles: int = 1, max_inner_iter: int = 30,
              mode: str = "fast", de_maxiter: int = 12, de_popsize: int = 5,
              de_polish: bool = False, workers: int = 1,
              loss_mode: str = "peak_shape", shape_offset_shared: bool = True,
              akta_traces: dict | None = None, akta_total_weight: float = 0.0,
              progress: Callable[[float], None] | None = None) -> ComponentSet:
    """Calibrate SMA params to uploaded data.

    ``mode='fast'`` keeps the MATLAB pushbutton9-style representative-experiment
    nu/kkin iteration. ``mode='high_precision'`` first obtains that warm start, then
    refines keq/kkin/nu/sigma jointly against all uploaded experiments.

    ``loss_mode`` selects the high-precision objective: ``'peak_shape'`` (default,
    unchanged) or ``'shape_corr'`` (offset-tolerant Pearson shape, see
    metrics.shape_correlation_loss). For ``'shape_corr'`` the hard retention anchor is
    dropped (its bounded tau search replaces it) and ``shape_offset_shared`` picks one
    time offset per experiment (True) vs per-peak (False).

    ``de_polish`` (high precision only) appends scipy's final L-BFGS-B local refine
    after the global DE. Default off (the DE alone, as the app uses it). Turn it ON
    when the DE is too small to step off the seed: at the preview/standard presets the
    global DE often returns the warm-start unchanged (too few generations for ~20 free
    params), so a local polish from that seed is what actually captures the nearby
    improvement (keq retention shift, kkin broadening). It costs an extra L-BFGS-B run
    (FD gradients -> ~n_params solves per step), so it is opt-in, not the default.

    ``progress`` (optional) is called with a 0..1 fraction to drive a UI progress
    bar. The serial warm-start block ticks once per ODE solve over ``[0,
    warmstart_frac]`` (1.0 for fast mode, 0.1 for high precision); high precision then
    advances ``[warmstart_frac, 0.98]`` once per ODE solve when serial (workers==1) or
    per DE generation when parallel, and reports 1.0 on return.
    """
    # The warm-start block runs serial in the main process; tick the bar once per ODE
    # solve against an upper-bound estimate so it advances smoothly instead of sitting
    # idle through the (dominant) nu/kkin iteration.
    warmstart_frac = 1.0 if mode != "high_precision" else 0.1
    warmstart_est = max(2 * len(experiments) + 2 * max_inner_iter * sim_cycles + 2, 1)
    _seen = {"n": 0}

    def _tick():
        _seen["n"] += 1
        progress(min(_seen["n"] / warmstart_est, 1.0) * warmstart_frac)

    # Fast warm-start keeps the MATLAB-style RMSE gate (its native metric). Only the
    # high-precision DE below switches to gating on the metric it actually optimizes.
    with count_solves(_tick if progress is not None else None):
        base_score = _mean_fit_rmse(column, init_components, experiments, correction, method)
        best = init_components

        if _is_one_to_one_observation(experiments, init_components):
            rep = max(experiments, key=lambda e: e.loading_g_l)
            cfg = FittingConfig(use_global=False, sim_cycles=sim_cycles, max_inner_iter=max_inner_iter)
            result = fit_sma_parameters(
                column, init_components, [rep.fit_dict()], cfg, correction=correction, method=method,
            )
            fitted = _restore_component_metadata(result.components, init_components)
            fast_score = _mean_fit_rmse(column, fitted, experiments, correction, method)
            best = fitted if fast_score <= base_score else init_components

    if mode != "high_precision":
        if progress is not None:
            progress(1.0)
        return best

    global_cfg = FittingConfig(
        use_global=True, sim_cycles=sim_cycles, max_inner_iter=max_inner_iter,
        de_popsize=de_popsize, de_maxiter=de_maxiter, de_polish=de_polish, de_tol=1e-3,
        random_seed=42, workers=workers, loss_mode=loss_mode,
        # Physical priors that break the wrong basin the basics fell into (see
        # PROJECT_STATE §4.4/§7.0 and the identifiability findings): keep nu in
        # elution order (basics' nu above the main's) and anchor each peak to its
        # experimental retention (the dominant peak cannot drift far for a lower
        # aggregate loss). Weights are large vs the ~O(1) peak_shape loss so they
        # dominate gross violations while barely touching a good fit.
        # ``shape_corr`` already tolerates a (bounded) time offset, so the hard
        # retention anchor is dropped for it (it would fight the offset-tolerance);
        # nu-monotonic stays (orthogonal). shape_offset_shared selects one tau per
        # experiment vs per-peak.
        nu_monotonic_weight=10.0,
        retention_weight=0.0 if loss_mode == "shape_corr" else 10.0,
        retention_tol_min=3.0, shape_offset_shared=shape_offset_shared,
        # Joint "C" recalibration: add the AKTA total-UV term when traces are supplied.
        akta_total_weight=(akta_total_weight if akta_traces else 0.0),
    )
    # DE-phase progress over [warmstart_frac, 0.98]. When serial (workers==1) tick once
    # per ODE solve (smooth bar) via the per-solve hook — it only fires in the main
    # process, so for parallel (workers!=1) fall back to scipy's per-generation callback
    # (also main-process). The bar no longer sits idle for minutes inside one generation.
    de_hook = None
    de_progress = None
    if progress is not None:
        span = 0.98 - warmstart_frac
        if workers == 1:
            de_est = max(estimate_high_precision_ode_count(
                best.n_protein, len(experiments), de_maxiter=de_maxiter, de_popsize=de_popsize), 1)
            de_seen = {"n": 0}

            def de_hook():
                de_seen["n"] += 1
                progress(warmstart_frac + span * min(de_seen["n"] / de_est, 1.0))
        else:
            de_progress = lambda f: progress(warmstart_frac + span * f)  # noqa: E731

    exp_dicts = []
    for e in experiments:
        d = e.fit_dict()
        if akta_traces and e.name in akta_traces:
            d["akta_trace"] = akta_traces[e.name]   # read by _FitObjective when weight > 0
        exp_dicts.append(d)
    with count_solves(de_hook):
        global_result = fit_sma_parameters(
            column, best, exp_dicts, global_cfg, correction=correction,
            method=method, bounds_override=_bounds_around(best.to_parameter_table(), wide=False),
            progress=de_progress,
        )
    global_fit = _restore_component_metadata(global_result.components, init_components)
    # Accept on the SAME metric the DE minimized (peak_shape, via history), not on
    # RMSE. Seeding the DE with the warm-start guarantees ``objective`` never
    # exceeds ``start_objective``, so high precision reliably improves the optimized
    # metric instead of being rejected by an RMSE it never optimized (and silently
    # falling back to the warm-start). RMSE is reported separately by the caller.
    hist = global_result.history[0] if global_result.history else {}
    de_obj = hist.get("objective", float("inf"))
    start_obj = hist.get("start_objective", float("inf"))
    if progress is not None:
        progress(1.0)
    return global_fit if de_obj <= start_obj else best


def simulate_elution(column: ColumnParameters, components: ComponentSet, *, buffer_a: float,
                     buffer_b: float, gradient_start_pct: float, gradient_end_pct: float,
                     elution_cv: float, loading_g_l: float, correction: LoadingCorrection,
                     method: str = "RK23", feed_cv: float = 10.0, hold_cv: float = 3.0,
                     n_time_points: int = 1200, backend=None) -> np.ndarray:
    """Simulate one elution curve -> [time_s, salt, comp1..n g/L] (proteins clipped >=0).

    Pass a trained surrogate ``backend`` to use the fast ANN forward (~ms/curve) instead of the
    ODE -- it honours the backend's fixed output grid. For Phase-2 speed (e.g. the robust-window
    Monte-Carlo); verify any adopted operating point on the ODE (backend=None) for a lossless result.

    ``hold_cv``/``n_time_points`` set the post-gradient hold length and the output grid; to re-plot a
    Phase-2 optimum so the collection window LINES UP, pass the SAME values ``optimize_process`` used
    (``ProcessOptimizationConfig.hold_cv`` / ``.n_time_points``) — a different hold_cv changes the curve
    DURATION and n_time_points changes the solver's first step, so a mismatch shifts the time axis.
    """
    if backend is not None:
        n_time_points = int(getattr(backend, "metadata", {}).get("n_time_points", n_time_points))
    inlet = build_fitting_inlet(
        buffer_a=buffer_a, buffer_b=buffer_b, gradient_start_pct=gradient_start_pct,
        gradient_end_pct=gradient_end_pct, elution_cv=elution_cv, rt_min=column.rt,
        load_amount_g_l=loading_g_l, component_fractions_pct=components.fraction_array(),
        feed_cv=feed_cv, hold_cv=hold_cv,
    )
    sim = ChromatographySimulator(column=column, components=components, method=method,
                                  correction=correction, n_time_points=n_time_points, backend=backend)
    curve = sim.simulate(inlet, loading_g_l, n_time_points=n_time_points, t_start=0.0).elution_curve()
    curve[:, 2:] = np.maximum(curve[:, 2:], 0.0)
    return curve


# ---- Optional ML residual layer (physical base + ML residual; see ------------------
# scripts/akta_ml_residual.py). Corrects the simulated TOTAL only; gated to do no harm
# outside the trained operating envelope. Returns None when a product has no trained
# corrector, so the app can offer the toggle only where it applies.
RESIDUAL_DIR = ROOT / "results" / "akta_ml_residual"


def load_residual_corrector(product_id: str):
    """Per-product :class:`ResidualCorrector`, or None if not trained / unreadable."""
    from cex_model.residual_corrector import ResidualCorrector

    path = RESIDUAL_DIR / f"{product_id}_corrector.npz"
    if not path.exists():
        return None
    try:
        return ResidualCorrector.load(path)
    except Exception:  # noqa: BLE001 -- a bad/old file must not break the app
        return None


def load_surrogate_backend(product_id: str):
    """Per-product surrogate backend (``models/<pid>_surrogate.pt``) for Phase-2 forward speed,
    or None if not trained / torch absent / unreadable. Train one with
    ``scripts/train_surrogate.py --output models/<pid>_surrogate.pt`` on a ``--design-space``
    dataset. Phase-2 runs the DE on it (~ms/curve) then verifies the optimum on the ODE (lossless).
    """
    path = ROOT / "models" / f"{product_id}_surrogate.pt"
    if not path.exists():
        return None
    try:
        from cex_model.surrogate.backend import TorchSurrogateBackend

        return TorchSurrogateBackend(path)
    except Exception:  # noqa: BLE001 -- torch missing / bad/old file must not break the app
        return None


def residual_corrected_total(corrector, sim_curve, *, loading_g_l, elution_cv,
                             gradient_start_pct, gradient_end_pct):
    """Apply the gated ML residual to a simulated curve's TOTAL protein.

    Returns ``(time_min, model_total, corrected_total, gate)`` -- the *same* gated, clipped
    correction Phase 2 uses (delegates to :meth:`ResidualCorrector.correct_curve`), so the
    displayed overlay matches the optimiser. ``gate``->0 outside the trained envelope.
    The condition feature order matches training (loading, elution_cv, grad_start, grad_end).
    """
    t_min = sim_curve[:, 0] / 60.0
    total = sim_curve[:, 2:].sum(axis=1)
    cond = [float(loading_g_l), float(elution_cv), float(gradient_start_pct), float(gradient_end_pct)]
    corrected_curve, gate = corrector.correct_curve(sim_curve, cond)
    return t_min, total, corrected_curve[:, 2:].sum(axis=1), gate


def residual_metrics_vs_akta(product_id, sim_curve, *, loading_g_l, elution_cv,
                             gradient_start_pct, gradient_end_pct, corrector):
    """RMSE of the TOTAL vs the real AKTA UV, physical model vs +ML, for a fitting condition.

    Returns ``{akta_file, rmse_model, rmse_corrected, gate}`` (gradient-start aligned, the
    same position-inclusive metric the LOO reports) or None if the condition has no AKTA
    fitting file. NOTE: a fitting condition is IN-SAMPLE for the deployed corrector, so this
    flatters the ML; the honest cross-condition number is :func:`residual_loo_summary`.
    """
    from cex_model import akta as _akta
    from cex_model.akta_compare import AKTA_FITTING, grad_start_min, gradient_aligned_grid

    files = AKTA_FITTING.get(product_id, {})
    cand = [(fn, c) for fn, c in files.items() if c[1] == elution_cv and c[2] == gradient_start_pct]
    if not cand:
        return None
    fname = min(cand, key=lambda kv: abs(kv[1][0] - loading_g_l))[0]
    akta_path = ROOT / "data" / "AKTA data" / product_id / "fitting AKTA" / fname
    if not akta_path.exists():
        return None
    tr = _akta.load_akta_uv(akta_path, eps_ml_mg_cm=_akta.extinction_for(product_id))
    t_min = sim_curve[:, 0] / 60.0
    salt = sim_curve[:, 1]
    m_y = sim_curve[:, 2:].sum(axis=1)
    rel, akta, model = gradient_aligned_grid(tr, t_min, m_y, salt)
    start = grad_start_min(t_min, salt)
    end = float(t_min[int(np.argmax(salt))])
    phi = rel / max(end - start, 1e-6)
    cond = [float(loading_g_l), float(elution_cv), float(gradient_start_pct), float(gradient_end_pct)]
    corrected, gate = corrector.apply(model, cond, phi, float(model.max()))
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))  # noqa: E731
    return {"akta_file": fname, "rmse_model": rmse(akta, model),
            "rmse_corrected": rmse(akta, corrected), "gate": gate}


def phase2_window_compare(sim_curve, components, corrector, *, loading_g_l, elution_cv,
                          gradient_start_pct, gradient_end_pct, acid_max, main_min, basic_max,
                          grid=40):
    """Physical vs +ML collection window/yield at one condition (same search as Phase 2).

    Returns ``{physical, ml, gate}`` where each side is the best purity-feasible window from
    :func:`optimize_collection_window` -- physical on the model curve, ml on the gated
    ML-corrected curve. Lets the app show how much the residual hybrid moves yield / the
    collection window at the chosen optimum (purity shown too, but it only shifts via the
    window / time-reweighting -- a total-only correction can't refine the component split).
    """
    from cex_model.collection import optimize_collection_window

    grp = group_indices(components)
    kw = dict(grid=grid, acid_max=acid_max, main_min=main_min, basic_max=basic_max,
              acid_idx=grp["acid"], main_idx=grp["main"], basic_idx=grp["basic"])
    cond = [float(loading_g_l), float(elution_cv), float(gradient_start_pct), float(gradient_end_pct)]
    corrected_curve, gate = corrector.correct_curve(sim_curve, cond)

    def _pack(w):
        return {"yield_g": float(w.yield_g), "start_s": float(w.start_time_s),
                "end_s": float(w.end_time_s), "feasible": bool(w.feasible),
                "acid": float(w.acid_fraction), "main": float(w.main_fraction),
                "basic": float(w.basic_fraction)}

    phys = optimize_collection_window(sim_curve, **kw).best
    ml = optimize_collection_window(corrected_curve, **kw).best
    return {"physical": _pack(phys), "ml": _pack(ml), "gate": gate}


def residual_loo_summary(product_id):
    """Per-product experiment-level LOO RMSE ``{before, after, noise}`` (honest, out-of-sample)."""
    path = RESIDUAL_DIR / "summary.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    for pr in data.get("products", []):
        if pr.get("product") == product_id:
            return {"before": pr.get("loo_median_rmse_before"),
                    "after": pr.get("loo_median_rmse_after"),
                    "noise": data.get("noise_floor_g_l")}
    return None


@dataclass
class RobustWindowResult:
    """Conservative fixed collection window + its validation across sampled feeds."""

    c_start_g_l: float          # rising-edge total-protein trigger
    c_end_g_l: float            # falling-edge total-protein trigger
    quantile: float
    n_samples: int
    n_optimal_feasible: int     # batches that had an optimal feasible window
    # The fixed window applied to every sampled batch:
    yields: np.ndarray
    acid: np.ndarray
    main: np.ndarray
    basic: np.ndarray
    feasible_mask: np.ndarray   # purity-spec met under the FIXED window
    start_time_s: np.ndarray
    end_time_s: np.ndarray


def robust_collection_window(
    column: ColumnParameters, components: ComponentSet, *,
    buffer_a: float, buffer_b: float, gradient_start_pct: float, gradient_end_pct: float,
    gradient_cv: float, loading_g_l: float, feed_cv: float,
    acid_idx: list[int], main_idx: list[int], basic_idx: list[int],
    acid_max: float = 0.10, main_min: float = 0.0, basic_max: float = 1.0,
    n_samples: int = 60, fraction_cv: float = 0.10, loading_cv: float = 0.0,
    quantile: float = 0.9, seed: int = 42, correction: LoadingCorrection | None = None,
    method: str = "RK23", n_time_points: int = 900, grid: int = 40,
    residual_corrector=None, backend=None,
    progress: Callable[[float], None] | None = None,
) -> RobustWindowResult:
    """Monte-Carlo a conservative FIXED collection window under feed variability.

    Ports/extends MATLAB ``gradient_opt_V1HLXSYN.m``'s "random batch -> fixed
    window": the gradient and loading are held at the chosen operating point and
    only the **feed composition** (and optionally loading) is perturbed. For each
    sampled batch the per-batch optimal window's edge concentrations are
    recorded; the conservative fixed total-protein triggers are then the
    ``quantile`` of those edges (vs MATLAB's raw ``max``). The fixed triggers are
    finally **validated** on every sampled batch -> yield/purity distribution
    (the robustness report MATLAB lacked). ``n_samples`` ODE solves dominate cost.
    """
    rng = np.random.default_rng(seed)
    nominal_frac = components.fraction_array()
    table = components.to_parameter_table()
    names = components.names

    curves: list[np.ndarray] = []
    start_concs: list[float] = []
    end_concs: list[float] = []
    for b in range(n_samples):
        fr = rng.normal(nominal_frac, fraction_cv * nominal_frac)
        fr = np.clip(fr, 1e-6, None)
        fr = fr / fr.sum() * 100.0
        load = loading_g_l
        if loading_cv > 0.0:
            load = max(loading_g_l * (1.0 + loading_cv * float(rng.standard_normal())), 1e-3)
        tbl = table.copy()
        tbl[0, :] = fr
        comps_b = ComponentSet.from_parameter_table(tbl, names=names)
        curve = simulate_elution(
            column, comps_b, buffer_a=buffer_a, buffer_b=buffer_b,
            gradient_start_pct=gradient_start_pct, gradient_end_pct=gradient_end_pct,
            elution_cv=gradient_cv, loading_g_l=load, correction=correction,
            method=method, feed_cv=feed_cv, n_time_points=n_time_points, backend=backend,
        )
        if residual_corrector is not None:  # gated ML total-residual (do-no-harm out-of-envelope)
            curve, _ = residual_corrector.correct_curve(
                curve, [load, gradient_cv, gradient_start_pct, gradient_end_pct])
        curves.append(curve)
        res = optimize_collection_window(
            curve, grid=grid, acid_max=acid_max, main_min=main_min, basic_max=basic_max,
            acid_idx=acid_idx, main_idx=main_idx, basic_idx=basic_idx,
        )
        if res.best.feasible:
            c_tot = curve[:, 2:].sum(axis=1)
            start_concs.append(float(c_tot[res.best.start_index]))
            end_concs.append(float(c_tot[res.best.end_index]))
        if progress is not None:
            progress((b + 1) / n_samples)

    c_start, c_end = fixed_window_thresholds(start_concs, end_concs, quantile)
    Y, AC, MA, BA, FE, T0, T1 = [], [], [], [], [], [], []
    for curve in curves:
        w = apply_concentration_window(
            curve, c_start, c_end, acid_max=acid_max, main_min=main_min, basic_max=basic_max,
            acid_idx=acid_idx, main_idx=main_idx, basic_idx=basic_idx,
        )
        Y.append(w.yield_g); AC.append(w.acid_fraction); MA.append(w.main_fraction)
        BA.append(w.basic_fraction); FE.append(w.feasible)
        T0.append(w.start_time_s); T1.append(w.end_time_s)

    return RobustWindowResult(
        c_start_g_l=c_start, c_end_g_l=c_end, quantile=quantile, n_samples=n_samples,
        n_optimal_feasible=len(start_concs),
        yields=np.array(Y), acid=np.array(AC), main=np.array(MA), basic=np.array(BA),
        feasible_mask=np.array(FE, dtype=bool), start_time_s=np.array(T0), end_time_s=np.array(T1),
    )


def fit_rmse(column, components, exp: Experiment, correction, method: str = "RK23") -> float:
    """Normalized RMSE of the calibrated model vs one experiment (MATLAB SMA.m metric)."""
    from cex_model.metrics import compute_rmse
    curve = simulate_elution(
        column, components, buffer_a=exp.buffer_a, buffer_b=exp.buffer_b,
        gradient_start_pct=exp.gradient_start_pct, gradient_end_pct=exp.gradient_end_pct,
        elution_cv=exp.elution_cv, loading_g_l=exp.loading_g_l, correction=correction, method=method,
    )
    return float(compute_rmse(curve, exp.curve, observation_groups=exp.observation_groups)[-1])


def group_indices(components: ComponentSet) -> dict[str, list[int]]:
    """acid / main / basic protein indices (for purity constraints), from component types."""
    acid = [j for j, c in enumerate(components.components) if c.component_type == ComponentType.ACID]
    main = [j for j, c in enumerate(components.components) if c.component_type == ComponentType.MAIN]
    basic = [j for j, c in enumerate(components.components) if c.component_type == ComponentType.BASIC]
    return {"acid": acid or [0], "main": main, "basic": basic}
