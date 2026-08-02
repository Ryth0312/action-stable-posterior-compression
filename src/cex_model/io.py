"""Configuration and data I/O."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cex_model.column import ColumnParameters
from cex_model.components import ComponentSet, ComponentType
from cex_model.corrections import LinearLoadingCorrection, LoadingCorrection
from cex_model.fitting import FittingConfig

_LAB_EXPERIMENT_ROW_MAP = {
    "loading_g_L": (2, 2),
    "buffer_a_mol_L": (8, 2),
    "buffer_b_mol_L": (9, 2),
    "gradient_start_pct": (10, 2),
    "gradient_end_pct": (11, 2),
    "gradient_length_CV": (12, 2),
    "elution_start_min": (13, 2),
}
_LAB_FRACTION_ROWS = range(3, 8)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_column_config(path: str | Path) -> ColumnParameters:
    return ColumnParameters.from_dict(load_yaml(path))


def load_components_config(path: str | Path) -> ComponentSet:
    data = load_yaml(path)
    items = data.get("components", data)
    cs = ComponentSet.from_dict_list(items)
    if isinstance(data, dict):
        cs.loading_correction = data.get("loading_correction")
    return cs


def load_project_config(path: str | Path) -> dict[str, Any]:
    """Load full project YAML (buffer, experiments, yamamoto, etc.)."""
    data = load_yaml(path)
    return data if isinstance(data, dict) else {"experiments": data}


def load_experiments_config(path: str | Path) -> list[dict[str, Any]]:
    data = load_project_config(path)
    exps = data.get("experiments", data)
    return exps if isinstance(exps, list) else [exps]


def load_fitting_config(path: str | Path) -> FittingConfig:
    data = load_yaml(path)
    return FittingConfig(**{k: v for k, v in data.items() if k in FittingConfig.__dataclass_fields__})


def load_correction_config(path: str | Path | None) -> LoadingCorrection | None:
    if path is None:
        return None
    data = load_yaml(path)
    if not data.get("enabled", True):
        return None
    return LinearLoadingCorrection.from_config(data)


def load_experiment_data_csv(path: str | Path) -> np.ndarray:
    """Load experimental chromatogram: time_s, component concentrations g/L."""
    df = pd.read_csv(path)
    time_col = "time_s" if "time_s" in df.columns else df.columns[0]
    cols = [time_col] + [c for c in df.columns if c != time_col]
    return df[cols].to_numpy(dtype=float)


def load_experiment_data_excel(path: str | Path, sheet: str | int = 0) -> np.ndarray:
    """Load from Excel matching MATLAB project format."""
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    arr = df.to_numpy(dtype=float)
    arr = arr[~np.isnan(arr[:, 3])] if arr.shape[1] > 3 else arr
    if arr.shape[1] >= 14:
        return arr[:, 13:]
    return arr


def _read_excel_cell(path: str | Path, row: int, col: int, sheet: str | int = 0) -> Any:
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    if row < 1 or col < 1 or row > df.shape[0] or col > df.shape[1]:
        raise IndexError(f"Cell ({row}, {col}) out of range for {path}")
    return df.iat[row - 1, col - 1]


def parse_lab_experiment_workbook(
    path: str | Path, *, sheet: str | int = 0
) -> dict[str, Any]:
    """Parse HLXSYN/HLXSYN experiment workbook metadata from the standard layout."""
    path = Path(path)
    parsed: dict[str, Any] = {"data_file": str(path).replace("\\", "/")}
    for key, (row, col) in _LAB_EXPERIMENT_ROW_MAP.items():
        value = _read_excel_cell(path, row, col, sheet=sheet)
        if value is None:
            raise ValueError(f"Missing {key} in {path} at row {row}")
        parsed[key] = float(value)

    fractions: list[float] = []
    for row in _LAB_FRACTION_ROWS:
        value = _read_excel_cell(path, row, 2, sheet=sheet)
        if value is None:
            continue
        fractions.append(float(value))
    parsed["component_fractions_pct"] = fractions
    return parsed


def load_hlxsyn_column_from_excel(path: str | Path) -> dict[str, float | int]:
    """Load HLXSYN column parameters from ``基本参数.xlsx``."""
    path = Path(path)
    return {
        "column_volume": float(_read_excel_cell(path, 2, 2)),
        "column_length": float(_read_excel_cell(path, 3, 2)),
        "flow_rate": float(_read_excel_cell(path, 12, 2)),
        "superficial_velocity": float(_read_excel_cell(path, 13, 2)),
        "total_porosity": float(_read_excel_cell(path, 11, 2)),
        "dax": float(_read_excel_cell(path, 14, 2)),
        "dead_volume": float(_read_excel_cell(path, 7, 2)),
        "ionic_capacity": 0.398,
        "particle_porosity": 0.5,
        "particle_radius": 0.045,
        "grid_size": 51,
    }


def load_hlxsyn_experiments_from_excel_dir(
    directory: str | Path,
    *,
    project_root: str | Path | None = None,
    id_prefix: str = "exp",
) -> list[dict[str, Any]]:
    """Build experiment config dicts from HLXSYN ``实验*.xlsx`` workbooks."""
    directory = Path(directory)
    root = Path(project_root) if project_root is not None else directory.parent.parent
    experiments: list[dict[str, Any]] = []
    for index, path in enumerate(sorted(directory.glob("实验*.xlsx")), start=1):
        meta = parse_lab_experiment_workbook(path)
        loading = int(meta["loading_g_L"]) if meta["loading_g_L"].is_integer() else meta["loading_g_L"]
        cv = int(meta["gradient_length_CV"]) if meta["gradient_length_CV"].is_integer() else meta["gradient_length_CV"]
        exp_id = (
            f"{id_prefix}{index}_"
            f"{loading}gL_{cv}CV_{int(meta['gradient_start_pct'])}_{int(meta['gradient_end_pct'])}"
        )
        entry = {
            "id": exp_id,
            "loading_g_L": float(meta["loading_g_L"]),
            "gradient_start_pct": float(meta["gradient_start_pct"]),
            "gradient_end_pct": float(meta["gradient_end_pct"]),
            "gradient_length_CV": float(meta["gradient_length_CV"]),
            "elution_start_min": float(meta["elution_start_min"]),
            "data_file": str(path.relative_to(root)).replace("\\", "/") if path.is_relative_to(root) else str(path).replace("\\", "/"),
            "component_fractions_pct": meta["component_fractions_pct"],
        }
        experiments.append(entry)
    return experiments


def load_fractionated_experiment_curve(
    path: str | Path,
    *,
    elution_start_min: float,
    n_components: int,
    sheet: str | int = 0,
    col14_is_absolute: bool = False,
) -> np.ndarray:
    """Load Excel fractionation data as [time_s, dummy_salt, proteins...].

    The uploaded HLXSYN/HLXSYN workbooks contain an "auto generated" table in columns
    O:T: time in seconds (col O) followed by component concentrations in g/L. The
    "自动生成" time column is the elution retention time to compare directly against
    the simulator's ``elution_curve`` (which re-zeros at the feed end). Negative
    rows are the pre-elution padding and are dropped.

    Most workbooks store retention time in col O already zeroed at elution start
    (绝对时间 minus col O equals ``elution_start_min``), so it is used as-is. A few
    workbooks instead have an *inflated* 绝对时间 column, which makes col O equal to
    the ABSOLUTE time rather than the retention time (cross-checked against the
    paper's Figure 4 retention axis: e.g. the 30 g/L 15 CV run's main peak sits at
    ~37 min in col O but ~10 min in the paper). For those, pass
    ``col14_is_absolute=True`` so the elution-start offset is subtracted here and the
    curve lands on the same retention axis as the rest. Verify any new workbook
    against a known reference before flagging it.
    """
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    arr = df.to_numpy()
    if arr.shape[1] < 15 + n_components:
        raise ValueError(
            f"Expected at least {15 + n_components} columns in {path}, got {arr.shape[1]}"
        )
    time_rel = pd.to_numeric(pd.Series(arr[:, 14]), errors="coerce").to_numpy()
    if col14_is_absolute:
        # col O holds absolute time (inflated 绝对时间); re-zero at elution start.
        time_rel = time_rel - elution_start_min * 60.0
    conc = np.column_stack(
        [
            pd.to_numeric(pd.Series(arr[:, 15 + j]), errors="coerce").to_numpy()
            for j in range(n_components)
        ]
    )
    valid = np.isfinite(time_rel) & np.all(np.isfinite(conc), axis=1)
    time_s = time_rel[valid].astype(float)
    conc = conc[valid].astype(float)
    keep = time_s >= 0.0
    time_s = time_s[keep]
    conc = conc[keep]
    order = np.argsort(time_s)
    time_s = time_s[order]
    conc = conc[order]
    salt = np.zeros_like(time_s)
    return np.column_stack([time_s, salt, conc])


def load_hlxsyn_hplc_curve(path: str | Path, sheet: str | int, *, n_components: int = 4) -> np.ndarray:
    """Load an HLXSYN hand-made HPLC fractionation sheet -> ``[time_s, c1..c_n]`` (elution-relative).

    These workbooks are hand-built and the column layout shifts between sheets, so columns
    are located by HEADER, not index:

    * components (g/L): the SECOND ``APS/MP/BP1/BP2`` group (the first group, near col B, is
      the HPLC percent composition; the second group is the per-fraction g/L). Taken as the
      ``n_components`` contiguous columns starting at that ``APS``.
    * time: the ``洗脱零点`` (elution-zero-referenced, i.e. already elution-relative) column
      if present, else the ``Time min`` column.

    Rows without a finite time / all-finite components (headers, blanks, the LOAD/wash block)
    are dropped; the curve is sorted by time. Returns minutes×60 as seconds to match the
    simulator's ``elution_curve`` axis (compare directly, like the xlsx grid loader).
    """
    arr = pd.read_excel(path, sheet_name=sheet, header=None).to_numpy()
    nrow, ncol = arr.shape

    def cell(r: int, c: int) -> str:
        return str(arr[r, c]).strip() if r < nrow and c < ncol else ""

    aps = [c for r in (0, 1) for c in range(ncol) if cell(r, c) == "APS"]
    if not aps:
        raise ValueError(f"no 'APS' header in {path} [{sheet}]")
    gl0 = max(aps)  # second APS group = the per-fraction g/L block
    if gl0 + n_components > ncol:
        raise ValueError(f"g/L component block at col {gl0} exceeds width in {path} [{sheet}]")
    xz = [c for r in (0, 1) for c in range(ncol) if cell(r, c) == "洗脱零点"]
    tmin = [c for r in (0, 1) for c in range(ncol) if "Time" in cell(r, c) and "min" in cell(r, c)]
    if xz:
        tcol = xz[0]
    elif tmin:
        tcol = tmin[0]
    else:
        # no time header (some sheets): pick the minutes column among those just left of the
        # g/L block -- a plausible-minutes median, and the smaller of a (min, ~60xseconds) pair.
        cand = []
        for c in range(max(0, gl0 - 3), gl0):
            col = pd.to_numeric(pd.Series(arr[:, c]), errors="coerce").to_numpy()
            fin = col[np.isfinite(col)]
            # require several points, not a lone stray numeric (e.g. a "体积 mL" volume
            # cell on the LOAD row alone) that could otherwise masquerade as a time axis
            if fin.size >= 5 and 3.0 < float(np.median(fin)) < 400.0:
                cand.append((float(np.median(fin)), c))
        if not cand:
            raise ValueError(f"no time column found in {path} [{sheet}]")
        tcol = min(cand)[1]  # minutes (smaller median) over the seconds column

    t = pd.to_numeric(pd.Series(arr[:, tcol]), errors="coerce").to_numpy()
    conc = np.column_stack(
        [pd.to_numeric(pd.Series(arr[:, gl0 + j]), errors="coerce").to_numpy() for j in range(n_components)]
    )
    valid = np.isfinite(t) & np.all(np.isfinite(conc), axis=1) & (t > 0.0)
    t = t[valid].astype(float)
    conc = np.maximum(conc[valid].astype(float), 0.0)
    # These sheets stack several fraction series in one column; split where time jumps
    # backwards and keep the FIRST block (the sheet's own primary series), so two stacked
    # blocks (e.g. a copied series pasted below) don't merge into a bogus double curve.
    if t.size:
        end = next((i for i in range(1, t.size) if t[i] < t[i - 1]), t.size)
        t, conc = t[:end], conc[:end]
        order = np.argsort(t)
        return np.column_stack([t[order] * 60.0, conc[order]])
    return np.zeros((0, 1 + n_components))


_HLXSYN_FRACTION_ID = re.compile(r"^\s*\d+\.[A-Z]\.\d+\s*$")


def load_hlxsyn_raw_curve(
    path: str | Path,
    sheet: str | int,
    *,
    elution_start_min: float = 0.0,
) -> np.ndarray:
    """Load one raw HLXSYN AKTA export sheet -> ``[time_s, cA1-1, cA2, cM, cB1]`` (g/L).

    HLXSYN ships as raw UNICORN exports (one multi-sheet workbook per experiment group).
    Each sheet embeds an HPLC fraction table alongside the continuous UV trace: a corrected
    fraction-time column (header ``减去图谱平前端负数时间显示``, minutes from injection), a
    ``Fraction`` id column (``1.A.1`` ...), per-fraction peak-area PERCENT for the four model
    peaks (headers ``酸1`` / ``酸2`` / ``主成分`` | ``MP`` / ``碱性`` | ``BPs``), and total
    ``PA (g/L)``. Component g/L = ``PA * percent / 100`` for ``[A1-1<-酸1, A2<-酸2, M<-主成分,
    B1<-碱性]``. The fraction block's start column shifts between sheets, so columns are
    located by HEADER, not index.

    ``Time`` is minutes from injection, so ``elution_start_min`` re-zeros onto the
    simulator's elution axis (pre-elution rows go negative and are dropped). Like every
    other product, ``elution_start_min`` is an initial estimate that must be calibrated
    against the run's AKTA UV / %B trace.
    """
    df = pd.read_excel(path, sheet_name=sheet, header=None)

    # Header row: the first row (<=5) carrying 'PA' and an acid/main token.
    hr = 2
    for i in range(min(6, len(df))):
        joined = " ".join(str(x) for x in df.iloc[i].tolist())
        if "PA" in joined and ("酸1" in joined or "MP" in joined or "主成分" in joined):
            hr = i
            break
    hdr = [("" if x is None else str(x)) for x in df.iloc[hr].tolist()]

    def find(*needles: str) -> int | None:
        return next((k for k, h in enumerate(hdr) if any(nd in h for nd in needles)), None)

    t_col = find("减去图谱平前端负数")
    a1, a2 = find("酸1"), find("酸2")
    mp, bp = find("主成分", "MP"), find("碱性", "BPs")
    pa = find("PA")
    if None in (t_col, a1, a2, mp, bp, pa):
        raise ValueError(
            f"missing fraction-table header in {path} [{sheet}]: "
            f"t={t_col} 酸1={a1} 酸2={a2} main={mp} basic={bp} PA={pa}"
        )
    # Fraction-id column = the one with the most 'N.X.N' values (its index shifts per sheet).
    fid = max(
        range(df.shape[1]),
        key=lambda c: int(
            df.iloc[hr + 1 : hr + 80, c]
            .apply(lambda v: isinstance(v, str) and bool(_HLXSYN_FRACTION_ID.match(v)))
            .sum()
        ),
    )
    sub = df.iloc[hr + 1 :]
    rows = sub[sub.iloc[:, fid].apply(lambda v: isinstance(v, str) and bool(_HLXSYN_FRACTION_ID.match(v)))]

    t = pd.to_numeric(rows.iloc[:, t_col], errors="coerce").to_numpy()
    pa_gl = pd.to_numeric(rows.iloc[:, pa], errors="coerce").to_numpy()
    pct = np.column_stack(
        [pd.to_numeric(rows.iloc[:, c], errors="coerce").to_numpy() for c in (a1, a2, mp, bp)]
    )
    conc = pa_gl[:, None] * pct / 100.0
    t = t * 60.0 - elution_start_min * 60.0
    valid = np.isfinite(t) & np.all(np.isfinite(conc), axis=1) & (t >= 0.0)
    t = t[valid].astype(float)
    conc = np.maximum(conc[valid].astype(float), 0.0)
    order = np.argsort(t)
    return np.column_stack([t[order], conc[order]])


def save_fitting_results(result: Any, path: str | Path) -> None:
    """Save fitted parameters to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = result.components.to_parameter_table()
    payload = {
        "components": [
            {
                "name": c.name,
                "type": c.component_type.value,
                "fraction": c.fraction,
                "keq": c.keq,
                "kkin": c.kkin,
                "nu": c.nu,
                "sigma": c.sigma,
            }
            for c in result.components.components
        ],
        "rmse_per_peak": result.rmse_per_peak.tolist(),
        "rmse_total": result.rmse_total,
        "parameter_table": table.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_matlab_reference(path: str | Path) -> np.ndarray:
    """Load MATLAB reference curve exported as CSV (time, salt, proteins)."""
    return load_experiment_data_csv(path)
