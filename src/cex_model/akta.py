"""Load AKTA/UNICORN total-UV chromatograms for residual diagnostics.

The mechanistic model is calibrated against manually-sampled fractionation points
(the xlsx workbooks), which under-sample the peak flanks and tail (low-concentration
fractions are not detected and recorded as zero). The AKTA UNICORN export instead
records the *continuous total-protein UV trace* of the very same runs, so it carries
the real peak shape -- including the flanks the xlsx misses. This module turns those
exports into ``[time, concentration]`` curves comparable to the simulator's total,
which is the input the residual-correction work needs.

UNICORN CSV layout (verified on the HLXSYN/HLXSYN exports)::

    row 1: Chrom.1,,Chrom.1,,Chrom.1,,Chrom.1,,Chrom.1,   <- curve name, paired cols
    row 2: Cond,,Conc B,,pH,,UV 2nd,,Run Log,             <- signal names
    row 3: min,mS/cm,min,%,min,,min,mAU,min,Logbook       <- units
    row 4+: data ...

Each signal is a ``(time, value)`` column pair on *its own* time axis; when one
signal runs out of samples its cells go empty (NaN) while others continue, so each
pair is extracted and NaN-dropped independently. Columns are located by matching the
name/unit header rows rather than hard-coded indices, so a re-ordered export still
loads.

Beer-Lambert: ``c[g/L] = c[mg/mL] = A[AU] / (eps * L)`` with ``A = mAU/1000``,
``eps`` the product's extinction coefficient (mL/mg/cm, from ``消光系数.txt``) and
``L`` the flow-cell path length (0.042 cm, per ``单抗UV吸光值计算.docx``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

# Extinction coefficients (mL/mg/cm) from data/AKTA data/消光系数.txt.
EXTINCTION_ML_MG_CM: dict[str, float] = {'HLXSYN': 1.4}
# Flow-cell path length (cm) from data/AKTA data/单抗UV吸光值计算.docx.
PATH_LENGTH_CM: float = 0.042


def extinction_for(product_id: str) -> float:
    """Extinction coefficient (mL/mg/cm) for a known product id."""
    try:
        return EXTINCTION_ML_MG_CM[product_id]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"未知产品消光系数: {product_id}") from exc


def mau_to_g_l(mau: np.ndarray, eps_ml_mg_cm: float, path_length_cm: float = PATH_LENGTH_CM) -> np.ndarray:
    """Convert a UV trace in mAU to concentration in g/L via Beer-Lambert."""
    return (np.asarray(mau, dtype=float) / 1000.0) / (eps_ml_mg_cm * path_length_cm)


@dataclass
class AktaTrace:
    """One AKTA run: UV (g/L) plus the Conc-B and conductivity context curves.

    UV is baseline-subtracted (pre-gradient median removed). The Conc-B and Cond
    curves keep their own time axes; use :meth:`concB_at` / :meth:`cond_at` to sample
    them on the UV grid.
    """

    name: str
    uv_min: np.ndarray          # UV retention time (min, absolute method clock)
    uv_g_l: np.ndarray          # UV concentration (g/L), baseline-subtracted
    uv_mau: np.ndarray          # raw UV (mAU)
    concB_min: np.ndarray
    concB_pct: np.ndarray
    cond_min: np.ndarray
    cond_ms_cm: np.ndarray
    eps_ml_mg_cm: float
    baseline_g_l: float

    def concB_at(self, t_min: np.ndarray | float) -> np.ndarray:
        return np.interp(t_min, self.concB_min, self.concB_pct)

    def cond_at(self, t_min: np.ndarray | float) -> np.ndarray:
        return np.interp(t_min, self.cond_min, self.cond_ms_cm)

    def _in_gradient_mask(self, cb_lo: float = 5.0) -> np.ndarray:
        """Boolean mask over the UV grid for samples eluting *within* the gradient.

        The product elutes mid-gradient -- at a moderate %-gradient (above the
        pre-gradient hold, below the 99 % strip step) and below the salt-strip
        conductivity. The two non-product UV features (flow-through before the gradient,
        and the salt strip/CIP after it) sit outside this band: the strip in particular
        is reached only after the line resets to 0 % or steps to 100 %, so the moderate
        Conc-B condition excludes it even when its UV spike precedes the Cond rise.

        Conc B is median-smoothed first so a single-sample ramp-start spike to 100 %
        (seen in some HLXSYN exports) does not poison the band.
        """
        cb_s = median_filter(self.concB_pct, size=9) if self.concB_pct.size >= 9 else self.concB_pct
        cb_uv = np.interp(self.uv_min, self.concB_min, cb_s)
        cond_uv = np.interp(self.uv_min, self.cond_min, self.cond_ms_cm)
        pre = cb_s[self.concB_min < 10.0]
        hold = float(np.median(pre)) if pre.size else float(cb_s[0]) if cb_s.size else 0.0
        ingrad = (cb_uv > hold + cb_lo) & (cb_uv < 95.0)
        cond_strip = max(40.0, 2.5 * float(np.median(cond_uv[ingrad]))) if np.any(ingrad) else 40.0
        return (cb_uv > hold + cb_lo) & (cb_uv < 99.0) & (cond_uv < cond_strip)

    def product_peak(self) -> tuple[float, float]:
        """``(t_min, g_l)`` of the product peak (largest UV inside the gradient band)."""
        m = self._in_gradient_mask()
        if not np.any(m):
            i = int(np.argmax(self.uv_g_l))
            return float(self.uv_min[i]), float(self.uv_g_l[i])
        idx = np.where(m)[0]
        i = idx[int(np.argmax(self.uv_g_l[idx]))]
        return float(self.uv_min[i]), float(self.uv_g_l[i])

    def gradient_window(self) -> tuple[float, float]:
        """``(lo_min, hi_min)`` contiguous in-gradient interval around the product peak.

        This is the time span over which to compare the AKTA total against the simulated
        total: the rise, apex and tail of the product peak, bounded by the gradient
        reset / strip on either side.
        """
        m = self._in_gradient_mask()
        if not np.any(m):
            return float(self.uv_min[0]), float(self.uv_min[-1])
        idx = np.where(m)[0]
        apex = idx[int(np.argmax(self.uv_g_l[idx]))]
        valid = np.zeros(self.uv_min.shape, dtype=bool)
        valid[idx] = True
        lo = apex
        while lo - 1 >= 0 and valid[lo - 1]:
            lo -= 1
        hi = apex
        while hi + 1 < valid.size and valid[hi + 1]:
            hi += 1
        return float(self.uv_min[lo]), float(self.uv_min[hi])


def _read_unicorn_csv(path: Path) -> pd.DataFrame:
    """Read a UNICORN CSV (3 header rows) trying common encodings."""
    last: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, skiprows=3, header=None, dtype=str,
                               encoding=enc, engine="python", on_bad_lines="skip")
        except Exception as exc:  # noqa: BLE001 - try the next encoding
            last = exc
    raise ValueError(f"无法读取 AKTA CSV: {path}") from last


def _locate_columns(path: Path) -> dict[str, int]:
    """Map signal -> value-column index using the *units* header row (row 3).

    Each ``(time, value)`` pair has unit ``min`` on the time column and the signal unit
    on the value column, e.g. ``min,mS/cm,min,%,min,,min,mAU,...``. Matching by unit
    rather than name is robust to naming differences across exports -- the percent
    gradient is labelled ``Conc B`` in most files but ``% Cond`` in some HLXSYN runs,
    yet both carry unit ``%``. Returns value-column indices for UV(mAU), the percent
    gradient(%) and conductivity(mS/cm); the matching time column is ``value_index-1``.
    """
    head = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            head = pd.read_csv(path, header=None, nrows=3, dtype=str, encoding=enc,
                               engine="python", on_bad_lines="skip")
            break
        except Exception:  # noqa: BLE001
            continue
    if head is None:
        raise ValueError(f"无法读取 AKTA 表头: {path}")
    units = [str(x).strip().lower() if x is not None else "" for x in head.iloc[2].tolist()]
    out: dict[str, int] = {}
    for k, un in enumerate(units):
        if un == "mau" and "uv" not in out:
            out["uv"] = k
        elif un == "%" and "concB" not in out:           # percent gradient (Conc B / % Cond)
            out["concB"] = k
        elif "ms/cm" in un and "cond" not in out:        # conductivity
            out["cond"] = k
    return out


def load_akta_uv(
    path: str | Path,
    *,
    eps_ml_mg_cm: float,
    path_length_cm: float = PATH_LENGTH_CM,
    baseline_window_min: float = 10.0,
) -> AktaTrace:
    """Load one UNICORN export as an :class:`AktaTrace` (UV in g/L + Conc B + Cond).

    ``eps_ml_mg_cm`` is the product extinction coefficient (see
    :data:`EXTINCTION_ML_MG_CM` / :func:`extinction_for`). The UV baseline is the
    median g/L over the first ``baseline_window_min`` minutes (pre-gradient) and is
    subtracted from the returned ``uv_g_l``.
    """
    path = Path(path)
    cols = _locate_columns(path)
    if "uv" not in cols:
        raise ValueError(f"AKTA 文件未找到 UV(mAU) 列: {path}")
    raw = _read_unicorn_csv(path)

    def _pair(value_col: int) -> tuple[np.ndarray, np.ndarray]:
        t = pd.to_numeric(raw.iloc[:, value_col - 1], errors="coerce").to_numpy()
        v = pd.to_numeric(raw.iloc[:, value_col], errors="coerce").to_numpy()
        ok = np.isfinite(t) & np.isfinite(v)
        order = np.argsort(t[ok])
        return t[ok][order], v[ok][order]

    uv_min, uv_mau = _pair(cols["uv"])
    cb_min, cb_pct = _pair(cols["concB"]) if "concB" in cols else (uv_min.copy(), np.zeros_like(uv_min))
    cd_min, cd_ms = _pair(cols["cond"]) if "cond" in cols else (uv_min.copy(), np.zeros_like(uv_min))

    uv_g_l = mau_to_g_l(uv_mau, eps_ml_mg_cm, path_length_cm)
    base_mask = uv_min < baseline_window_min
    baseline = float(np.median(uv_g_l[base_mask])) if np.any(base_mask) else 0.0
    uv_g_l = uv_g_l - baseline

    return AktaTrace(
        name=path.name, uv_min=uv_min, uv_g_l=uv_g_l, uv_mau=uv_mau,
        concB_min=cb_min, concB_pct=cb_pct, cond_min=cd_min, cond_ms_cm=cd_ms,
        eps_ml_mg_cm=eps_ml_mg_cm, baseline_g_l=baseline,
    )
