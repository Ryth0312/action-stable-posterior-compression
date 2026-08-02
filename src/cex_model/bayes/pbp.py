"""Lightweight Yamamoto / parameter-by-parameter (PbP) baseline for nu, keq.

The PbP method (Chen, Yao & Lin, J. Chromatogr. A 2022 1680:463418 / 2023
1687:463655) determines the SMA parameters SEQUENTIALLY from a PRESCRIBED
experiment set: LR1/LR2 (the Yamamoto plot) get nu and keq from linear-gradient-
elution runs at SEVERAL gradient slopes, then LA->sigma, IM->kkin. It is fast,
deterministic, and identifiable BY DESIGN -- provided you can run the gradient-
slope ladder.

This implements the LR1/LR2 core (nu, keq) as a comparison baseline and --
crucially -- reports "not applicable" when the data lacks the slope ladder (e.g.
existing process data that only varied loading, like HLXSYN): exactly the regime
where a posterior + adaptive design is needed instead. The sigma/kkin PbP steps
are not reimplemented (cited as a simplified baseline).

Yamamoto relation (linear gradient, SDM): with normalized gradient slope GH and
peak-elution salt concentration c_s,R,
    log(GH) = (nu+1)*log(c_s,R) - log[ keq * Lambda^nu * (nu+1) ],
so a regression of log(GH) on log(c_s,R) gives nu (slope-1) and keq (intercept).
"""

from __future__ import annotations

import numpy as np

__all__ = ["yamamoto_nu_keq", "pbp_applicability"]


def pbp_applicability(op_matrix, *, min_slopes: int = 3) -> dict:
    """Can the PbP/Yamamoto LR1 (nu, keq) regression run on this experiment design?

    LR1 needs >= ``min_slopes`` DISTINCT normalized gradient slopes
    ``(gradient_end - gradient_start) / elution_cv`` across the experiments.
    Existing process data that only varied loading (e.g. HLXSYN/03/04) lacks this
    ladder -> not applicable, which is exactly the regime where a posterior +
    adaptive design is needed instead.

    ``op_matrix`` is ``(n_exp, 4)`` in OP_FEATURES order
    ``(loading_g_l, gradient_start_pct, gradient_end_pct, elution_cv)``.
    """
    M = np.asarray(op_matrix, float)
    if M.ndim != 2 or M.shape[1] < 4:
        raise ValueError("op_matrix must be (n_exp, 4) in OP_FEATURES order")
    slopes = (M[:, 2] - M[:, 1]) / np.clip(M[:, 3], 1e-9, None)
    n_distinct = int(np.unique(np.round(slopes, 6)).size)
    applicable = n_distinct >= min_slopes
    return {"applicable": applicable, "n_distinct_slopes": n_distinct,
            "slopes": [float(s) for s in slopes],
            "reason": ("" if applicable else
                       f"needs >= {min_slopes} distinct gradient slopes (the LGE ladder for LR1); "
                       f"got {n_distinct} -- the data did not vary gradient slope")}


def yamamoto_nu_keq(gradient_slopes, retention_salts, ionic_capacity: float, *, min_points: int = 3) -> dict:
    """Estimate ``(nu, keq)`` by the Yamamoto LR1/LR2 regression.

    ``gradient_slopes`` = normalized gradient slope GH per LGE; ``retention_salts``
    = peak-elution salt concentration c_s,R per LGE; ``ionic_capacity`` = Lambda.
    Returns ``{"applicable": False, ...}`` when fewer than ``min_points`` distinct
    slopes are present (no LGE ladder -> the PbP regression cannot run).
    """
    GH = np.asarray(gradient_slopes, float)
    cs = np.asarray(retention_salts, float)
    if GH.shape != cs.shape:
        raise ValueError("gradient_slopes and retention_salts must have the same shape")
    n_distinct = int(np.unique(np.round(GH, 9)).size)
    if n_distinct < min_points:
        return {"applicable": False, "n_distinct_slopes": n_distinct,
                "reason": f"needs >= {min_points} distinct gradient slopes (the LGE ladder); got {n_distinct}"}
    x, y = np.log(cs), np.log(GH)
    slope, intercept = np.polyfit(x, y, 1)
    nu = float(slope - 1.0)
    keq = float(np.exp(-intercept) / (ionic_capacity**nu * (nu + 1.0)))
    yhat = slope * x + intercept
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - yhat) ** 2)) / ss_tot if ss_tot > 0 else 0.0
    return {"applicable": True, "nu": nu, "keq": keq, "r2": r2, "n_distinct_slopes": n_distinct}
