"""Error metrics for model validation and fitting."""

from __future__ import annotations

import numpy as np

from cex_model.sma import MOL_TO_G_L


def aggregate_observation_curve(curve: np.ndarray, observation_groups: list[list[int]] | None) -> np.ndarray:
    """Aggregate simulated protein columns according to observed peak groups."""
    if not observation_groups:
        return curve
    proteins = [curve[:, 2 + np.asarray(group, dtype=int)].sum(axis=1) for group in observation_groups]
    return np.column_stack([curve[:, 0], curve[:, 1], *proteins])


def peak_features(time_s: np.ndarray, conc: np.ndarray) -> dict[str, float]:
    """Peak retention, height, area and half-height width."""
    time_s = np.asarray(time_s, dtype=float)
    conc = np.maximum(np.asarray(conc, dtype=float), 0.0)
    if time_s.size == 0 or conc.size == 0 or float(np.max(conc)) <= 0.0:
        return {"tr_min": np.nan, "height_g_l": 0.0, "area": 0.0, "width_min": np.nan}
    idx = int(np.argmax(conc))
    height = float(conc[idx])
    half = 0.5 * height
    above = np.where(conc >= half)[0]
    width = float((time_s[above[-1]] - time_s[above[0]]) / 60.0) if above.size >= 2 else 0.0
    return {"tr_min": float(time_s[idx] / 60.0), "height_g_l": height,
            "area": float(np.trapezoid(conc, time_s)), "width_min": width}


def peak_shape_loss(curve: np.ndarray, exp_data: np.ndarray,
                    observation_groups: list[list[int]] | None = None) -> float:
    """Composite visual/process-oriented loss over peak time, height, area and width."""
    sim_obs = aggregate_observation_curve(curve, observation_groups)
    losses = []
    n_obs = exp_data.shape[1] - 1
    tr_scale = max(float(np.nanmax(exp_data[:, 0]) - np.nanmin(exp_data[:, 0])) / 60.0, 1.0)
    for j in range(n_obs):
        exp_f = peak_features(exp_data[:, 0], exp_data[:, 1 + j])
        sim_f = peak_features(sim_obs[:, 0], sim_obs[:, 2 + j])
        if exp_f["height_g_l"] <= 0.0 and exp_f["area"] <= 0.0:
            continue
        tr_err = 0.0 if np.isnan(exp_f["tr_min"]) or np.isnan(sim_f["tr_min"]) else (sim_f["tr_min"] - exp_f["tr_min"]) / tr_scale
        h_err = np.log((sim_f["height_g_l"] + 1e-9) / (exp_f["height_g_l"] + 1e-9))
        a_err = np.log((sim_f["area"] + 1e-9) / (exp_f["area"] + 1e-9))
        w_err = 0.0 if np.isnan(exp_f["width_min"]) or np.isnan(sim_f["width_min"]) else np.log((sim_f["width_min"] + 1e-6) / (exp_f["width_min"] + 1e-6))
        losses.append(2.0 * tr_err**2 + h_err**2 + 0.5 * a_err**2 + 0.25 * w_err**2)
    return float(np.sqrt(np.mean(losses))) if losses else 0.0


def retention_penalty(
    curve: np.ndarray,
    exp_data: np.ndarray,
    observation_groups: list[list[int]] | None = None,
    *,
    tol_min: float = 0.0,
) -> float:
    """Hinged squared retention-time error per observed peak (anchoring term).

    For each observed peak the error is ``max(|tr_sim - tr_exp| - tol, 0)``,
    normalized by the run length (so it is comparable to ``peak_shape_loss``'s
    retention term) and squared; the mean over observed peaks is returned. Inside
    ``tol_min`` minutes the peak is unpenalized (absorbs grid/feed_cv jitter);
    beyond it the penalty grows quadratically, which is what stops an optimizer
    from drifting a peak far from its experimental position to lower an aggregate
    loss. A *missing* simulated peak (NaN retention while the experiment has one)
    scores a full unit so a peak cannot simply be made to vanish.

    This penalizes only retention; height/area/width are left to the data loss.
    """
    sim_obs = aggregate_observation_curve(curve, observation_groups)
    n_obs = exp_data.shape[1] - 1
    tr_scale = max(float(np.nanmax(exp_data[:, 0]) - np.nanmin(exp_data[:, 0])) / 60.0, 1.0)
    pen, count = 0.0, 0
    for j in range(n_obs):
        exp_f = peak_features(exp_data[:, 0], exp_data[:, 1 + j])
        if exp_f["height_g_l"] <= 0.0 and exp_f["area"] <= 0.0:
            continue
        count += 1
        sim_f = peak_features(sim_obs[:, 0], sim_obs[:, 2 + j])
        if np.isnan(exp_f["tr_min"]) or np.isnan(sim_f["tr_min"]):
            pen += 1.0  # experimental peak present but simulated one absent
            continue
        d = max(abs(sim_f["tr_min"] - exp_f["tr_min"]) - tol_min, 0.0) / tr_scale
        pen += d * d
    return pen / count if count else 0.0


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; 0 for degenerate (near-constant or <2-point) inputs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    na = float(np.sqrt(a @ a))
    nb = float(np.sqrt(b @ b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a @ b) / (na * nb))


def shape_correlation_loss(
    curve: np.ndarray,
    exp_data: np.ndarray,
    observation_groups: list[list[int]] | None = None,
    *,
    max_shift_s: float = 180.0,
    n_shift: int = 61,
    shared_offset: bool = True,
    height_weight: float = 1.0,
) -> tuple[float, dict]:
    """Offset-tolerant peak-shape loss (CADET-Match style; Heymann/von Lieres 2022).

    For each observed peak the simulated curve is shifted in time by ``tau`` and the shape
    similarity is scored by Pearson correlation against the experiment, searched over a grid:
    ``shape_j = 1 - max_tau Pearson(exp_j, sim(t - tau))``. The shape term therefore ignores
    a pure time offset (pump delay / feed-timing / grid jitter) instead of penalising it, so
    a right-shape / slightly-mis-positioned candidate is no longer dominated by a wrong-shape
    / right-position one (which is how SSD/RMSE and the tr-term of :func:`peak_shape_loss`
    drive keq/nu unphysical). A separate log-height term keeps a right-shape / wrong-height
    peak penalised (Pearson is scale invariant). The chosen ``tau`` (minutes) is returned per
    peak as the "position" diagnostic; ``tau > 0`` means the model elutes EARLIER than the data
    (must be delayed by ``tau`` to align).

    ``shared_offset=True`` uses ONE tau per experiment (argmax of the present-peak-mean
    correlation -- a single physical pump-delay/clock cause, more constrained); ``False`` lets
    each peak pick its own tau (CADET-Match style; more flexible but can mask binding errors).

    Returns ``(loss, info)`` where ``loss`` is the mean over present peaks of
    ``(1 - pearson) + height_weight * log(h_sim/h_exp)^2`` (peaks with no experimental
    signal are skipped, as in :func:`peak_shape_loss`) and ``info`` has per-peak
    ``offset_min`` / ``pearson`` / ``shape``.
    """
    sim_obs = aggregate_observation_curve(curve, observation_groups)
    sim_t = sim_obs[:, 0]
    exp_t = exp_data[:, 0]
    n_obs = exp_data.shape[1] - 1
    taus = np.linspace(-max_shift_s, max_shift_s, max(int(n_shift), 1))

    present: list[int] = []
    corr_rows: list[np.ndarray] = []
    heights: list[tuple[float, float]] = []
    for j in range(n_obs):
        exp_j = np.maximum(exp_data[:, 1 + j], 0.0)
        exp_f = peak_features(exp_t, exp_j)
        if exp_f["height_g_l"] <= 0.0 and exp_f["area"] <= 0.0:
            continue  # no experimental peak -> no penalty (matches peak_shape_loss)
        sim_j = np.maximum(sim_obs[:, 2 + j], 0.0)
        sim_f = peak_features(sim_t, sim_j)
        corr = np.array([_pearson(exp_j, np.interp(exp_t - tau, sim_t, sim_j)) for tau in taus])
        present.append(j)
        corr_rows.append(corr)
        heights.append((sim_f["height_g_l"], exp_f["height_g_l"]))
    if not present:
        return 0.0, {"offset_min": [], "pearson": [], "shape": []}

    corr = np.vstack(corr_rows)  # (n_present, n_tau)
    if shared_offset:
        idx = np.full(len(present), int(np.argmax(corr.mean(axis=0))))
    else:
        idx = np.argmax(corr, axis=1)

    losses, offs, pears, shapes = [], [], [], []
    for r in range(len(present)):
        p = float(corr[r, idx[r]])
        shape = 1.0 - p
        sim_h, exp_h = heights[r]
        h_err = np.log((sim_h + 1e-9) / (exp_h + 1e-9))
        losses.append(shape + height_weight * h_err * h_err)
        offs.append(float(taus[idx[r]] / 60.0))
        pears.append(p)
        shapes.append(shape)
    loss = float(np.mean(losses))
    return loss, {"offset_min": offs, "pearson": pears, "shape": shapes}


def compute_rmse(
    curve: np.ndarray,
    exp_data: np.ndarray,
    lag_s: float = 0.0,
    min_conc_g_l: float = 0.3,
    observation_groups: list[list[int]] | None = None,
) -> np.ndarray:
    """Compute per-peak and total RMSE (MATLAB ``SMA.m``).

    Parameters
    ----------
    curve : array (n_time, 1 + 1 + n_protein) — time, salt, proteins in g/L
    exp_data : array (n_points, 1 + n_protein) — time, proteins in g/L
    lag_s : time lag subtracted from experimental times
    min_conc_g_l : ignore residuals where experimental conc below threshold

    Returns
    -------
    mse : array length n_protein + 1 (per-peak RMSE, then total RMSE)
    """
    curve = aggregate_observation_curve(curve, observation_groups)

    n_protein = exp_data.shape[1] - 1
    find_index = np.zeros(exp_data.shape[0], dtype=int)
    for j in range(exp_data.shape[0]):
        idx = np.where(curve[:, 0] > exp_data[j, 0] - lag_s)[0]
        find_index[j] = idx[0] if len(idx) else len(curve) - 1

    sim = curve[find_index, 2 : 2 + n_protein]
    exp = exp_data[:, 1:]
    denom = np.mean(exp, axis=0)
    denom = np.maximum(denom, 1e-12)
    error = (sim - exp) / denom
    mask = exp < min_conc_g_l
    error[mask] = 0.0

    mse_peak = np.sqrt(np.sum(error**2, axis=0) / error.shape[0])
    mse_total = np.sqrt(np.sum(error**2) / error.size)
    return np.append(mse_peak, mse_total)


def peak_retention_and_height(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Peak retention time (s) and height (g/L) from simulated curve."""
    tr, ht = [], []
    for i in range(2, curve.shape[1]):
        col = curve[:, i]
        idx = int(np.argmax(col))
        tr.append(curve[idx, 0])
        ht.append(col[idx])
    return np.array(tr), np.array(ht)


def delta_tr_height(
    sim_curve: np.ndarray, tr_exp: np.ndarray, height_exp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Retention and height deviations (MATLAB ``cal_delta_tr_delta_height.m``)."""
    tr_sim, ht_sim = peak_retention_and_height(sim_curve)
    return tr_sim - tr_exp, ht_sim - height_exp
