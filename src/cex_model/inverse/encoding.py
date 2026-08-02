"""Encoding for the inverse (curve -> SMA parameters) network.

The inverse problem maps an experimental elution curve (plus the *known*
operating conditions it was measured under) to the mechanistic SMA parameters
``keq/kkin/nu/sigma`` of each protein component.  This module provides the
pure-numpy building blocks shared by training, prediction and evaluation:

* parameter packing/unpacking with a log10 transform on the cross-decade
  ``keq``/``kkin`` (so the regression is not dominated by the largest values),
  using the **same packed order as** :func:`cex_model.fitting._pack_params`
  (``[keq..., kkin..., nu..., sigma...]``) so a predicted vector can warm-start
  the mechanistic optimizer without reshuffling;
* curve encoding: per-channel standardization -> flatten -> PCA projection;
* low-dimensional peak descriptors (height / retention time / width per
  protein) that stay sharp where PCA tends to smooth the tall, narrow
  high-loading peaks;
* the operating-condition slice of the forward feature vector (the
  ``component_*`` entries are stripped because those are the prediction target).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Table rows 1..4 from ComponentSet.to_parameter_table() (row 0 is fraction).
PARAM_ROWS: tuple[str, ...] = ("keq", "kkin", "nu", "sigma")
# Parameters predicted/sampled in log10 space (they span several decades).
LOG_PARAMS: frozenset[str] = frozenset({"keq", "kkin"})
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Parameter <-> vector transforms
# --------------------------------------------------------------------------- #
def param_names(component_names: list[str]) -> list[str]:
    """Names aligned with :func:`params_table_to_vector` (ravel order)."""
    return [f"{p}_{c}" for p in PARAM_ROWS for c in component_names]


def log_mask(n_protein: int) -> np.ndarray:
    """Boolean mask over the packed vector marking log10-transformed entries."""
    rows = np.array([p in LOG_PARAMS for p in PARAM_ROWS], dtype=bool)
    return np.repeat(rows, n_protein)


def params_table_to_vector(table: np.ndarray) -> np.ndarray:
    """(5, n_protein) parameter table -> packed (4*n_protein,) transformed vector.

    Order matches ``table[1:5].ravel()`` = ``[keq.., kkin.., nu.., sigma..]``;
    ``keq``/``kkin`` are stored as ``log10``.
    """
    sub = np.array(table[1:5, :], dtype=float)
    sub[0] = np.log10(np.clip(sub[0], _EPS, None))  # keq
    sub[1] = np.log10(np.clip(sub[1], _EPS, None))  # kkin
    return sub.ravel()


def params_batch_to_vectors(tables: np.ndarray) -> np.ndarray:
    """Vectorized :func:`params_table_to_vector` over a (N, 5, n_protein) stack."""
    sub = np.array(tables[:, 1:5, :], dtype=float)
    sub[:, 0] = np.log10(np.clip(sub[:, 0], _EPS, None))
    sub[:, 1] = np.log10(np.clip(sub[:, 1], _EPS, None))
    return sub.reshape(sub.shape[0], -1)


def vector_to_params_table(vec: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Inverse of :func:`params_table_to_vector` -> (5, n_protein) table.

    ``fractions`` (length n_protein) fills row 0; it is held fixed (known feed
    composition, not predicted) to match the mechanistic optimizer.
    """
    fractions = np.asarray(fractions, dtype=float).ravel()
    n_protein = fractions.size
    sub = np.array(vec, dtype=float).reshape(4, n_protein)
    sub[0] = np.power(10.0, sub[0])  # keq
    sub[1] = np.power(10.0, sub[1])  # kkin
    table = np.zeros((5, n_protein), dtype=float)
    table[0] = fractions
    table[1:5] = sub
    return table


# --------------------------------------------------------------------------- #
# Curve encoding
#
# The curve encoder uses the PROTEIN channels only (g/L).  The salt outlet curve
# is dropped: it is effectively determined by the gradient (already carried by
# the condition vector) and, crucially, real fractionation data has no salt
# trace -- so proteins-only keeps synthetic training and real inference inputs
# on the same footing.
# --------------------------------------------------------------------------- #
def proteins_from_targets(targets: np.ndarray) -> np.ndarray:
    """Drop the salt channel: (N, n_time, 1+n_protein) -> (N, n_time, n_protein)."""
    return np.asarray(targets, dtype=float)[..., 1:]


def channel_stats(proteins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-protein mean/std over a (N, n_time, n_protein) stack."""
    mean = proteins.mean(axis=(0, 1))
    std = np.clip(proteins.std(axis=(0, 1)), 1e-8, None)
    return mean.astype(float), std.astype(float)


def normalize_curves(proteins: np.ndarray, chan_mean: np.ndarray, chan_std: np.ndarray) -> np.ndarray:
    """Per-protein standardize then flatten -> (N, n_time*n_protein)."""
    arr = np.asarray(proteins, dtype=float)
    norm = (arr - chan_mean) / chan_std
    return norm.reshape(arr.shape[0], -1)


def peak_feature_matrix(proteins: np.ndarray, output_time_s: np.ndarray) -> np.ndarray:
    """Per-protein [height, retention_time, width] -> (N, 3*n_protein).

    ``proteins`` is (N, n_time, n_protein) in g/L.  ``width`` is the
    concentration-weighted second moment (a robust, gradient-free peak width).
    """
    arr = np.asarray(proteins, dtype=float)
    t = np.asarray(output_time_s, dtype=float)
    n, _, n_protein = arr.shape
    feats = np.zeros((n, 3 * n_protein), dtype=float)
    for p in range(n_protein):
        col = arr[:, :, p]  # (N, n_time)
        idx = np.argmax(col, axis=1)
        feats[:, 3 * p + 0] = col[np.arange(n), idx]  # height
        feats[:, 3 * p + 1] = t[idx]                  # retention time
        mass = np.clip(col.sum(axis=1), _EPS, None)
        mean_t = (col * t).sum(axis=1) / mass
        var_t = (col * (t[None, :] - mean_t[:, None]) ** 2).sum(axis=1) / mass
        feats[:, 3 * p + 2] = np.sqrt(np.clip(var_t, 0.0, None))  # width
    return feats


def condition_indices(feature_names: list[str]) -> np.ndarray:
    """Indices of forward-feature entries that are *not* component parameters."""
    return np.array(
        [i for i, name in enumerate(feature_names) if not name.startswith("component_")],
        dtype=int,
    )


# --------------------------------------------------------------------------- #
# Self-contained encoder (stored in the model artifact)
# --------------------------------------------------------------------------- #
@dataclass
class InverseEncoder:
    """Deterministic curve+conditions -> normalized model input encoder.

    All statistics (channel norm, PCA basis, input z-score) are fit on the
    training split only and frozen here so prediction/evaluation reproduce the
    exact same encoding without re-fitting (no leakage).
    """

    output_time_s: np.ndarray
    n_protein: int
    cond_index: np.ndarray
    chan_mean: np.ndarray
    chan_std: np.ndarray
    pca_mean: np.ndarray            # (n_curve_flat,)
    pca_components: np.ndarray      # (K, n_curve_flat)
    x_mean: np.ndarray             # (input_dim,)
    x_std: np.ndarray
    use_peak_features: bool = True

    def raw_input_matrix(
        self, proteins: np.ndarray, features: np.ndarray
    ) -> np.ndarray:
        """Assemble un-normalized inputs for a batch (PCA + peaks + conditions).

        ``proteins`` is (N, n_time, n_protein) on :attr:`output_time_s`;
        ``features`` is the (N, n_feature) forward feature matrix (the
        condition slice is taken with :attr:`cond_index`).
        """
        flat = normalize_curves(proteins, self.chan_mean, self.chan_std)
        coeffs = (flat - self.pca_mean) @ self.pca_components.T
        blocks = [coeffs]
        if self.use_peak_features:
            blocks.append(peak_feature_matrix(proteins, self.output_time_s))
        blocks.append(np.asarray(features, dtype=float)[:, self.cond_index])
        return np.concatenate(blocks, axis=1)

    def encode(self, proteins: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Normalized model inputs for a batch -> (N, input_dim)."""
        raw = self.raw_input_matrix(proteins, features)
        return (raw - self.x_mean) / self.x_std

    def encode_single(self, protein_curve: np.ndarray, feature_vec: np.ndarray) -> np.ndarray:
        """Normalized model input for one curve -> (input_dim,).

        ``protein_curve`` is (n_time, n_protein) on :attr:`output_time_s`.
        """
        return self.encode(protein_curve[None, ...], feature_vec[None, :])[0]
