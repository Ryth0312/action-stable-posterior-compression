"""Shared synthetic SMA bundle builder for the design / engine demos (P1).

De-duplicates the near-identical 1|2-component builders that were copy-pasted
across ``scripts/bayes_active_design.py`` and the bayes tests, and generalises
them to any number of components with optional realism knobs so the multi-component
design demo is not a toy:

  * ``fractions``  -- unequal feed fractions (a dominant main peak + minors);
  * ``keq_ladder`` -- a per-component keq ladder injected AFTER ``generic_components``
    (whose keq is identical across components), so the per-component keq<->nu
    degeneracy is *load-bearing* rather than a-priori separable;
  * ``nu`` / ``sigma`` -- explicit overrides (peak overlap is otherwise emergent
    from the default nu ladder).

Returns ``(bundle, components, u_true)`` so callers stop recomputing the ground
truth.  Pure construction -- no solver call.  The bundle is the duck-typed shape
(``column``/``components``/``correction``/``experiments``/``observation_groups``/
``product_id``) that ``run_adaptive_design`` / ``mechanistic_model`` already accept.
"""

from __future__ import annotations

import dataclasses
import types

import numpy as np

from cex_model.app_support import generic_components
from cex_model.bayes.prior import components_to_u
from cex_model.column import ColumnParameters
from cex_model.components import ComponentSet
from cex_model.corrections import LinearLoadingCorrection

__all__ = ["synthetic_sma_bundle"]


def _column() -> ColumnParameters:
    return ColumnParameters(grid_size=21, column_volume=1.0, flow_rate=2.0, column_length=50.0,
                            total_porosity=0.84, superficial_velocity=0.6, dax=0.12, ionic_capacity=0.398)


def synthetic_sma_bundle(n_comp: int, *, fractions=None, keq_ladder=None, nu=None, sigma=None,
                         loading: float = 10.0, gradient=(20.0, 80.0), elution_cv: float = 12.0,
                         product_id: str = "SYN"):
    """A synthetic per-product bundle + its ground-truth ``u`` (model units).

    ``keq_ladder=True`` injects a mild geometric keq ladder (table units) so the
    per-component keq<->nu degeneracy is genuine; pass an explicit length-``n_comp``
    sequence to set it.  ``nu`` / ``sigma`` accept a scalar or length-``n_comp``
    override.  Returns ``(bundle, components, u_true)``.
    """
    if n_comp < 1:
        raise ValueError("n_comp must be >= 1")
    if fractions is None:
        fractions = [100.0] if n_comp == 1 else list(np.full(n_comp, 100.0 / n_comp))
    if len(fractions) != n_comp:
        raise ValueError(f"fractions must have length n_comp={n_comp}, got {len(fractions)}")

    comps = generic_components(n_comp, list(fractions))

    # Optional per-component overrides. generic_components sets identical keq across
    # components, so without a ladder the keq<->nu degeneracy is a-priori separable.
    overrides: dict[str, np.ndarray] = {}
    if keq_ladder is not None and keq_ladder is not False:
        ladder = (np.geomspace(0.02, 0.13, n_comp) if keq_ladder is True
                  else np.asarray(keq_ladder, dtype=float))
        if ladder.size != n_comp:
            raise ValueError(f"keq_ladder must have length n_comp={n_comp}")
        overrides["keq"] = ladder  # table units (keq_array scales by 1e-2)
    if nu is not None:
        overrides["nu"] = np.broadcast_to(np.asarray(nu, float), (n_comp,)).copy()
    if sigma is not None:
        overrides["sigma"] = np.broadcast_to(np.asarray(sigma, float), (n_comp,)).copy()
    if overrides:
        new = [dataclasses.replace(c, **{k: float(v[j]) for k, v in overrides.items()})
               for j, c in enumerate(comps.components)]
        comps = ComponentSet(new, loading_correction=comps.loading_correction)

    exp0 = types.SimpleNamespace(loading_g_l=float(loading), gradient_start_pct=float(gradient[0]),
                                 gradient_end_pct=float(gradient[1]), elution_cv=float(elution_cv),
                                 buffer_a=0.02, buffer_b=0.5)
    bundle = types.SimpleNamespace(column=_column(), components=comps,
                                   correction=LinearLoadingCorrection(), experiments=[exp0],
                                   observation_groups=None, product_id=product_id)
    return bundle, comps, components_to_u(comps)
