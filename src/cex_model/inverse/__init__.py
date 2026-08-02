"""Inverse parameter-estimation network (experimental curve -> SMA par).

A neural network that maps an elution curve plus its known operating conditions
to mechanistic SMA parameters.  Used both as a standalone amortized estimator
and -- primarily -- to warm-start the mechanistic optimizer
(:func:`cex_model.fitting.fit_sma_parameters`) so the final parameters are still
guaranteed by the physics model.
"""

from cex_model.inverse.encoding import (
    InverseEncoder,
    param_names,
    params_table_to_vector,
    proteins_from_targets,
    vector_to_params_table,
)
from cex_model.inverse.model import load_inverse_artifact, save_inverse_artifact

__all__ = [
    "InverseEncoder",
    "param_names",
    "params_table_to_vector",
    "vector_to_params_table",
    "proteins_from_targets",
    "save_inverse_artifact",
    "load_inverse_artifact",
]
