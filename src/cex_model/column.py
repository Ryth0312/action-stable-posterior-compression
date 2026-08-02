"""Column geometry and transport parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ColumnParameters:
    """Physical column parameters with explicit units.

    Units
    -----
    column_volume : mL
    column_length : mm
    flow_rate : mL/min
    superficial_velocity : mm/s  (interstitial / linear velocity in bed)
    total_porosity : dimensionless (bed void fraction, eps_bed)
    particle_porosity : dimensionless (optional, for future extensions)
    dax : mm^2/s (axial dispersion coefficient)
    ionic_capacity : M (mol/L resin phase)
    particle_radius : mm (optional)
    dead_volume : mL
    grid_size : number of axial finite elements (default 51, matches MATLAB)
    """

    column_volume: float = 14.5  # mL
    column_length: float = 185.0  # mm
    flow_rate: float = 2.88  # mL/min
    total_porosity: float = 0.84
    superficial_velocity: float = 0.617  # mm/s
    dax: float = 0.085  # mm^2/s
    ionic_capacity: float = 0.398  # M
    particle_porosity: float = 0.5
    particle_radius: float = 0.045  # mm
    dead_volume: float = 0.71  # mL
    grid_size: int = 51

    @property
    def rt(self) -> float:
        """Residence time per column volume (1 CV), minutes."""
        return self.column_volume / self.flow_rate

    @property
    def velocity(self) -> float:
        """Interstitial velocity used in EDM, mm/s (matches MATLAB ``velocity``)."""
        return self.superficial_velocity

    @property
    def epsbed(self) -> float:
        """Alias for total porosity (MATLAB ``epsbed``)."""
        return self.total_porosity

    @property
    def inocap(self) -> float:
        """Alias for ionic capacity (MATLAB ``inocap``)."""
        return self.ionic_capacity

    @property
    def length(self) -> float:
        """Alias for column length (MATLAB ``length``)."""
        return self.column_length

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnParameters:
        """Build from YAML/JSON dict with flexible key names."""
        mapping = {
            "column_volume": data.get("column_volume", data.get("volume_ml")),
            "column_length": data.get("column_length", data.get("length_mm")),
            "flow_rate": data.get("flow_rate", data.get("velocity_ml_min")),
            "total_porosity": data.get("total_porosity", data.get("epsbed")),
            "superficial_velocity": data.get(
                "superficial_velocity", data.get("velocity_mm_s")
            ),
            "dax": data.get("dax", data.get("Dax")),
            "ionic_capacity": data.get("ionic_capacity", data.get("Lambda")),
            "particle_porosity": data.get("particle_porosity"),
            "particle_radius": data.get("particle_radius"),
            "dead_volume": data.get("dead_volume"),
            "grid_size": data.get("grid_size", 51),
        }
        clean = {k: v for k, v in mapping.items() if v is not None}
        return cls(**clean)
