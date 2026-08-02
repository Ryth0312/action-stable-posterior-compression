"""CEX/IEC ion-exchange chromatography mechanistic model (EDM + SMA)."""

from cex_model.column import ColumnParameters
from cex_model.components import SMAComponent, ComponentSet
from cex_model.simulator import ChromatographySimulator, SimulationResult

__all__ = [
    "ColumnParameters",
    "SMAComponent",
    "ComponentSet",
    "ChromatographySimulator",
    "SimulationResult",
]

__version__ = "0.1.0"
