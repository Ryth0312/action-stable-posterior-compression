"""Simulation backend interfaces."""

from __future__ import annotations

from typing import Any, Protocol


class SimulationBackend(Protocol):
    """Backend that can produce a simulator-compatible result."""

    def simulate(
        self,
        *,
        simulator: Any,
        inlet: Any,
        loading_g_l: float,
        n_time_points: int | None = None,
        method: str | None = None,
        t_start: float = 0.0,
    ) -> Any:
        """Return an object compatible with ``SimulationResult``."""
