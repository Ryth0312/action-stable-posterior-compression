"""Protein component definitions and SMA parameter containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np


class ComponentType(str, Enum):
    """Peak classification for downstream optimization."""

    ACID = "acid"
    MAIN = "main"
    BASIC = "basic"
    OTHER = "other"


@dataclass
class SMAComponent:
    """Steric Mass Action model parameters for one protein species.

    Units / scaling (MATLAB convention)
    -----------------------------------
    keq, kkin stored internally in model units:
        keq_model = keq_table * 1e-2
        kkin_model = kkin_table * 1e-5
    where table values are those shown in GUI / config YAML.

    nu : characteristic charge (dimensionless)
    keq : equilibrium constant (table units, scaled in simulator)
    sigma : steric shielding factor (dimensionless)
    kkin : kinetic constant (table units, scaled in simulator)
    fraction : mass fraction in feed (%), 0-100
    """

    name: str
    nu: float
    keq: float
    sigma: float
    kkin: float
    fraction: float = 25.0
    component_type: ComponentType = ComponentType.OTHER
    fit: bool = True

    def to_model_arrays(self) -> dict[str, float]:
        """Return parameters in internal model units."""
        return {
            "nu": self.nu,
            "keq": self.keq * 1e-2,
            "sigma": self.sigma,
            "kkin": self.kkin * 1e-5,
            "fraction": self.fraction,
        }


@dataclass
class ComponentSet:
    """Collection of SMA components (excluding salt)."""

    components: list[SMAComponent] = field(default_factory=list)
    # Optional per-product linear loading-correction coefficients {"a": [...], "b": [...]}.
    # When present (e.g. HLXSYN), make_loading_correction uses these instead of the
    # hard-coded HLXSYN defaults. Salt is index 0; a/b length = n_protein + 1.
    loading_correction: dict | None = None

    @property
    def n_protein(self) -> int:
        return len(self.components)

    @property
    def n_total(self) -> int:
        """Total species including salt (index 0)."""
        return self.n_protein + 1

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.components]

    def nu_array(self) -> np.ndarray:
        arr = np.zeros(self.n_total)
        for i, c in enumerate(self.components, start=1):
            arr[i] = c.nu
        return arr

    def keq_array(self) -> np.ndarray:
        arr = np.zeros(self.n_total)
        for i, c in enumerate(self.components, start=1):
            arr[i] = c.keq * 1e-2
        return arr

    def kkin_array(self) -> np.ndarray:
        arr = np.zeros(self.n_total)
        for i, c in enumerate(self.components, start=1):
            arr[i] = c.kkin * 1e-5
        return arr

    def sigma_array(self) -> np.ndarray:
        arr = np.zeros(self.n_total)
        for i, c in enumerate(self.components, start=1):
            arr[i] = c.sigma
        return arr

    def fraction_array(self) -> np.ndarray:
        return np.array([c.fraction for c in self.components])

    def to_parameter_table(self) -> np.ndarray:
        """Return 5 x n_protein table matching MATLAB GUI layout.

        Rows: fraction(%), keq, kkin, nu, sigma
        """
        n = self.n_protein
        table = np.zeros((5, n))
        for j, c in enumerate(self.components):
            table[0, j] = c.fraction
            table[1, j] = c.keq
            table[2, j] = c.kkin
            table[3, j] = c.nu
            table[4, j] = c.sigma
        return table

    @classmethod
    def from_parameter_table(
        cls,
        table: np.ndarray,
        names: Sequence[str] | None = None,
        types: Sequence[ComponentType] | None = None,
    ) -> ComponentSet:
        n = table.shape[1]
        if names is None:
            names = [f"C{j+1}" for j in range(n)]
        comps = []
        for j in range(n):
            ctype = types[j] if types else ComponentType.OTHER
            comps.append(
                SMAComponent(
                    name=names[j],
                    fraction=float(table[0, j]),
                    keq=float(table[1, j]),
                    kkin=float(table[2, j]),
                    nu=float(table[3, j]),
                    sigma=float(table[4, j]),
                    component_type=ctype,
                )
            )
        return cls(components=comps)

    @classmethod
    def from_dict_list(cls, items: list[dict[str, Any]]) -> ComponentSet:
        comps = []
        for item in items:
            ctype = ComponentType(item.get("type", "other"))
            comps.append(
                SMAComponent(
                    name=item["name"],
                    nu=float(item["nu"]),
                    keq=float(item["keq"]),
                    sigma=float(item["sigma"]),
                    kkin=float(item["kkin"]),
                    fraction=float(item.get("fraction", 25.0)),
                    component_type=ctype,
                    fit=bool(item.get("fit", True)),
                )
            )
        return cls(components=comps)
