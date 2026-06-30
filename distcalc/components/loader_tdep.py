"""Loader for the T-dependent PR-EOS component database
(components_tdep.json). Parallel to loader.py — does not modify it.

Builds a MixtureTD runtime object whose Kij is a function of T,
Kij(T) = a + b/T, instead of the constant matrix used by Mixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .loader import ComponentRecord

_DATA_FILE = Path(__file__).parent / "components_tdep.json"


# ---------------------------------------------------------------------------
# Runtime mixture object (T-dependent Kij)
# ---------------------------------------------------------------------------

class MixtureTD:
    """Lightweight runtime container with temperature-dependent Kij(T)=a+b/T.

    Construct from load_mixture_tdep().
    """

    def __init__(self, components: list[ComponentRecord], kij_ab: np.ndarray):
        self.components = components
        self._kij_ab = kij_ab     # (n, n, 2) float64 ndarray: [...,0]=a, [...,1]=b
        self.n = len(components)
        self.names = [c.formula for c in components]
        self.MW = np.array([c.MW for c in components])  # kg/mol

    def Kij_matrix(self, T: float) -> np.ndarray:
        """Kij(T) = a + b/T for every pair, shape (n,n), zero diagonal."""
        a = self._kij_ab[:, :, 0]
        b = self._kij_ab[:, :, 1]
        return a + b / T

    def dKij_dT_matrix(self, T: float) -> np.ndarray:
        """d[Kij(T)]/dT = -b/T², shape (n,n), zero diagonal."""
        b = self._kij_ab[:, :, 1]
        return -b / T**2

    def __repr__(self) -> str:
        return f"MixtureTD({self.names})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mixture_tdep(
    formulas: list[str] | None = None,
    path: Path = _DATA_FILE,
) -> MixtureTD:
    """Return a MixtureTD for the given component formulas (default: all).

    Parameters
    ----------
    formulas:
        Ordered list of formula strings, e.g. ['N2', 'O2', 'Ar'].
        Order defines index 0, 1, 2 used everywhere else.
    path:
        Path to components_tdep.json (default: bundled file).
    """
    raw: dict[str, Any] = json.loads(path.read_text())
    all_comps = {c["formula"]: ComponentRecord.model_validate(c) for c in raw["components"]}
    pairs: dict[str, list[float]] = raw["Kij_params"]["pairs"]

    if formulas is None:
        formulas = list(all_comps.keys())

    for f in formulas:
        if f not in all_comps:
            raise KeyError(f"Component '{f}' not found. Available: {list(all_comps)}")

    selected = [all_comps[f] for f in formulas]

    n = len(formulas)
    kij_ab = np.zeros((n, n, 2))
    for i, fi in enumerate(formulas):
        for j, fj in enumerate(formulas):
            if i == j:
                continue
            key = f"{fi}-{fj}"
            rkey = f"{fj}-{fi}"
            if key in pairs:
                kij_ab[i, j] = pairs[key]
            elif rkey in pairs:
                kij_ab[i, j] = pairs[rkey]
            # else: leave as [0, 0] (no fitted data for this pair)

    return MixtureTD(selected, kij_ab)
