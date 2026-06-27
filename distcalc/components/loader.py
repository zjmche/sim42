"""Component data loader. Reads components.json, validates with Pydantic,
builds the Mixture runtime object used by the thermo engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, field_validator, model_validator

_TREF = 298.15  # K — ideal-gas reference for H/S
_DATA_FILE = Path(__file__).parent / "components.json"


# ---------------------------------------------------------------------------
# Pydantic data models (pure validation / serialization)
# ---------------------------------------------------------------------------

class ShomateCoeffs(BaseModel):
    Tmin: float
    Tmax: float
    A: float
    B: float
    C: float
    D: float
    E: float

    def cp(self, T: float) -> float:
        """Ideal-gas Cp at T [J/(mol·K)] via Shomate equation."""
        t = T / 1000.0
        return self.A + self.B * t + self.C * t**2 + self.D * t**3 + self.E / t**2

    def delta_h(self, T: float) -> float:
        """∫(Tref→T) Cp dT [J/mol] relative to Tref=298.15 K."""
        def _sh(t: float) -> float:
            return self.A * t + self.B * t**2 / 2 + self.C * t**3 / 3 + self.D * t**4 / 4 - self.E / t
        t0 = _TREF / 1000.0
        return (_sh(T / 1000.0) - _sh(t0)) * 1000.0  # kJ/mol → J/mol

    def delta_s(self, T: float) -> float:
        """∫(Tref→T) Cp/T dT [J/(mol·K)]."""
        def _sh(t: float) -> float:
            return self.A * np.log(t) + self.B * t + self.C * t**2 / 2 + self.D * t**3 / 3 - self.E / (2 * t**2)
        t0 = _TREF / 1000.0
        return _sh(T / 1000.0) - _sh(t0)


class ComponentRecord(BaseModel):
    name: str
    formula: str
    CAS: str
    MW: float          # kg/mol
    Tc: float          # K
    Pc: float          # Pa
    omega: float
    shomate: list[ShomateCoeffs]

    @field_validator("shomate")
    @classmethod
    def _nonempty(cls, v: list) -> list:
        if not v:
            raise ValueError("shomate list must not be empty")
        return v

    def _get_shomate(self, T: float) -> ShomateCoeffs:
        for s in self.shomate:
            if s.Tmin <= T <= s.Tmax:
                return s
        # nearest-range extrapolation (handles cryogenic edge)
        return min(self.shomate, key=lambda s: min(abs(T - s.Tmin), abs(T - s.Tmax)))

    def h_ig(self, T: float) -> float:
        """Ideal-gas molar enthalpy relative to Tref [J/mol]."""
        return self._get_shomate(T).delta_h(T)

    def s_ig_ref(self, T: float) -> float:
        """Ideal-gas molar entropy change ∫(Tref→T) Cp/T dT [J/(mol·K)]."""
        return self._get_shomate(T).delta_s(T)

    def cp_ig(self, T: float) -> float:
        """Ideal-gas Cp [J/(mol·K)]."""
        return self._get_shomate(T).cp(T)


class KijRecord(BaseModel):
    citation: str
    names: list[str]
    matrix: list[list[float]]

    @model_validator(mode="after")
    def _check_symmetric(self) -> "KijRecord":
        m = np.array(self.matrix)
        if m.shape != (len(self.names), len(self.names)):
            raise ValueError("Kij matrix shape must match number of names")
        if not np.allclose(m, m.T):
            raise ValueError("Kij matrix must be symmetric")
        return self


class ComponentDatabase(BaseModel):
    _version: str = "1.0.0"
    components: list[ComponentRecord]
    Kij: KijRecord


# ---------------------------------------------------------------------------
# Runtime mixture object (used by thermo engine)
# ---------------------------------------------------------------------------

class Mixture:
    """Lightweight runtime container. Construct from load_mixture()."""

    def __init__(self, components: list[ComponentRecord], Kij: np.ndarray):
        self.components = components
        self.Kij = Kij            # (n, n) float64 ndarray
        self.n = len(components)
        self.names = [c.formula for c in components]
        self.MW = np.array([c.MW for c in components])  # kg/mol

    def __repr__(self) -> str:
        return f"Mixture({self.names})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_database(path: Path = _DATA_FILE) -> ComponentDatabase:
    raw: dict[str, Any] = json.loads(path.read_text())
    # Strip comment keys before passing to Pydantic
    clean = {k: v for k, v in raw.items() if not k.startswith("_")}
    return ComponentDatabase.model_validate(clean)


def load_mixture(
    formulas: list[str] | None = None,
    path: Path = _DATA_FILE,
) -> Mixture:
    """Return a Mixture for the given component formulas (default: all).

    Parameters
    ----------
    formulas:
        Ordered list of formula strings, e.g. ['N2', 'O2', 'Ar'].
        Order defines index 0, 1, 2 used everywhere else.
    path:
        Path to components.json (default: bundled file).
    """
    db = load_database(path)
    all_comps = {c.formula: c for c in db.components}
    kij_names = db.Kij.names
    full_Kij = np.array(db.Kij.matrix)

    if formulas is None:
        formulas = [c.formula for c in db.components]

    # Validate all requested formulas are in the database
    for f in formulas:
        if f not in all_comps:
            raise KeyError(f"Component '{f}' not found. Available: {list(all_comps)}")

    selected = [all_comps[f] for f in formulas]

    # Build submatrix of Kij in the requested order
    kij_idx = {name: i for i, name in enumerate(kij_names)}
    n = len(formulas)
    Kij_sub = np.zeros((n, n))
    for i, fi in enumerate(formulas):
        for j, fj in enumerate(formulas):
            if fi in kij_idx and fj in kij_idx:
                Kij_sub[i, j] = full_Kij[kij_idx[fi], kij_idx[fj]]

    return Mixture(selected, Kij_sub)
