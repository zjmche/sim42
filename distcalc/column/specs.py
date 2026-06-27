"""Column configuration and specification objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class FeedSpec:
    stage: int           # 1-based stage index (1=condenser)
    flow: float          # total molar feed rate [mol/s]
    z: np.ndarray        # feed mole fractions
    T: float             # feed temperature [K]
    P: float             # feed pressure [Pa]
    q: float = 1.0       # feed quality (1=sat. liquid, 0=sat. vapor)


@dataclass
class ColumnConfig:
    """Complete specification of a single-feed equilibrium-stage column.

    Stage numbering: 1 = condenser (or top stage), N = reboiler.
    For a total condenser: stage 1 is the condenser, equilibrium stages
    are 2..N-1, and the partial reboiler is stage N.
    """
    N_stages: int
    feed: FeedSpec
    P_profile: np.ndarray          # pressure [Pa] on each stage, shape (N,)
    condenser_type: Literal["total", "partial"] = "total"
    distillate_rate: float = 1.0   # D [mol/s]
    reflux_ratio: float = 3.0      # L/D
    bottoms_rate: float | None = None   # if None, computed from feed and D

    def __post_init__(self) -> None:
        if self.P_profile is None or len(self.P_profile) != self.N_stages:
            raise ValueError("P_profile must have length N_stages")
        if self.bottoms_rate is None:
            self.bottoms_rate = self.feed.flow - self.distillate_rate
        if self.bottoms_rate < 0:
            raise ValueError("Bottoms rate < 0: check D and feed flow")

    @property
    def D(self) -> float:
        return self.distillate_rate

    @property
    def B(self) -> float:
        return self.bottoms_rate   # type: ignore[return-value]

    @property
    def RR(self) -> float:
        return self.reflux_ratio


@dataclass
class ColumnResult:
    """Output from a MESH solver iteration."""
    T: np.ndarray        # stage temperatures [K], shape (N,)
    V: np.ndarray        # vapor flow from each stage [mol/s], shape (N,)
    L: np.ndarray        # liquid flow from each stage [mol/s], shape (N,)
    x: np.ndarray        # liquid mole fractions, shape (n_comp, N)
    y: np.ndarray        # vapor mole fractions, shape (n_comp, N)
    K: np.ndarray        # K-values, shape (n_comp, N)
    H_L: np.ndarray      # liquid molar enthalpy [J/mol], shape (N,)
    H_V: np.ndarray      # vapor molar enthalpy [J/mol], shape (N,)
    Q_condenser: float   # condenser duty [W] (negative = heat removal)
    Q_reboiler: float    # reboiler duty [W] (positive = heat addition)
    converged: bool
    n_iter: int
    max_dT: float        # max |ΔT| at convergence [K]

    @property
    def x_distillate(self) -> np.ndarray:
        return self.x[:, 0]

    @property
    def x_bottoms(self) -> np.ndarray:
        return self.x[:, -1]

    def summary(self) -> str:
        lines = [
            f"Converged: {self.converged}  iterations: {self.n_iter}  max|ΔT|: {self.max_dT:.4f} K",
            f"Q_cond: {self.Q_condenser/1e3:.3f} kW    Q_reb: {self.Q_reboiler/1e3:.3f} kW",
            f"Distillate: {self.x_distillate}",
            f"Bottoms:    {self.x_bottoms}",
        ]
        return "\n".join(lines)
