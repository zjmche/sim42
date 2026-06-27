"""Pydantic I/O schemas for MCP tools."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, field_validator, model_validator


class FlashInput(BaseModel):
    T: float                  # K
    P: float                  # Pa
    composition: list[float]  # mole fractions (will be normalised)
    components: list[str] = ["N2", "O2", "Ar"]
    tol: float = 1e-8
    max_iter: int = 200

    @field_validator("composition")
    @classmethod
    def _nonempty(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("composition must not be empty")
        return v


class FlashOutput(BaseModel):
    phase: str
    beta: float               # vapor fraction
    x: list[float]            # liquid mole fractions
    y: list[float]            # vapor mole fractions
    K: list[float]            # K-values
    converged: bool
    n_iter: int


class BubblePointInput(BaseModel):
    P: float                  # Pa (fixed)
    composition: list[float]  # liquid mole fractions
    components: list[str] = ["N2", "O2", "Ar"]
    T0: float | None = None   # initial guess [K]
    tol: float = 1e-7


class BubblePointOutput(BaseModel):
    T: float                  # bubble-point temperature [K]
    P: float                  # Pa
    x: list[float]            # liquid mole fractions (input)
    y: list[float]            # vapor mole fractions
    converged: bool
    n_iter: int


class FeedInput(BaseModel):
    stage: int
    flow: float               # mol/s
    composition: list[float]  # mole fractions
    T: float                  # K
    P: float                  # Pa
    q: float = 1.0            # feed quality


class ColumnInput(BaseModel):
    N_stages: int
    feed: FeedInput
    P_top: float              # Pa
    P_bot: float | None = None    # if None, same as P_top
    condenser_type: Literal["total", "partial"] = "total"
    distillate_rate: float = 1.0   # mol/s
    reflux_ratio: float = 3.0
    components: list[str] = ["N2", "O2", "Ar"]
    tol_T: float = 1e-4
    max_iter: int = 150


class ColumnOutput(BaseModel):
    converged: bool
    n_iter: int
    max_dT: float
    T_profile: list[float]         # K per stage
    V_profile: list[float]         # mol/s per stage
    L_profile: list[float]         # mol/s per stage
    x_distillate: list[float]
    x_bottoms: list[float]
    Q_condenser_kW: float
    Q_reboiler_kW: float
    components: list[str]
