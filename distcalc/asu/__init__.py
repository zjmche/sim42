"""Air Separation Unit (ASU) double-column + argon column distillation.

Three-column coupled system:
  Lower column  — high-pressure (~6 bar), separates compressed air
  Upper column  — low-pressure (~1.3 bar), produces N₂ and O₂ products
  Argon column  — crude-Ar side draw from upper column
  MCHE          — upper condenser = lower reboiler (thermal coupling)

New files only; original distcalc modules are imported read-only.
"""

from .specs import (
    ASUConfig,
    ASUResult,
    ASUStreams,
    MultiColumnConfig,
    SideDraw,
    StreamSpec,
)
from .system import solve_asu

__all__ = [
    "ASUConfig",
    "ASUResult",
    "ASUStreams",
    "MultiColumnConfig",
    "SideDraw",
    "StreamSpec",
    "solve_asu",
]
