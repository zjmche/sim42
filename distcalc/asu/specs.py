"""ASU-specific specification and result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..column.specs import ColumnResult, FeedSpec


# ---------------------------------------------------------------------------
# Multi-feed / side-draw column configuration
# ---------------------------------------------------------------------------

@dataclass
class SideDraw:
    """Side draw from a single column stage.

    phase="liquid" draws at the stage liquid composition x[j] and is
    subtracted from the descending L profile (as before).
    phase="vapor" draws at the stage vapor composition y[j] and is
    subtracted from the *ascending* V profile instead — used e.g. for a
    "waste GAN" (gaseous nitrogen) vent a few stages below the top of the
    upper column, distinct from the high-purity N2 distillate.
    """
    stage: int      # 1-based stage index
    flow: float     # molar flow rate [mol/s]
    phase: str = "liquid"


@dataclass
class MultiColumnConfig:
    """Column configuration with multiple feed stages and optional side draws.

    Extends the single-feed concept from ColumnConfig to support:
      - Multiple FeedSpec objects on different stages
      - Liquid side draws at specified stages

    Stage numbering: 1 = condenser (top), N = reboiler (bottom).
    """
    N_stages: int
    feeds: list[FeedSpec]           # one or more feeds (1-based stage indices)
    P_profile: np.ndarray           # pressure [Pa], shape (N,)
    condenser_type: str = "total"   # "total" or "partial"
    distillate_rate: float = 1.0    # D [mol/s]
    reflux_ratio: float = 3.0       # L/D
    bottoms_rate: float | None = None
    side_draws: list[SideDraw] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.P_profile) != self.N_stages:
            raise ValueError("P_profile length must equal N_stages")
        if not self.feeds:
            raise ValueError("At least one feed is required")
        total_feed = sum(f.flow for f in self.feeds)
        if self.bottoms_rate is None:
            S_total = sum(sd.flow for sd in self.side_draws)
            self.bottoms_rate = total_feed - self.distillate_rate - S_total
        if self.bottoms_rate < 0:
            raise ValueError("Bottoms rate < 0: check D, side draws, and feed flow")

    @property
    def D(self) -> float:
        return self.distillate_rate

    @property
    def B(self) -> float:
        return self.bottoms_rate  # type: ignore[return-value]

    @property
    def RR(self) -> float:
        return self.reflux_ratio

    @property
    def total_feed_flow(self) -> float:
        return sum(f.flow for f in self.feeds)

    @property
    def z_feed_avg(self) -> np.ndarray:
        """Flow-weighted average feed composition."""
        F_total = self.total_feed_flow
        z = np.zeros(len(self.feeds[0].z))
        for f in self.feeds:
            z += f.flow / F_total * np.asarray(f.z)
        return z


# ---------------------------------------------------------------------------
# Stream connectors between ASU columns
# ---------------------------------------------------------------------------

@dataclass
class StreamSpec:
    """A process stream connecting ASU columns."""
    flow: float           # molar flow [mol/s]
    z: np.ndarray         # mole fractions [N2, O2, Ar]
    T: float              # temperature [K]
    P: float              # pressure [Pa]
    phase: str = "liquid" # "liquid", "vapor", or "mixed"

    def __repr__(self) -> str:
        comp = ", ".join(f"{v:.4f}" for v in self.z)
        return (
            f"StreamSpec(flow={self.flow:.4f} mol/s, "
            f"z=[{comp}], T={self.T:.2f} K, P={self.P/1e5:.2f} bar, {self.phase})"
        )


# ---------------------------------------------------------------------------
# ASU top-level configuration
# ---------------------------------------------------------------------------

@dataclass
class ASUConfig:
    """Configuration for a complete double-column ASU with argon side draw.

    Process overview
    ----------------
    Compressed air enters the LOWER (HP) column where it pre-separates into:
      - N₂-rich vapour overhead → condensed in MCHE → provides lower reflux
        and liquid N₂ feed to the top of the UPPER (LP) column
      - Crude liquid O₂ bottoms → J-T expanded → mid-column feed to UPPER column

    The MCHE couples: Q_cond_lower ≅ Q_reb_upper  (thermal duty balance).

    The UPPER column produces:
      - Pure N₂ overhead (distillate)
      - Pure O₂ bottoms
      - Crude Ar side draw (from stage of peak Ar concentration)

    The ARGON column refines the crude Ar side draw.
    """
    # ------------------------------------------------------------------
    # Air feed
    # ------------------------------------------------------------------
    air_flow: float = 1.0                # total air molar flow [mol/s]
    z_air: Optional[np.ndarray] = None   # [N2, O2, Ar] — default: dry air
    T_air: float = 100.0                 # K (after main heat exchanger cooling)
    P_air: float = 6.0e5                 # Pa (HP compression, ~6 bar)

    # ------------------------------------------------------------------
    # Lower column (HP)
    # ------------------------------------------------------------------
    P_lower: float = 6.0e5     # HP column pressure [Pa]
    N_lower: int = 25          # number of equilibrium stages
    feed_stage_lower: int = 13 # 1-based air feed stage
    RR_lower: float = 2.5      # reflux ratio
    D_frac_lower: float = 0.50 # distillate fraction of air feed (N₂-rich overhead)

    # ------------------------------------------------------------------
    # Upper column (LP)
    # ------------------------------------------------------------------
    P_upper: float = 1.3e5    # LP column pressure [Pa]
    # Equilibrium stages. Real LP columns run ~60-100 trays; this needs to be
    # high enough that the rectifying section below the crude-O2 feed can
    # strip N2 down to trace (~ppm) levels before the Ar side draw, which a
    # 40-stage column cannot do (see ar_draw_n2_ppm_target below).
    N_upper: int = 60
    RR_upper: float = 3.0     # reflux ratio
    D_frac_upper: float = 0.65  # high-purity N₂ distillate fraction of TOTAL upper feed
    # Feed stage placement (1-based)
    n2_feed_stage_upper: int = 2    # liquid N₂ from lower distillate enters near top
    co2_feed_stage_upper: int = 0   # crude O₂ from lower bottoms (0 → auto: 2/3 from top)

    # Argon side draw
    ar_draw_flow: float = 0.0       # mol/s (0 → auto-estimated from Ar balance)
    ar_draw_stage: int = 0          # 1-based (0 → auto: see ar_draw_n2_ppm_target)
    # The draw stage is NOT the stage of maximum Ar concentration — N2 there
    # is still far too contaminated. Real plants draw a few stages further
    # down, where N2 has rectified out to a trace level, accepting a lower
    # Ar fraction in trade. Auto-stage picks the shallowest stage where the
    # liquid N2 mole fraction first drops to/below this ppm target.
    ar_draw_n2_ppm_target: float = 150.0

    # ------------------------------------------------------------------
    # Waste GAN (gaseous nitrogen) vent — second, lower-purity N₂ product
    # ------------------------------------------------------------------
    # Real double-column ASUs withdraw a second nitrogen stream as VAPOR a
    # few stages below the high-purity top product. This relieves the
    # rectifying section above the Ar peak from having to drive N2 recovery
    # to ~100%, which otherwise pushes N2 down into the Ar side-draw region.
    # The waste GAN is vented to atmosphere via the main heat exchanger
    # (regenerating the air dryer mol-sieve beds and, in older plants,
    # cooling DCAC water) rather than sold as product.
    waste_gan_stage: int = 0        # 1-based (0 → auto: a few stages below top)
    waste_gan_flow: float = 0.0     # mol/s (0 → auto: fraction of upper feed)
    waste_gan_frac: float = 0.13    # used only when waste_gan_flow == 0

    # ------------------------------------------------------------------
    # Argon column
    # ------------------------------------------------------------------
    N_argon: int = 170          # equilibrium stages (Ar/O2 α≈1.07, needs many stages)
    # Reflux ratio. With the N2-ppm-targeted side-draw stage (see
    # ar_draw_n2_ppm_target above), the crude Ar feed lands richer in Ar
    # (~17% vs the ~8% feed this was originally tuned against), but the
    # Ar/O2 cut still needs much more internal reflux than that to reach
    # high purity — RR=5 only reached ~53% Ar in the distillate against
    # this feed; RR=20 reaches ~98%.
    RR_argon: float = 20.0      # reflux ratio
    # Crude Ar distillate fraction. A high distillate fraction can't reach
    # high purity (mass balance caps purity at ~z_Ar_feed/D_frac_argon).
    # D_frac_argon well below z_Ar_feed is needed to concentrate the dilute
    # Ar into a ~97%+ purity cut.
    D_frac_argon: float = 0.07

    # ------------------------------------------------------------------
    # Solver / MCHE coupling
    # ------------------------------------------------------------------
    max_outer_iter: int = 30   # outer MCHE coupling iterations
    tol_duty: float = 0.02     # relative duty imbalance tolerance

    def __post_init__(self) -> None:
        if self.z_air is None:
            self.z_air = np.array([0.7812, 0.2096, 0.0092])
        self.z_air = np.asarray(self.z_air, dtype=float)
        self.z_air /= self.z_air.sum()

        if self.co2_feed_stage_upper == 0:
            # Default: feed crude O₂ about 2/3 down the upper column
            self.co2_feed_stage_upper = max(2, int(self.N_upper * 2 // 3))

        if self.ar_draw_flow == 0.0:
            # Estimate: draw enough crude side-draw flow to capture ~all the
            # feed Ar, given the draw point sits at the real-plant Ar-peak
            # composition (~90% O2 / ~10% Ar / trace N2) rather than at
            # high purity — the crude Ar column then refines this further,
            # capping overall Ar recovery well below 100%.
            if len(self.z_air) > 2 and self.z_air[2] > 0:
                self.ar_draw_flow = float(self.air_flow * self.z_air[2] / 0.10)
            else:
                self.ar_draw_flow = 0.0

        if self.waste_gan_stage == 0:
            # Default: a few stages below the high-purity N2 top product,
            # above the Ar-peak region — typical industrial placement.
            self.waste_gan_stage = max(2, min(self.N_upper // 8, 6))

        if self.waste_gan_flow == 0.0 and self.waste_gan_frac > 0.0:
            total_upper_feed = self.air_flow  # ~equal to total upper feed (N2+crude O2 from lower col)
            self.waste_gan_flow = float(total_upper_feed * self.waste_gan_frac)


# ---------------------------------------------------------------------------
# ASU intermediate streams (populated during solve)
# ---------------------------------------------------------------------------

@dataclass
class ASUStreams:
    """Intermediate process streams in the ASU solve."""
    # Lower column outputs
    lower_distillate: Optional[StreamSpec] = None   # N₂-rich top product
    lower_bottoms: Optional[StreamSpec] = None      # crude O₂ bottom product

    # Upper column inputs (after pressure reduction)
    upper_n2_feed: Optional[StreamSpec] = None      # liquid N₂ from lower top
    upper_co2_feed: Optional[StreamSpec] = None     # crude O₂ from lower bottom

    # Upper column outputs
    upper_distillate: Optional[StreamSpec] = None   # pure N₂ product
    upper_bottoms: Optional[StreamSpec] = None      # pure O₂ product
    upper_ar_draw: Optional[StreamSpec] = None      # crude Ar side draw
    waste_gan: Optional[StreamSpec] = None          # vented low-purity N₂ (vapor)

    # Argon column outputs
    ar_distillate: Optional[StreamSpec] = None      # pure / crude Ar product
    ar_bottoms: Optional[StreamSpec] = None         # O₂-rich waste

    # MCHE
    Q_mche: float = 0.0          # MCHE duty [W] (positive = heat transferred)
    duty_imbalance: float = 0.0  # |Q_cond_upper - Q_reb_lower| [W]


# ---------------------------------------------------------------------------
# Top-level ASU result
# ---------------------------------------------------------------------------

@dataclass
class ASUResult:
    """Complete ASU calculation output."""
    streams: ASUStreams
    lower_col: Optional[ColumnResult] = None
    upper_col: Optional[ColumnResult] = None
    argon_col: Optional[ColumnResult] = None

    converged: bool = False
    n_outer_iter: int = 0
    duty_imbalance: float = float("inf")  # [W]

    N2_purity: float = 0.0      # mole fraction in upper distillate
    O2_purity: float = 0.0      # mole fraction in upper bottoms
    Ar_purity: float = 0.0      # mole fraction in Ar column distillate

    N2_recovery: float = 0.0    # fraction of feed N₂ in N₂ product
    O2_recovery: float = 0.0    # fraction of feed O₂ in O₂ product
    Ar_recovery: float = 0.0    # fraction of feed Ar in Ar column distillate

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  ASU DOUBLE-COLUMN + ARGON COLUMN RESULT",
            "=" * 55,
            f"  Converged          : {self.converged}",
            f"  Outer iterations   : {self.n_outer_iter}",
            f"  MCHE duty imbalance: {self.duty_imbalance/1e3:.2f} kW",
            "",
            f"  N₂ purity (upper distillate) : {self.N2_purity*100:.3f}%",
            f"  O₂ purity (upper bottoms)    : {self.O2_purity*100:.3f}%",
            f"  Ar purity (Ar col distillate): {self.Ar_purity*100:.3f}%",
            "",
            f"  N₂ recovery : {self.N2_recovery*100:.1f}%",
            f"  O₂ recovery : {self.O2_recovery*100:.1f}%",
            f"  Ar recovery : {self.Ar_recovery*100:.1f}%",
        ]
        if self.streams.waste_gan:
            lines.append(f"  Waste GAN   : {self.streams.waste_gan!r}")
        if self.streams.Q_mche:
            lines.append(f"  MCHE duty   : {self.streams.Q_mche/1e3:.1f} kW")
        if self.lower_col:
            lines += ["", "[Lower Column]", self.lower_col.summary()]
        if self.upper_col:
            lines += ["", "[Upper Column]", self.upper_col.summary()]
        if self.argon_col:
            lines += ["", "[Argon Column]", self.argon_col.summary()]
        return "\n".join(lines)
