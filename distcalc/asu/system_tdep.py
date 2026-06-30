"""Top-level ASU system solver.

Orchestrates three coupled columns:
  1. Lower column (HP)   — separates compressed air
  2. Upper column (LP)   — produces N₂ and O₂ (multi-feed + Ar side draw)
  3. Argon column        — refines crude Ar side draw

MCHE coupling constraint: Q_cond_lower ≈ Q_reb_upper
Outer iterations adjust upper column distillate rate to satisfy this.
"""

from __future__ import annotations

import numpy as np

from ..column.specs import FeedSpec, ColumnResult
from ..components.loader_tdep import MixtureTD
from ..equilibrium.bubble_dew_tdep import bubble_T as _bubble_T
from .mesh_asu_tdep import find_ar_draw_stage, find_ar_draw_stage_near, solve_bp_asu
from .specs import (
    ASUConfig,
    ASUResult,
    ASUStreams,
    MultiColumnConfig,
    SideDraw,
    StreamSpec,
)

# Re-export standard single-feed solver for lower/argon columns
from ..column.mesh_bp_tdep import solve_bp
from ..column.specs import ColumnConfig


# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------

def _bubble_T_safe(P: float, z: np.ndarray, mix: MixtureTD, T0: float = 90.0) -> float:
    """Bubble temperature with fallback."""
    try:
        return float(_bubble_T(P, z, mix, T0=T0).T)
    except Exception:
        return T0


def _column_result_to_stream(
    res: ColumnResult,
    flow: float,
    P: float,
    end: str = "top",  # "top" = distillate, "bottom" = bottoms
) -> StreamSpec:
    """Extract a StreamSpec from a ColumnResult."""
    if end == "top":
        z = res.x_distillate.copy()
        T = float(res.T[0])
    else:
        z = res.x_bottoms.copy()
        T = float(res.T[-1])
    z = np.clip(z, 0.0, 1.0)
    z /= z.sum()
    return StreamSpec(flow=flow, z=z, T=T, P=P, phase="liquid")


def _expand_stream(stream: StreamSpec, P_new: float, mix: MixtureTD) -> StreamSpec:
    """J-T expand stream to new pressure (isenthalpic, approximate as isocomposition).

    For cryogenic streams the temperature change is small; we recalculate the
    bubble point at the new pressure to get the new T.
    """
    T_new = _bubble_T_safe(P_new, stream.z, mix, T0=stream.T)
    return StreamSpec(
        flow=stream.flow,
        z=stream.z.copy(),
        T=T_new,
        P=P_new,
        phase="liquid",
    )


# ---------------------------------------------------------------------------
# Lower column (HP)
# ---------------------------------------------------------------------------

def _solve_lower_column(cfg: ASUConfig, mix: MixtureTD) -> ColumnResult:
    """Solve the HP lower column: compressed air → N₂-rich top + crude O₂ bottom."""
    N  = cfg.N_lower
    P  = cfg.P_lower
    D  = cfg.air_flow * cfg.D_frac_lower
    B  = cfg.air_flow - D

    T_feed = _bubble_T_safe(P, cfg.z_air, mix)
    feed = FeedSpec(
        stage=cfg.feed_stage_lower,
        flow=cfg.air_flow,
        z=cfg.z_air.copy(),
        T=T_feed,
        P=P,
        q=1.0,   # saturated liquid at HP
    )
    col_cfg = ColumnConfig(
        N_stages=N,
        feed=feed,
        P_profile=np.full(N, P),
        condenser_type="total",
        distillate_rate=D,
        reflux_ratio=cfg.RR_lower,
    )
    return solve_bp(col_cfg, mix, max_iter=300, n_cmo_iter=60)


# ---------------------------------------------------------------------------
# Upper column (LP, multi-feed + Ar side draw)
# ---------------------------------------------------------------------------

def _build_upper_config(
    cfg: ASUConfig,
    lower_res: ColumnResult,
    D_upper: float,
    ar_draw_flow: float,
    ar_draw_stage: int,
    mix: MixtureTD,
    ar_bottoms_recycle: "StreamSpec | None" = None,
    waste_gan_stage: int = 0,
    waste_gan_flow: float = 0.0,
) -> MultiColumnConfig:
    """Build the MultiColumnConfig for the LP upper column."""
    N  = cfg.N_upper
    P  = cfg.P_upper
    D_lower = cfg.air_flow * cfg.D_frac_lower
    B_lower = cfg.air_flow - D_lower

    # Feed 1: liquid N₂ from lower distillate, expanded to LP
    z_n2 = np.clip(lower_res.x_distillate.copy(), 1e-10, 1.0)
    z_n2 /= z_n2.sum()
    T_n2 = _bubble_T_safe(P, z_n2, mix, T0=lower_res.T[0])
    feed_n2 = FeedSpec(
        stage=cfg.n2_feed_stage_upper,
        flow=D_lower,
        z=z_n2,
        T=T_n2,
        P=P,
        q=1.0,
    )

    # Feed 2: crude O₂ from lower bottoms, J-T expanded to LP
    z_co2 = np.clip(lower_res.x_bottoms.copy(), 1e-10, 1.0)
    z_co2 /= z_co2.sum()
    T_co2 = _bubble_T_safe(P, z_co2, mix, T0=lower_res.T[-1])
    feed_co2 = FeedSpec(
        stage=cfg.co2_feed_stage_upper,
        flow=B_lower,
        z=z_co2,
        T=T_co2,
        P=P,
        q=0.95,   # mostly liquid after J-T expansion
    )

    # Build feed list; optionally include O2-rich Ar-column bottoms recycle
    feeds: list[FeedSpec] = [feed_n2, feed_co2]
    if ar_bottoms_recycle is not None and ar_draw_stage > 0:
        recycle_stage = min(ar_draw_stage, cfg.N_upper)
        feed_recycle = FeedSpec(
            stage=recycle_stage,
            flow=ar_bottoms_recycle.flow,
            z=ar_bottoms_recycle.z.copy(),
            T=ar_bottoms_recycle.T,
            P=P,
            q=1.0,
        )
        feeds.append(feed_recycle)

    side_draws: list[SideDraw] = []
    if ar_draw_flow > 0 and ar_draw_stage > 0:
        side_draws.append(SideDraw(stage=ar_draw_stage, flow=ar_draw_flow))
    if waste_gan_flow > 0 and waste_gan_stage > 0:
        # Vapor draw: vents a lower-purity N2 stream a few stages below the
        # top, distinct from the high-purity N2 distillate (phase="vapor").
        side_draws.append(SideDraw(stage=waste_gan_stage, flow=waste_gan_flow, phase="vapor"))

    # B is auto-computed from mass balance: total_feed - D_upper - side_draw_total
    # When recycle is included: B = air_flow + B_ar - D_upper - ar_draw_flow
    #                             = air_flow - D_upper - D_ar_product  (correct)
    return MultiColumnConfig(
        N_stages=N,
        feeds=feeds,
        P_profile=np.full(N, P),
        condenser_type="total",
        distillate_rate=D_upper,
        reflux_ratio=cfg.RR_upper,
        side_draws=side_draws,
    )


def _solve_upper_column(
    cfg: ASUConfig,
    lower_res: ColumnResult,
    D_upper: float,
    ar_draw_flow: float,
    ar_draw_stage: int,
    mix: MixtureTD,
    ar_bottoms_recycle: "StreamSpec | None" = None,
    waste_gan_stage: int = 0,
    waste_gan_flow: float = 0.0,
) -> ColumnResult:
    upper_cfg = _build_upper_config(
        cfg, lower_res, D_upper, ar_draw_flow, ar_draw_stage, mix,
        ar_bottoms_recycle=ar_bottoms_recycle,
        waste_gan_stage=waste_gan_stage, waste_gan_flow=waste_gan_flow,
    )
    return solve_bp_asu(upper_cfg, mix, max_iter=400, n_cmo_iter=60)


# ---------------------------------------------------------------------------
# Argon column
# ---------------------------------------------------------------------------

def _solve_argon_column(
    cfg: ASUConfig,
    upper_res: ColumnResult,
    ar_draw_stage: int,
    ar_draw_flow: float,
    mix: MixtureTD,
) -> ColumnResult:
    """Solve the argon side-rectifier column."""
    N = cfg.N_argon
    P = cfg.P_upper   # same pressure as upper column (or slightly below)

    # Side draw composition from upper column
    j = min(ar_draw_stage - 1, upper_res.x.shape[1] - 1)
    z_ar = np.clip(upper_res.x[:, j].copy(), 1e-10, 1.0)
    z_ar /= z_ar.sum()

    T_ar = _bubble_T_safe(P, z_ar, mix, T0=float(upper_res.T[j]))

    D_ar = ar_draw_flow * cfg.D_frac_argon
    B_ar = ar_draw_flow - D_ar

    feed = FeedSpec(
        stage=max(1, N // 2),
        flow=ar_draw_flow,
        z=z_ar,
        T=T_ar,
        P=P,
        q=1.0,
    )
    ar_cfg = ColumnConfig(
        N_stages=N,
        feed=feed,
        P_profile=np.full(N, P),
        condenser_type="total",
        distillate_rate=D_ar,
        reflux_ratio=cfg.RR_argon,
        bottoms_rate=B_ar,
    )
    return solve_bp(ar_cfg, mix, max_iter=600, n_cmo_iter=120)


# ---------------------------------------------------------------------------
# Top-level solver
# ---------------------------------------------------------------------------

def solve_asu(cfg: ASUConfig, mix: MixtureTD) -> ASUResult:
    """Solve the full ASU double-column + argon column system.

    Algorithm
    ---------
    1. Solve lower (HP) column independently.
    2. Expand lower products to LP; form two feeds for upper column.
    3. Solve upper (LP) column without Ar side draw first to find Ar peak stage.
    4. Re-solve upper column with the Ar side draw.
    5. Solve argon column with crude Ar feed.
    6. Check MCHE duty balance: |Q_cond_lower + Q_reb_upper|.
    7. Outer-iterate: adjust D_upper to improve MCHE balance.
    """
    streams = ASUStreams()

    # ---- Step 1: Lower column ----
    lower_res = _solve_lower_column(cfg, mix)
    D_lower = cfg.air_flow * cfg.D_frac_lower
    B_lower = cfg.air_flow - D_lower

    streams.lower_distillate = _column_result_to_stream(
        lower_res, D_lower, cfg.P_lower, "top"
    )
    streams.lower_bottoms = _column_result_to_stream(
        lower_res, B_lower, cfg.P_lower, "bottom"
    )
    streams.upper_n2_feed  = _expand_stream(streams.lower_distillate, cfg.P_upper, mix)
    streams.upper_co2_feed = _expand_stream(streams.lower_bottoms,    cfg.P_upper, mix)

    # MCHE duty from lower column condenser (heat removed = positive)
    Q_mche_target = -lower_res.Q_condenser   # >0 (heat released by HP N₂ condensation)
    streams.Q_mche = Q_mche_target

    # ---- Step 2: Upper column — pilot solve (waste GAN, no Ar side draw yet) ----
    # Waste GAN vents a lower-purity N2 stream a few stages below the top so
    # the rectifying section isn't forced toward ~100% N2 recovery, which
    # would otherwise push N2 contamination down into the Ar-peak region.
    D_upper = cfg.air_flow * cfg.D_frac_upper
    waste_gan_stage = cfg.waste_gan_stage
    waste_gan_flow = cfg.waste_gan_flow
    upper_res0 = _solve_upper_column(
        cfg, lower_res, D_upper, 0.0, 0, mix,
        waste_gan_stage=waste_gan_stage, waste_gan_flow=waste_gan_flow,
    )

    # Locate Ar side-draw stage: shallowest point where N2 has rectified
    # out to the target trace level (NOT the Ar-concentration peak — see
    # find_ar_draw_stage docstring).
    ar_idx = 2  # Ar is the third component [N2, O2, Ar]
    n2_idx = 0
    if mix.n > 2:
        ar_draw_stage = find_ar_draw_stage(
            upper_res0.x, n2_idx=n2_idx, ar_idx=ar_idx,
            n2_ppm_target=cfg.ar_draw_n2_ppm_target,
        )
    else:
        ar_draw_stage = cfg.N_upper // 2

    ar_draw_flow = cfg.ar_draw_flow

    # ---- Step 3: Outer recycle iteration ----
    # The outer loop serves two purposes:
    #   (a) Propagate the Ar-column O2-rich bottoms recycle back into the upper
    #       column until the recycle composition converges.
    #   (b) Report the MCHE duty imbalance as a design metric (NOT used to drive
    #       D_upper — adjusting D_upper for MCHE balance destroys product purity
    #       because Q_reb_upper ∝ (RR+1)*D_upper and the required drop in D_upper
    #       to match Q_cond_lower conflicts with the N2 mass balance).
    #
    # Convergence criterion: max change in Ar recycle composition < tol_duty
    # (reusing the same tolerance parameter for both metrics).
    upper_res = upper_res0
    argon_res: ColumnResult | None = None
    ar_bottoms_recycle: StreamSpec | None = None   # O2-rich bottoms recycled from Ar column
    ar_recycle_prev: StreamSpec | None = None      # previous iteration recycle (for convergence)
    n_outer = 0
    duty_imbalance = float("inf")
    recycle_change = float("inf")

    for n_outer in range(1, cfg.max_outer_iter + 1):
        # Solve upper column with Ar + waste GAN side draws (+ recycle from Ar column on iter ≥ 2)
        upper_res = _solve_upper_column(
            cfg, lower_res, D_upper, ar_draw_flow, ar_draw_stage, mix,
            ar_bottoms_recycle=ar_bottoms_recycle,
            waste_gan_stage=waste_gan_stage, waste_gan_flow=waste_gan_flow,
        )

        # The draw stage was first picked from a zero-extraction pilot
        # profile. Actually pulling ar_draw_flow out of the column reduces
        # the descending L below the draw point, which can leave the
        # originally-picked stage above the target N2 ppm. Re-evaluate
        # within a bounded window of the current stage (a global re-scan can
        # jump straight past the Ar peak across a steep N2 cliff — see
        # find_ar_draw_stage_near docstring) and re-solve until the stage
        # stabilizes (bounded retries so this can't outrun the outer
        # recycle loop).
        if mix.n > 2 and ar_draw_flow > 0:
            for _ in range(5):
                corrected_stage = find_ar_draw_stage_near(
                    upper_res.x, ar_draw_stage, n2_idx=n2_idx, ar_idx=ar_idx,
                    n2_ppm_target=cfg.ar_draw_n2_ppm_target,
                )
                if corrected_stage == ar_draw_stage:
                    break
                ar_draw_stage = corrected_stage
                upper_res = _solve_upper_column(
                    cfg, lower_res, D_upper, ar_draw_flow, ar_draw_stage, mix,
                    ar_bottoms_recycle=ar_bottoms_recycle,
                    waste_gan_stage=waste_gan_stage, waste_gan_flow=waste_gan_flow,
                )

        # Solve argon column; extract O2-rich bottoms as recycle back to upper column
        if mix.n > 2 and ar_draw_flow > 0 and ar_draw_stage > 0:
            argon_res = _solve_argon_column(
                cfg, upper_res, ar_draw_stage, ar_draw_flow, mix
            )
            D_ar_flow = ar_draw_flow * cfg.D_frac_argon
            B_ar_flow = ar_draw_flow - D_ar_flow
            z_ar_bot = np.clip(argon_res.x_bottoms.copy(), 0.0, 1.0)
            if z_ar_bot.sum() > 1e-10:
                z_ar_bot /= z_ar_bot.sum()
            new_recycle = StreamSpec(
                flow=B_ar_flow,
                z=z_ar_bot,
                T=float(argon_res.T[-1]),
                P=cfg.P_upper,
                phase="liquid",
            )

            # Check recycle convergence
            if ar_recycle_prev is not None:
                recycle_change = float(np.max(np.abs(new_recycle.z - ar_recycle_prev.z)))
            ar_recycle_prev   = ar_bottoms_recycle
            ar_bottoms_recycle = new_recycle
        else:
            recycle_change = 0.0   # no Ar column → trivially converged

        # Compute MCHE duty imbalance (informational; not used to adjust D_upper)
        Q_reb_upper  = upper_res.Q_reboiler
        Q_cond_lower = lower_res.Q_condenser
        duty_imbalance = abs(abs(Q_cond_lower) - Q_reb_upper)

        # Converge when the Ar recycle composition is stable
        if recycle_change < cfg.tol_duty:
            break

    # ---- Step 4: Collect streams ----
    streams.duty_imbalance = duty_imbalance

    z_upper_dist = np.clip(upper_res.x_distillate.copy(), 0.0, 1.0)
    z_upper_dist /= z_upper_dist.sum()
    streams.upper_distillate = StreamSpec(
        flow=D_upper, z=z_upper_dist,
        T=float(upper_res.T[0]), P=cfg.P_upper, phase="liquid"
    )

    z_upper_bot = np.clip(upper_res.x_bottoms.copy(), 0.0, 1.0)
    z_upper_bot /= z_upper_bot.sum()
    # Net O2 product = air in - N2 product - waste GAN - crude Ar product leaving the system
    # (Ar column bottoms is recycled so only D_ar_product is a net loss)
    D_ar_product = ar_draw_flow * cfg.D_frac_argon if ar_draw_flow > 0 else 0.0
    B_upper = cfg.air_flow - D_upper - waste_gan_flow - D_ar_product
    streams.upper_bottoms = StreamSpec(
        flow=max(B_upper, 0.0), z=z_upper_bot,
        T=float(upper_res.T[-1]), P=cfg.P_upper, phase="liquid"
    )

    if ar_draw_stage > 0 and ar_draw_stage <= upper_res.x.shape[1]:
        j_ar = ar_draw_stage - 1
        z_ar_draw = np.clip(upper_res.x[:, j_ar].copy(), 0.0, 1.0)
        z_ar_draw /= z_ar_draw.sum()
        streams.upper_ar_draw = StreamSpec(
            flow=ar_draw_flow, z=z_ar_draw,
            T=float(upper_res.T[j_ar]), P=cfg.P_upper, phase="liquid"
        )

    if waste_gan_stage > 0 and waste_gan_flow > 0 and waste_gan_stage <= upper_res.y.shape[1]:
        j_wg = waste_gan_stage - 1
        z_wg = np.clip(upper_res.y[:, j_wg].copy(), 0.0, 1.0)
        z_wg /= z_wg.sum()
        streams.waste_gan = StreamSpec(
            flow=waste_gan_flow, z=z_wg,
            T=float(upper_res.T[j_wg]), P=cfg.P_upper, phase="vapor"
        )

    if argon_res is not None:
        D_ar = D_ar_product
        z_ar_top = np.clip(argon_res.x_distillate.copy(), 0.0, 1.0)
        z_ar_top /= z_ar_top.sum()
        streams.ar_distillate = StreamSpec(
            flow=D_ar, z=z_ar_top,
            T=float(argon_res.T[0]), P=cfg.P_upper, phase="liquid"
        )
        z_ar_bot = np.clip(argon_res.x_bottoms.copy(), 0.0, 1.0)
        z_ar_bot /= z_ar_bot.sum()
        streams.ar_bottoms = StreamSpec(
            flow=ar_draw_flow - D_ar, z=z_ar_bot,
            T=float(argon_res.T[-1]), P=cfg.P_upper, phase="liquid"
        )

    # ---- Step 5: Key metrics ----
    n2_idx, o2_idx, ar_idx2 = 0, 1, 2

    N2_purity = float(z_upper_dist[n2_idx]) if mix.n > n2_idx else 0.0
    O2_purity = float(z_upper_bot[o2_idx])  if mix.n > o2_idx else 0.0

    Ar_purity = 0.0
    if argon_res is not None and mix.n > ar_idx2:
        z_ar_top_arr = np.clip(argon_res.x_distillate.copy(), 0.0, 1.0)
        z_ar_top_arr /= z_ar_top_arr.sum()
        Ar_purity = float(z_ar_top_arr[ar_idx2])

    F_N2_in = float(cfg.air_flow * cfg.z_air[n2_idx])
    F_O2_in = float(cfg.air_flow * cfg.z_air[o2_idx])
    N2_prod  = D_upper * N2_purity
    O2_prod  = max(B_upper, 0.0) * O2_purity
    N2_recovery = N2_prod / F_N2_in if F_N2_in > 0 else 0.0
    O2_recovery = O2_prod / F_O2_in if F_O2_in > 0 else 0.0

    Ar_recovery = 0.0
    if argon_res is not None and mix.n > ar_idx2 and cfg.z_air[ar_idx2] > 0:
        F_Ar_in = float(cfg.air_flow * cfg.z_air[ar_idx2])
        Ar_prod = D_ar_product * Ar_purity
        Ar_recovery = Ar_prod / F_Ar_in if F_Ar_in > 0 else 0.0

    # Outer convergence = Ar recycle composition converged within tolerance.
    outer_converged = recycle_change < cfg.tol_duty

    return ASUResult(
        streams=streams,
        lower_col=lower_res,
        upper_col=upper_res,
        argon_col=argon_res,
        converged=outer_converged,
        n_outer_iter=n_outer,
        duty_imbalance=duty_imbalance,
        N2_purity=N2_purity,
        O2_purity=O2_purity,
        Ar_purity=Ar_purity,
        N2_recovery=N2_recovery,
        O2_recovery=O2_recovery,
        Ar_recovery=Ar_recovery,
    )
