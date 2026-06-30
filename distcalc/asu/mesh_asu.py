"""Extended Wang-Henke BP MESH solver for multi-feed columns with side draws.

Generalises distcalc/column/mesh_bp.py to support:
  - Multiple feed stages at different positions and qualities
  - Liquid side draws from specified stages

The tridiagonal equations (0-indexed, with side draw S[j] at stage j):

  j=0  (total condenser):
    (D + L[0]) x₀  −  V[1]·K[1]·x₁  =  0

  j=1…N-2  (interior):
    −L[j-1]·x_{j-1}  +  (V[j]·K[j] + L[j] + S[j]) xⱼ  −  V[j+1]·K[j+1]·x_{j+1}
      =  Σₖ Fₖ·zₖᵢ   (sum over feeds on this stage)

  j=N-1  (reboiler):
    −L[N-2]·x_{N-2}  +  (V[N-1]·K[N-1] + B + S[N-1])·x_{N-1}  =  0

Here L[j] is the NET liquid flowing from stage j to stage j+1 (excluding
side draws). S[j] is the liquid side draw flow at stage j (appears in the
main diagonal and carries composition x[j]).

Energy balance (algebraically correct form)
-------------------------------------------
Substituting the mass balance  L[j]+S[j] = V[j+1]+L[j-1]+F[j]-V[j]  into
the adiabatic stage energy balance eliminates the circular dependence on
L[j] and gives the denominator as the latent heat at stage j:

  V[j+1] · (H_V[j+1] − H_L[j])
    = V[j]·(H_V[j]−H_L[j]) + L[j-1]·(H_L[j]−H_L[j-1]) + F[j]·H_L[j] − HF[j]

This is always well-conditioned because H_V − H_L ≈ latent heat > 0.
After each sequential (top-down) sweep the outer Broyden loop accelerates
convergence of the vapour-flow profile across outer iterations.

Reference: Wang & Henke (1966) Hydrocarbon Processing 45(8):155.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from ..column.specs import ColumnResult, FeedSpec
from ..components.loader import Mixture
from ..equilibrium.bubble_dew import bubble_T as _bubble_T
from ..thermo.pr_eos import enthalpy as _enthalpy, wilson_K
from .specs import MultiColumnConfig, SideDraw

_DAMP_T = 0.65
_DAMP_V = 0.70
_XMIN   = 1e-15


# ---------------------------------------------------------------------------
# CMO initialization
# ---------------------------------------------------------------------------

def _initialize_multi(cfg: MultiColumnConfig, mix: Mixture):
    """CMO initialisation for multi-feed, side-draw column."""
    N   = cfg.N_stages
    D   = cfg.D
    B   = cfg.B
    RR  = cfg.RR

    # Sort feeds top-to-bottom (0-based stage index ascending)
    feeds_sorted = sorted(cfg.feeds, key=lambda f: f.stage)
    sd_by_stage: dict[int, float] = {}   # liquid side draws
    vd_by_stage: dict[int, float] = {}   # vapor side draws (e.g. waste GAN)
    for sd in cfg.side_draws:
        if sd.phase == "vapor":
            vd_by_stage[sd.stage - 1] = vd_by_stage.get(sd.stage - 1, 0.0) + sd.flow
        else:
            sd_by_stage[sd.stage - 1] = sd_by_stage.get(sd.stage - 1, 0.0) + sd.flow

    # ---- Build L profile (CMO-like, top to bottom) ----
    L = np.zeros(N)
    L[0] = RR * D  # condenser reflux

    L_val = RR * D
    for j in range(1, N):
        # Add liquid component of feeds at stage j (1-based: j+1)
        for f in feeds_sorted:
            if f.stage - 1 == j:
                L_val += f.q * f.flow
        # Subtract liquid side draws
        L_val -= sd_by_stage.get(j, 0.0)
        L[j] = max(L_val, 1e-6)

    L[N - 1] = B  # enforce bottoms at reboiler

    # ---- Build V profile (top-down total balance) ----
    # V[j] = TOTAL vapor generated at stage j (continuing-up + any vapor draw)
    V = np.zeros(N)
    # V[0] = 0 for total condenser
    # V[1] = (RR+1)*D  (vapor from stage 1 into condenser)
    V[1] = (RR + 1.0) * D

    V_val = V[1]
    L_prev = L[0]
    for j in range(1, N - 1):
        F_j  = sum(f.flow for f in feeds_sorted if f.stage - 1 == j)
        S_j  = sd_by_stage.get(j, 0.0)
        # V_val is TOTAL vapor at stage j (already includes any draw at j);
        # solving for V_up arriving at j-1, then add back the draw at j+1
        # to store TOTAL vapor generated at j+1.
        V_up_next = V_val + L[j] + S_j - L_prev - F_j
        W_next = vd_by_stage.get(j + 1, 0.0)
        V_next = V_up_next + W_next
        V[j + 1] = max(V_next, 1e-6)
        V_val = V[j + 1]
        L_prev = L[j]

    # ---- Temperature profile: bubble-point of flow-weighted average feed ----
    z_avg = cfg.z_feed_avg
    P_top = cfg.P_profile[0]
    P_bot = cfg.P_profile[-1]
    try:
        bub_top = _bubble_T(P_top, z_avg, mix)
        T_top = bub_top.T * 0.97
    except Exception:
        T_top = 77.0
    try:
        bub_bot = _bubble_T(P_bot, z_avg, mix)
        T_bot = bub_bot.T * 1.03
    except Exception:
        T_bot = 90.0
    T = np.linspace(T_top, T_bot, N)

    # ---- Initial K from Wilson at mid temperature ----
    T_mid = 0.5 * (T_top + T_bot)
    P_mid = cfg.P_profile[N // 2]
    K0 = wilson_K(T_mid, P_mid, mix)
    K = np.outer(K0, np.ones(N))

    x = np.outer(z_avg, np.ones(N))
    y = K * x
    y /= y.sum(axis=0, keepdims=True)

    return T, V, L, K, x, y


# ---------------------------------------------------------------------------
# Tridiagonal builder (multi-feed + side draw)
# ---------------------------------------------------------------------------

def _build_tridiag_asu(
    V: np.ndarray,
    L: np.ndarray,
    K_i: np.ndarray,
    cfg: MultiColumnConfig,
    z_i_by_stage: dict[int, float],  # stage_j (0-based) → Σ Fk*zk_i
    sd_flow: np.ndarray,             # shape (N,) liquid side draw flow at each stage
    vd_flow: np.ndarray,             # shape (N,) vapor side draw flow at each stage
) -> tuple[np.ndarray, np.ndarray]:
    """Build banded (3×N) matrix and RHS for one component.

    Parameters
    ----------
    z_i_by_stage : mapping from 0-based stage index j to total feed source
                   term Σ_k F_k * z_k_i  for feeds on that stage.
    sd_flow      : liquid side draw flow at each stage [mol/s], shape (N,).
    vd_flow      : vapor side draw flow at each stage [mol/s], shape (N,).
                   A vapor draw at stage j removes vd_flow[j] of the V[j]
                   vapor (composition y[j]) before it continues up into
                   stage j-1, so only (V[j]-vd_flow[j]) arrives there. The
                   stage-j component balance itself is unaffected since the
                   total vapor leaving stage j (draw + continuing) is still
                   V[j]·K[j]·x[j] regardless of how it's split.
    """
    N = cfg.N_stages
    D = cfg.D
    B = cfg.B

    ab  = np.zeros((3, N))
    rhs = np.zeros(N)

    for j in range(N):
        # Main diagonal
        if j == 0:
            ab[1, 0] = D + L[0]
        elif j == N - 1:
            ab[1, j] = V[j] * K_i[j] + B + sd_flow[j]
        else:
            ab[1, j] = V[j] * K_i[j] + L[j] + sd_flow[j]

        # Superdiagonal c[j] = −(V[j+1]-vd_flow[j+1])·K[j+1]  (vapor arriving
        # at row j is only what's left after the vapor draw at j+1)
        if j < N - 1:
            ab[0, j + 1] = -(V[j + 1] - vd_flow[j + 1]) * K_i[j + 1]

        # Subdiagonal a[j] = −L[j-1]  (net L, no side draw)
        if j > 0:
            ab[2, j - 1] = -L[j - 1]

        # Feed source term
        if j in z_i_by_stage:
            rhs[j] = z_i_by_stage[j]

    return ab, rhs


def _solve_component_balances_asu(
    V: np.ndarray,
    L: np.ndarray,
    K: np.ndarray,
    cfg: MultiColumnConfig,
    n_comp: int,
    sd_flow: np.ndarray,
    vd_flow: np.ndarray,
) -> np.ndarray:
    """Solve tridiagonal per component. Returns x: shape (n_comp, N)."""
    N = cfg.N_stages
    x = np.zeros((n_comp, N))

    # Pre-build feed source maps per component
    feeds_sorted = sorted(cfg.feeds, key=lambda f: f.stage)
    for i in range(n_comp):
        z_i_by_stage: dict[int, float] = {}
        for f in feeds_sorted:
            j = f.stage - 1
            z_i_by_stage[j] = z_i_by_stage.get(j, 0.0) + f.flow * float(f.z[i])

        ab, rhs = _build_tridiag_asu(V, L, K[i, :], cfg, z_i_by_stage, sd_flow, vd_flow)
        try:
            x[i, :] = solve_banded((1, 1), ab, rhs)
        except Exception:
            pass  # keep previous x on singular matrix
    return x


# ---------------------------------------------------------------------------
# Energy balance — algebraically correct (no circular L[j] dependency)
# ---------------------------------------------------------------------------

def _energy_balance_asu(
    V: np.ndarray,
    L: np.ndarray,
    H_L: np.ndarray,
    H_V: np.ndarray,
    cfg: MultiColumnConfig,
    mix: Mixture,
    sd_flow: np.ndarray,
    vd_flow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Top-down energy balance for multi-feed + side-draw column.

    Derives V[j+1] by substituting the total mass balance into the adiabatic
    stage energy balance, eliminating the circular dependence on L[j]:

        V_up[j+1]*(H_V[j+1] - H_L[j])
            = V[j]*(H_V[j] - H_L[j])
              + L[j-1]*(H_L[j] - H_L[j-1])
              + F_j*H_L[j] - HF_j

    where V[j] is the TOTAL vapor generated at stage j (continuing-up +
    any vapor side draw at j) and V_up[j+1] is the vapor that actually
    arrives at stage j from below (i.e. V[j+1] minus any vapor draw at
    j+1). The vapor array V_new stores the TOTAL at each stage (V_up plus
    the draw added back), consistent with its use in the component
    balance and K-value weighting elsewhere.

    Denominator H_V[j+1]-H_L[j] ≈ latent heat (always > 0 for boiling
    mixtures), so no oscillation can arise from this formula.
    """
    N  = cfg.N_stages
    D  = cfg.D
    B  = cfg.B

    feeds_sorted = sorted(cfg.feeds, key=lambda f: f.stage)
    feed_H: dict[int, float] = {}
    feed_F: dict[int, float] = {}
    for f in feeds_sorted:
        j = f.stage - 1
        hf = _enthalpy(f.T, f.P, f.z, mix, "liquid") * f.flow
        feed_H[j]  = feed_H.get(j, 0.0) + hf
        feed_F[j]  = feed_F.get(j, 0.0) + f.flow

    V_new = V.copy()
    L_new = L.copy()

    # Condenser duty
    if N > 1:
        Q_cond = V_new[1] * H_V[1] - (D + L_new[0]) * H_L[0]
    else:
        Q_cond = 0.0

    for j in range(1, N - 1):
        HF_j = feed_H.get(j, 0.0)
        F_j  = feed_F.get(j, 0.0)
        S_j  = float(sd_flow[j])
        W_j1 = float(vd_flow[j + 1])  # vapor draw AT stage j+1

        # Denominator = latent heat at stage j evaluated between j+1 and j.
        # Always positive for vapour/liquid cryogenic systems.
        denom = H_V[j + 1] - H_L[j]
        if abs(denom) < 200.0:
            # Fallback: use latent heat at stage j directly
            denom = max(H_V[j] - H_L[j], 200.0)

        numerator = (
            V_new[j]   * (H_V[j]   - H_L[j])
            + L_new[j - 1] * (H_L[j]   - H_L[j - 1])
            + F_j          *  H_L[j]
            - HF_j
        )
        V_up_j1 = numerator / denom
        V_up_j1 = max(V_up_j1, 0.05 * max(V[j + 1] - vd_flow[j + 1], 1e-6))
        V_new[j + 1] = V_up_j1 + W_j1   # store TOTAL vapor generated at j+1

        # L[j] from total mass balance (no circular dependency); uses
        # V_up_j1 (vapor actually arriving from below), not the total.
        L_new[j] = V_up_j1 + L_new[j - 1] + F_j - V_new[j] - S_j
        L_new[j] = max(L_new[j], 0.05 * max(L[j], 1e-6))

    if N > 1:
        Q_reb = (
            V_new[N - 1] * H_V[N - 1]
            + (B + sd_flow[N - 1]) * H_L[N - 1]
            - L_new[N - 2] * H_L[N - 2]
        )
    else:
        Q_reb = 0.0

    return V_new, L_new, Q_cond, Q_reb


def _eb_residuals(
    V: np.ndarray,
    L: np.ndarray,
    H_L: np.ndarray,
    H_V: np.ndarray,
    feed_H: dict[int, float],
    feed_F: dict[int, float],
    sd_flow: np.ndarray,
    B: float,
) -> np.ndarray:
    """Energy balance residuals for interior stages 1..N-2.

    R[j] = V[j+1]*H_V[j+1] + L[j-1]*H_L[j-1] + HF_j
           - V[j]*H_V[j] - (L[j]+S[j])*H_L[j]

    where L[j]+S[j] is obtained from the mass balance to avoid circular
    dependence:  L[j]+S[j] = V[j+1]+L[j-1]+F[j]-V[j].
    """
    N = len(V)
    res = np.zeros(N - 2)
    for idx, j in enumerate(range(1, N - 1)):
        HF_j = feed_H.get(j, 0.0)
        F_j  = feed_F.get(j, 0.0)
        S_j  = float(sd_flow[j])
        LS_j = V[j + 1] + L[j - 1] + F_j - V[j]  # = L[j]+S[j] from mass balance
        res[idx] = (
            V[j + 1] * H_V[j + 1]
            + L[j - 1] * H_L[j - 1]
            + HF_j
            - V[j] * H_V[j]
            - LS_j * H_L[j]
        )
    return res


# ---------------------------------------------------------------------------
# Broyden update (following Numerical Recipes 9.7 / sim42 Tower.py)
# ---------------------------------------------------------------------------

def _broyden_update(B: np.ndarray, dx: np.ndarray, dF: np.ndarray) -> np.ndarray:
    """Rank-1 Broyden update of the inverse Jacobian.

    B  : current inverse Jacobian (n×n)
    dx : parameter step taken (n)
    dF : change in residuals  (n)
    """
    dotdxB = dx @ B
    denom  = dotdxB @ dF
    if abs(denom) < 1e-30:
        return B
    return B + np.outer(dx - B @ dF, dotdxB) / denom


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_bp_asu(
    cfg: MultiColumnConfig,
    mix: Mixture,
    tol_T: float = 1e-4,
    tol_x: float = 1e-6,
    max_iter: int = 200,
    damp_T: float = _DAMP_T,
    damp_V: float = _DAMP_V,
    n_cmo_iter: int = 80,
) -> ColumnResult:
    """Wang-Henke bubble-point solver for multi-feed + side-draw columns.

    Improvements over the baseline algorithm
    -----------------------------------------
    1. Algebraically correct energy balance (no circular L[j] dependency).
       The denominator is H_V[j+1]-H_L[j] ≈ latent heat, always positive.
    2. After CMO warm-up, Broyden-accelerated Newton outer loop following
       the inside-out structure of sim42 Tower.py (UpdateJacobian method).
       Inner variables: log(V[1:N]) for positivity; outer errors: log of
       the ratio of successive V corrections.

    Parameters
    ----------
    cfg        : MultiColumnConfig (multiple feeds, optional side draws)
    mix        : Mixture
    n_cmo_iter : iterations with CMO flows (no energy balance V/L update)
                 before switching to full energy balance; prevents early
                 blow-up from inconsistent compositions.
    """
    N = cfg.N_stages
    n = mix.n

    # Side draw flow arrays (0-indexed): liquid (sd_flow) and vapor (vd_flow)
    sd_flow = np.zeros(N)
    vd_flow = np.zeros(N)
    for sd in cfg.side_draws:
        if 1 <= sd.stage <= N:
            if sd.phase == "vapor":
                vd_flow[sd.stage - 1] += sd.flow
            else:
                sd_flow[sd.stage - 1] += sd.flow

    T, V, L, K, x, y = _initialize_multi(cfg, mix)
    V_ref = V.copy()
    L_ref = L.copy()
    H_L = np.zeros(N)
    H_V = np.zeros(N)
    Q_cond = Q_reb = 0.0
    converged = False
    max_dT = np.inf

    # Broyden state for outer-loop acceleration of V profile
    n_inner = N - 1           # log(V[1:N]) — N-1 components
    B_inv: np.ndarray | None = None
    log_V_prev: np.ndarray | None = None
    F_prev: np.ndarray | None = None

    # Pre-compute feed enthalpy maps (constant across iterations)
    feeds_sorted = sorted(cfg.feeds, key=lambda f: f.stage)
    feed_H: dict[int, float] = {}
    feed_F: dict[int, float] = {}
    for f in feeds_sorted:
        j = f.stage - 1
        hf = _enthalpy(f.T, f.P, f.z, mix, "liquid") * f.flow
        feed_H[j] = feed_H.get(j, 0.0) + hf
        feed_F[j] = feed_F.get(j, 0.0) + f.flow

    for n_iter in range(1, max_iter + 1):
        T_old = T.copy()
        x_old = x.copy()

        # Step A: tridiagonal material balance → x
        x_new = _solve_component_balances_asu(V, L, K, cfg, n, sd_flow, vd_flow)
        x_new = np.maximum(x_new, _XMIN)
        col_sums = x_new.sum(axis=0)
        col_sums = np.where(col_sums > 0, col_sums, 1.0)
        x = x_new / col_sums

        # Step B: bubble-point per stage → new T, K
        K_new = K.copy()
        T_new = T.copy()
        for j in range(N):
            try:
                res = _bubble_T(cfg.P_profile[j], x[:, j], mix, T0=float(T[j]))
                T_new[j] = res.T
                K_new[:, j] = np.maximum(res.y, _XMIN) / np.maximum(x[:, j], _XMIN)
            except Exception:
                pass

        dT = T_new - T_old
        T = T_old + damp_T * dT
        K = K_new

        # Step C: update y
        y = K * x
        y_sums = y.sum(axis=0)
        y /= np.where(y_sums > 1e-20, y_sums, 1.0)

        # Step D: enthalpies
        for j in range(N):
            H_L[j] = _enthalpy(T[j], cfg.P_profile[j], x[:, j], mix, "liquid")
            H_V[j] = _enthalpy(T[j], cfg.P_profile[j], y[:, j], mix, "vapor")

        # Step E: algebraically correct energy balance
        V_new, L_new, Q_cond, Q_reb = _energy_balance_asu(
            V, L, H_L, H_V, cfg, mix, sd_flow, vd_flow
        )

        if n_iter > n_cmo_iter:
            # Clip to safe bounds (±5× CMO reference)
            V_lo = np.where(V_ref > 0, 0.2 * V_ref, 0.0)
            V_hi = np.where(V_ref > 0, 5.0 * V_ref, 1e6)
            L_lo = np.where(L_ref > 0, 0.2 * L_ref, 0.0)
            L_hi = np.where(L_ref > 0, 5.0 * L_ref, 1e6)
            V_new = np.clip(V_new, V_lo, V_hi)
            L_new = np.clip(L_new, L_lo, L_hi)

            # Broyden outer-loop acceleration on log(V[1:])
            # Following sim42 Tower.py UpdateJacobian / inner loop pattern:
            #   dx = current log(V) - previous log(V)
            #   dF = change in fixed-point residual  F(logV) = log(V_new) - log(V)
            #   Newton step: Δlog(V) = -B_inv · F_current
            log_V     = np.log(np.maximum(V[1:],     1e-15))
            log_V_new = np.log(np.maximum(V_new[1:], 1e-15))
            F_cur     = log_V_new - log_V  # fixed-point residual

            if B_inv is None:
                B_inv = np.eye(n_inner)
            elif log_V_prev is not None and F_prev is not None:
                dx = log_V - log_V_prev
                dF = F_cur  - F_prev
                B_inv = _broyden_update(B_inv, dx, dF)

            log_V_prev = log_V.copy()
            F_prev     = F_cur.copy()

            # Newton step in log space; clip to prevent huge steps
            step = -B_inv @ F_cur
            step = np.clip(step, -1.0, 1.0)
            log_V_acc = log_V + step
            V_acc = np.empty_like(V)
            V_acc[0]  = 0.0
            V_acc[1:] = np.exp(log_V_acc)
            V_acc = np.clip(V_acc, V_lo, V_hi)

            # Blend Broyden step with direct energy-balance result
            V = V + damp_V * (V_acc  - V)
            L = L + damp_V * (L_new  - L)
        # else: CMO warmup — don't update V/L yet

        max_dT = float(np.max(np.abs(T - T_old)))
        max_dx = float(np.max(np.abs(x - x_old)))
        if max_dT < tol_T and max_dx < tol_x:
            converged = True
            break

    # Build ColumnResult (reuse existing dataclass)
    return ColumnResult(
        T=T, V=V, L=L, x=x, y=y, K=K,
        H_L=H_L, H_V=H_V,
        Q_condenser=Q_cond,
        Q_reboiler=Q_reb,
        converged=converged,
        n_iter=n_iter,
        max_dT=max_dT,
    )


# ---------------------------------------------------------------------------
# Helper: side draw stage finder
# ---------------------------------------------------------------------------

def find_ar_peak_stage(x: np.ndarray, ar_idx: int = 2) -> int:
    """Return 1-based stage index where component ar_idx peaks in liquid."""
    return int(np.argmax(x[ar_idx, :])) + 1
