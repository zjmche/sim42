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
    sd_by_stage: dict[int, float] = {}
    for sd in cfg.side_draws:
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
    # V[j] = vapor leaving stage j upward
    V = np.zeros(N)
    # V[0] = 0 for total condenser
    # V[1] = (RR+1)*D  (vapor from stage 1 into condenser)
    V[1] = (RR + 1.0) * D

    V_val = V[1]
    L_prev = L[0]
    for j in range(1, N - 1):
        F_j  = sum(f.flow for f in feeds_sorted if f.stage - 1 == j)
        S_j  = sd_by_stage.get(j, 0.0)
        V_next = V_val + L[j] + S_j - L_prev - F_j
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
    sd_flow: np.ndarray,             # shape (N,) side draw flow at each stage
) -> tuple[np.ndarray, np.ndarray]:
    """Build banded (3×N) matrix and RHS for one component.

    Parameters
    ----------
    z_i_by_stage : mapping from 0-based stage index j to total feed source
                   term Σ_k F_k * z_k_i  for feeds on that stage.
    sd_flow      : liquid side draw flow at each stage [mol/s], shape (N,).
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

        # Superdiagonal c[j] = −V[j+1]·K[j+1]
        if j < N - 1:
            ab[0, j + 1] = -V[j + 1] * K_i[j + 1]

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

        ab, rhs = _build_tridiag_asu(V, L, K[i, :], cfg, z_i_by_stage, sd_flow)
        try:
            x[i, :] = solve_banded((1, 1), ab, rhs)
        except Exception:
            pass  # keep previous x on singular matrix
    return x


# ---------------------------------------------------------------------------
# Energy balance (multi-feed + side draw)
# ---------------------------------------------------------------------------

def _energy_balance_asu(
    V: np.ndarray,
    L: np.ndarray,
    H_L: np.ndarray,
    H_V: np.ndarray,
    cfg: MultiColumnConfig,
    mix: Mixture,
    sd_flow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Top-down energy balance for multi-feed + side-draw column."""
    N  = cfg.N_stages
    D  = cfg.D
    B  = cfg.B

    feeds_sorted = sorted(cfg.feeds, key=lambda f: f.stage)
    feed_H: dict[int, float] = {}  # 0-based stage → total feed enthalpy rate [W]
    feed_F: dict[int, float] = {}  # 0-based stage → total feed flow [mol/s]
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

        numerator = (
            (L_new[j] + S_j) * H_L[j]
            + V_new[j] * H_V[j]
            - L_new[j - 1] * H_L[j - 1]
            - HF_j
        )
        if j + 1 <= N - 1 and abs(H_V[j + 1]) > 1.0:
            V_new[j + 1] = numerator / H_V[j + 1]
            V_new[j + 1] = max(V_new[j + 1], 0.05 * V[j + 1])

        L_new[j] = V_new[j + 1] + L_new[j - 1] + F_j - V_new[j] - S_j
        L_new[j] = max(L_new[j], 0.05 * L[j])

    if N > 1:
        Q_reb = (
            V_new[N - 1] * H_V[N - 1]
            + (B + sd_flow[N - 1]) * H_L[N - 1]
            - L_new[N - 2] * H_L[N - 2]
        )
    else:
        Q_reb = 0.0

    return V_new, L_new, Q_cond, Q_reb


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

    # Side draw flow array (0-indexed)
    sd_flow = np.zeros(N)
    for sd in cfg.side_draws:
        if 1 <= sd.stage <= N:
            sd_flow[sd.stage - 1] += sd.flow

    T, V, L, K, x, y = _initialize_multi(cfg, mix)
    V_ref = V.copy()
    L_ref = L.copy()
    H_L = np.zeros(N)
    H_V = np.zeros(N)
    Q_cond = Q_reb = 0.0
    converged = False
    max_dT = np.inf

    for n_iter in range(1, max_iter + 1):
        T_old = T.copy()
        x_old = x.copy()

        # Step A: tridiagonal material balance → x
        x_new = _solve_component_balances_asu(V, L, K, cfg, n, sd_flow)
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

        # Step E: energy balance (after CMO warmup)
        V_new, L_new, Q_cond, Q_reb = _energy_balance_asu(
            V, L, H_L, H_V, cfg, mix, sd_flow
        )
        if n_iter > n_cmo_iter:
            V_lo = np.where(V_ref > 0, 0.2 * V_ref, 0.0)
            V_hi = np.where(V_ref > 0, 5.0 * V_ref, 0.0)
            L_lo = np.where(L_ref > 0, 0.2 * L_ref, 0.0)
            L_hi = np.where(L_ref > 0, 5.0 * L_ref, 0.0)
            V_new = np.clip(V_new, V_lo, V_hi)
            L_new = np.clip(L_new, L_lo, L_hi)
            V = V + damp_V * (V_new - V)
            L = L + damp_V * (L_new - L)

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
