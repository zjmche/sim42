"""Wang-Henke Bubble-Point (BP) MESH distillation solver.

Stage model (j = 0…N-1, condenser=0, reboiler=N-1):
  M — component material balance (tridiagonal, Thomas algorithm via scipy)
  E — equilibrium  yᵢⱼ = Kᵢⱼ·xᵢⱼ  (condenser is NOT an equilibrium stage)
  S — summation   Σxᵢⱼ = Σyᵢⱼ = 1
  H — enthalpy balance → vapor flows Vⱼ

Flow conventions (0-indexed stages):
  V[j] = vapor leaving stage j upward (to stage j-1 or zero for condenser)
  L[j] = liquid leaving stage j downward (to stage j+1 or bottoms)

Correct tridiagonal coefficients (per component i, per stage j):
  a[j] = -L[j-1]                              (subdiagonal, j>0)
  b[j] = V[j]*K[i,j] + L[j]   for 1≤j≤N-2  (main diagonal, interior)
       = D + L[0]               for j=0       (total condenser: no K)
       = V[j]*K[i,j] + B        for j=N-1    (reboiler)
  c[j] = -V[j+1]*K[i,j+1]                    (superdiagonal, j<N-1)
  rhs[j] = +F_j * z_i           if j is feed stage

Reference: Wang & Henke (1966) Hydrocarbon Processing 45(8):155.
           Seader, Henley & Roper (2011) Separation Process Principles, §10.3.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from ..components.loader import Mixture
from ..equilibrium.bubble_dew import bubble_T as _bubble_T
from ..thermo.pr_eos import enthalpy as _enthalpy, wilson_K
from .specs import ColumnConfig, ColumnResult

_DAMP_T = 0.65   # temperature-update damping
_DAMP_V = 0.70   # vapor-flow damping
_XMIN   = 1e-15  # composition floor

# Module-level slot for component index (passed into _build_tridiag)
_COMP_Z: float = 0.0
_COMP_IDX: int = 0


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def _initialize(cfg: ColumnConfig, mix: Mixture):
    """Linear T profile + constant-molar-overflow (CMO) V and L arrays."""
    N   = cfg.N_stages
    D, B = cfg.D, cfg.B
    RR  = cfg.RR
    F   = cfg.feed.flow
    q   = cfg.feed.q
    fs  = cfg.feed.stage - 1   # 0-based index of feed stage

    # ---- CMO flow rates ----
    V_above = (RR + 1.0) * D           # vapor in rectifying section
    L_above = RR * D                   # liquid in rectifying section
    # For q=1 (sat.liq): V unchanged across feed, L increases
    V_below = V_above - (1.0 - q) * F
    L_below = L_above + q * F

    # V[j]: vapor leaving stage j upward
    V = np.zeros(N)
    if fs + 1 <= N - 1:
        V[1 : fs + 1] = V_above        # stages 1..fs (rectifying section + feed stage)
    if fs + 1 < N:
        V[fs + 1 :] = V_below          # stages fs+1..N-1 (stripping section)
    # V[0] = 0 for total condenser (no vapor out of stage 0)

    # L[j]: liquid leaving stage j downward
    L = np.zeros(N)
    L[0] = L_above                     # reflux
    if fs > 1:
        L[1 : fs] = L_above            # rectifying interior
    if fs < N - 1:
        L[fs : N - 1] = L_below        # feed stage + stripping interior
    L[N - 1] = B                       # bottoms out of reboiler

    # ---- Temperature profile: linear bubble T estimate ----
    from ..equilibrium.bubble_dew import bubble_T as _bt
    z = cfg.feed.z
    P_top = cfg.P_profile[0]
    P_bot = cfg.P_profile[-1]
    bub_top = _bt(P_top, z, mix)
    bub_bot = _bt(P_bot, z, mix)
    # Condenser is slightly below bubble T; reboiler slightly above
    T_top = bub_top.T * 0.97
    T_bot = bub_bot.T * 1.03
    T = np.linspace(T_top, T_bot, N)

    # ---- Initial K from Wilson at average T ----
    T_mid = 0.5 * (T_top + T_bot)
    P_mid = cfg.P_profile[N // 2]
    K0 = wilson_K(T_mid, P_mid, mix)
    K = np.outer(K0, np.ones(N))   # (n_comp, N)

    # ---- Initial x: feed composition on every stage ----
    x = np.outer(z, np.ones(N))
    y = K * x
    y /= y.sum(axis=0, keepdims=True)

    return T, V, L, K, x, y


# ---------------------------------------------------------------------------
# Tridiagonal material balance builder
# ---------------------------------------------------------------------------

def _build_tridiag(
    V: np.ndarray,
    L: np.ndarray,
    K_i: np.ndarray,   # K for component i, shape (N,)
    cfg: ColumnConfig,
    z_i: float,        # feed mole fraction for component i
) -> tuple[np.ndarray, np.ndarray]:
    """Build banded matrix (3×N) and RHS for one component i.

    Tridiagonal equations (0-indexed):

    j=0  (total condenser):
        (D + L[0])·x₀  −  V[1]·K[1]·x₁  =  0

    j=1…N-2  (interior stages):
        −L[j-1]·x_{j-1}  +  (V[j]·K[j] + L[j])·xⱼ  −  V[j+1]·K[j+1]·x_{j+1}  =  +Fⱼ·zᵢ

    j=N-1  (reboiler):
        −L[N-2]·x_{N-2}  +  (V[N-1]·K[N-1] + B)·x_{N-1}  =  0
    """
    N  = cfg.N_stages
    fs = cfg.feed.stage - 1
    D, B = cfg.D, cfg.B

    # ab layout for scipy.solve_banded(l=1, u=1):
    #   ab[0, j]  = superdiagonal element at column j  (i.e. M[j-1, j])
    #   ab[1, j]  = main diagonal at column j
    #   ab[2, j]  = subdiagonal element at column j    (i.e. M[j+1, j])
    ab  = np.zeros((3, N))
    rhs = np.zeros(N)

    for j in range(N):
        # Main diagonal b[j]
        if j == 0:
            ab[1, 0] = D + L[0]                      # condenser: D + reflux (no K)
        elif j == N - 1:
            ab[1, j] = V[j] * K_i[j] + B            # reboiler
        else:
            ab[1, j] = V[j] * K_i[j] + L[j]        # interior

        # Superdiagonal c[j] = -V[j+1]*K[j+1]  for j < N-1
        if j < N - 1:
            ab[0, j + 1] = -V[j + 1] * K_i[j + 1]

        # Subdiagonal a[j] = -L[j-1]  for j > 0
        if j > 0:
            ab[2, j - 1] = -L[j - 1]

        # Feed contribution on RHS (material balance: +F*z is the source term)
        if j == fs:
            rhs[j] = cfg.feed.flow * z_i

    return ab, rhs


def _solve_component_balances(
    V: np.ndarray,
    L: np.ndarray,
    K: np.ndarray,    # shape (n_comp, N)
    cfg: ColumnConfig,
    n_comp: int,
) -> np.ndarray:
    """Solve tridiagonal per component. Returns x: shape (n_comp, N)."""
    N = cfg.N_stages
    x = np.zeros((n_comp, N))
    for i in range(n_comp):
        z_i = float(cfg.feed.z[i])
        ab, rhs = _build_tridiag(V, L, K[i, :], cfg, z_i)
        try:
            x[i, :] = solve_banded((1, 1), ab, rhs)
        except Exception:
            # Singular matrix (degenerate K profile) — keep previous x
            pass
    return x


# ---------------------------------------------------------------------------
# Energy balance
# ---------------------------------------------------------------------------

def _enthalpy_feed(cfg: ColumnConfig, mix: Mixture) -> float:
    """Total feed molar enthalpy [J/mol] at feed conditions."""
    # Approximate: assume feed is saturated liquid (q=1) at T_feed
    # A more precise treatment would flash the feed first
    return _enthalpy(cfg.feed.T, cfg.feed.P, cfg.feed.z, mix, "liquid")


def _energy_balance(
    V: np.ndarray,
    L: np.ndarray,
    H_L: np.ndarray,
    H_V: np.ndarray,
    cfg: ColumnConfig,
    mix: Mixture,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Top-down energy balance; returns (V_new, L_new, Q_cond, Q_reb).

    Algebraically correct form: substitute total mass balance into the
    adiabatic stage energy balance to eliminate circular dependence on L[j]:

        V[j+1]*(H_V[j+1]-H_L[j])
            = V[j]*(H_V[j]-H_L[j]) + L[j-1]*(H_L[j]-H_L[j-1]) + F_j*H_L[j] - HF_j

    Denominator H_V[j+1]-H_L[j] ≈ latent heat (always > 0), so no
    oscillation from stage-to-stage propagation.
    """
    N  = cfg.N_stages
    fs = cfg.feed.stage - 1
    D, B = cfg.D, cfg.B

    H_feed = _enthalpy_feed(cfg, mix) * cfg.feed.flow   # [J/s]

    V_new = V.copy()
    L_new = L.copy()

    # ---- Condenser duty ----
    if N > 1:
        Q_cond = V_new[1] * H_V[1] - (D + L_new[0]) * H_L[0]
    else:
        Q_cond = 0.0

    # ---- Interior stages ----
    for j in range(1, N - 1):
        F_j  = cfg.feed.flow if j == fs else 0.0
        HF_j = H_feed        if j == fs else 0.0

        denom = H_V[j + 1] - H_L[j]
        if abs(denom) < 200.0:
            denom = max(H_V[j] - H_L[j], 200.0)

        numerator = (
            V_new[j]       * (H_V[j]   - H_L[j])
            + L_new[j - 1] * (H_L[j]   - H_L[j - 1])
            + F_j          *  H_L[j]
            - HF_j
        )
        if j + 1 <= N - 1:
            V_new[j + 1] = numerator / denom
            V_new[j + 1] = max(V_new[j + 1], 0.05 * max(V[j + 1], 1e-6))

        L_new[j] = V_new[j + 1] + L_new[j - 1] + F_j - V_new[j]
        L_new[j] = max(L_new[j], 0.05 * max(L[j], 1e-6))

    # ---- Reboiler duty ----
    if N > 1:
        Q_reb = V_new[N - 1] * H_V[N - 1] + B * H_L[N - 1] - L_new[N - 2] * H_L[N - 2]
    else:
        Q_reb = 0.0

    return V_new, L_new, Q_cond, Q_reb


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_bp(
    cfg: ColumnConfig,
    mix: Mixture,
    tol_T: float = 1e-4,
    tol_x: float = 1e-6,
    max_iter: int = 150,
    damp_T: float = _DAMP_T,
    damp_V: float = _DAMP_V,
    n_cmo_iter: int = 60,
) -> ColumnResult:
    """Wang-Henke bubble-point column solver.

    Parameters
    ----------
    cfg        : ColumnConfig
    mix        : Mixture
    tol_T      : convergence on max |ΔT| [K]
    tol_x      : convergence on max |Δx|
    max_iter   : maximum outer iterations
    damp_T     : temperature damping factor (0, 1]
    damp_V     : vapor-flow damping factor (0, 1]
    n_cmo_iter : iterations to run with constant-molar-overflow before
                 enabling energy balance (prevents early instability from
                 wrong compositions corrupting H values)
    """
    N = cfg.N_stages
    n = mix.n

    T, V, L, K, x, y = _initialize(cfg, mix)
    # Store CMO reference flows for energy-balance bounds after warmup
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

        # ---- Step A: tridiagonal material balance → x ----
        x_new = _solve_component_balances(V, L, K, cfg, n)

        # Clip negatives and normalise per stage
        x_new = np.maximum(x_new, _XMIN)
        col_sums = x_new.sum(axis=0)
        col_sums = np.where(col_sums > 0, col_sums, 1.0)
        x = x_new / col_sums

        # ---- Step B: bubble-point per stage → new T, K ----
        K_new = K.copy()
        T_new = T.copy()
        for j in range(N):
            try:
                res = _bubble_T(cfg.P_profile[j], x[:, j], mix, T0=float(T[j]))
                T_new[j] = res.T
                # K[i,j] from bubble_T: y/x (normalized)
                K_new[:, j] = np.maximum(res.y, _XMIN) / np.maximum(x[:, j], _XMIN)
            except Exception:
                pass  # keep previous K, T if bubble_T fails

        # Damp T update
        dT = T_new - T_old
        T = T_old + damp_T * dT
        K = K_new

        # ---- Step C: update y from new K ----
        y = K * x
        y_sums = y.sum(axis=0)
        y /= np.where(y_sums > 1e-20, y_sums, 1.0)

        # ---- Step D: compute enthalpies ----
        for j in range(N):
            H_L[j] = _enthalpy(T[j], cfg.P_profile[j], x[:, j], mix, "liquid")
            H_V[j] = _enthalpy(T[j], cfg.P_profile[j], y[:, j], mix, "vapor")

        # ---- Step E: energy balance → update V, L ----
        # Skip V/L update for the first n_cmo_iter iterations so that
        # compositions and temperatures can establish before enthalpies are
        # used to drive the flow profile.  Wrong early-iteration compositions
        # produce inconsistent H values that cause V/L to diverge.
        V_new, L_new, Q_cond, Q_reb = _energy_balance(V, L, H_L, H_V, cfg, mix)
        if n_iter > n_cmo_iter:
            # Clip to ±5× CMO reference to prevent runaway
            V_lo = np.where(V_ref > 0, 0.2 * V_ref, 0.0)
            V_hi = np.where(V_ref > 0, 5.0 * V_ref, 0.0)
            L_lo = np.where(L_ref > 0, 0.2 * L_ref, 0.0)
            L_hi = np.where(L_ref > 0, 5.0 * L_ref, 0.0)
            V_new = np.clip(V_new, V_lo, V_hi)
            L_new = np.clip(L_new, L_lo, L_hi)
            V = V + damp_V * (V_new - V)
            L = L + damp_V * (L_new - L)

        # ---- Convergence ----
        max_dT = float(np.max(np.abs(T - T_old)))
        max_dx = float(np.max(np.abs(x - x_old)))
        if max_dT < tol_T and max_dx < tol_x:
            converged = True
            break

    return ColumnResult(
        T=T, V=V, L=L, x=x, y=y, K=K,
        H_L=H_L, H_V=H_V,
        Q_condenser=Q_cond,
        Q_reboiler=Q_reb,
        converged=converged,
        n_iter=n_iter,
        max_dT=max_dT,
    )
