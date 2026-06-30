"""Bubble-point and dew-point calculations (T and P).

All solvers use Newton iteration on the Σ(y-x)=0 summation residual,
with fugacity coefficients updated at each step (φ-corrected).

Reference: Seader, Henley & Roper, Separation Process Principles, 3rd ed.
(2011) §4.4; Prausnitz et al. Molecular Thermodynamics of Fluid-Phase
Equilibria, 3rd ed. (1999) §8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..components.loader_tdep import MixtureTD
from ..thermo.pr_eos_tdep import ln_phi, wilson_K


@dataclass
class BubbleResult:
    T: float           # K
    P: float           # Pa
    x: np.ndarray      # liquid mole fractions
    y: np.ndarray      # vapor mole fractions
    converged: bool
    n_iter: int


@dataclass
class DewResult:
    T: float
    P: float
    x: np.ndarray
    y: np.ndarray
    converged: bool
    n_iter: int


# ---------------------------------------------------------------------------
# Internal helper — IMPORTANT: both x and y must be normalized before calling
# ---------------------------------------------------------------------------

def _K_from_phi(T: float, P: float, x: np.ndarray, y: np.ndarray,
                mix: MixtureTD) -> np.ndarray:
    """K = φ_L / φ_V.  x and y must already be normalised (sum=1)."""
    phi_L = np.exp(ln_phi(T, P, x, mix, "liquid"))
    phi_V = np.exp(ln_phi(T, P, y, mix, "vapor"))
    return phi_L / phi_V


def _normalize(v: np.ndarray) -> np.ndarray:
    s = v.sum()
    return v / s if s > 0 else v


# ---------------------------------------------------------------------------
# Bubble-point temperature
# ---------------------------------------------------------------------------

def bubble_T(
    P: float,
    x: np.ndarray,
    mix: MixtureTD,
    T0: float | None = None,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> BubbleResult:
    """Given P and liquid x, find bubble-point T and vapor y.

    Newton iteration on f(T) = Σᵢ Kᵢ(T,P,x,y)·xᵢ − 1 = 0.
    φ-corrected: K updated with PR-EOS each step.
    y must be normalized before each K computation — crucial for correctness.
    """
    x = np.asarray(x, dtype=float)
    x = _normalize(x)

    if T0 is None:
        T0 = float(np.dot(x, [c.Tc for c in mix.components])) * 0.7

    T = float(T0)
    K = wilson_K(T, P, mix)
    y = _normalize(K * x)

    for n_iter in range(1, max_iter + 1):
        # Update K with normalized y (fugacity coefficients need normalized compositions)
        K = _K_from_phi(T, P, x, y, mix)
        S = float(K @ x)          # Σ Kᵢxᵢ; bubble point when = 1
        y = _normalize(K * x)     # normalize immediately; keep normalized throughout
        f = S - 1.0

        # Finite-difference dS/dT using normalized y
        dT_fd = T * 1e-4
        K_p = _K_from_phi(T + dT_fd, P, x, y, mix)  # y is normalized here ✓
        S_p = float(K_p @ x)
        df_dT = (S_p - S) / dT_fd

        if abs(df_dT) < 1e-30:
            break

        step = np.clip(-f / df_dT, -20.0, 20.0)
        T += step

        if abs(step) < tol:
            return BubbleResult(T=T, P=P, x=x, y=y, converged=True, n_iter=n_iter)

    return BubbleResult(T=T, P=P, x=x, y=y, converged=False, n_iter=max_iter)


# ---------------------------------------------------------------------------
# Bubble-point pressure
# ---------------------------------------------------------------------------

def bubble_P(
    T: float,
    x: np.ndarray,
    mix: MixtureTD,
    P0: float | None = None,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> BubbleResult:
    """Given T and liquid x, find bubble-point P and vapor y."""
    x = np.asarray(x, dtype=float)
    x = _normalize(x)

    if P0 is None:
        K_est = wilson_K(T, 101325.0, mix)
        P0 = 101325.0 / float((x / K_est).sum())

    P = float(P0)
    K = wilson_K(T, P, mix)
    y = _normalize(K * x)

    for n_iter in range(1, max_iter + 1):
        K = _K_from_phi(T, P, x, y, mix)
        S = float(K @ x)
        y = _normalize(K * x)
        f = S - 1.0

        dP_fd = P * 1e-4
        K_p = _K_from_phi(T, P + dP_fd, x, y, mix)
        S_p = float(K_p @ x)
        df_dP = (S_p - S) / dP_fd

        if abs(df_dP) < 1e-40:
            break

        step = np.clip(-f / df_dP, -0.5 * P, 2.0 * P)
        P = max(P + step, 1.0)

        if abs(step) / P < tol:
            return BubbleResult(T=T, P=P, x=x, y=y, converged=True, n_iter=n_iter)

    return BubbleResult(T=T, P=P, x=x, y=y, converged=False, n_iter=max_iter)


# ---------------------------------------------------------------------------
# Dew-point temperature
# ---------------------------------------------------------------------------

def dew_T(
    P: float,
    y: np.ndarray,
    mix: MixtureTD,
    T0: float | None = None,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> DewResult:
    """Given P and vapor y, find dew-point T and liquid x.

    Newton on f(T) = Σᵢ yᵢ/Kᵢ − 1 = 0.
    """
    y = np.asarray(y, dtype=float)
    y = _normalize(y)

    if T0 is None:
        T0 = float(np.dot(y, [c.Tc for c in mix.components])) * 0.75

    T = float(T0)
    K = wilson_K(T, P, mix)
    x = _normalize(y / K)

    for n_iter in range(1, max_iter + 1):
        K = _K_from_phi(T, P, x, y, mix)
        S = float((y / K).sum())      # Σ yᵢ/Kᵢ; dew point when = 1
        x = _normalize(y / K)
        f = S - 1.0

        dT_fd = T * 1e-4
        K_p = _K_from_phi(T + dT_fd, P, x, y, mix)
        S_p = float((y / K_p).sum())
        df_dT = (S_p - S) / dT_fd

        if abs(df_dT) < 1e-30:
            break

        step = np.clip(-f / df_dT, -20.0, 20.0)
        T += step

        if abs(step) < tol:
            return DewResult(T=T, P=P, x=x, y=y, converged=True, n_iter=n_iter)

    return DewResult(T=T, P=P, x=x, y=y, converged=False, n_iter=max_iter)


# ---------------------------------------------------------------------------
# Dew-point pressure
# ---------------------------------------------------------------------------

def dew_P(
    T: float,
    y: np.ndarray,
    mix: MixtureTD,
    P0: float | None = None,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> DewResult:
    """Given T and vapor y, find dew-point P and liquid x."""
    y = np.asarray(y, dtype=float)
    y = _normalize(y)

    if P0 is None:
        K_est = wilson_K(T, 101325.0, mix)
        P0 = 101325.0 * float((y * K_est).sum())

    P = float(P0)
    K = wilson_K(T, P, mix)
    x = _normalize(y / K)

    for n_iter in range(1, max_iter + 1):
        K = _K_from_phi(T, P, x, y, mix)
        S = float((y / K).sum())
        x = _normalize(y / K)
        f = S - 1.0

        dP_fd = P * 1e-4
        K_p = _K_from_phi(T, P + dP_fd, x, y, mix)
        S_p = float((y / K_p).sum())
        df_dP = (S_p - S) / dP_fd

        if abs(df_dP) < 1e-40:
            break

        step = np.clip(-f / df_dP, -0.5 * P, 2.0 * P)
        P = max(P + step, 1.0)

        if abs(step) / P < tol:
            return DewResult(T=T, P=P, x=x, y=y, converged=True, n_iter=n_iter)

    return DewResult(T=T, P=P, x=x, y=y, converged=False, n_iter=max_iter)
