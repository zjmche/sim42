"""Isothermal PT-flash (Rachford-Rice + successive substitution).

Algorithm:
  1. Wilson K-values as initial guess.
  2. Rachford-Rice equation for β (vapor fraction) via Newton.
  3. x, y from β and K.
  4. Update K = φ_L / φ_V via PR-EOS.
  5. Repeat 2-4 until convergence.
  6. Trivial-solution guard: reject K→1 collapse.

Reference: Rachford & Rice (1952); Reid, Prausnitz & Poling (1987) §3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..components.loader import Mixture
from ..thermo.pr_eos import K_values as _K_values, ln_phi, wilson_K


@dataclass
class FlashResult:
    beta: float             # vapor fraction [0, 1]
    x: np.ndarray           # liquid mole fractions
    y: np.ndarray           # vapor mole fractions
    K: np.ndarray           # K-values at convergence
    converged: bool
    n_iter: int
    phase: str              # 'two-phase', 'liquid', 'vapor'


def _rachford_rice(beta: float, z: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    """g(β) and g'(β) for Rachford-Rice equation."""
    denom = 1.0 + beta * (K - 1.0)
    g = float(np.sum(z * (K - 1.0) / denom))
    gp = float(-np.sum(z * (K - 1.0) ** 2 / denom**2))
    return g, gp


def _solve_rr(z: np.ndarray, K: np.ndarray, tol: float = 1e-10) -> float:
    """Solve Rachford-Rice for β by Newton with bounds [0, 1]."""
    # Boundary check: single-phase?
    g0, _ = _rachford_rice(0.0, z, K)
    g1, _ = _rachford_rice(1.0, z, K)
    if g0 <= 0.0:
        return 0.0   # subcooled liquid
    if g1 >= 0.0:
        return 1.0   # superheated vapor

    beta = 0.5  # start in middle
    for _ in range(50):
        g, gp = _rachford_rice(beta, z, K)
        if abs(g) < tol:
            break
        step = -g / gp
        # Clip to (0, 1) with margin
        beta = np.clip(beta + step, 1e-6, 1.0 - 1e-6)
    return beta


def _split(beta: float, z: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute x, y from β and K via Rachford-Rice split."""
    denom = 1.0 + beta * (K - 1.0)
    x = z / denom
    x = np.maximum(x, 0.0)
    x /= x.sum()
    y = K * x
    y = np.maximum(y, 0.0)
    y /= y.sum()
    return x, y


def flash_PT(
    T: float,
    P: float,
    z: np.ndarray,
    mix: Mixture,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> FlashResult:
    """Isothermal PT-flash.

    Parameters
    ----------
    T : float  Temperature [K]
    P : float  Pressure [Pa]
    z : array  Overall mole fractions, shape (n,)
    mix : Mixture
    tol : float  Convergence on max |ln(K_new/K_old)|
    max_iter : int

    Returns
    -------
    FlashResult
    """
    z = np.asarray(z, dtype=float)
    z = z / z.sum()

    # Initial K from Wilson
    K = wilson_K(T, P, mix)

    beta = _solve_rr(z, K)

    if beta <= 0.0:
        # Pure liquid
        phi_L = np.exp(ln_phi(T, P, z, mix, "liquid"))
        return FlashResult(
            beta=0.0, x=z.copy(), y=z.copy(),
            K=K, converged=True, n_iter=0, phase="liquid"
        )
    if beta >= 1.0:
        # Pure vapor
        phi_V = np.exp(ln_phi(T, P, z, mix, "vapor"))
        return FlashResult(
            beta=1.0, x=z.copy(), y=z.copy(),
            K=K, converged=True, n_iter=0, phase="vapor"
        )

    x, y = _split(beta, z, K)
    converged = False

    for n_iter in range(1, max_iter + 1):
        K_new = _K_values(T, P, x, y, mix)

        # Trivial-solution guard: if all K→1, reject
        if np.max(np.abs(K_new - 1.0)) < 1e-5:
            # Flash has collapsed; determine phase from enthalpy vs dew/bubble
            break

        err = np.max(np.abs(np.log(K_new / K)))
        K = K_new

        beta = _solve_rr(z, K)
        if beta <= 0.0:
            return FlashResult(beta=0.0, x=z.copy(), y=z.copy(),
                               K=K, converged=True, n_iter=n_iter, phase="liquid")
        if beta >= 1.0:
            return FlashResult(beta=1.0, x=z.copy(), y=z.copy(),
                               K=K, converged=True, n_iter=n_iter, phase="vapor")

        x, y = _split(beta, z, K)

        if err < tol:
            converged = True
            break

    return FlashResult(
        beta=beta, x=x, y=y, K=K,
        converged=converged, n_iter=n_iter, phase="two-phase"
    )
