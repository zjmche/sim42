"""Fenske-Underwood-Gilliland (FUG) shortcut column calculator.

Computes minimum stages (N_min), minimum reflux ratio (R_min), and
estimates actual stages / actual reflux for a binary or multi-component
column.

References:
  Fenske (1932) Ind. Eng. Chem. 24:482
  Underwood (1948) Chem. Eng. Prog. 44:603
  Gilliland (1940) Ind. Eng. Chem. 32:1220
  Seader & Henley (2011) §9.3
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..components.loader import Mixture
from ..equilibrium.bubble_dew import bubble_T


@dataclass
class ShortcutResult:
    N_min: float         # Fenske minimum stages (including reboiler)
    R_min: float         # Underwood minimum reflux ratio
    N_actual: float      # Gilliland estimate (at given R/R_min)
    R_actual: float      # actual reflux ratio used
    alpha: np.ndarray    # relative volatilities at avg T
    converged: bool


def fenske_Nmin(
    x_dist: np.ndarray,
    x_bot: np.ndarray,
    alpha: np.ndarray,
    hk: int,
) -> float:
    """Fenske equation for N_min.

    N_min = ln[(xD_lk/xD_hk)·(xB_hk/xB_lk)] / ln(α_lk)

    Parameters
    ----------
    x_dist : distillate mole fractions
    x_bot  : bottoms mole fractions
    alpha  : relative volatilities (to heavy key, component hk)
    hk     : index of heavy key component
    """
    lk = np.argmax(alpha[alpha > 1.0])  # light key = highest alpha > 1
    # Guard against zero fractions
    eps = 1e-12
    ratio = ((x_dist[lk] + eps) / (x_dist[hk] + eps)) * (
        (x_bot[hk] + eps) / (x_bot[lk] + eps)
    )
    return np.log(ratio) / np.log(float(alpha[lk]))


def underwood_Rmin(
    z_feed: np.ndarray,
    alpha: np.ndarray,
    q: float,
    x_dist: np.ndarray | None = None,
    theta_bounds: tuple[float, float] | None = None,
) -> float:
    """Underwood minimum reflux ratio.

    Solves Σ αᵢzᵢ/(αᵢ − θ) = 1 − q  for θ  (root between α_hk and α_lk),
    then  R_min + 1 = Σ αᵢxᵢ_dist/(αᵢ − θ).

    Parameters
    ----------
    z_feed  : feed mole fractions
    alpha   : relative volatilities normalised so α_hk = 1.0 (K / K_min)
    q       : feed quality (1=sat. liquid, 0=sat. vapor)
    x_dist  : distillate mole fractions; defaults to z_feed if None
    """
    if x_dist is None:
        x_dist = z_feed

    # alpha is normalised by K_min, so the heavy-key alpha is always 1.0
    # (minimum alpha). The light-key alpha is the maximum.
    hk_idx = int(np.argmin(alpha))   # α_hk = 1.0
    lk_idx = int(np.argmax(alpha))   # α_lk = max

    if theta_bounds is None:
        # Underwood root lies in the open interval (α_hk, α_lk)
        lo = float(alpha[hk_idx]) + 1e-8
        hi = float(alpha[lk_idx]) - 1e-8
    else:
        lo, hi = theta_bounds

    if lo >= hi:
        # Degenerate (e.g. all components have the same alpha) — no separation
        return 0.0

    def f_theta(theta: float) -> float:
        return float(np.sum(alpha * z_feed / (alpha - theta))) - (1.0 - q)

    try:
        # Check that the bracket is valid (function changes sign)
        if f_theta(lo) * f_theta(hi) < 0:
            theta = brentq(f_theta, lo, hi, xtol=1e-10)
        else:
            theta = (lo + hi) / 2.0
    except (ValueError, ZeroDivisionError):
        theta = (lo + hi) / 2.0

    R_min = float(np.sum(alpha * x_dist / (alpha - theta))) - 1.0
    return max(R_min, 0.0)


def gilliland_N(N_min: float, R_min: float, R: float) -> float:
    """Gilliland correlation (Molokanov approximation).

    X = (R - R_min)/(R + 1)
    Y = (N - N_min)/(N + 1)
    Y = 1 - exp[(1+54.4X)/(11+117.2X)·(X-1)/X^0.5]
    """
    if R <= R_min or R_min < 0:
        return float("inf")
    X = (R - R_min) / (R + 1.0)
    if X < 1e-12:
        return float("inf")
    Y = 1.0 - np.exp((1.0 + 54.4 * X) / (11.0 + 117.2 * X) * (X - 1.0) / X**0.5)
    if Y >= 1.0 - 1e-12:
        return float("inf")
    N = (N_min + Y) / (1.0 - Y)
    return float(N)


def shortcut_column(
    z_feed: np.ndarray,
    x_dist: np.ndarray,
    x_bot: np.ndarray,
    P: float,
    q: float,
    mix: Mixture,
    R_ratio: float = 1.5,   # R/R_min
    hk: int | None = None,
) -> ShortcutResult:
    """Run Fenske-Underwood-Gilliland shortcut calculation.

    Parameters
    ----------
    z_feed  : feed mole fractions
    x_dist  : distillate mole fractions (specification)
    x_bot   : bottoms mole fractions (specification)
    P       : column pressure [Pa]
    q       : feed quality
    mix     : Mixture
    R_ratio : actual / minimum reflux ratio multiplier
    hk      : heavy-key component index (None → auto-detect as 2nd highest α)
    """
    z_feed = np.asarray(z_feed, dtype=float); z_feed /= z_feed.sum()
    x_dist = np.asarray(x_dist, dtype=float); x_dist /= x_dist.sum()
    x_bot  = np.asarray(x_bot,  dtype=float); x_bot  /= x_bot.sum()

    # Average temperature via bubble point of feed
    bub = bubble_T(P, z_feed, mix)
    T_avg = bub.T

    # Relative volatilities at average T
    from ..thermo.pr_eos import K_values as _kv
    K_avg = _kv(T_avg, P, z_feed, bub.y, mix)
    alpha = K_avg / K_avg.min()   # normalise to heavy key (smallest K)

    if hk is None:
        hk = int(np.argmin(K_avg))

    N_min = fenske_Nmin(x_dist, x_bot, alpha, hk)
    R_min = underwood_Rmin(z_feed, alpha, q, x_dist=x_dist)
    R_actual = R_ratio * R_min
    N_actual = gilliland_N(N_min, R_min, R_actual)

    return ShortcutResult(
        N_min=N_min,
        R_min=R_min,
        N_actual=N_actual,
        R_actual=R_actual,
        alpha=alpha,
        converged=bub.converged,
    )
