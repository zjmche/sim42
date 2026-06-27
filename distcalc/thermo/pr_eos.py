"""Peng-Robinson EOS engine.

Implements Z, ln(φ), H_departure, S_departure for mixtures with vdW
one-fluid mixing rules and a symmetric Kij matrix.

All inputs/outputs in SI units (Pa, K, J/mol, J/(mol·K)).

Reference: Peng & Robinson (1976) Ind. Eng. Chem. Fundam. 15:59.
Departure functions: Poling, Prausnitz, O'Connell, The Properties of
Gases and Liquids, 5th ed. (2001), Chapter 6.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..components.loader import Mixture

Phase = Literal["vapor", "liquid", "auto"]

R = 8.314462        # J/(mol·K)
SQRT2 = np.sqrt(2.0)
_1P = 1.0 + SQRT2   # 1 + √2
_1M = 1.0 - SQRT2   # 1 − √2
_2S2 = 2.0 * SQRT2  # 2√2

_TREF = 298.15  # K — must match loader.py


# ---------------------------------------------------------------------------
# Per-component PR parameters
# ---------------------------------------------------------------------------

def _kappa(omega: float) -> float:
    return 0.37464 + 1.54226 * omega - 0.26992 * omega**2


def _ai_vec(T: float, mix: Mixture) -> np.ndarray:
    """aᵢ(T) for every component, shape (n,)."""
    out = np.empty(mix.n)
    for i, c in enumerate(mix.components):
        kap = _kappa(c.omega)
        sq = 1.0 + kap * (1.0 - (T / c.Tc) ** 0.5)
        out[i] = 0.45724 * R**2 * c.Tc**2 / c.Pc * sq**2
    return out


def _bi_vec(mix: Mixture) -> np.ndarray:
    """bᵢ for every component (T-independent), shape (n,)."""
    return np.array([0.07780 * R * c.Tc / c.Pc for c in mix.components])


def _dai_dT_vec(T: float, mix: Mixture, a_vec: np.ndarray) -> np.ndarray:
    """d(aᵢ)/dT for every component, shape (n,).

    From dαᵢ/dT = −κᵢ·√αᵢ / √(T·Tcᵢ)  →  daᵢ/dT = 0.45724·R²Tc²/Pc · dαᵢ/dT.
    """
    out = np.empty(mix.n)
    for i, c in enumerate(mix.components):
        kap = _kappa(c.omega)
        sq = 1.0 + kap * (1.0 - (T / c.Tc) ** 0.5)   # = √αᵢ
        out[i] = -0.45724 * R**2 * c.Tc**2 / c.Pc * kap * sq / (T * c.Tc) ** 0.5
    return out


# ---------------------------------------------------------------------------
# Mixture parameters
# ---------------------------------------------------------------------------

def _mix_params(
    T: float, P: float, x: np.ndarray, mix: Mixture
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """
    Returns (a_vec, b_vec, a_ij, a_mix, b_mix, A, B).

    a_ij[i,j] = √(aᵢaⱼ)·(1−kᵢⱼ)  (symmetric, shape n×n)
    A = a_mix·P/(RT)²
    B = b_mix·P/(RT)
    """
    a_vec = _ai_vec(T, mix)
    b_vec = _bi_vec(mix)
    sqrt_a = np.sqrt(a_vec)
    a_ij = np.outer(sqrt_a, sqrt_a) * (1.0 - mix.Kij)
    a_mix = float(x @ a_ij @ x)
    b_mix = float(x @ b_vec)
    RT = R * T
    A = a_mix * P / RT**2
    B = b_mix * P / RT
    return a_vec, b_vec, a_ij, a_mix, b_mix, A, B


def _da_mix_dT(T: float, x: np.ndarray, mix: Mixture, a_vec: np.ndarray) -> float:
    """d(a_mix)/dT.

    d(√aᵢ·√aⱼ)/dT = (daᵢ/dT)/(2√aᵢ)·√aⱼ + √aᵢ·(daⱼ/dT)/(2√aⱼ)
    """
    da_vec = _dai_dT_vec(T, mix, a_vec)
    sqrt_a = np.sqrt(a_vec)
    d_sqrt_a = da_vec / (2.0 * sqrt_a)
    # outer(d√a, √a) + outer(√a, d√a) gives d(√aᵢ·√aⱼ)/dT for each (i,j)
    da_ij_dT = (np.outer(d_sqrt_a, sqrt_a) + np.outer(sqrt_a, d_sqrt_a)) * (
        1.0 - mix.Kij
    )
    return float(x @ da_ij_dT @ x)


# ---------------------------------------------------------------------------
# Cubic solver
# ---------------------------------------------------------------------------

def _log_ratio(Z: float, B: float) -> float:
    """ln[(Z+(1+√2)B)/(Z+(1-√2)B)] — appears in departure functions."""
    return np.log((Z + _1P * B) / (Z + _1M * B))


def _g_residual(Z: float, A: float, B: float) -> float:
    """Dimensionless residual Gibbs energy. Used for root selection."""
    return Z - 1.0 - np.log(Z - B) - A / (_2S2 * B) * _log_ratio(Z, B)


def _solve_cubic(A: float, B: float, phase: Phase) -> float:
    """
    Solve Z³ − (1−B)Z² + (A−3B²−2B)Z − (AB−B²−B³) = 0 and
    return the physically correct root.

    Largest real root → vapor; smallest positive > B → liquid.
    When three real roots exist and phase='auto', select by minimum
    residual Gibbs energy (Michelsen criterion).
    """
    coeffs = [1.0, -(1.0 - B), A - 3*B**2 - 2*B, -(A*B - B**2 - B**3)]
    roots = np.roots(coeffs)
    # Keep only real, physically admissible roots (Z > B ensures V > b)
    real_roots = sorted(
        [r.real for r in roots if abs(r.imag) < 1e-8 and r.real > B + 1e-10],
        reverse=True,
    )
    if not real_roots:
        raise ValueError(f"No physical Z root: A={A:.5g}, B={B:.5g}")

    if len(real_roots) == 1:
        return real_roots[0]

    # Three roots: largest=vapor, smallest=liquid
    Z_V, Z_L = real_roots[0], real_roots[-1]

    if phase == "vapor":
        return Z_V
    if phase == "liquid":
        return Z_L
    # 'auto': minimum residual Gibbs (correct stable phase)
    return Z_V if _g_residual(Z_V, A, B) <= _g_residual(Z_L, A, B) else Z_L


# ---------------------------------------------------------------------------
# Public thermodynamic functions
# ---------------------------------------------------------------------------

def solve_Z(
    T: float, P: float, x: np.ndarray, mix: Mixture, phase: Phase = "auto"
) -> float:
    """Compressibility factor Z for the specified phase."""
    _, _, _, _, _, A, B = _mix_params(T, P, x, mix)
    return _solve_cubic(A, B, phase)


def ln_phi(
    T: float, P: float, x: np.ndarray, mix: Mixture, phase: Phase = "auto"
) -> np.ndarray:
    """Log fugacity coefficients for each component, shape (n,).

    ln(φᵢ) = bᵢ/b·(Z−1) − ln(Z−B) − A/(2√2·B)·[2·Σⱼxⱼaᵢⱼ/a − bᵢ/b]·ln[…]
    """
    a_vec, b_vec, a_ij, a_mix, b_mix, A, B = _mix_params(T, P, x, mix)
    Z = _solve_cubic(A, B, phase)
    lt = _log_ratio(Z, B)
    sum_xa = a_ij @ x   # Σⱼ xⱼ·aᵢⱼ for each i
    return (
        b_vec / b_mix * (Z - 1.0)
        - np.log(Z - B)
        - A / (_2S2 * B) * (2.0 * sum_xa / a_mix - b_vec / b_mix) * lt
    )


def H_departure(
    T: float, P: float, x: np.ndarray, mix: Mixture, phase: Phase = "auto"
) -> float:
    """Molar departure enthalpy H_dep = H − H_ig [J/mol].

    H_dep = RT(Z−1) + [T·(da/dT) − a]/(2√2·b) · ln[…]
    """
    a_vec, _, _, a_mix, b_mix, A, B = _mix_params(T, P, x, mix)
    Z = _solve_cubic(A, B, phase)
    da_dT = _da_mix_dT(T, x, mix, a_vec)
    lt = _log_ratio(Z, B)
    return R * T * (Z - 1.0) + (T * da_dT - a_mix) / (_2S2 * b_mix) * lt


def S_departure(
    T: float, P: float, x: np.ndarray, mix: Mixture, phase: Phase = "auto"
) -> float:
    """Molar departure entropy S_dep = S − S_ig(T,P) [J/(mol·K)].

    S_dep = R·ln(Z−B) + (da/dT)/(2√2·b)·ln[…]
    """
    a_vec, _, _, _, b_mix, A, B = _mix_params(T, P, x, mix)
    Z = _solve_cubic(A, B, phase)
    da_dT = _da_mix_dT(T, x, mix, a_vec)
    lt = _log_ratio(Z, B)
    return R * np.log(Z - B) + da_dT / (_2S2 * b_mix) * lt


def enthalpy(
    T: float, P: float, x: np.ndarray, mix: Mixture, phase: Phase = "auto"
) -> float:
    """Total molar enthalpy [J/mol].

    H(T,P,x) = Σxᵢ·∫(Tref→T) Cp_ig,i dT + H_dep(T,P,x,phase)

    Reference state: ideal gas at Tref=298.15 K, Href=0 for all components.
    Same reference for both phases ensures energy-balance consistency.
    """
    h_ig = float(sum(xi * c.h_ig(T) for xi, c in zip(x, mix.components)))
    return h_ig + H_departure(T, P, x, mix, phase)


def K_values(
    T: float, P: float, x: np.ndarray, y: np.ndarray, mix: Mixture
) -> np.ndarray:
    """Equilibrium K-values: Kᵢ = φᵢ_L / φᵢ_V, shape (n,)."""
    phi_L = np.exp(ln_phi(T, P, x, mix, "liquid"))
    phi_V = np.exp(ln_phi(T, P, y, mix, "vapor"))
    return phi_L / phi_V


def wilson_K(T: float, P: float, mix: Mixture) -> np.ndarray:
    """Wilson (1969) K-value estimate. Fast init for flash/bubble/dew solvers.

    Kᵢ = (Pcᵢ/P)·exp[5.373(1+ωᵢ)(1−Tcᵢ/T)]
    """
    return np.array(
        [
            c.Pc / P * np.exp(5.373 * (1.0 + c.omega) * (1.0 - c.Tc / T))
            for c in mix.components
        ]
    )


def density(T: float, P: float, x: np.ndarray, mix: Mixture, phase: Phase = "auto") -> float:
    """Molar density [mol/m³]."""
    Z = solve_Z(T, P, x, mix, phase)
    return P / (R * T * Z)
