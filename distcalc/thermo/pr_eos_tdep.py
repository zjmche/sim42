"""Peng-Robinson EOS engine — temperature-dependent Kij variant.

Parallel to pr_eos.py — does not modify it. Same Z/ln(phi)/H_dep/S_dep
formulation, but the binary interaction parameter is Kij(T) = a + b/T
(regressed against CoolProp HEOS data; see components_tdep.json) instead
of a constant matrix.

Because Kij now depends on T, the mixing-rule temperature derivative
da_mix/dT picks up an extra term from dKij/dT:

    a_ij(T)    = sqrt(ai(T)*aj(T)) * (1 - Kij(T))
    da_ij/dT   = d[sqrt(ai*aj)]/dT * (1 - Kij(T))  -  sqrt(ai*aj) * dKij/dT

This is required so H_departure and S_departure (which are built from
da_mix/dT) stay analytically consistent with the Gibbs energy surface
implied by a_mix(T,x) — see Poling/Prausnitz/O'Connell Ch. 6 and the
Gibbs-Duhem relation d(ln phi)/dT = -H_dep/(R T^2).

All inputs/outputs in SI units (Pa, K, J/mol, J/(mol*K)).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..components.loader_tdep import MixtureTD

Phase = Literal["vapor", "liquid", "auto"]

R = 8.314462        # J/(mol*K)
SQRT2 = np.sqrt(2.0)
_1P = 1.0 + SQRT2   # 1 + sqrt(2)
_1M = 1.0 - SQRT2   # 1 - sqrt(2)
_2S2 = 2.0 * SQRT2  # 2*sqrt(2)


# ---------------------------------------------------------------------------
# Per-component PR parameters (identical physics to pr_eos.py)
# ---------------------------------------------------------------------------

def _kappa(omega: float) -> float:
    return 0.37464 + 1.54226 * omega - 0.26992 * omega**2


def _ai_vec(T: float, mix: MixtureTD) -> np.ndarray:
    """aᵢ(T) for every component, shape (n,). Mathias-Copeman alpha."""
    out = np.empty(mix.n)
    for i, c in enumerate(mix.components):
        if c.mc_alpha is not None:
            c1, c2, c3 = c.mc_alpha
            x = 1.0 - (T / c.Tc) ** 0.5
            sq = 1.0 + c1 * x + c2 * x**2 + c3 * x**3
        else:
            kap = _kappa(c.omega)
            sq = 1.0 + kap * (1.0 - (T / c.Tc) ** 0.5)
        out[i] = 0.45724 * R**2 * c.Tc**2 / c.Pc * sq**2
    return out


def _bi_vec(mix: MixtureTD) -> np.ndarray:
    """bᵢ for every component (T-independent), shape (n,)."""
    return np.array([0.07780 * R * c.Tc / c.Pc for c in mix.components])


def _dai_dT_vec(T: float, mix: MixtureTD, a_vec: np.ndarray) -> np.ndarray:
    """d(aᵢ)/dT for every component, shape (n,)."""
    out = np.empty(mix.n)
    for i, c in enumerate(mix.components):
        if c.mc_alpha is not None:
            c1, c2, c3 = c.mc_alpha
            x = 1.0 - (T / c.Tc) ** 0.5
            sq = 1.0 + c1 * x + c2 * x**2 + c3 * x**3   # = sqrt(alpha_i)
            dsq_dx = c1 + 2.0 * c2 * x + 3.0 * c3 * x**2
            out[i] = -0.45724 * R**2 * c.Tc**2 / c.Pc * sq * dsq_dx / (T * c.Tc) ** 0.5
        else:
            kap = _kappa(c.omega)
            sq = 1.0 + kap * (1.0 - (T / c.Tc) ** 0.5)   # = sqrt(alpha_i)
            out[i] = -0.45724 * R**2 * c.Tc**2 / c.Pc * kap * sq / (T * c.Tc) ** 0.5
    return out


# ---------------------------------------------------------------------------
# Mixture parameters (Kij now a function of T)
# ---------------------------------------------------------------------------

def _mix_params(
    T: float, P: float, x: np.ndarray, mix: MixtureTD
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """
    Returns (a_vec, b_vec, a_ij, a_mix, b_mix, A, B).

    a_ij[i,j] = √(aᵢaⱼ)·(1−kᵢⱼ(T))  (symmetric, shape n×n)
    A = a_mix·P/(RT)²
    B = b_mix·P/(RT)
    """
    a_vec = _ai_vec(T, mix)
    b_vec = _bi_vec(mix)
    sqrt_a = np.sqrt(a_vec)
    Kij_T = mix.Kij_matrix(T)
    a_ij = np.outer(sqrt_a, sqrt_a) * (1.0 - Kij_T)
    a_mix = float(x @ a_ij @ x)
    b_mix = float(x @ b_vec)
    RT = R * T
    A = a_mix * P / RT**2
    B = b_mix * P / RT
    return a_vec, b_vec, a_ij, a_mix, b_mix, A, B


def _da_mix_dT(T: float, x: np.ndarray, mix: MixtureTD, a_vec: np.ndarray) -> float:
    """d(a_mix)/dT, including the dKij/dT term (analytic consistency).

    d(√aᵢ·√aⱼ)/dT = (daᵢ/dT)/(2√aᵢ)·√aⱼ + √aᵢ·(daⱼ/dT)/(2√aⱼ)
    da_ij/dT = d(√aᵢ·√aⱼ)/dT·(1−Kij(T)) − √aᵢ·√aⱼ·dKij/dT
    """
    da_vec = _dai_dT_vec(T, mix, a_vec)
    sqrt_a = np.sqrt(a_vec)
    d_sqrt_a = da_vec / (2.0 * sqrt_a)
    d_sqrt_aiaj = np.outer(d_sqrt_a, sqrt_a) + np.outer(sqrt_a, d_sqrt_a)
    sqrt_aiaj = np.outer(sqrt_a, sqrt_a)
    Kij_T = mix.Kij_matrix(T)
    dKij_dT = mix.dKij_dT_matrix(T)
    da_ij_dT = d_sqrt_aiaj * (1.0 - Kij_T) - sqrt_aiaj * dKij_dT
    return float(x @ da_ij_dT @ x)


# ---------------------------------------------------------------------------
# Cubic solver (identical to pr_eos.py)
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
    """
    coeffs = [1.0, -(1.0 - B), A - 3*B**2 - 2*B, -(A*B - B**2 - B**3)]
    roots = np.roots(coeffs)
    real_roots = sorted(
        [r.real for r in roots if abs(r.imag) < 1e-8 and r.real > B + 1e-10],
        reverse=True,
    )
    if not real_roots:
        raise ValueError(f"No physical Z root: A={A:.5g}, B={B:.5g}")

    if len(real_roots) == 1:
        return real_roots[0]

    Z_V, Z_L = real_roots[0], real_roots[-1]

    if phase == "vapor":
        return Z_V
    if phase == "liquid":
        return Z_L
    return Z_V if _g_residual(Z_V, A, B) <= _g_residual(Z_L, A, B) else Z_L


# ---------------------------------------------------------------------------
# Public thermodynamic functions
# ---------------------------------------------------------------------------

def solve_Z(
    T: float, P: float, x: np.ndarray, mix: MixtureTD, phase: Phase = "auto"
) -> float:
    """Compressibility factor Z for the specified phase."""
    _, _, _, _, _, A, B = _mix_params(T, P, x, mix)
    return _solve_cubic(A, B, phase)


def ln_phi(
    T: float, P: float, x: np.ndarray, mix: MixtureTD, phase: Phase = "auto"
) -> np.ndarray:
    """Log fugacity coefficients for each component, shape (n,)."""
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
    T: float, P: float, x: np.ndarray, mix: MixtureTD, phase: Phase = "auto"
) -> float:
    """Molar departure enthalpy H_dep = H − H_ig [J/mol]."""
    a_vec, _, _, a_mix, b_mix, A, B = _mix_params(T, P, x, mix)
    Z = _solve_cubic(A, B, phase)
    da_dT = _da_mix_dT(T, x, mix, a_vec)
    lt = _log_ratio(Z, B)
    return R * T * (Z - 1.0) + (T * da_dT - a_mix) / (_2S2 * b_mix) * lt


def S_departure(
    T: float, P: float, x: np.ndarray, mix: MixtureTD, phase: Phase = "auto"
) -> float:
    """Molar departure entropy S_dep = S − S_ig(T,P) [J/(mol·K)]."""
    a_vec, _, _, _, b_mix, A, B = _mix_params(T, P, x, mix)
    Z = _solve_cubic(A, B, phase)
    da_dT = _da_mix_dT(T, x, mix, a_vec)
    lt = _log_ratio(Z, B)
    return R * np.log(Z - B) + da_dT / (_2S2 * b_mix) * lt


def enthalpy(
    T: float, P: float, x: np.ndarray, mix: MixtureTD, phase: Phase = "auto"
) -> float:
    """Total molar enthalpy [J/mol]. Same ideal-gas reference as pr_eos.py."""
    h_ig = float(sum(xi * c.h_ig(T) for xi, c in zip(x, mix.components)))
    return h_ig + H_departure(T, P, x, mix, phase)


def K_values(
    T: float, P: float, x: np.ndarray, y: np.ndarray, mix: MixtureTD
) -> np.ndarray:
    """Equilibrium K-values: Kᵢ = φᵢ_L / φᵢ_V, shape (n,)."""
    phi_L = np.exp(ln_phi(T, P, x, mix, "liquid"))
    phi_V = np.exp(ln_phi(T, P, y, mix, "vapor"))
    return phi_L / phi_V


def wilson_K(T: float, P: float, mix: MixtureTD) -> np.ndarray:
    """Wilson (1969) K-value estimate. Fast init for flash/bubble/dew solvers."""
    return np.array(
        [
            c.Pc / P * np.exp(5.373 * (1.0 + c.omega) * (1.0 - c.Tc / T))
            for c in mix.components
        ]
    )


def density(T: float, P: float, x: np.ndarray, mix: MixtureTD, phase: Phase = "auto") -> float:
    """Molar density [mol/m³]."""
    Z = solve_Z(T, P, x, mix, phase)
    return P / (R * T * Z)
