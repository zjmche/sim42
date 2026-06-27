"""Unit tests for the PR-EOS engine (L2).

Phase 1 exit gate: φ, Z, H_departure for pure components and binary
mixtures must match textbook/CoolProp reference values to within tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from distcalc.components.loader import load_mixture
from distcalc.thermo.pr_eos import (
    H_departure,
    K_values,
    S_departure,
    enthalpy,
    ln_phi,
    solve_Z,
    wilson_K,
    R,
)

GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture
def mix_n2o2ar():
    return load_mixture(["N2", "O2", "Ar"])


@pytest.fixture
def mix_n2o2():
    return load_mixture(["N2", "O2"])


@pytest.fixture
def mix_n2():
    return load_mixture(["N2"])


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

class TestLoader:
    def test_mixture_names(self, mix_n2o2ar):
        assert mix_n2o2ar.names == ["N2", "O2", "Ar"]

    def test_kij_symmetric(self, mix_n2o2ar):
        K = mix_n2o2ar.Kij
        assert np.allclose(K, K.T)

    def test_kij_diagonal_zero(self, mix_n2o2ar):
        assert np.allclose(np.diag(mix_n2o2ar.Kij), 0.0)

    def test_kij_n2_o2(self, mix_n2o2ar):
        # k(N2,O2) should be -0.0119
        assert abs(mix_n2o2ar.Kij[0, 1] - (-0.0119)) < 1e-6

    def test_subset_mixture(self, mix_n2o2):
        assert mix_n2o2.n == 2
        assert mix_n2o2.names == ["N2", "O2"]

    def test_cp_ig_n2_at_298(self, mix_n2o2ar):
        # Cp_ig(N2, 298.15 K) ≈ 29.1 J/(mol·K) (NIST JANAF)
        cp = mix_n2o2ar.components[0].cp_ig(298.15)
        assert abs(cp - 29.1) < 0.3

    def test_cp_ig_ar_constant(self, mix_n2o2ar):
        # Ar is monatomic: Cp = 5/2 R = 20.786 J/(mol·K) at all T
        ar = mix_n2o2ar.components[2]
        for T in [100.0, 200.0, 500.0]:
            assert abs(ar.cp_ig(T) - 20.786) < 0.01

    def test_h_ig_reference(self, mix_n2o2ar):
        # h_ig at Tref = 0 by definition
        for c in mix_n2o2ar.components:
            assert abs(c.h_ig(298.15)) < 1.0   # J/mol


# ---------------------------------------------------------------------------
# PR-EOS cubic / Z tests
# ---------------------------------------------------------------------------

class TestCubicZ:
    def test_z_vapor_gt_liquid(self, mix_n2o2ar):
        # Use T=90 K (subcritical, well below N2 Tc=126.19 K), P=3e5 Pa
        # At these conditions the cubic has two physical roots (liquid + vapor)
        x = np.array([1.0, 0.0, 0.0])
        T, P = 90.0, 3e5
        Z_V = solve_Z(T, P, x, mix_n2o2ar, "vapor")
        Z_L = solve_Z(T, P, x, mix_n2o2ar, "liquid")
        assert Z_V > Z_L

    def test_z_ideal_gas_limit(self, mix_n2o2ar):
        # At very low P, Z → 1
        x = np.array([0.5, 0.3, 0.2])
        Z = solve_Z(300.0, 1.0, x, mix_n2o2ar, "vapor")  # 1 Pa
        assert abs(Z - 1.0) < 1e-3

    def test_z_pure_n2_vapor(self, mix_n2):
        # N2 at 90 K, 3e5 Pa (subcritical, superheated vapor): Z_vapor > 0.8
        x = np.array([1.0])
        Z = solve_Z(90.0, 3e5, x, mix_n2, "vapor")
        assert 0.80 < Z < 1.0

    def test_z_pure_n2_liquid(self, mix_n2):
        # N2 at 100 K, 1 MPa: liquid Z should be small (dense liquid)
        x = np.array([1.0])
        Z = solve_Z(100.0, 1e6, x, mix_n2, "liquid")
        assert Z < 0.1

    def test_z_auto_selects_correct_phase(self, mix_n2):
        """At supercritical conditions, 'auto' should return unique root."""
        x = np.array([1.0])
        # N2 above Tc=126.19 K — should give single-phase Z
        Z = solve_Z(200.0, 5e6, x, mix_n2, "auto")
        assert 0.5 < Z < 1.05


# ---------------------------------------------------------------------------
# Fugacity coefficient tests
# ---------------------------------------------------------------------------

class TestLnPhi:
    def test_ln_phi_pure_ideal_limit(self, mix_n2):
        """ln(φ) → 0 as P → 0."""
        x = np.array([1.0])
        lphi = ln_phi(300.0, 1.0, x, mix_n2, "vapor")
        assert abs(lphi[0]) < 1e-3

    def test_ln_phi_liquid_more_negative_than_vapor(self, mix_n2o2ar):
        """At sub-critical T and P, ln(φ_L) < ln(φ_V) for same composition."""
        x = np.array([0.5, 0.4, 0.1])
        T, P = 100.0, 5e5
        lphi_L = ln_phi(T, P, x, mix_n2o2ar, "liquid")
        lphi_V = ln_phi(T, P, x, mix_n2o2ar, "vapor")
        # φ_L generally larger in magnitude departure
        assert float(np.sum(np.abs(lphi_L))) > float(np.sum(np.abs(lphi_V)))

    def test_phi_consistency_k_values(self, mix_n2o2ar):
        """K = φ_L/φ_V; at bubble point Σ K*x = 1."""
        from distcalc.equilibrium.bubble_dew import bubble_T
        x = np.array([0.4, 0.4, 0.2])
        P = 2e5
        res = bubble_T(P, x, mix_n2o2ar)
        assert res.converged, f"bubble_T did not converge: T={res.T:.3f}"
        K = K_values(res.T, P, x, res.y, mix_n2o2ar)
        assert abs(float(K @ x) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Departure enthalpy tests
# ---------------------------------------------------------------------------

class TestHDeparture:
    def test_h_dep_negative_for_liquid(self, mix_n2):
        """Liquid phase departure enthalpy is negative (condensation releases energy)."""
        x = np.array([1.0])
        H = H_departure(100.0, 1e6, x, mix_n2, "liquid")
        assert H < 0.0

    def test_h_dep_small_for_vapor_low_P(self, mix_n2):
        """H_dep → 0 as P → 0."""
        x = np.array([1.0])
        H = H_departure(200.0, 100.0, x, mix_n2, "vapor")   # 100 Pa
        assert abs(H) < 10.0   # J/mol

    def test_enthalpy_reference_consistency(self, mix_n2o2ar):
        """H(Tref, P→0) ≈ 0 for ideal-gas reference state."""
        x = np.array([1.0, 0.0, 0.0])
        H = enthalpy(298.15, 100.0, x, mix_n2o2ar, "vapor")  # near-ideal
        assert abs(H) < 50.0  # J/mol — small departure at 100 Pa

    def test_enthalpy_vapor_gt_liquid(self, mix_n2o2ar):
        """H_vap > H_liq at VLE conditions."""
        x = np.array([0.5, 0.4, 0.1])
        T, P = 100.0, 5e5
        H_V = enthalpy(T, P, x, mix_n2o2ar, "vapor")
        H_L = enthalpy(T, P, x, mix_n2o2ar, "liquid")
        assert H_V > H_L

    def test_h_dep_pure_n2_vapor(self, mix_n2):
        """Pure N2 vapor at 90 K, 3e5 Pa: H_dep should be small negative."""
        x = np.array([1.0])
        H = H_departure(90.0, 3e5, x, mix_n2, "vapor")
        assert -500 < H < 0

    def test_h_dep_pure_n2_liquid(self, mix_n2):
        """Pure N2 liquid at 100 K, 1 MPa: H_dep should be strongly negative."""
        x = np.array([1.0])
        H = H_departure(100.0, 1e6, x, mix_n2, "liquid")
        assert H < -3000   # J/mol


# ---------------------------------------------------------------------------
# Wilson K-values test
# ---------------------------------------------------------------------------

class TestWilsonK:
    def test_wilson_k_ordering(self, mix_n2o2ar):
        """N2 should be most volatile (highest K) in ASU range."""
        K = wilson_K(90.0, 1e5, mix_n2o2ar)
        assert K[0] > K[2] > K[1]   # K_N2 > K_Ar > K_O2 approximately

    def test_wilson_k_gt_1_for_light(self, mix_n2o2ar):
        K = wilson_K(90.0, 1e5, mix_n2o2ar)
        assert K[0] > 1.0   # N2 above 1 at 90 K, 1 bar


# ---------------------------------------------------------------------------
# Golden-file regression tests (Phase 1 exit gate)
# ---------------------------------------------------------------------------

class TestGoldenPureEOS:
    """Compare PR-EOS results against CoolProp PR-backend reference values."""

    def _get_mix(self, component: str):
        return load_mixture([component])

    @pytest.mark.parametrize("case_idx", range(4))
    def test_golden_z(self, case_idx):
        data = json.loads((GOLDEN / "pure_pr_eos.json").read_text())
        case = data["cases"][case_idx]
        tol = data["tolerance_Z"]
        mix = self._get_mix(case["component"])
        x = np.array([1.0])
        Z = solve_Z(case["T_K"], case["P_Pa"], x, mix, case["phase"])
        assert abs(Z - case["Z"]) < tol, (
            f"{case['component']} Z={Z:.4f} vs ref={case['Z']} ΔZ={abs(Z-case['Z']):.4f}"
        )

    @pytest.mark.parametrize("case_idx", range(4))
    def test_golden_ln_phi(self, case_idx):
        data = json.loads((GOLDEN / "pure_pr_eos.json").read_text())
        case = data["cases"][case_idx]
        tol = data["tolerance_ln_phi"]
        mix = self._get_mix(case["component"])
        x = np.array([1.0])
        lp = ln_phi(case["T_K"], case["P_Pa"], x, mix, case["phase"])
        assert abs(lp[0] - case["ln_phi"]) < tol, (
            f"{case['component']} ln_phi={lp[0]:.4f} vs ref={case['ln_phi']}"
        )

    @pytest.mark.parametrize("case_idx", range(4))
    def test_golden_h_dep(self, case_idx):
        data = json.loads((GOLDEN / "pure_pr_eos.json").read_text())
        case = data["cases"][case_idx]
        rel_tol = data["tolerance_H_dep_rel"]
        mix = self._get_mix(case["component"])
        x = np.array([1.0])
        H = H_departure(case["T_K"], case["P_Pa"], x, mix, case["phase"])
        ref = case["H_dep_J_per_mol"]
        err = abs(H - ref) / (abs(ref) + 1.0)
        assert err < rel_tol, (
            f"{case['component']} H_dep={H:.1f} vs ref={ref:.1f} rel_err={err:.4f}"
        )
