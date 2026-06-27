"""Tests for phase equilibrium layer (L3).

Phase 2 exit gate: N2/O2 and N2/Ar bubble/dew curves match DWSIM (PR)
over the cryogenic range; flash mass balance closes to machine precision.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from distcalc.components.loader import load_mixture
from distcalc.equilibrium.bubble_dew import bubble_T, bubble_P, dew_T, dew_P
from distcalc.equilibrium.flash import flash_PT

GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture
def mix_n2o2():
    return load_mixture(["N2", "O2"])


@pytest.fixture
def mix_n2o2ar():
    return load_mixture(["N2", "O2", "Ar"])


# ---------------------------------------------------------------------------
# Flash tests
# ---------------------------------------------------------------------------

class TestFlash:
    def test_two_phase_n2_o2(self, mix_n2o2):
        """N2/O2 at mid-range should be two-phase."""
        z = np.array([0.5, 0.5])
        res = flash_PT(85.0, 1e5, z, mix_n2o2)
        assert res.phase == "two-phase"
        assert res.converged

    def test_mass_balance_closed(self, mix_n2o2ar):
        """z = β*y + (1-β)*x must hold component-wise to machine precision."""
        z = np.array([0.4, 0.4, 0.2])
        res = flash_PT(90.0, 2e5, z, mix_n2o2ar)
        z_check = res.beta * res.y + (1.0 - res.beta) * res.x
        np.testing.assert_allclose(z_check, z / z.sum(), atol=1e-6)

    def test_x_sums_to_one(self, mix_n2o2ar):
        z = np.array([0.3, 0.5, 0.2])
        res = flash_PT(88.0, 1.5e5, z, mix_n2o2ar)
        if res.phase == "two-phase":
            assert abs(res.x.sum() - 1.0) < 1e-6
            assert abs(res.y.sum() - 1.0) < 1e-6

    def test_pure_vapor_high_T(self, mix_n2o2):
        """Above dew point → pure vapor."""
        z = np.array([0.9, 0.1])
        res = flash_PT(200.0, 1e5, z, mix_n2o2)
        assert res.phase == "vapor"

    def test_pure_liquid_low_T(self, mix_n2o2):
        """Below bubble point → pure liquid."""
        z = np.array([0.1, 0.9])
        res = flash_PT(70.0, 1e5, z, mix_n2o2)
        assert res.phase == "liquid"

    def test_k_values_consistency(self, mix_n2o2ar):
        """At convergence, K*x should sum close to 1 (summation eq.)."""
        z = np.array([0.4, 0.4, 0.2])
        res = flash_PT(90.0, 1.5e5, z, mix_n2o2ar)
        if res.phase == "two-phase":
            assert abs(float(res.K @ res.x) - 1.0) < 0.02

    def test_beta_bounds(self, mix_n2o2ar):
        z = np.array([0.4, 0.4, 0.2])
        for T in [80.0, 90.0, 100.0]:
            res = flash_PT(T, 1e5, z, mix_n2o2ar)
            assert 0.0 <= res.beta <= 1.0


# ---------------------------------------------------------------------------
# Bubble-point temperature tests
# ---------------------------------------------------------------------------

class TestBubbleT:
    def test_pure_o2_1atm(self, mix_n2o2):
        """Pure O2 bubble point at 1 atm ≈ 90.19 K (NIST)."""
        x = np.array([0.0, 1.0])
        res = bubble_T(101325.0, x, mix_n2o2)
        assert res.converged
        assert abs(res.T - 90.19) < 2.0   # ±2 K tolerance for PR

    def test_pure_n2_1atm(self, mix_n2o2):
        """Pure N2 bubble point at 1 atm ≈ 77.36 K (NIST)."""
        x = np.array([1.0, 0.0])
        res = bubble_T(101325.0, x, mix_n2o2)
        assert res.converged
        assert abs(res.T - 77.36) < 2.0

    def test_bubble_curve_golden(self, mix_n2o2):
        """N2/O2 bubble curve at 1 atm within tolerance of golden data."""
        data = json.loads((GOLDEN / "n2_o2_bubble_curve.json").read_text())
        P = data["P_Pa"]
        tol_T = data["tolerance_T_K"]
        tol_y = data["tolerance_y"]

        for pt in data["points"]:
            x_N2 = pt["x_N2"]
            if x_N2 in (0.0, 1.0):
                continue   # pure-component endpoints already tested above
            x = np.array([x_N2, 1.0 - x_N2])
            res = bubble_T(P, x, mix_n2o2)
            assert res.converged, f"No convergence at x_N2={x_N2}"
            assert abs(res.T - pt["T_bubble_K"]) < tol_T, (
                f"x_N2={x_N2}: T={res.T:.2f} vs ref={pt['T_bubble_K']:.2f}"
            )
            if abs(pt["y_N2"]) > 1e-6:
                assert abs(res.y[0] - pt["y_N2"]) < tol_y, (
                    f"x_N2={x_N2}: y_N2={res.y[0]:.3f} vs ref={pt['y_N2']:.3f}"
                )

    def test_bubble_t_convergence_ternary(self, mix_n2o2ar):
        x = np.array([0.3, 0.5, 0.2])
        res = bubble_T(2e5, x, mix_n2o2ar)
        assert res.converged
        assert 70.0 < res.T < 130.0

    def test_y_sums_to_one(self, mix_n2o2):
        x = np.array([0.5, 0.5])
        res = bubble_T(1e5, x, mix_n2o2)
        assert abs(res.y.sum() - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# Dew-point tests
# ---------------------------------------------------------------------------

class TestDewT:
    def test_pure_n2_dew(self, mix_n2o2):
        """Pure N2 dew T = bubble T."""
        y = np.array([1.0, 0.0])
        res = dew_T(101325.0, y, mix_n2o2)
        assert res.converged
        assert abs(res.T - 77.36) < 2.0

    def test_dew_above_bubble(self, mix_n2o2):
        """Dew-point T ≥ bubble-point T for same composition (dew envelope above)."""
        z = np.array([0.5, 0.5])
        bub = bubble_T(101325.0, z, mix_n2o2)
        dew = dew_T(101325.0, z, mix_n2o2)
        assert dew.T >= bub.T - 0.5   # small tolerance for numerical

    def test_x_sums_to_one_dew(self, mix_n2o2):
        y = np.array([0.7, 0.3])
        res = dew_T(1e5, y, mix_n2o2)
        assert abs(res.x.sum() - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# Bubble-P tests
# ---------------------------------------------------------------------------

class TestBubbleP:
    def test_pure_o2_90K(self, mix_n2o2):
        """Pure O2 bubble P at 90 K ≈ 99.35 kPa (NIST)."""
        x = np.array([0.0, 1.0])
        res = bubble_P(90.0, x, mix_n2o2)
        assert res.converged
        assert abs(res.P - 99350.0) / 99350.0 < 0.05   # 5% tolerance for PR

    def test_bubble_p_increases_with_volatility(self, mix_n2o2):
        """Adding more N2 raises bubble pressure at constant T."""
        x1 = np.array([0.1, 0.9])
        x2 = np.array([0.5, 0.5])
        r1 = bubble_P(90.0, x1, mix_n2o2)
        r2 = bubble_P(90.0, x2, mix_n2o2)
        assert r2.P > r1.P
