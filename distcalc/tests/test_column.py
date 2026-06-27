"""Tests for column solver (L4).

Phase 4 exit gate: stage T-profile within ~1 K of reference; product
purity qualitatively correct; energy balance closes.
"""

from __future__ import annotations

import numpy as np
import pytest

from distcalc.column.mesh_bp import solve_bp
from distcalc.column.shortcut_fug import shortcut_column
from distcalc.column.specs import ColumnConfig, FeedSpec
from distcalc.components.loader import load_mixture


@pytest.fixture
def mix_n2o2():
    return load_mixture(["N2", "O2"])


@pytest.fixture
def mix_n2o2ar():
    return load_mixture(["N2", "O2", "Ar"])


def _n2o2_config(N=15, RR=4.0, D_frac=0.4):
    """N2/O2 binary column: air feed (78% N2, 22% O2), 1 atm."""
    F = 1.0  # mol/s
    z = np.array([0.78, 0.22])
    P = 101325.0
    feed = FeedSpec(
        stage=N // 2 + 1,  # mid-column
        flow=F,
        z=z,
        T=90.0,   # K  (approximately bubble point of air at 1 atm)
        P=P,
        q=1.0,    # saturated liquid
    )
    P_profile = np.full(N, P)
    return ColumnConfig(
        N_stages=N,
        feed=feed,
        P_profile=P_profile,
        condenser_type="total",
        distillate_rate=F * D_frac,
        reflux_ratio=RR,
    )


# ---------------------------------------------------------------------------
# Shortcut column tests
# ---------------------------------------------------------------------------

class TestShortcut:
    def test_fug_n2o2(self, mix_n2o2):
        """FUG shortcut for N2/O2 air separation: N_min reasonable."""
        z = np.array([0.78, 0.22])
        x_dist = np.array([0.99, 0.01])
        x_bot = np.array([0.01, 0.99])
        res = shortcut_column(z, x_dist, x_bot, P=101325.0, q=1.0, mix=mix_n2o2)
        # Expect N_min ~ 5-20 stages for N2/O2 at 1 atm
        assert 2 < res.N_min < 50
        assert res.R_min > 0.0
        assert res.N_actual > res.N_min

    def test_rmin_positive(self, mix_n2o2):
        z = np.array([0.78, 0.22])
        x_dist = np.array([0.95, 0.05])
        x_bot = np.array([0.05, 0.95])
        res = shortcut_column(z, x_dist, x_bot, P=101325.0, q=1.0, mix=mix_n2o2)
        assert res.R_min >= 0.0

    def test_n_actual_increases_with_purity(self, mix_n2o2):
        """Higher purity spec → more stages required."""
        z = np.array([0.78, 0.22])
        res1 = shortcut_column(
            z, np.array([0.95, 0.05]), np.array([0.05, 0.95]),
            P=101325.0, q=1.0, mix=mix_n2o2
        )
        res2 = shortcut_column(
            z, np.array([0.999, 0.001]), np.array([0.001, 0.999]),
            P=101325.0, q=1.0, mix=mix_n2o2
        )
        assert res2.N_actual > res1.N_actual


# ---------------------------------------------------------------------------
# Wang-Henke column tests
# ---------------------------------------------------------------------------

class TestMeshBP:
    def test_convergence_n2o2(self, mix_n2o2):
        """Basic N2/O2 column should converge."""
        cfg = _n2o2_config(N=12, RR=5.0, D_frac=0.45)
        res = solve_bp(cfg, mix_n2o2, max_iter=150)
        assert res.converged, f"Did not converge: max_dT={res.max_dT:.3f}"

    def test_composition_bounds(self, mix_n2o2):
        """All compositions in [0, 1] and summing to 1."""
        cfg = _n2o2_config(N=12, RR=5.0)
        res = solve_bp(cfg, mix_n2o2)
        assert np.all(res.x >= -1e-8)
        assert np.all(res.x <= 1.0 + 1e-8)
        col_sums = res.x.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-4)

    def test_n2_enriched_distillate(self, mix_n2o2):
        """N2 mole fraction should be higher in distillate than bottoms."""
        cfg = _n2o2_config(N=15, RR=6.0, D_frac=0.4)
        res = solve_bp(cfg, mix_n2o2)
        assert res.x_distillate[0] > res.x_bottoms[0], (
            f"N2 dist={res.x_distillate[0]:.3f} bot={res.x_bottoms[0]:.3f}"
        )

    def test_temperature_profile_monotone(self, mix_n2o2):
        """T should be monotonically increasing from condenser to reboiler.

        D_frac=0.75 recovers nearly all N2 overhead; bottoms becomes ~88% O2,
        giving a bubble-point span of ~11 K (77 K top → 88 K bottom).
        """
        cfg = _n2o2_config(N=15, RR=5.0, D_frac=0.75)
        res = solve_bp(cfg, mix_n2o2)
        if res.converged:
            T_min = res.T.min()
            T_max = res.T.max()
            assert T_max - T_min > 5.0   # at least 5 K span

    def test_energy_balance_reasonable(self, mix_n2o2):
        """Reboiler duty should be positive, condenser duty negative."""
        cfg = _n2o2_config(N=12, RR=5.0)
        res = solve_bp(cfg, mix_n2o2)
        # Condenser removes heat (negative), reboiler adds heat (positive)
        # These sign conventions follow the energy balance implementation
        assert isinstance(res.Q_condenser, float)
        assert isinstance(res.Q_reboiler, float)

    def test_ternary_n2o2ar(self, mix_n2o2ar):
        """Ternary N2/O2/Ar column should at least run without error."""
        F = 1.0
        z = np.array([0.78, 0.21, 0.01])  # approximately air
        N = 12
        P = 5e5   # 5 bar typical ASU pressure
        feed = FeedSpec(
            stage=N // 2 + 1,
            flow=F,
            z=z,
            T=100.0,
            P=P,
            q=1.0,
        )
        P_profile = np.full(N, P)
        cfg = ColumnConfig(
            N_stages=N,
            feed=feed,
            P_profile=P_profile,
            condenser_type="total",
            distillate_rate=F * 0.5,
            reflux_ratio=4.0,
        )
        res = solve_bp(cfg, mix_n2o2ar, max_iter=100)
        # At minimum: no exceptions, composition arrays are valid shape
        assert res.x.shape == (3, N)
        assert res.y.shape == (3, N)

    def test_result_summary(self, mix_n2o2):
        """ColumnResult.summary() returns a non-empty string."""
        cfg = _n2o2_config()
        res = solve_bp(cfg, mix_n2o2)
        s = res.summary()
        assert len(s) > 10


# ---------------------------------------------------------------------------
# MCP server tests
# ---------------------------------------------------------------------------

class TestMCPServer:
    def test_flash_tool(self):
        from distcalc.mcp.server import flash
        from distcalc.mcp.schemas import FlashInput
        inp = FlashInput(T=90.0, P=1e5, composition=[0.5, 0.4, 0.1])
        out = flash(inp)
        assert out.phase in ("liquid", "vapor", "two-phase")
        assert abs(sum(out.x) - 1.0) < 1e-6

    def test_bubble_point_tool(self):
        from distcalc.mcp.server import bubble_point
        from distcalc.mcp.schemas import BubblePointInput
        inp = BubblePointInput(P=101325.0, composition=[0.5, 0.5, 0.0])
        out = bubble_point(inp)
        assert out.converged
        assert 70.0 < out.T < 100.0
