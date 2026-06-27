"""Tests for the full ASU double-column + argon column system.

Tests are deliberately tolerant: the ASU MESH solve is iterative and the
results depend on convergence of three coupled columns.  The exit criteria
here are qualitative correctness rather than tight numerical values.
"""

from __future__ import annotations

import numpy as np
import pytest

from distcalc.asu.mesh_asu import (
    MultiColumnConfig,
    SideDraw,
    find_ar_peak_stage,
    solve_bp_asu,
)
from distcalc.asu.specs import ASUConfig, StreamSpec
from distcalc.asu.system import solve_asu
from distcalc.column.specs import FeedSpec
from distcalc.components.loader import load_mixture


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mix_n2o2():
    return load_mixture(["N2", "O2"])


@pytest.fixture(scope="module")
def mix_n2o2ar():
    return load_mixture(["N2", "O2", "Ar"])


# ---------------------------------------------------------------------------
# MultiColumnConfig / mesh_asu tests
# ---------------------------------------------------------------------------

class TestMultiColumnConfig:
    def test_construction(self, mix_n2o2):
        """MultiColumnConfig with two feeds should build without error."""
        N = 12
        P = 1.3e5
        feed1 = FeedSpec(stage=2,  flow=0.5, z=np.array([0.97, 0.03]), T=77.0, P=P, q=1.0)
        feed2 = FeedSpec(stage=8,  flow=0.5, z=np.array([0.35, 0.65]), T=80.0, P=P, q=0.9)
        cfg = MultiColumnConfig(
            N_stages=N,
            feeds=[feed1, feed2],
            P_profile=np.full(N, P),
            distillate_rate=0.78,
            reflux_ratio=3.0,
        )
        assert cfg.B > 0
        assert abs(cfg.total_feed_flow - 1.0) < 1e-10
        np.testing.assert_allclose(cfg.z_feed_avg.sum(), 1.0, atol=1e-10)

    def test_solve_binary_multifeed(self, mix_n2o2):
        """Two-feed binary column should converge and enrich N2 overhead."""
        N = 15
        P = 1.3e5
        feed1 = FeedSpec(stage=2,  flow=0.5, z=np.array([0.97, 0.03]), T=77.0, P=P, q=1.0)
        feed2 = FeedSpec(stage=10, flow=0.5, z=np.array([0.40, 0.60]), T=80.0, P=P, q=0.95)
        cfg = MultiColumnConfig(
            N_stages=N,
            feeds=[feed1, feed2],
            P_profile=np.full(N, P),
            distillate_rate=0.78,
            reflux_ratio=3.0,
        )
        res = solve_bp_asu(cfg, mix_n2o2, max_iter=250)
        # N2 fraction in distillate should be higher than in bottoms
        assert res.x_distillate[0] > res.x_bottoms[0], (
            f"N2 dist={res.x_distillate[0]:.3f} bot={res.x_bottoms[0]:.3f}"
        )

    def test_side_draw_stage_accepted(self, mix_n2o2ar):
        """Column with liquid side draw should not crash."""
        N = 20
        P = 1.3e5
        z_air = np.array([0.7812, 0.2096, 0.0092])
        feed1 = FeedSpec(stage=2,  flow=0.50, z=np.array([0.97, 0.025, 0.005]), T=77.0, P=P, q=1.0)
        feed2 = FeedSpec(stage=13, flow=0.50, z=np.array([0.58, 0.40,  0.02]),  T=80.0, P=P, q=0.95)
        cfg = MultiColumnConfig(
            N_stages=N,
            feeds=[feed1, feed2],
            P_profile=np.full(N, P),
            distillate_rate=0.78,
            reflux_ratio=3.0,
            side_draws=[SideDraw(stage=15, flow=0.01)],
        )
        res = solve_bp_asu(cfg, mix_n2o2ar, max_iter=200)
        assert res.x.shape == (3, N)
        assert np.all(res.x >= -1e-6)


class TestFindArPeak:
    def test_returns_valid_stage(self, mix_n2o2ar):
        """Fake Ar profile peaking at stage 10 should return stage 10."""
        n = 3
        N = 20
        x = np.zeros((n, N))
        x[0] = 0.7   # N2
        x[1] = 0.25  # O2
        x[2] = 0.05  # Ar, small everywhere
        x[2, 9] = 0.30  # peak at stage 10 (0-indexed: 9)
        x /= x.sum(axis=0, keepdims=True)
        stage = find_ar_peak_stage(x, ar_idx=2)
        assert stage == 10


# ---------------------------------------------------------------------------
# ASU system tests
# ---------------------------------------------------------------------------

class TestASUBinary:
    """Tests with N2/O2 only (no Ar) for speed."""

    def test_asu_binary_runs(self, mix_n2o2):
        """ASU solve with 2-component air (no Ar) should complete."""
        cfg = ASUConfig(
            air_flow=1.0,
            z_air=np.array([0.79, 0.21]),
            N_lower=15,
            N_upper=25,
            N_argon=15,
            max_outer_iter=5,
        )
        res = solve_asu(cfg, mix_n2o2)
        assert res.lower_col is not None
        assert res.upper_col is not None
        assert res.lower_col.x.shape == (2, 15)
        assert res.upper_col.x.shape == (2, 25)

    def test_n2_enriched_in_upper_distillate(self, mix_n2o2):
        """Upper column N2 mole fraction must be higher in distillate than in bottoms."""
        cfg = ASUConfig(
            air_flow=1.0,
            z_air=np.array([0.79, 0.21]),
            N_lower=15,
            N_upper=25,
            max_outer_iter=5,
        )
        res = solve_asu(cfg, mix_n2o2)
        x_d_n2 = res.upper_col.x_distillate[0]
        x_b_n2 = res.upper_col.x_bottoms[0]
        assert x_d_n2 > x_b_n2, (
            f"N2 in distillate={x_d_n2:.4f} should exceed N2 in bottoms={x_b_n2:.4f}"
        )

    def test_streams_populated(self, mix_n2o2):
        """Key streams should be populated after solve."""
        cfg = ASUConfig(
            air_flow=1.0,
            z_air=np.array([0.79, 0.21]),
            N_lower=12,
            N_upper=20,
            max_outer_iter=3,
        )
        res = solve_asu(cfg, mix_n2o2)
        assert res.streams.lower_distillate is not None
        assert res.streams.lower_bottoms is not None
        assert res.streams.upper_distillate is not None
        assert res.streams.upper_bottoms is not None

    def test_composition_bounds_binary(self, mix_n2o2):
        """All compositions in both columns must be in [0, 1]."""
        cfg = ASUConfig(
            air_flow=1.0,
            z_air=np.array([0.79, 0.21]),
            N_lower=12,
            N_upper=20,
            max_outer_iter=3,
        )
        res = solve_asu(cfg, mix_n2o2)
        for col_name, col in [("lower", res.lower_col), ("upper", res.upper_col)]:
            assert np.all(col.x >= -1e-6), f"{col_name} has x < 0"
            assert np.all(col.x <= 1.0 + 1e-6), f"{col_name} has x > 1"


class TestASUTernary:
    """Full N2/O2/Ar system tests.

    Uses small stage counts to keep CI runtime manageable (< 60 s per test).
    Separation quality assertions are intentionally lenient.
    """

    @staticmethod
    def _fast_cfg():
        """Return a fresh config dict for each test (avoids z_air array mutation)."""
        return dict(
            air_flow=1.0,
            z_air=np.array([0.7812, 0.2096, 0.0092]),
            N_lower=12,
            N_upper=20,
            N_argon=12,
            max_outer_iter=3,
        )

    def test_asu_ternary_runs(self, mix_n2o2ar):
        """Full ASU with N2/O2/Ar should complete without exception."""
        cfg = ASUConfig(**self._fast_cfg())
        res = solve_asu(cfg, mix_n2o2ar)
        assert res.lower_col is not None
        assert res.upper_col is not None
        assert res.upper_col.x.shape == (3, 20)

    def test_argon_column_runs(self, mix_n2o2ar):
        """Ar column should produce a result when side draw is active."""
        cfg = ASUConfig(**self._fast_cfg())
        res = solve_asu(cfg, mix_n2o2ar)
        assert res.argon_col is not None, "Argon column result not populated"
        assert res.argon_col.x.shape == (3, 12)

    def test_n2_o2_separation(self, mix_n2o2ar):
        """Upper column N2 in distillate > N2 in bottoms (qualitative separation)."""
        cfg = ASUConfig(**self._fast_cfg())
        res = solve_asu(cfg, mix_n2o2ar)
        x_d_n2 = res.upper_col.x_distillate[0]
        x_b_n2 = res.upper_col.x_bottoms[0]
        assert x_d_n2 > x_b_n2, (
            f"N2 in distillate={x_d_n2:.4f} should exceed N2 in bottoms={x_b_n2:.4f}"
        )

    def test_summary_nonempty(self, mix_n2o2ar):
        """ASUResult.summary() should return a non-empty string."""
        cfg = ASUConfig(**self._fast_cfg())
        res = solve_asu(cfg, mix_n2o2ar)
        s = res.summary()
        assert len(s) > 50

    def test_stream_compositions_valid(self, mix_n2o2ar):
        """All stream compositions must be in [0, 1] and sum to 1."""
        cfg = ASUConfig(**self._fast_cfg())
        res = solve_asu(cfg, mix_n2o2ar)
        for name, s in [
            ("lower_distillate", res.streams.lower_distillate),
            ("lower_bottoms",    res.streams.lower_bottoms),
            ("upper_distillate", res.streams.upper_distillate),
            ("upper_bottoms",    res.streams.upper_bottoms),
        ]:
            assert s is not None, f"{name} stream is None"
            assert np.all(s.z >= -1e-8), f"{name} z < 0"
            assert np.all(s.z <= 1.0 + 1e-8), f"{name} z > 1"
            np.testing.assert_allclose(s.z.sum(), 1.0, atol=1e-6, err_msg=f"{name} z sums != 1")
