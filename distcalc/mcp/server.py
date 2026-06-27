"""MCP server exposing distillation tools.

Tools:
  flash        — isothermal PT-flash
  bubble_point — bubble-point temperature at fixed P
  rate_column  — rigorous MESH column (Wang-Henke BP solver)

Run with:
    python -m distcalc.mcp.server

Or use as a library:
    from distcalc.mcp.server import flash, bubble_point, rate_column
"""

from __future__ import annotations

import json

import numpy as np

from ..components.loader import load_mixture
from ..column.mesh_bp import solve_bp
from ..column.specs import ColumnConfig, FeedSpec
from ..equilibrium.bubble_dew import bubble_T as _bubble_T
from ..equilibrium.flash import flash_PT
from .schemas import (
    BubblePointInput, BubblePointOutput,
    ColumnInput, ColumnOutput,
    FlashInput, FlashOutput,
)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def flash(inp: FlashInput) -> FlashOutput:
    """Isothermal PT-flash for a mixture.

    Returns phase fractions, compositions, K-values, and convergence info.
    """
    mix = load_mixture(inp.components)
    z = np.array(inp.composition, dtype=float)
    z /= z.sum()

    result = flash_PT(inp.T, inp.P, z, mix, tol=inp.tol, max_iter=inp.max_iter)

    return FlashOutput(
        phase=result.phase,
        beta=result.beta,
        x=result.x.tolist(),
        y=result.y.tolist(),
        K=result.K.tolist(),
        converged=result.converged,
        n_iter=result.n_iter,
    )


def bubble_point(inp: BubblePointInput) -> BubblePointOutput:
    """Find bubble-point temperature for a liquid mixture at fixed pressure."""
    mix = load_mixture(inp.components)
    x = np.array(inp.composition, dtype=float)
    x /= x.sum()

    result = _bubble_T(inp.P, x, mix, T0=inp.T0, tol=inp.tol)

    return BubblePointOutput(
        T=result.T,
        P=result.P,
        x=result.x.tolist(),
        y=result.y.tolist(),
        converged=result.converged,
        n_iter=result.n_iter,
    )


def rate_column(inp: ColumnInput) -> ColumnOutput:
    """Rigorous equilibrium-stage column rating (Wang-Henke BP solver).

    Returns stage profiles (T, V, L, x, y), duties, and convergence report.
    """
    mix = load_mixture(inp.components)

    P_bot = inp.P_bot if inp.P_bot is not None else inp.P_top
    P_profile = np.linspace(inp.P_top, P_bot, inp.N_stages)

    feed_z = np.array(inp.feed.composition, dtype=float)
    feed_z /= feed_z.sum()

    feed = FeedSpec(
        stage=inp.feed.stage,
        flow=inp.feed.flow,
        z=feed_z,
        T=inp.feed.T,
        P=inp.feed.P,
        q=inp.feed.q,
    )

    cfg = ColumnConfig(
        N_stages=inp.N_stages,
        feed=feed,
        P_profile=P_profile,
        condenser_type=inp.condenser_type,
        distillate_rate=inp.distillate_rate,
        reflux_ratio=inp.reflux_ratio,
    )

    result = solve_bp(cfg, mix, tol_T=inp.tol_T, max_iter=inp.max_iter)

    return ColumnOutput(
        converged=result.converged,
        n_iter=result.n_iter,
        max_dT=result.max_dT,
        T_profile=result.T.tolist(),
        V_profile=result.V.tolist(),
        L_profile=result.L.tolist(),
        x_distillate=result.x_distillate.tolist(),
        x_bottoms=result.x_bottoms.tolist(),
        Q_condenser_kW=result.Q_condenser / 1000.0,
        Q_reboiler_kW=result.Q_reboiler / 1000.0,
        components=inp.components,
    )


# ---------------------------------------------------------------------------
# CLI entry point (JSON stdin/stdout protocol)
# ---------------------------------------------------------------------------

_TOOLS = {
    "flash": (flash, FlashInput),
    "bubble_point": (bubble_point, BubblePointInput),
    "rate_column": (rate_column, ColumnInput),
}


def _handle_request(request: dict) -> dict:
    tool = request.get("tool")
    if tool not in _TOOLS:
        return {"error": f"Unknown tool: {tool}. Available: {list(_TOOLS)}"}
    fn, InputModel = _TOOLS[tool]
    try:
        inp = InputModel.model_validate(request.get("params", {}))
        out = fn(inp)
        return {"result": out.model_dump()}
    except Exception as exc:
        return {"error": str(exc)}


def main() -> None:
    import sys
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"JSON parse error: {e}"}))
            continue
        response = _handle_request(request)
        print(json.dumps(response))


if __name__ == "__main__":
    main()
