"""Internal helper: solve the ASU system with one engine ('orig' or 'tdep')
and dump the result to a JSON file. Used by compare_asu_tdep.py to run both
engines in parallel background processes (170-stage Ar column is slow).

Usage:
    python scripts/_run_asu_side.py orig /path/to/out.json
    python scripts/_run_asu_side.py tdep /path/to/out.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distcalc.asu.specs import ASUConfig


def _stream_to_dict(s):
    if s is None:
        return None
    return {"flow": float(s.flow), "z": list(map(float, s.z)), "T": float(s.T), "P": float(s.P), "phase": s.phase}


def main() -> None:
    engine, out_path = sys.argv[1], sys.argv[2]
    cfg = ASUConfig()

    if engine == "orig":
        from distcalc.asu.system import solve_asu
        from distcalc.components.loader import load_mixture
        mix = load_mixture(["N2", "O2", "Ar"])
    else:
        from distcalc.asu.system_tdep import solve_asu
        from distcalc.components.loader_tdep import load_mixture_tdep as load_mixture
        mix = load_mixture(["N2", "O2", "Ar"])

    t0 = time.time()
    res = solve_asu(cfg, mix)
    wall = time.time() - t0

    out = {
        "engine": engine,
        "wall_s": wall,
        "converged": res.converged,
        "n_outer_iter": res.n_outer_iter,
        "duty_imbalance": float(res.duty_imbalance),
        "N2_purity": float(res.N2_purity),
        "O2_purity": float(res.O2_purity),
        "Ar_purity": float(res.Ar_purity),
        "N2_recovery": float(res.N2_recovery),
        "O2_recovery": float(res.O2_recovery),
        "Q_mche": float(res.streams.Q_mche),
        "lower_Q_cond": float(res.lower_col.Q_condenser) if res.lower_col else None,
        "lower_Q_reb": float(res.lower_col.Q_reboiler) if res.lower_col else None,
        "upper_Q_cond": float(res.upper_col.Q_condenser) if res.upper_col else None,
        "upper_Q_reb": float(res.upper_col.Q_reboiler) if res.upper_col else None,
        "argon_Q_cond": float(res.argon_col.Q_condenser) if res.argon_col else None,
        "argon_Q_reb": float(res.argon_col.Q_reboiler) if res.argon_col else None,
        "streams": {
            "upper_distillate": _stream_to_dict(res.streams.upper_distillate),
            "upper_bottoms": _stream_to_dict(res.streams.upper_bottoms),
            "ar_distillate": _stream_to_dict(res.streams.ar_distillate),
            "ar_bottoms": _stream_to_dict(res.streams.ar_bottoms),
        },
    }
    Path(out_path).write_text(json.dumps(out, indent=1))
    print(f"[{engine}] done in {wall:.1f}s, converged={res.converged}, "
          f"outer_iter={res.n_outer_iter} -> {out_path}")


if __name__ == "__main__":
    main()
