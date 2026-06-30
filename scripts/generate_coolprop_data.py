"""Generate reference VLE/saturation data from CoolProp (HEOS backend) for
regressing an improved PR-EOS model (T-dependent alpha + T-dependent Kij)
for the N2/O2/Ar ASU system.

Output: scripts/data/coolprop_asu_data.json
  - pure: {formula: [[T, Psat], ...]}  T=60-130K, 0.5K step
  - binary: {pair: [[T, x0, P, y0], ...]}  T=65-125K (5K step), x0=0.05-0.95 (0.05 step)

Usage:
    python scripts/generate_coolprop_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import CoolProp.CoolProp as CP

_CP_NAME = {"N2": "Nitrogen", "O2": "Oxygen", "Ar": "Argon"}
_OUT = Path(__file__).parent / "data" / "coolprop_asu_data.json"

# ASU operating envelope: LP column ~1.3 bar (T~79-93K), HP column ~6 bar (T~97-112K)
# Use a temperature grid that spans and slightly brackets both regimes.
T_PURE = np.arange(60.0, 130.5, 0.5)
T_BINARY = np.arange(65.0, 126.0, 5.0)
X_GRID = np.arange(0.05, 1.00, 0.05)


def gen_pure() -> dict:
    out = {}
    for formula, cp_name in _CP_NAME.items():
        rows = []
        for T in T_PURE:
            try:
                P = CP.PropsSI("P", "T", float(T), "Q", 0, cp_name)
                rows.append([float(T), float(P)])
            except Exception:
                continue
        out[formula] = rows
        print(f"  pure {formula}: {len(rows)} points")
    return out


def gen_binary() -> dict:
    out = {}
    pairs = [("N2", "O2"), ("N2", "Ar"), ("O2", "Ar")]
    for f1, f2 in pairs:
        key = f"{f1}-{f2}"
        fluid_str = f"{_CP_NAME[f1]}&{_CP_NAME[f2]}"
        rows = []
        for T in T_BINARY:
            AS = CP.AbstractState("HEOS", fluid_str)
            for x0 in X_GRID:
                try:
                    AS.set_mole_fractions([float(x0), float(1.0 - x0)])
                    AS.update(CP.QT_INPUTS, 0.0, float(T))
                    P = AS.p()
                    y0 = AS.mole_fractions_vapor()[0]
                    rows.append([float(T), float(x0), float(P), float(y0)])
                except Exception:
                    continue
        out[key] = rows
        print(f"  binary {key}: {len(rows)} points")
    return out


def main() -> None:
    print("Generating pure-component saturation data...")
    pure = gen_pure()
    print("Generating binary VLE data...")
    binary = gen_binary()

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps({"pure": pure, "binary": binary}, indent=1))
    print(f"\nSaved to {_OUT}")


if __name__ == "__main__":
    main()
