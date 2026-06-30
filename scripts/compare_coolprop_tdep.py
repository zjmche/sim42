"""Compare distcalc's T-DEPENDENT PR-EOS (refitted MC alpha + Kij(T)) against CoolProp's
reference Helmholtz-energy mixture model (HEOS backend) over the ASU
operating envelope, with special focus on O2/Ar where relative volatility
is close to 1 and Kij accuracy matters most.

Requires the optional `CoolProp` dependency:  pip install CoolProp

Usage:
    python scripts/compare_coolprop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import CoolProp.CoolProp as CP

from distcalc.components.loader_tdep import load_mixture_tdep as load_mixture
from distcalc.equilibrium.bubble_dew_tdep import bubble_T as distcalc_bubble_T

# CoolProp fluid names
_CP_NAME = {"N2": "Nitrogen", "O2": "Oxygen", "Ar": "Argon"}


# ---------------------------------------------------------------------------
# CoolProp bubble-T helper
# ---------------------------------------------------------------------------

def coolprop_bubble_T(P: float, x: np.ndarray, formulas: list[str], T_guess: float) -> tuple[float, np.ndarray]:
    """Bubble-point T and vapor y at given P, liquid x, via CoolProp HEOS."""
    fluid_str = "&".join(_CP_NAME[f] for f in formulas)
    AS = CP.AbstractState("HEOS", fluid_str)
    AS.set_mole_fractions(list(x))
    AS.update(CP.PQ_INPUTS, P, 0.0)
    y = np.array(AS.mole_fractions_vapor())
    return AS.T(), y


# ---------------------------------------------------------------------------
# Comparison cases
# ---------------------------------------------------------------------------

def compare_binary(formulas: list[str], P_bar: float, label: str) -> list[dict]:
    mix = load_mixture(formulas)
    P = P_bar * 1e5
    rows = []
    for x0 in np.arange(0.05, 1.0, 0.05):
        x = np.array([x0, 1.0 - x0])
        try:
            res = distcalc_bubble_T(P, x, mix, T0=90.0)
            T_dc, y_dc = res.T, res.y
        except Exception as e:
            continue
        try:
            T_cp, y_cp = coolprop_bubble_T(P, x, formulas, T_guess=T_dc)
        except Exception:
            continue
        rows.append({
            "system": label,
            "P_bar": P_bar,
            "x0": x0,
            "T_distcalc": T_dc,
            "T_coolprop": T_cp,
            "dT": T_dc - T_cp,
            "y0_distcalc": y_dc[0],
            "y0_coolprop": y_cp[0],
            "dy0": y_dc[0] - y_cp[0],
        })
    return rows


def compare_ternary_ar_zone(P_bar: float = 1.3) -> list[dict]:
    """Ternary VLE near the Ar side-draw / Ar-column operating zone:
    dilute N2 (0-7%), O2 majority, Ar 8-15%."""
    formulas = ["N2", "O2", "Ar"]
    mix = load_mixture(formulas)
    P = P_bar * 1e5
    rows = []
    # Sweep N2 contamination level at fixed Ar=10%, rest O2 — this is the
    # exact axis that determines Ar product purity in the ASU model.
    for n2_frac in [0.0001, 0.001, 0.01, 0.03, 0.05, 0.07, 0.10]:
        ar_frac = 0.10
        o2_frac = 1.0 - n2_frac - ar_frac
        x = np.array([n2_frac, o2_frac, ar_frac])
        try:
            res = distcalc_bubble_T(P, x, mix, T0=90.0)
            T_dc, y_dc = res.T, res.y
        except Exception:
            continue
        try:
            T_cp, y_cp = coolprop_bubble_T(P, x, formulas, T_guess=T_dc)
        except Exception:
            continue
        K_ar_o2_dc = (y_dc[2] / x[2]) / (y_dc[1] / x[1])
        K_ar_o2_cp = (y_cp[2] / x[2]) / (y_cp[1] / x[1])
        rows.append({
            "x_N2": n2_frac,
            "x_O2": o2_frac,
            "x_Ar": ar_frac,
            "T_distcalc": T_dc,
            "T_coolprop": T_cp,
            "dT": T_dc - T_cp,
            "alpha_ArO2_distcalc": K_ar_o2_dc,
            "alpha_ArO2_coolprop": K_ar_o2_cp,
            "d_alpha_pct": (K_ar_o2_dc - K_ar_o2_cp) / K_ar_o2_cp * 100,
        })
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict], cols: list[str], header: str) -> None:
    print(f"\n{header}")
    print("-" * len(header))
    widths = {c: max(len(c), 9) for c in cols}
    print("  ".join(f"{c:>{widths[c]}s}" for c in cols))
    for r in rows:
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:>{widths[c]}.4f}" if isinstance(v, float) else f"{v:>{widths[c]}}")
        print("  ".join(vals))


def main() -> None:
    print("=" * 78)
    print("  distcalc PR-EOS (MC alpha + Kij) vs CoolProp HEOS — ASU operating range")
    print("=" * 78)

    all_dT = []

    # --- N2/O2 binary at LP (1.3 bar) and HP (6 bar) ---
    for P_bar in (1.3, 6.0):
        rows = compare_binary(["N2", "O2"], P_bar, "N2/O2")
        _print_table(
            rows, ["x0", "T_distcalc", "T_coolprop", "dT", "y0_distcalc", "y0_coolprop"],
            f"N2/O2 bubble curve @ {P_bar:.1f} bar",
        )
        all_dT += [abs(r["dT"]) for r in rows]

    # --- O2/Ar binary — the low-relative-volatility, high-sensitivity pair ---
    for P_bar in (1.3, 6.0):
        rows = compare_binary(["O2", "Ar"], P_bar, "O2/Ar")
        _print_table(
            rows, ["x0", "T_distcalc", "T_coolprop", "dT", "y0_distcalc", "y0_coolprop"],
            f"O2/Ar bubble curve @ {P_bar:.1f} bar  (x0 = x_O2)",
        )
        all_dT += [abs(r["dT"]) for r in rows]

    # --- N2/Ar binary ---
    for P_bar in (1.3, 6.0):
        rows = compare_binary(["N2", "Ar"], P_bar, "N2/Ar")
        _print_table(
            rows, ["x0", "T_distcalc", "T_coolprop", "dT", "y0_distcalc", "y0_coolprop"],
            f"N2/Ar bubble curve @ {P_bar:.1f} bar  (x0 = x_N2)",
        )
        all_dT += [abs(r["dT"]) for r in rows]

    # --- Ternary, Ar-column operating zone: focus on N2 contamination effect ---
    rows_tern = compare_ternary_ar_zone(P_bar=1.3)
    _print_table(
        rows_tern,
        ["x_N2", "x_Ar", "T_distcalc", "T_coolprop", "dT",
         "alpha_ArO2_distcalc", "alpha_ArO2_coolprop", "d_alpha_pct"],
        "Ternary N2/O2/Ar @ 1.3 bar — Ar-draw zone (Ar=10%, N2 swept 0.01-10%)",
    )

    # --- Summary ---
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    dT_arr = np.array(all_dT)
    print(f"  Binary bubble-T comparisons : {len(dT_arr)} points")
    print(f"  Mean |dT|                   : {dT_arr.mean():.3f} K")
    print(f"  Max  |dT|                   : {dT_arr.max():.3f} K")
    print(f"  RMS  dT                     : {np.sqrt((dT_arr**2).mean()):.3f} K")

    d_alpha = np.array([abs(r["d_alpha_pct"]) for r in rows_tern])
    print(f"\n  O2/Ar relative volatility error in Ar-draw zone:")
    print(f"  Mean |d_alpha|              : {d_alpha.mean():.2f} %")
    print(f"  Max  |d_alpha|              : {d_alpha.max():.2f} %")
    print("=" * 78)


if __name__ == "__main__":
    main()
