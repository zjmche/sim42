"""Regress an improved PR-EOS model for N2/O2/Ar from CoolProp-generated data:

  1. Refit Mathias-Copeman alpha [c1,c2,c3] per component against the dense
     CoolProp pure-saturation curve (60-130K) — fixes the +0.35-0.43K bias
     found at 6 bar with the original NIST-spot-point fit.
  2. Fit a temperature-dependent Kij(T) = a + b/T per binary pair against
     CoolProp binary VLE data, replacing the constant literature Kij values.

Output: distcalc/components/components_tdep.json

Does NOT modify distcalc/components/components.json or any original code —
this produces a new, parallel component database for the new T-dependent
engine (pr_eos_tdep.py).

Usage:
    python scripts/fit_pr_eos_tdep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distcalc.components.loader import load_database, load_mixture
from distcalc.equilibrium.bubble_dew import bubble_P

_DATA = Path(__file__).parent / "data" / "coolprop_asu_data.json"
_OUT = Path(__file__).parent.parent / "distcalc" / "components" / "components_tdep.json"

R = 8.314462
SQRT2 = np.sqrt(2.0)
_1P, _1M, _2S2 = 1.0 + SQRT2, 1.0 - SQRT2, 2.0 * SQRT2

# Key ASU operating temperatures (approx Tsat at 1.3 bar / 6 bar) used to
# weight the alpha refit, same spirit as the original NIST-weighted fit.
_KEY_T = {"N2": (79.5, 96.4), "O2": (92.6, 111.5), "Ar": (89.7, 108.4)}


# ---------------------------------------------------------------------------
# Step 1: refit Mathias-Copeman alpha per pure component
# ---------------------------------------------------------------------------

def _pure_residual_lnphi(T: float, P: float, Tc: float, Pc: float, c: np.ndarray) -> float:
    """ln(phi_L) - ln(phi_V) for a pure component at given (T,P), MC alpha c=[c1,c2,c3].
    Zero at the true saturation point; used as a cheap (no-Newton) fit residual."""
    c1, c2, c3 = c
    x = 1.0 - (T / Tc) ** 0.5
    sq = 1.0 + c1 * x + c2 * x**2 + c3 * x**3
    a = 0.45724 * R**2 * Tc**2 / Pc * sq**2
    b = 0.07780 * R * Tc / Pc
    RT = R * T
    A = a * P / RT**2
    B = b * P / RT
    coeffs = [1.0, -(1.0 - B), A - 3 * B**2 - 2 * B, -(A * B - B**2 - B**3)]
    roots = np.roots(coeffs)
    real = sorted([r.real for r in roots if abs(r.imag) < 1e-8 and r.real > B + 1e-10])
    if not real:
        return 0.0
    Z_L, Z_V = real[0], real[-1]

    def ln_phi(Z):
        lt = np.log((Z + _1P * B) / (Z + _1M * B))
        return (Z - 1.0) - np.log(Z - B) - A / _2S2 / B * lt

    return ln_phi(Z_L) - ln_phi(Z_V)


def refit_alpha(formula: str, Tc: float, Pc: float, c0: list[float], pure_data: list) -> list[float]:
    T1, T2 = _KEY_T[formula]
    pts = np.array(pure_data)  # [[T, P], ...]
    T_arr, P_arr = pts[:, 0], pts[:, 1]
    weight = 1.0 + 4.0 * np.exp(-((T_arr - T1) / 3.0) ** 2) + 4.0 * np.exp(-((T_arr - T2) / 3.0) ** 2)

    def obj(c):
        res = np.array([_pure_residual_lnphi(T, P, Tc, Pc, c) for T, P in zip(T_arr, P_arr)])
        return float(np.sum(weight * res**2))

    out = minimize(obj, x0=np.array(c0), method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 3000})
    return out.x.tolist()


# ---------------------------------------------------------------------------
# Step 2: fit Kij(T) per binary pair
# ---------------------------------------------------------------------------

def fit_kij_T(f1: str, f2: str, binary_data: list, new_alpha: dict) -> tuple[list[float], list]:
    """Returns ([a, b] for Kij(T)=a+b/T, [[T, Kij_opt], ...] local fit table)."""
    mix = load_mixture([f1, f2])
    mix.components[0].mc_alpha = new_alpha[f1]
    mix.components[1].mc_alpha = new_alpha[f2]

    pts = np.array(binary_data)  # [[T, x0, P, y0], ...]
    T_unique = sorted(set(pts[:, 0].tolist()))

    table = []
    for T in T_unique:
        sub = pts[pts[:, 0] == T]
        x0s, Ps, y0s = sub[:, 1], sub[:, 2], sub[:, 3]

        def obj(kij):
            mix.Kij[0, 1] = mix.Kij[1, 0] = kij
            err = 0.0
            for x0, P_act in zip(x0s, Ps):
                x = np.array([x0, 1.0 - x0])
                try:
                    res = bubble_P(T, x, mix, P0=P_act, max_iter=60)
                    err += ((res.P - P_act) / P_act) ** 2
                except Exception:
                    err += 1.0
            return err

        out = minimize_scalar(obj, bounds=(-0.06, 0.06), method="bounded",
                               options={"xatol": 1e-5})
        table.append([float(T), float(out.x)])

    # Fit Kij(T) = a + b/T via least squares
    T_arr = np.array([t for t, _ in table])
    K_arr = np.array([k for _, k in table])
    X = np.column_stack([np.ones_like(T_arr), 1.0 / T_arr])
    (a, b), *_ = np.linalg.lstsq(X, K_arr, rcond=None)
    return [float(a), float(b)], table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    raw = json.loads(_DATA.read_text())
    db = load_database()
    comp_by_formula = {c.formula: c for c in db.components}

    print("Step 1: refitting Mathias-Copeman alpha against CoolProp pure data...")
    new_alpha = {}
    for formula in ["N2", "O2", "Ar"]:
        c = comp_by_formula[formula]
        c0 = c.mc_alpha
        fitted = refit_alpha(formula, c.Tc, c.Pc, c0, raw["pure"][formula])
        new_alpha[formula] = fitted
        print(f"  {formula}: {[f'{v:.4f}' for v in c0]} -> {[f'{v:.4f}' for v in fitted]}")

    print("\nStep 2: fitting Kij(T) = a + b/T per binary pair...")
    kij_params = {}
    kij_tables = {}
    for f1, f2 in [("N2", "O2"), ("N2", "Ar"), ("O2", "Ar")]:
        key = f"{f1}-{f2}"
        (a, b), table = fit_kij_T(f1, f2, raw["binary"][key], new_alpha)
        kij_params[key] = [a, b]
        kij_tables[key] = table
        print(f"  {key}: Kij(T) = {a:.6f} + {b:.4f}/T   "
              f"[T={table[0][0]:.0f}K: {table[0][1]:+.5f}, T={table[-1][0]:.0f}K: {table[-1][1]:+.5f}]")

    # Build the new component database
    out_components = []
    for formula in ["N2", "O2", "Ar"]:
        c = comp_by_formula[formula]
        out_components.append({
            "name": c.name, "formula": c.formula, "CAS": c.CAS, "MW": c.MW,
            "Tc": c.Tc, "Pc": c.Pc, "omega": c.omega,
            "mc_alpha": new_alpha[formula],
            "shomate": [s.model_dump() for s in c.shomate],
        })

    out = {
        "_comment": "T-dependent PR-EOS model regressed from CoolProp HEOS data. "
                    "Parallel to components.json (v1.1.0) — does not replace it.",
        "_version": "2.0.0-tdep",
        "_regression_source": "CoolProp HEOS backend, pure sat. 60-130K (0.5K step), "
                               "binary VLE 65-125K (5K step) x=0.05-0.95",
        "components": out_components,
        "Kij_params": {
            "_comment": "Kij(T) = a + b/T, T in K. Fitted by least-squares to local "
                        "per-temperature Kij optimized against CoolProp binary bubble-P data.",
            "names": ["N2", "O2", "Ar"],
            "pairs": kij_params,
        },
        "_kij_fit_tables": kij_tables,
    }
    _OUT.write_text(json.dumps(out, indent=1))
    print(f"\nSaved to {_OUT}")


if __name__ == "__main__":
    main()
