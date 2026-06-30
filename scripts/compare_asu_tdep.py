"""Run the full ASU double-column + argon-column system with both the
original constant-Kij PR-EOS engine and the new T-dependent (Kij(T),
refitted MC alpha) engine, and report a side-by-side mass-balance and
purity/recovery comparison.

Uses the default ASUConfig (N_argon=170 stages, the same configuration
used for the validated 170-stage Ar column result).

Does not modify any original code — imports distcalc.asu.system (original)
and distcalc.asu.system_tdep (new) side by side.

Usage:
    python scripts/compare_asu_tdep.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distcalc.asu.specs import ASUConfig
from distcalc.asu.system import solve_asu as solve_asu_orig
from distcalc.asu.system_tdep import solve_asu as solve_asu_tdep
from distcalc.components.loader import load_mixture
from distcalc.components.loader_tdep import load_mixture_tdep


def _row(label: str, a, b, fmt: str = "{:.4f}") -> str:
    return f"  {label:<28s} {fmt.format(a):>16s} {fmt.format(b):>16s}"


def main() -> None:
    cfg = ASUConfig()  # defaults: N_argon=170, RR_argon=5.0, D_frac_argon=0.07

    print("=" * 70)
    print("  ASU system: original PR-EOS vs T-dependent (Kij(T)) PR-EOS")
    print("=" * 70)
    print(f"  air_flow={cfg.air_flow} mol/s  z_air={cfg.z_air}")
    print(f"  N_lower={cfg.N_lower}  N_upper={cfg.N_upper}  N_argon={cfg.N_argon}")

    mix_orig = load_mixture(["N2", "O2", "Ar"])
    mix_tdep = load_mixture_tdep(["N2", "O2", "Ar"])

    print("\nSolving with ORIGINAL engine (constant Kij)...")
    t0 = time.time()
    res_orig = solve_asu_orig(cfg, mix_orig)
    t_orig = time.time() - t0
    print(f"  done in {t_orig:.1f}s, converged={res_orig.converged}, "
          f"outer_iter={res_orig.n_outer_iter}")

    print("\nSolving with T-DEPENDENT engine (Kij(T), refitted alpha)...")
    t0 = time.time()
    res_tdep = solve_asu_tdep(cfg, mix_tdep)
    t_tdep = time.time() - t0
    print(f"  done in {t_tdep:.1f}s, converged={res_tdep.converged}, "
          f"outer_iter={res_tdep.n_outer_iter}")

    # ---- Comparison table ----
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print(f"  {'metric':<28s} {'original':>16s} {'tdep':>16s}")
    print("  " + "-" * 64)
    print(_row("N2 purity [%]", res_orig.N2_purity * 100, res_tdep.N2_purity * 100, "{:.3f}"))
    print(_row("O2 purity [%]", res_orig.O2_purity * 100, res_tdep.O2_purity * 100, "{:.3f}"))
    print(_row("Ar purity [%]", res_orig.Ar_purity * 100, res_tdep.Ar_purity * 100, "{:.3f}"))
    print(_row("N2 recovery [%]", res_orig.N2_recovery * 100, res_tdep.N2_recovery * 100, "{:.2f}"))
    print(_row("O2 recovery [%]", res_orig.O2_recovery * 100, res_tdep.O2_recovery * 100, "{:.2f}"))
    print(_row("MCHE duty imbalance [kW]", res_orig.duty_imbalance / 1e3, res_tdep.duty_imbalance / 1e3, "{:.3f}"))
    print(_row("MCHE duty [kW]", res_orig.streams.Q_mche / 1e3, res_tdep.streams.Q_mche / 1e3, "{:.2f}"))
    if res_orig.lower_col and res_tdep.lower_col:
        print(_row("Lower Q_cond [kW]", res_orig.lower_col.Q_condenser / 1e3, res_tdep.lower_col.Q_condenser / 1e3, "{:.2f}"))
        print(_row("Lower Q_reb [kW]", res_orig.lower_col.Q_reboiler / 1e3, res_tdep.lower_col.Q_reboiler / 1e3, "{:.2f}"))
    if res_orig.upper_col and res_tdep.upper_col:
        print(_row("Upper Q_cond [kW]", res_orig.upper_col.Q_condenser / 1e3, res_tdep.upper_col.Q_condenser / 1e3, "{:.2f}"))
        print(_row("Upper Q_reb [kW]", res_orig.upper_col.Q_reboiler / 1e3, res_tdep.upper_col.Q_reboiler / 1e3, "{:.2f}"))
    if res_orig.argon_col and res_tdep.argon_col:
        print(_row("Argon col Q_cond [kW]", res_orig.argon_col.Q_condenser / 1e3, res_tdep.argon_col.Q_condenser / 1e3, "{:.4f}"))
        print(_row("Argon col Q_reb [kW]", res_orig.argon_col.Q_reboiler / 1e3, res_tdep.argon_col.Q_reboiler / 1e3, "{:.4f}"))

    # ---- Mass balance tables ----
    def mass_balance(res, mix, label):
        print(f"\n[{label}] Mass balance — product streams")
        F = cfg.air_flow
        rows = [
            ("Air feed", F, cfg.z_air),
        ]
        s = res.streams
        if s.upper_distillate:
            rows.append(("N2 product (upper D)", s.upper_distillate.flow, s.upper_distillate.z))
        if s.upper_bottoms:
            rows.append(("O2 product (upper B)", s.upper_bottoms.flow, s.upper_bottoms.z))
        if s.ar_distillate:
            rows.append(("Ar product (ar col D)", s.ar_distillate.flow, s.ar_distillate.z))
        if s.ar_bottoms:
            rows.append(("Ar col bottoms (waste)", s.ar_bottoms.flow, s.ar_bottoms.z))
        print(f"  {'stream':<26s} {'flow [mol/s]':>14s} {'N2':>8s} {'O2':>8s} {'Ar':>8s}")
        for name, flow, z in rows:
            zz = list(z) + [0.0] * (3 - len(z))
            print(f"  {name:<26s} {flow:>14.5f} {zz[0]:>8.4f} {zz[1]:>8.4f} {zz[2]:>8.4f}")

        out_flow = sum(flow for _, flow, _ in rows[1:])
        print(f"  {'Total out':<26s} {out_flow:>14.5f}   (in={F:.5f}, balance err={out_flow-F:+.5f})")

    mass_balance(res_orig, mix_orig, "ORIGINAL")
    mass_balance(res_tdep, mix_tdep, "TDEP")

    print("\n" + "=" * 70)
    print(f"  Wall time: original={t_orig:.1f}s  tdep={t_tdep:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
