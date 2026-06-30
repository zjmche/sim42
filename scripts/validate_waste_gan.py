"""Validate the waste-GAN (vapor side draw) ASU upper-column fix.

Real double-column ASUs withdraw a second, lower-purity N2 stream as
VAPOR a few stages below the high-purity N2 top product ("waste GAN" —
vented to atmosphere via the main heat exchanger, used to regenerate
the air-dryer mol-sieve beds). This relieves the rectifying section
from having to drive N2 recovery to ~100%, which would otherwise push
N2 contamination down into the Ar side-draw region.

Without the waste GAN, the crude Ar side draw in this simulator was
~71% N2-contaminated (see the historical baseline below). With it, the
draw point lands close to the real-plant target of ~90% O2 / ~10% Ar /
trace N2, and overall Ar recovery is naturally capped well under the
~60% ceiling typical of real plants.

Usage:
    python scripts/validate_waste_gan.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distcalc.asu.specs import ASUConfig
from distcalc.asu.system import solve_asu
from distcalc.components.loader import load_mixture

# Historical baseline (no waste GAN, ar_draw_flow assumed 90% purity at the
# draw point, D_frac_argon=0.12): crude Ar side draw was 71.3% N2 / 1.1% O2 /
# 27.6% Ar — see scripts/data/asu_result_orig.json.
BASELINE_AR_DRAW_Z = (0.713, 0.011, 0.276)


def main() -> None:
    mix = load_mixture(["N2", "O2", "Ar"])
    cfg = ASUConfig()  # N_argon=170, waste_gan_stage/flow auto, D_frac_argon=0.07

    t0 = time.time()
    res = solve_asu(cfg, mix)
    wall = time.time() - t0

    ar_draw = res.streams.upper_ar_draw
    wg = res.streams.waste_gan

    print("=" * 70)
    print("  ASU waste-GAN / Ar side-draw validation")
    print("=" * 70)
    print(f"  Wall time: {wall:.1f}s   Converged: {res.converged}   Outer iters: {res.n_outer_iter}")
    print()
    print(f"  waste_gan_stage={cfg.waste_gan_stage}  waste_gan_flow={cfg.waste_gan_flow:.4f} mol/s")
    print(f"  ar_draw_flow={cfg.ar_draw_flow:.4f} mol/s  D_frac_argon={cfg.D_frac_argon}")
    print()
    print(f"  N2 purity/recovery : {res.N2_purity*100:.3f}% / {res.N2_recovery*100:.1f}%")
    print(f"  O2 purity/recovery : {res.O2_purity*100:.3f}% / {res.O2_recovery*100:.1f}%")
    print(f"  Ar purity (Ar col distillate) / overall recovery: "
          f"{res.Ar_purity*100:.1f}% / {res.Ar_recovery*100:.1f}%")
    print()
    if ar_draw is not None:
        z = ar_draw.z
        print(f"  Ar side-draw composition (N2, O2, Ar): "
              f"[{z[0]*100:.1f}%, {z[1]*100:.1f}%, {z[2]*100:.1f}%]")
        b = BASELINE_AR_DRAW_Z
        print(f"  Baseline (no waste GAN, pre-fix)      : "
              f"[{b[0]*100:.1f}%, {b[1]*100:.1f}%, {b[2]*100:.1f}%]")
    if wg is not None:
        print(f"  Waste GAN composition (N2, O2, Ar)    : "
              f"[{wg.z[0]*100:.2f}%, {wg.z[1]*100:.4f}%, {wg.z[2]*100:.4f}%]"
              f"  flow={wg.flow:.4f} mol/s")
    print()
    print(f"  Target  : Ar side-draw ~90% O2 / ~10% Ar / trace N2; overall Ar recovery <= 60%")
    ok_draw = ar_draw is not None and ar_draw.z[1] > 0.80 and ar_draw.z[2] > 0.05 and ar_draw.z[0] < 0.10
    ok_recovery = res.Ar_recovery <= 0.60
    print(f"  Result  : draw composition {'OK' if ok_draw else 'OFF TARGET'}, "
          f"Ar recovery {'OK' if ok_recovery else 'EXCEEDS CEILING'}")


if __name__ == "__main__":
    main()
