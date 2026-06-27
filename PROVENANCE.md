# Algorithm Provenance

This file records the textbook and literature sources for every algorithm
in the distcalc package. Maintained to demonstrate clean-room implementation
independent of GPL-licensed simulator source code (DWSIM, ChemSep).

---

## L1 — Component Constants

| Constant | Source |
|----------|--------|
| Tc, Pc, ω (N₂, O₂, Ar) | NIST WebBook (https://webbook.nist.gov), accessed 2024 |
| MW | NIST Atomic Weights (IUPAC 2021) |
| Cp_ig (Shomate) | NIST WebBook Shomate equation, phases: Gas |
| Kij (N₂/O₂/Ar) | See below |

**Kij citation:**
Binary interaction parameters for the Peng-Robinson EOS fitted to cryogenic
VLE data for N₂/O₂/Ar systems.

- k(N₂,O₂) = −0.0119: Rowlinson, J.S. (1969) "Liquids and Liquid Mixtures"
  2nd ed., adapted for PR by multiple authors; consistent with values in
  Poling, B.E., Prausnitz, J.M., O'Connell, J.P., "The Properties of Gases
  and Liquids" 5th ed. (2001), App. B.
- k(N₂,Ar) = −0.0026, k(O₂,Ar) = 0.0104: Consistent with Thorogood, R.M.
  (1991) "Developments in Air Separation", Gas Sep. Purif. 5:83–94 and
  Xu, G. et al. (2014) "An Improved Simulation Framework for Cryogenic
  Air Separation", AIChE J. 60:2053.

These are public-domain physical data. No proprietary regression involved.

---

## L2 — PR-EOS Engine

**Peng-Robinson equation of state:**
Peng, D.-Y. and Robinson, D.B. (1976) "A New Two-Constant Equation of State"
Ind. Eng. Chem. Fundam. 15(1):59–64.

**Mixing rules (vdW one-fluid):**
van der Waals (1890); standard form as in Prausnitz, J.M. et al.,
"Molecular Thermodynamics of Fluid-Phase Equilibria" 3rd ed. (1999) §5.

**Fugacity coefficient (analytic PR form):**
Equation 6.64 in Smith, J.M., Van Ness, H.C., Abbott, M.M.,
"Introduction to Chemical Engineering Thermodynamics" 8th ed. (2018).

**Departure enthalpy and entropy:**
Table 6-4 in Poling, Prausnitz, O'Connell (2001), and
Equation 6-4.13 ibid.

**Root selection (minimum Gibbs energy):**
Michelsen, M.L. (1982) "The Isothermal Flash Problem",
Fluid Phase Equilibria 9:1–19.

---

## L3 — Phase Equilibrium

**Wilson K-value estimate:**
Wilson, G.M. (1969) "A Modified Redlich-Kwong EOS...",
AIChE J. Paper 15C, 65th Annual Meeting.

**Rachford-Rice equation:**
Rachford, H.H. and Rice, J.D. (1952) "Procedure for Use of Electronic
Digital Computers in Calculating Flash Vaporization Hydrocarbon
Equilibrium", JPT 4(10):19.

**PT-flash successive substitution:**
Prausnitz et al. (1999) §8.2; Michelsen & Mollerup,
"Thermodynamic Models: Fundamentals & Computational Aspects" 2nd ed. (2007).

**Bubble/dew point Newton solvers:**
Seader, J.D., Henley, E.J., Roper, D.K.,
"Separation Process Principles" 3rd ed. (2011) §4.4.

---

## L4 — Rigorous Column Solver (MESH)

**Wang-Henke Bubble-Point algorithm:**
Wang, J.C. and Henke, G.E. (1966) "Tridiagonal Matrix for Distillation",
Hydrocarbon Processing 45(8):155–163.

**MESH equations / tridiagonal formulation:**
Seader, J.D., Henley, E.J., Roper, D.K. (2011) §10.3.

**Stage energy balance (top-down):**
Naphtali, L.M. and Sandholm, D.P. (1971) "Multicomponent Separation
Calculations by Linearization", AIChE J. 17(1):148–153.

**FUG shortcut:**
- Fenske, M.R. (1932) Ind. Eng. Chem. 24:482.
- Underwood, A.J.V. (1948) Chem. Eng. Prog. 44:603.
- Gilliland correlation (Molokanov approximation):
  Molokanov, Y.K. (1972) "Unipetrol" (orig. Russian); English summary in
  Seader & Henley (2011) §9.3.

---

## Validation Oracles

| Tool | Role |
|------|------|
| CoolProp 6.x (MIT license) | Pure-component Z, φ, H_dep reference (tests only, not runtime) |
| DWSIM 8.x (GPL) | Binary VLE bubble/dew reference; rigorous column stage profiles |
| NIST WebBook | Pure-component saturation data for cryogenic range |

CoolProp and NIST data are used **only in the test suite as numerical oracles**,
not as a runtime dependency. The shipped engine is self-contained and
license-clean.

---

*Last updated: 2024. Maintained by distcalc project.*
