"""distcalc — Pure-Python PR-EOS rigorous distillation calculator.

Layer architecture:
  L1  components/   — component constants + Kij matrix
  L2  thermo/       — PR-EOS engine (Z, φ, H_dep, S_dep)
  L3  equilibrium/  — PT-flash, bubble/dew T and P
  L4  column/       — MESH solver (Wang-Henke BP), FUG shortcut
  L6  mcp/          — MCP-callable tools

Units: SI throughout (Pa, K, J/mol, mol/s). Convert only at I/O boundary.
"""

__version__ = "0.1.0"
