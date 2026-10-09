# Atelier Evidence Labels

Version 1.0 (draft), 2026-10-09. Code: `atelier_spec/evidence.py`.

Every number Atelier reports (a cycle count, an energy, an area, a frequency) carries the label of where it came from. Weakest first:

| Label | Meaning |
|---|---|
| `estimate` | a formula or a literature value, not checked on this design |
| `datasheet` | a vendor's datasheet |
| `rtl-sim` | measured in RTL simulation of this design |
| `synthesis` | from synthesis on a named kit |
| `post-layout` | from a placed and routed layout on a named kit |
| `transistor-sim` | from transistor-level simulation |
| `silicon` | measured on silicon |

## Records

`{value | expr, unit, evidence, source, conditions}`:
- `source` names the run, file or document.
- `conditions` gives corner, voltage, clock, workload and kit.

## Combining

- A number computed from several records carries the **weakest** label among them (`evidence.weakest`).
- A Pareto point built from estimates is an estimate.
- Public results show the label.
- Numbers for our custom CIM macro come only from transistor-level simulation (sutra-cim-macro).
