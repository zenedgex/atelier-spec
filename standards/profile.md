# Atelier Profile

Version 0.1 (draft), 2026-10-15.
- Schema: `atelier_spec/data/schemas/profile-0.1.schema.json`
- Checker: `atelier-spec profile check FILE... [--machine M]`
- Conformance suite: `conformance/profile/` (checked against `conformance/machine/good/gemmini.yaml`)

A profile is what one run of a network on an accelerator measured, layer by layer: cycles, what each
unit did, the bytes moved. It is evidence, not a model. Its numbers are as the accelerator's simulator,
its silicon or the customer's tools reported them, with the run's evidence label.

Atelier reads profiles to analyze and calibrate designs it did not generate (atelier-platform spec 20).
Its own `soc_top` counters are written in the same form, so a design whose truth we know checks the
analysis of designs whose truth we don't.

## 1. Sections

| Section | Required | Content |
|---|---|---|
| `standard` | yes | `atelier-profile/0.1` |
| `run` | yes | `machine: {name, hash, params}` (the machine description it ran on, and the generator's parameters), `network`, `batch`, `compiler: {name, version, options}`, `clock`, `evidence` (required: an `atelier-evidence/1.0` label), `tool`, `date` |
| `layers` | yes | at least one layer (below) |
| `total` | no | `{cycles}` for the whole run. Layers may overlap in time (pipelined frames), so the total is given, not summed |

**A layer:**

| Field | Required | Content |
|---|---|---|
| `name` | yes | unique in the profile |
| `op` | yes | an op-set name (`conv2d`, `matmul`, ...) or the vendor's |
| `cycles` | yes | the layer's cycles, in the run's clock |
| `start`, `end` | no | on the run's timeline; when both are given, `end - start = cycles` |
| `shapes`, `formats` | no | tensor shapes by role; formats as the machine writes them |
| `units` | no | the units it ran on (machine unit ids; `engine.3` for a cluster instance) |
| `macs` | no | multiply-accumulates done |
| `activity` | no | unit → `{busy, stall, start, end}`. `busy`: doing work; `stall`: has work but waits; the rest is idle. `busy + stall <= cycles` |
| `bytes` | no | per memory (`spad`) or per DMA route (`l2>spad`): bytes moved |
| `counters` | no | any vendor counters, as they come |

## 2. Consistency checks

The checker reports an error when:
- the run's evidence is not a label;
- a layer name is used twice;
- a layer ends before it starts, or `end - start` is not its cycles;
- a unit's `busy + stall` exceeds the layer's cycles, or its activity ends before it starts.

With a machine (`--machine`), also when:
- the run's machine name is not the machine's;
- a unit is not in the machine;
- a memory or a route's end is not a memory or a stream port of the machine.

## 3. Versioning

As the other standards: `major.minor`. A minor version adds fields; a major version changes a rule.
