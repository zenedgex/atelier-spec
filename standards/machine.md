# Atelier Machine Description

Version 0.1 (draft), 2026-10-09.
- Schema: `atelier_spec/data/schemas/machine-0.1.schema.json`
- Checker: `atelier-spec machine check`
- Conformance suite: `conformance/machine/`
- Examples: `conformance/machine/good/`, one each of an NPU, a GPU-like chip, a DSP and a CIM chip

A machine description is one YAML (or JSON) file that describes a whole chip:
- its units;
- how they group into clusters;
- its memory hierarchy, data movement and interconnect;
- its control model, address map and clock domains;
- optionally, the axes to explore.

It names blocks by their contracts (`standards/block.md`); it contains no RTL. The generator
(atelier-gen) builds the RTL from it, the compiler (atelier-compiler) targets it, and the runtime
(atelier-runtime) gets its board support from it.

## 1. Sections

| Section | Required | Content |
|---|---|---|
| `standard` | yes | `atelier-machine/0.1` |
| `name` | yes | |
| `units` | yes | id → `{class and/or block, params, capabilities, control: command \| isa \| none, core}` |
| `clusters` | no | id → `{count, contains: [unit and memory ids], wiring: template id}` |
| `memory` | yes | a list of `{name, level, kind, per, size, banks, width, ports, ways, line, latency, energy, policy, type, bandwidth, block}` |
| `movement` | no | `dma: [{engine, from, to}]`, `bridges: [{from_bus, to_bus}]` |
| `interconnect` | no | `{topology: bus \| crossbar \| noc \| auto, protocol, width, masters, slaves}` |
| `control` | yes | `{model: microcontroller \| application \| host \| computing, core: unit id, os: none \| rtos \| linux}` |
| `address_map` | no | region → `{base, size, stride, target}` |
| `clocks` | no | domain → `{freq, units}` |
| `space` | no | dotted path → list of values, for exploration |

**Memory fields:**
- `kind`: register_file, scratchpad, cache, tcm, compute (CIM weights), dram, rom.
- `per`:
  - `chip`;
  - `cluster` (one per cluster instance; the memory must be in a cluster's `contains`);
  - a unit id (private to that unit).
- `level`: 0 is closest to compute.

**Sizes and references:**
- Sizes are bytes, or strings with units (`64B`, `128KiB`, `2MiB`, `1GB`).
- References in `interconnect`, `clocks` and `movement` are ids. `cluster.*` and `cluster.*.regs` refer to every instance of a cluster.

## 2. Consistency checks

The checker reports an error when:
- an id is used twice across units, clusters and memories;
- a class is not a slot class of the block standard, or a unit's block is of another class (when a block library is given);
- a cluster contains something that is not a unit or a memory;
- a `per: cluster` memory is in no cluster, or `per` names nothing;
- a memory's size doesn't split into its banks, or a non-DRAM memory has no size;
- a DMA's engine is not a unit, or its ends are not memories;
- an interconnect master or slave names nothing, or a protocol is unknown;
- the control core is not a unit, or is not a `control.core`, `compute.vliw` or `compute.simt`;
- two address regions overlap;
- a clock domain names nothing, or a unit is in two domains;
- a `space` path names nothing in the machine.

Warnings: a cache without a line size; a gap in memory levels; an application processor without an OS; a block not found in the libraries given.

**Planned for 0.2:**
- the tensor unit's tile fits its L1;
- every master reaches the memories its units need;
- clock-domain crossings have synchronisers;
- bandwidth per level against the compiler's plan.

## 3. Counts

No count is limited. `count`, the number of units, memories, masters, slaves,
regions and domains are free. `tests/test_units.py` checks a machine with 2,000 units, 2,000
memories, 2,000 regions and a 256-instance cluster.

## 4. Versioning

As the block standard: `major.minor`. A minor version adds sections or fields; a major version
changes a rule.
