# Atelier Block Standard

Version 1.0 (draft), 2026-10-09. Machine-readable form: `atelier_spec/data/schemas/block-1.0.schema.json`;
checker: `atelier-spec block check`; conformance suite: `conformance/block/`.

Changes from 0.2:
- slot classes for every architecture: tensor kinds, vector, SIMT, VLIW, operator accelerators, caches, TCM, bridges, NoC (section 5.1);
- the **programming contract**: `none`, `command` or `isa` (section 5.3);
- protocols for array dataflows and standard buses (section 4);
- capabilities name ops of the op set, `atelier-ops/0.1` (section 5.2);
- the memory and core contracts (section 5.4);
- no fixed counts (section 5.5);
- YAML accepted besides JSON;
- 0.x files migrate on load (section 9).

Atelier composes accelerators from **blocks**: a compute-in-memory macro, a systolic array, a
scratchpad, a vector unit, a DMA, a control core. Each block has a **contract**: its interface, its
behaviour in integer arithmetic, its timing, its cost, and the evidence behind those numbers.
**Templates** wire blocks into **slots**. A block from Atelier's library and a block from a
customer or a vendor are treated the same way: anything that meets its slot's contract can fill it.

This standard defines that contract. A whole chip — which blocks, how many, which memories and
buses — is described by the machine standard (`standards/machine.md`), which names blocks by this
contract. Section 9 says how 0.x contracts read under 1.0.

The key words MUST, SHOULD and MAY are used as in RFC 2119.

---

## 1. Terms

| Term | Meaning |
|---|---|
| **Block** | One hardware unit with a contract: RTL (or a macro's views) plus the files of section 3 |
| **Slot** | A position in a template that accepts any block of one **slot class** (section 5) |
| **Template** | A fixed wiring of slots, with its own parameters (section 7) |
| **Composition** | A template with every slot filled and every parameter set: one design point |
| **Port group** | A set of ports following one protocol of section 4 |
| **Reference model** | Executable integer semantics of a block, the bit-exact source of truth |
| **Evidence** | Where a number comes from (section 8) |

## 2. Principles

1. **The reference model is the truth.** RTL is correct when it matches its reference model bit for
   bit; the compiler targets reference models, never RTL. This is how Sutra works today (`isa.py`).
2. **Contracts, not implementations.** A slot fixes ports, protocols and semantics; the block's
   inside is free (standard cells, a vendor macro, a customer's RTL).
3. **Every number carries its evidence.** A cost or timing figure without an evidence label is
   invalid.
4. **Formats are part of the contract.** A block states the integer formats and scaling it supports;
   the compiler and the quantizer work within them.
5. **Nothing leaves the customer's site.** A customer block's contract can be checked where its
   RTL lives; the contract files are all Atelier needs.

## 3. The block contract

A block is a directory with these files:

```
<block>/
  block.json        identity, parameters, port groups, capabilities, timing, cost, evidence
  model.py          reference model (integer semantics)
  rtl/              SystemVerilog (or a macro's Verilog model, LEF, Liberty)
  tests/            contract tests (section 6)
  README.md         what it is, limits, known differences from its slot class
```

### 3.1 `block.json`

```json
{
  "standard": "atelier-block/1.0",
  "name": "zenedgex.cim_macro",
  "version": "1.0.0",
  "slot_class": "compute.tensor",
  "kind": "cim",
  "owner": "ZenEdgeX",
  "top": "cim_macro",
  "parameters": {
    "R":  {"values": [16, 32, 64, 128], "meaning": "reduction length of a tile"},
    "C":  {"values": [8, 16, 32, 64],   "meaning": "output columns"},
    "DW": {"values": [8],               "meaning": "operand width"},
    "IB": {"values": [1, 2, 4, 8],      "meaning": "input bits per cycle; divides DW"},
    "DEPTH": {"expr": "multiple of R",  "meaning": "weight rows held"}
  },
  "port_groups": [
    {"name": "clk_rst", "protocol": "clock_reset", "ports": ["clk", "rst_n"]},
    {"name": "wr", "protocol": "write_vr", "role": "sink",
     "ports": {"valid": "wr_valid", "ready": "wr_ready", "addr": "wr_addr", "data": "wr_data"}},
    {"name": "tile", "protocol": "tile_req_hold",
     "ports": {"req": "tile_req", "slot": "slot", "hold": "hold"}},
    {"name": "x", "protocol": "stream_v", "role": "sink", "ports": {"valid": "x_valid", "data": "x_row"}},
    {"name": "y", "protocol": "stream_v", "role": "source", "ports": {"valid": "y_valid", "data": "y_row"}}
  ],
  "capabilities": [
    {"op": "matmul", "a": "int8", "w": "int8", "acc": "int(2*DW+clog2(R))"}
  ],
  "timing": {
    "tile_latency": {"value": 2, "evidence": "rtl-sim"},
    "row_interval": {"expr": "DW / IB", "evidence": "rtl-sim"},
    "result_latency": {"expr": "see model.py", "evidence": "rtl-sim"}
  },
  "cost": {
    "area_um2": {"expr": "...", "evidence": "synthesis"},
    "energy_pj": {"mac": {"value": 0.48, "evidence": "post-layout"}},
    "leakage_uw": {"expr": "...", "evidence": "estimate"}
  },
  "files": {"rtl": ["rtl/cim_macro.sv", "rtl/dimc_mem.sv", "rtl/gemm_dimc.sv"]},
  "programming": {"model": "none"}
}
```

- `parameters`: every RTL parameter the composition may set, with its allowed values or a rule.
- `port_groups`: every port of `top` MUST belong to exactly one group; a group names its protocol
  (section 4) and maps the protocol's roles to the block's port names.
- `capabilities`: the ops the block performs, by name from the op set (section 5.2), with operand
  and result formats.
- `programming`: how the block is driven (section 5.3). Omitted means `none`.
- `kind` (`compute.tensor` only): `systolic_ws`, `systolic_os`, `mac_array`, `cim`, `tensor_core` or `hmx`.
- A port group gives its ports as a role map, a list (role = port name), or a `prefix`/`suffix` that
  finds them in the RTL (`prefix: s_axil_` maps role `awvalid` to port `s_axil_awvalid`).
- `timing` and `cost`: numbers or expressions in the parameters, each with an `evidence` label.

### 3.2 Reference model (`model.py`)

A Python module exposing:

```python
def step(state, inputs, params) -> (state, outputs)   # cycle-level, OPTIONAL
def op(name, operands, params) -> result               # op-level, REQUIRED for each capability
def latency(name, shape, params) -> int                # REQUIRED: cycles, matching "timing"
```

`op` MUST be exact integer arithmetic (NumPy int64 or Python int), including rounding, saturation
and overflow. `latency` MUST be the formula the cycle model uses. A block MAY instead name an
external golden model (C or Python) the customer already has, wrapped to this interface.

## 4. Port protocols

All blocks use one clock `clk` (rising edge) and an active-low reset `rst_n`. Control state resets
asynchronously (`always_ff @(posedge clk or negedge rst_n)`); memories and datapath registers MAY
have no reset. Multi-element buses pack element *i* at bits `[i*W +: W]`.

| Protocol | Ports (roles) | Rule |
|---|---|---|
| `clock_reset` | `clk`, `rst_n` | as above |
| `stream_vr` | `valid`, `ready`, `data` | a transfer happens on a cycle with `valid && ready`; `valid` MUST NOT wait for `ready`; `data` is held while `valid && !ready` |
| `stream_v` | `valid`, `data` | no back-pressure: the sink MUST accept every cycle `valid` is high; the composition guarantees the rate (e.g. at most one row every `row_interval` cycles) |
| `write_vr` | `valid`, `ready`, `addr`, `data` (+ `mask`) | `stream_vr` carrying an address; the block MAY hold `ready` low while busy |
| `read_sync` | `re`, `addr`, `rdata` | `rdata` is valid exactly `read_latency` cycles after `re` (declared in `timing`; 1 for today's SRAMs) |
| `config` | `valid`/`we`, `ready`, `addr[15:0]`, `data[63:0]` | memory-mapped parameters; the address map is part of the reference model |
| `command` | `valid`, `ready`, `cmd[63:0]`, `done` | 64-bit commands in order; `done` pulses once per finished command |
| `tile_req_hold` | `req`, `slot`, `hold` | `req` pulses to select tile `slot`; the first input MAY follow `tile_latency` cycles later; `hold` stays high until the last result and blocks writes to that tile |
| `status` | `busy` (+ counters) | `busy` high while any accepted work is unfinished |
| `write_plain` | `we`, `addr`, `data` | a write on every cycle `we` is high; no back-pressure (accumulator memories) |
| `row_pipe` | `re`, `row`, `data_in`, `en`, `data_out` | a fixed pipeline: `en[i]` advances stage *i*; `data_out` follows `data_in` after the stages enabled (the post-op unit) |
| `start_done` | `start`, `done` | `start` pulses with the operation's fields; `done` pulses once when its last result is written |
| `obi` | `req`, `gnt`, `rvalid`, `addr` (+ `rdata`, `we`, `be`, `wdata`) | OpenHW Bus Interface: a transfer is accepted on `req && gnt`; its response (read data, or the write's completion) arrives with `rvalid` one or more cycles later, in order. Role `source` is the manager (the core) |
| `interrupt` | `irq` (+ `ack`, `id`) | level-sensitive interrupt lines; `ack` pulses with the `id` of the interrupt taken |
| `debug` | `req` (+ status fields) | debug request and the core's debug status outputs |
| `run_control` | `enable` (+ `sleep`) | `enable` lets the block run (a core fetches); `sleep` reports it idle |
| `strap` | fields only | inputs a SoC ties at integration: boot and trap addresses, ids, test-mode and clock-gate enables. Their values belong to the template |

A port group MAY carry **fields**: extra ports that travel with the protocol's transfer (for
example the row and channel indices of a drained row). Fields are listed by name in `block.json`.

New in 1.0:

| Protocol | Ports (roles) | Rule |
|---|---|---|
| `weight_load` | `load`, `row` | weight-stationary arrays: while `load` is high, one weight row enters per cycle, shifting the array (`gemm_ws`) |
| `os_stream` | `in_valid`, `in_first`, `a_col`, `b_row`, `drain`, `out_row` | output-stationary arrays: `in_first` marks k = 0; `drain` shifts results out (`gemm_os`) |
| `axi4` | AMBA AXI4 channel signals, without the `a`/`m_`/`s_` prefixes: `awvalid` ... `rdata` required; ids, bursts, cache, prot, qos, region, user, `wstrb`, `wlast`, `bresp`, `rresp`, `rlast` optional | as AMBA AXI4 |
| `axi4_lite` | the AXI4-Lite subset | as AMBA AXI4-Lite |
| `ahb` | `haddr`, `htrans`, `hwrite`, `hrdata`, `hready` (+ `hsize`, `hburst`, `hprot`, `hwdata`, `hreadyout`, `hresp`, `hsel`, `hmastlock`) | as AMBA AHB |
| `apb` | `paddr`, `psel`, `penable`, `pwrite`, `prdata`, `pready` (+ `pwdata`, `pstrb`, `pprot`, `pslverr`) | as AMBA APB |
| `axi_stream` | `tvalid`, `tready` (+ `tdata`, `tstrb`, `tkeep`, `tlast`, `tid`, `tdest`, `tuser`) | as AMBA AXI4-Stream |
| `mem_req` | `req`, `gnt`, `addr`, `rvalid` (+ `we`, `wdata`, `be`, `rdata`, `id`, `rid`) | a generic request/grant memory port; responses MAY return out of order when `id`/`rid` are present |
| `dma_desc` | `valid`, `ready`, `desc` (+ `done`) | DMA descriptors in order |

Roles `master`/`slave` are used with the buses; `source`/`sink` with streams. A block MAY use a
protocol not listed only through a versioned addition to this table.

## 5. Slot classes and capabilities

### 5.1 Slot classes

| Category | Slot class | Does | Typical port groups |
|---|---|---|---|
| compute | `compute.tensor` | matrix multiplies; `kind` says how (WS, OS, MAC array, CIM, tensor core, matrix unit) | `weight_load` / `os_stream` / `tile_req_hold` + streams, or a bus |
| compute | `compute.vector` | a programmable SIMD unit | `config` or a bus, a memory port |
| compute | `compute.simt` | a GPU-like cluster of SIMT lanes | buses, a core contract |
| compute | `compute.vliw` | a DSP core | buses, a core contract |
| compute | `compute.engine` | a whole engine: a template composition used as one slot | the engine's command and data ports |
| accelerator | `accelerator.op` | whole operators behind commands | `command` or a bus, memory ports |
| memory | `memory.scratchpad`, `memory.cache`, `memory.tcm`, `memory.compute`, `memory.accumulator`, `memory.activation`, `memory.sram`, `memory.l2`, `memory.dram_ctrl` | storage at each level; `memory.compute` holds CIM weights | `read_sync`, `write_vr`, `write_plain`, `mem_req`, buses |
| vector | `vector.post_op`, `vector.elementwise`, `vector.row` | fixed-function units driven by a template | `row_pipe`, `config`, `command`, `start_done` |
| control | `control.core`, `control.sequencer`, `control.irq` | runs the runtime; issues commands; collects interrupts | buses, `interrupt`, `debug`, `run_control`, `strap` |
| movement | `movement.dma`, `movement.bridge`, `movement.interconnect`, `movement.noc` | moves data; converts buses; connects masters and slaves | `dma_desc`, buses, streams |

`compute.matmul_tile` (0.x) is `compute.tensor` with `kind: cim`.

### 5.2 Capabilities and the op set

Capabilities name ops of the op set (`atelier-ops/0.1`, `atelier_spec/data/opset-0.1.yaml`): their
semantics, attributes, formats and bit-exact references. A block MAY add its own ops in a namespace
(`acme.fft256`) under `programming.ops`, each with a reference; the compiler side of such an op is a
native plugin (atelier-compiler).

Formats are written `int8`, `int16`, `int32`, `uint7` (Q7), with scales `per_tensor`,
`per_channel` or `per_token`. New ops and formats are added by a minor version of this standard.

### 5.3 Programming contract

Every block says how it is programmed. The compiler learns a block from this section; no compiler
code names a block.

| `programming.model` | Driven by | The block provides | The compiler uses it to |
|---|---|---|---|
| `none` | the template's sequencer, through the slot's protocols | nothing more | tile and schedule from parameters and timing |
| `command` | descriptors written by a runtime or a sequencer | a **command set** | generate the descriptor encoder; match graph ops to commands; schedule by the cycle expressions |
| `isa` | a program | an **ISA binding** | emit kernels in C or LLVM IR with the mapped intrinsics and micro-kernels, built by the named toolchain |

**Command set:**
- `word` (descriptor bits), `doorbell` (where descriptors go), `completion` (interrupt or poll);
- `commands`, each with:
  - `name` and `opcode`;
  - `fields` (`bits: [low, high]`, `kind`: opcode, addr, count, imm, flag, enum, reserved; `space` for addresses);
  - `semantics` (a sequence of op-set ops whose inputs are fields or earlier results);
  - `constraints` and `cycles` (expressions in fields and parameters), and an `evidence` label for `cycles`.

**ISA binding:**
- `isa`, `toolchain` (`cc`, `flags`, `link`, `objcopy`, `iss`), `headers`;
- `intrinsics` (op, formats, intrinsic, vector length);
- `microkernels` (op, symbol, library, C signature, tile, cycles, evidence);
- `abi`.

Rules checked at L0:
- fields fit the word and don't overlap;
- one opcode field at most, and the opcode fits it;
- names and opcodes are unique;
- every semantic op, intrinsic op and micro-kernel op is in the op set or the block's extension;
- every name in a constraint or cycle expression is a field, a parameter, a tile dimension or a signature argument;
- `accelerator.op` blocks use `command`; `control.core`, `compute.vliw` and `compute.simt` blocks use `isa`.

At L1, `model.py` executes the commands, and the results equal the op-set references composed per the semantics; each mapped intrinsic run on the ISS equals its op's reference.

### 5.4 Memory and core contracts

- `memory`: `size`, `banks`, `width`, `ports`, `read_latency`, `mask_granularity`, `retention`,
  `access` (rows, words, bytes), `realisation` (rtl, macro, compiler).
- `core` (required for `control.core`, `compute.vliw`, `compute.simt`):
  - `isa`;
  - `boot_addr` (a parameter `{param: X}` or a strap port `{strap: port}`);
  - `irq` (protocol and line count);
  - `bus` (per port group).

### 5.5 Counts

No rule of this standard fixes a count: of parameters, port groups, commands, fields, opcodes,
intrinsics or ops. Widths derive from parameters.

## 6. Conformance

| Level | Name | Requires |
|---|---|---|
| **L0** | Described | the contract is valid and consistent (`atelier-spec block check`), and against the RTL every port is in exactly one port group and the parameters exist (atelier-blocks) |
| **L1** | Functional | contract tests pass: random and directed stimulus on the RTL in simulation, every output equal to the reference model |
| **L2** | Timed | measured latencies and intervals equal `latency()` (exactly, or within a stated bound) over a characterization sweep |
| **L3** | Implemented | area, timing and power from synthesis or place-and-route on a named kit, with evidence labels |

A composition is only as conformant as its lowest block. Atelier MUST show the level of each block
next to any result that depends on it.

## 7. Templates

**Engine v1** (today's `accel_top`):

```
 command ─► sequencer ─► compute.tensor ──────►  memory.accumulator ─► vector.post_op ─► out / activation
               │               ▲  ▲                                          │
               │   memory.scratchpad (A)  weights (write_vr)                 ▼
               └──────────────► vector.elementwise (+ memory.activation) ◄───┘
                               vector.row (in development)
```

The tensor slot takes any `kind`; today's `accel_top` builds WS, OS and CIM (`DATAFLOW` 0, 1, 2).

**SoC v1** (today's `multi_top` and `soc_top`): `control.core` broadcasting one command stream to
any number of Engine v1 instances (tested with 1, 2, 4 and 8); host writes per engine;
`movement.dma` and `memory.l2` are planned slots.

Templates live in atelier-architectures, and are data plus SystemVerilog skeletons. The machine
standard decides the counts.

A template declares its own parameters (engine count, memory sizes, bus widths) and which slot
classes each slot accepts. The template, not the block, owns the sequencer, the arbitration and the
wiring.

## 8. Evidence labels

| Label | Meaning |
|---|---|
| `estimate` | a formula or a literature value, not checked on this design |
| `datasheet` | a vendor's datasheet |
| `rtl-sim` | measured in RTL simulation of this block |
| `synthesis` | from synthesis on a named kit |
| `post-layout` | from a placed and routed layout on a named kit |
| `transistor-sim` | from transistor-level simulation |
| `silicon` | measured on silicon |

Public results MUST show the weakest label among the numbers they use.

## 9. Migration from 0.x

A 0.x file is read as 1.0 by these rules (`atelier_spec.block.migrate`):

| 0.x | 1.0 |
|---|---|
| `compute.matmul_tile` | `compute.tensor`, `kind: cim` |
| capability ops `matmul_tile`, `elt_add`, `elt_mul`, `elt_max`, `pool_max`, `map_lut`, `im2col_load`, `row_norm`, `row_softmax` | `matmul`, `add`, `mul`, `max`, `maxpool`, `lut`, `im2col`, `rmsnorm`, `softmax` |
| a memory's `store_rows` capability | `memory: {access: rows, data}` |
| a core's ISA as a capability (`rv32imc`) | `programming: {model: isa, isa}`, `core: {isa}` |
| no `programming` | `programming: {model: none}` |

Migration adds nothing a 0.x file didn't say. A core still needs its toolchain and boot address
written in, and the check asks for them. Today's eight library blocks: seven pass after migration;
`cv32e40p` passes once those two facts are added (`tests/fixtures/v1.0/cv32e40p.json`).

## 10. Versioning

The standard is versioned `major.minor`. A minor version adds protocols, ops or formats; a major
version changes an existing rule. `block.json` names the version it follows; Atelier refuses a
block whose major version it does not know.
