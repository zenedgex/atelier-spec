"""Protocols, slot classes and the 0.x migration table of atelier-block/1.0 (standards/block.md)."""

from __future__ import annotations

# protocol -> (required roles, optional roles); section 4 of the block standard
PROTOCOLS: dict[str, tuple[frozenset[str], frozenset[str]]] = {}


def _p(name: str, required: str, optional: str = "") -> None:
    PROTOCOLS[name] = (frozenset(required.split()), frozenset(optional.split()))


# engine level (from 0.2)
_p("clock_reset", "clk", "rst_n")
_p("stream_vr", "valid ready", "data")
_p("stream_v", "valid", "data")
_p("write_vr", "valid ready addr data", "mask")
_p("write_plain", "we addr data")
_p("read_sync", "re addr rdata")
_p("config", "addr data", "valid we ready")
_p("command", "valid ready cmd", "done")
_p("tile_req_hold", "req slot hold")
_p("row_pipe", "re row data_in en data_out")
_p("start_done", "start", "done")
_p("status", "busy")
# SoC level (from 0.2)
_p("obi", "req gnt rvalid addr", "rdata we be wdata")
_p("interrupt", "irq", "ack id")
_p("debug", "req")
_p("run_control", "enable", "sleep")
_p("strap", "")
# new in 1.0: array dataflows
_p("weight_load", "load row")
_p("os_stream", "in_valid in_first a_col b_row drain out_row")
# new in 1.0: standard buses
_p("axi4",
   "awvalid awready awaddr wvalid wready wdata bvalid bready arvalid arready araddr rvalid rready rdata",
   "awid awlen awsize awburst awlock awcache awprot awqos awregion awuser wstrb wlast wuser "
   "bid bresp buser arid arlen arsize arburst arlock arcache arprot arqos arregion aruser rid rresp rlast ruser")
_p("axi4_lite",
   "awvalid awready awaddr wvalid wready wdata bvalid bready arvalid arready araddr rvalid rready rdata",
   "awprot wstrb bresp arprot rresp")
_p("ahb", "haddr htrans hwrite hrdata hready", "hsize hburst hprot hwdata hreadyout hresp hsel hmastlock")
_p("apb", "paddr psel penable pwrite prdata pready", "pwdata pstrb pprot pslverr")
_p("axi_stream", "tvalid tready", "tdata tstrb tkeep tlast tid tdest tuser")
# new in 1.0: generic memory and movement
_p("mem_req", "req gnt addr rvalid", "we wdata be rdata id rid")
_p("dma_desc", "valid ready desc", "done")

ROLES = {"source", "sink", "master", "slave"}

# slot class -> category (the part before the dot) and a readable name; section 5.1
SLOT_CLASSES: dict[str, str] = {
    "compute.tensor": "Tensor unit (systolic, MAC array, CIM, tensor core, matrix unit)",
    "compute.vector": "Programmable vector unit (SIMD)",
    "compute.simt": "SIMT cluster (GPU-like)",
    "compute.vliw": "VLIW core (DSP)",
    "compute.engine": "Engine (a composition)",
    "accelerator.op": "Operator accelerator",
    "memory.scratchpad": "Scratchpad",
    "memory.cache": "Cache",
    "memory.tcm": "Tightly coupled memory",
    "memory.compute": "Compute memory (CIM)",
    "memory.accumulator": "Accumulator memory",
    "memory.activation": "Activation memory",
    "memory.sram": "System RAM",
    "memory.l2": "Shared L2",
    "memory.dram_ctrl": "DRAM controller",
    "vector.post_op": "Requantize (post-op)",
    "vector.elementwise": "Elementwise and activation",
    "vector.row": "Row reductions (norm, softmax)",
    "control.core": "Control core",
    "control.sequencer": "Sequencer",
    "control.irq": "Interrupt controller",
    "movement.dma": "DMA",
    "movement.bridge": "Bus bridge",
    "movement.interconnect": "Interconnect",
    "movement.noc": "Network on chip",
}

CATEGORIES = ("compute", "accelerator", "memory", "vector", "control", "movement")

TENSOR_KINDS = {"systolic_ws", "systolic_os", "mac_array", "cim", "tensor_core", "hmx"}

# classes whose blocks run programs: they need a core contract when programmed by an ISA
CORE_CLASSES = {"control.core", "compute.vliw", "compute.simt"}

PROGRAMMING_MODELS = {"none", "command", "isa"}

# 0.x -> 1.0: slot class renames, and what a renamed class implies
MIGRATE_CLASS = {
    "compute.matmul_tile": ("compute.tensor", {"kind": "cim"}),
}
