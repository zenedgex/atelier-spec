"""Expressions, sizes, evidence, op references, and scale (no count limits)."""

import time

import numpy as np
import pytest

from atelier_spec import block, evidence, expr, machine, ops
from atelier_spec.ops_ref import round_shift


# ------------------------------------------------------------------------------- expressions
def test_expr_arithmetic_and_functions():
    assert expr.evaluate("2*DW + clog2(R)", {"DW": 8, "R": 32}) == 21
    assert expr.evaluate("4 + len / LANES", {"len": 256, "LANES": 64}) == 8
    assert expr.evaluate("len % 64 == 0 and len <= 4096", {"len": 128}) is True
    assert expr.names("m*n*k/1024 + 40") == {"m", "n", "k"}


@pytest.mark.parametrize("bad", ["__import__('os')", "a.b", "[1, 2]", "'x'", "f(1)", "lambda: 1"])
def test_expr_refuses_anything_but_arithmetic(bad):
    with pytest.raises(expr.ExprError):
        expr.parse(bad)


# ------------------------------------------------------------------------------------ sizes
@pytest.mark.parametrize("text,n", [("128KiB", 131072), ("2MiB", 2 << 20), (4096, 4096), ("64B", 64), ("1GB", 10**9)])
def test_size(text, n):
    assert machine.size(text) == n


# --------------------------------------------------------------------------------- evidence
def test_weakest_label_wins():
    assert evidence.weakest(["silicon", "rtl-sim", "post-layout"]) == "rtl-sim"
    with pytest.raises(ValueError):
        evidence.weakest(["guess"])


# ------------------------------------------------------------------------------ references
def test_round_shift_modes():
    x = np.array([5, 6, 7, -5, -6, -7, 4, -4])        # / 4 -> 1.25 1.5 1.75 -1.25 -1.5 -1.75 1 -1
    assert round_shift(x, 2, "floor").tolist() == [1, 1, 1, -2, -2, -2, 1, -1]
    assert round_shift(x, 2, "half_up").tolist() == [1, 2, 2, -1, -1, -2, 1, -1]
    assert round_shift(x, 2, "half_even").tolist() == [1, 2, 2, -1, -2, -2, 1, -1]


def test_requant_saturates_and_is_per_channel():
    x = np.array([[1000, -1000], [100, 300]])
    y = ops.reference("requant")(x, multiplier=np.array([1, 2]), shift=2, zero_point=0, fmt="int8")
    assert y.tolist() == [[127, -128], [25, 127]]


def test_matmul_is_exact_and_refuses_overflow():
    a = np.full((2, 4), 127)
    assert ops.reference("matmul")(a, a.T, fmt="int32")[0, 0] == 4 * 127 * 127
    with pytest.raises(OverflowError):
        ops.reference("matmul")(a, a.T, fmt="int16")


def test_lut_and_alias():
    table = np.arange(256) - 128
    assert ops.reference("exp_lut")(np.array([-128, 0, 127]), table).tolist() == [-128, 0, 127]


def test_pending_reference_says_so():
    with pytest.raises(NotImplementedError, match="pending"):
        ops.reference("softmax")


# ------------------------------------------------------------------------------- scale
def test_machine_has_no_count_limit():
    """Thousands of units, memories, regions and masters check fast and without a limit."""
    n = 2000
    m = {"standard": "atelier-machine/0.1", "name": "big",
         "units": {f"u{i}": {"class": "compute.tensor"} for i in range(n)} | {"core": {"class": "control.core"}},
         "clusters": {"c": {"count": 256, "contains": ["u0", "m0"]}},
         "memory": [{"name": f"m{i}", "level": 1, "kind": "scratchpad", "per": "cluster" if i == 0 else "chip",
                     "size": "64KiB", "banks": 4} for i in range(n)],
         "interconnect": {"topology": "noc", "protocol": "axi4", "masters": [f"u{i}" for i in range(n)],
                          "slaves": [f"m{i}" for i in range(n)]},
         "address_map": {f"r{i}": {"base": i * 0x10000, "size": "64KiB"} for i in range(n)},
         "control": {"model": "microcontroller", "core": "core"}}
    t = time.perf_counter()
    assert block.errors(machine.check(m)) == []
    assert time.perf_counter() - t < 5


def test_command_set_has_no_count_limit():
    cmds = [{"name": f"c{i}", "opcode": i, "fields": [{"name": "op", "bits": [0, 15], "kind": "opcode"}],
             "semantics": [{"op": "relu"}]} for i in range(5000)]
    s = {"standard": "atelier-block/1.0", "name": "big", "version": "1.0.0", "slot_class": "compute.vector",
         "top": "t", "parameters": [], "port_groups": [], "files": {"rtl": []},
         "programming": {"model": "command", "word": 64, "doorbell": {}, "completion": {}, "commands": cmds}}
    assert block.errors(block.check(s)) == []


def test_queues_and_spaces_follow_the_description():
    """Spec 16: a queue per command unit and per DMA channel, in a fixed order; a launch queue for an
    ISA core, none for its coprocessor or the dispatcher; spaces are the memories then the streams."""
    m = {"standard": "atelier-machine/0.2", "name": "q",
         "units": {"t": {"class": "compute.tensor", "control": "command"},
                   "mx": {"class": "compute.tensor", "control": "isa"},
                   "sm": {"class": "compute.simt", "control": "isa"},
                   "d": {"class": "movement.dma", "params": {"CHANNELS": 3}},
                   "q": {"class": "control.dispatcher"}, "c": {"class": "control.core"}},
         "memory": [{"name": "a", "level": 1, "kind": "scratchpad", "per": "chip", "size": 1024},
                    {"name": "m", "level": 2, "kind": "dram", "per": "chip"}],
         "movement": {"dma": [{"engine": "d", "from": ["m", "s"], "to": "a"}],
                      "streams": [{"name": "s", "unit": "t", "direction": "in"}]},
         "control": {"model": "microcontroller", "core": "c"},
         "dispatch": {"by": "q", "tokens": 8}}
    assert machine.check(m) == []
    assert [tuple(q) for q in machine.queues(m)] == [("t", "command", 0), ("sm", "launch", 0),
                                                     ("d", "dma", 0), ("d", "dma", 1), ("d", "dma", 2)]
    assert machine.spaces(m) == ["a", "m", "s"]
    m["standard"] = "atelier-machine/0.1"                      # 0.1 still reads
    assert machine.check(m) == []
    del m["dispatch"]
    assert machine.queues(m) == []


def test_ten_thousand_queues():
    """R15: no limit on queues or tokens."""
    units = {f"e{i}": {"class": "compute.tensor", "control": "command"} for i in range(10_000)}
    units |= {"q": {"class": "control.dispatcher"}, "c": {"class": "control.core"}}
    m = {"standard": "atelier-machine/0.2", "name": "big", "units": units,
         "memory": [{"name": "a", "level": 1, "kind": "scratchpad", "per": "chip", "size": 1024}],
         "control": {"model": "microcontroller", "core": "c"}, "dispatch": {"by": "q", "tokens": 1 << 20}}
    assert machine.check(m) == [] and len(machine.queues(m)) == 10_000
