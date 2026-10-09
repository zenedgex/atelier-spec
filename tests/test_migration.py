"""Today's eight library blocks (atelier-block/0.x, Sutra's library as it was) under 1.0."""

from pathlib import Path

import pytest

from atelier_spec import block

FIX = Path(__file__).resolve().parent / "fixtures"
V02 = sorted((FIX / "v0.2").glob("*.json"))


def test_all_eight_are_here():
    assert len(V02) == 8


@pytest.mark.parametrize("f", [f for f in V02 if f.stem != "cv32e40p"], ids=lambda f: f.stem)
def test_migrated_block_passes(f):
    s = block.load(f)
    assert s["standard"] == block.STANDARD
    assert block.errors(block.check(s)) == []


def test_cim_macro_becomes_a_cim_tensor_unit():
    s = block.load(FIX / "v0.2/cim_macro.json")
    assert (s["slot_class"], s["kind"]) == ("compute.tensor", "cim")
    assert s["capabilities"][0]["op"] == "matmul"


def test_memory_capability_moves_to_the_memory_contract():
    s = block.load(FIX / "v0.2/spad_banked.json")
    assert s["memory"]["access"] == "rows" and s["capabilities"] == []


def test_core_needs_what_0x_did_not_say():
    """Migration does not invent a toolchain or a boot address: the check asks for both."""
    s = block.load(FIX / "v0.2/cv32e40p.json")
    assert s["programming"] == {"model": "isa", "isa": "rv32imc"}
    wheres = {p.where for p in block.errors(block.check(s))}
    assert wheres == {"core.boot_addr", "programming.toolchain"}


def test_completed_core_passes():
    assert block.errors(block.check(block.load(FIX / "v1.0/cv32e40p.json"))) == []


def test_1x_is_not_migrated():
    s = {"standard": "atelier-block/1.0", "slot_class": "compute.matmul_tile"}
    assert block.migrate(s) == s
