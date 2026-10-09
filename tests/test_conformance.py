"""The conformance suite: every good file passes, every bad file fails with its expected error."""

from pathlib import Path

import pytest

from atelier_spec import block, machine

ROOT = Path(__file__).resolve().parent.parent
GOOD_BLOCKS = sorted((ROOT / "conformance/block/good").glob("*.yaml"))
BAD_BLOCKS = sorted((ROOT / "conformance/block/bad").glob("*.yaml"))
GOOD_MACHINES = sorted((ROOT / "conformance/machine/good").glob("*.yaml"))
BAD_MACHINES = sorted((ROOT / "conformance/machine/bad").glob("*.yaml"))


def _expect(f: Path) -> str:
    first = f.read_text().splitlines()[0]
    assert first.startswith("# expect: "), f
    return first.removeprefix("# expect: ")


def test_suite_is_not_empty():
    assert len(GOOD_BLOCKS) >= 2 and len(BAD_BLOCKS) >= 10
    assert len(GOOD_MACHINES) == 4 and len(BAD_MACHINES) >= 10


@pytest.mark.parametrize("f", GOOD_BLOCKS, ids=lambda f: f.name)
def test_good_block(f):
    assert block.errors(block.check(block.load(f))) == []


@pytest.mark.parametrize("f", BAD_BLOCKS, ids=lambda f: f.stem)
def test_bad_block(f):
    errs = [str(p) for p in block.errors(block.check(block.load(f)))]
    assert any(_expect(f) in e for e in errs), errs


@pytest.mark.parametrize("f", GOOD_MACHINES, ids=lambda f: f.name)
def test_good_machine(f):
    assert block.errors(machine.check(machine.load(f))) == []


@pytest.mark.parametrize("f", BAD_MACHINES, ids=lambda f: f.stem)
def test_bad_machine(f):
    errs = [str(p) for p in block.errors(machine.check(machine.load(f)))]
    assert any(_expect(f) in e for e in errs), errs


def test_machine_with_block_library():
    """With a library, a unit's class comes from its block, and a mismatch is an error."""
    lib = {"systolic_ws": {"slot_class": "compute.tensor"}, "cv32e40p": {"slot_class": "control.core"},
           "simd_vec": {"slot_class": "vector.post_op"}}
    m = machine.load(ROOT / "conformance/machine/good/npu.yaml")
    errs = [str(p) for p in block.errors(machine.check(m, lib.get))]
    assert any("units.vec" in e and "vector.post_op" in e for e in errs), errs
