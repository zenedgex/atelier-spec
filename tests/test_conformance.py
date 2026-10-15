"""The conformance suite: every good file passes, every bad file fails with its expected error."""

from pathlib import Path

import pytest

from atelier_spec import block, machine, profile

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
    assert len(GOOD_MACHINES) == 6 and len(BAD_MACHINES) >= 10


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


GOOD_PROFILES = sorted((ROOT / "conformance/profile/good").glob("*.yaml"))
BAD_PROFILES = sorted((ROOT / "conformance/profile/bad").glob("*.yaml"))


def _profile_machine(f: Path):
    """`# machine: gemmini.yaml` in a profile's first lines: check it against that good machine."""
    for line in f.read_text().splitlines()[:2]:
        if line.startswith("# machine: "):
            return machine.load(ROOT / "conformance/machine/good" / line.removeprefix("# machine: "))
    return None


def test_profile_suite_is_not_empty():
    assert len(GOOD_PROFILES) >= 1 and len(BAD_PROFILES) >= 8


@pytest.mark.parametrize("f", GOOD_PROFILES, ids=lambda f: f.name)
def test_good_profile(f):
    assert profile.check(profile.load(f), _profile_machine(f)) == []


@pytest.mark.parametrize("f", BAD_PROFILES, ids=lambda f: f.stem)
def test_bad_profile(f):
    errs = [str(p) for p in block.errors(profile.check(profile.load(f), _profile_machine(f)))]
    assert any(_expect(f) in e for e in errs), errs


def test_profile_of_another_machine():
    f = GOOD_PROFILES[0]
    other = machine.load(ROOT / "conformance/machine/good/nvdla_small.yaml")
    errs = [p.where for p in profile.check(profile.load(f), other)]
    assert "run.machine.name" in errs
