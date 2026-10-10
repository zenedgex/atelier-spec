"""The op set (atelier-ops/0.1): names, signatures, formats, and the bit-exact references written so far.

    from atelier_spec import ops
    ops.known("requant")                         # True
    ops.reference("requant")(x, shift=7, round="half_even", fmt="int8")

References need NumPy (the `ops` extra); the names and signatures don't.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path


OPSET = Path(__file__).resolve().parent / "data" / "opset-0.1.yaml"
_NS = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_.]*$")      # a customer namespace: acme.fft256


@cache
def opset() -> dict:
    import yaml
    return yaml.safe_load(OPSET.read_text())


def known(name: str, extra: dict | None = None) -> bool:
    """An op of the set, or one declared in `extra` (a customer's op set extension)."""
    return name in opset()["ops"] or bool(extra and name in extra)


def namespaced(name: str) -> bool:
    return bool(_NS.match(name))


def formats() -> list[str]:
    return opset()["formats"]


def fmt_range(fmt: str) -> tuple[int, int]:
    m = re.fullmatch(r"(u?)int(\d+)", fmt)
    if not m:
        raise ValueError(f"unknown format {fmt!r}")
    bits = int(m.group(2))
    return (0, 2**bits - 1) if m.group(1) else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)


def reference(name: str):
    entry = opset()["ops"][name]
    if entry.get("reference") != "ref":
        raise NotImplementedError(f"op {name}: reference pending")
    from atelier_spec import ops_ref
    return getattr(ops_ref, entry.get("alias_of", name))
