"""Configuration words from a block's register table (programming.registers), by its bit fields alone.

    rs = registers.RegisterSet(spec["programming"])
    rs.encode("post_scale", index=3, m=1 << 30, n=12)     # (address, 64-bit word)
    rs.encode("post_lut", index=0, entry=[-128, ..., -121])  # a lane field takes a list

A register is one address, or an array of them (count, stride: index i at addr + i * stride). Field
kinds: uint (must fit), int (two's complement in its width), flag (0 or 1); a field with lanes is that
many fields of its width side by side. Fields not given are 0. The compiler's C++ encoder
(Target/RegisterSet) reads the same table and is checked against this one.
"""

from __future__ import annotations


class RegisterError(ValueError):
    pass


def _field_bits(f: dict, v: int, where: str) -> int:
    lo, hi = f["bits"]
    width = (hi - lo + 1) // f.get("lanes", 1)
    v = int(v)
    if f["kind"] == "int":
        if not -(1 << (width - 1)) <= v < 1 << (width - 1):
            raise RegisterError(f"{where} = {v} does not fit a signed {width}-bit field")
        v &= (1 << width) - 1
    elif not 0 <= v < 1 << width:
        raise RegisterError(f"{where} = {v} does not fit {width} bits")
    return v


class RegisterSet:
    def __init__(self, programming: dict):
        self.word = programming.get("word", 64)
        self.registers = {r["name"]: r for r in programming.get("registers", [])}

    def address(self, name: str, index: int = 0) -> int:
        r = self.registers[name]
        count = r.get("count", 1)
        if not 0 <= index < count:
            raise RegisterError(f"{name}[{index}]: {name} has {count} entries")
        return r["addr"] + index * r.get("stride", 1)

    def encode(self, name: str, index: int = 0, **values) -> tuple[int, int]:
        r = self.registers[name]
        word = 0
        names = set()
        for f in r["fields"]:
            names.add(f["name"])
            lo, hi = f["bits"]
            lanes = f.get("lanes", 1)
            width = (hi - lo + 1) // lanes
            v = values.get(f["name"], [0] * lanes if lanes > 1 else 0)
            vs = list(v) if lanes > 1 else [v]
            if len(vs) != lanes:
                raise RegisterError(f"{name}.{f['name']}: {lanes} lanes, {len(vs)} values")
            for i, x in enumerate(vs):
                word |= _field_bits(f, x, f"{name}.{f['name']}") << (lo + i * width)
        unknown = set(values) - names
        if unknown:
            raise RegisterError(f"{name} has no fields {sorted(unknown)}")
        return self.address(name, index), word
