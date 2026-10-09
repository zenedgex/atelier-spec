"""Machine descriptions (atelier-machine/0.1): load and check a whole chip.

    from atelier_spec import machine
    m = machine.load("npu.yaml")
    problems = machine.check(m, blocks=lookup)     # lookup(name) -> block contract or None

`blocks` is optional. Without it, block names are not resolved and the checks that need a block's
contract (its slot class) are skipped. Nothing here limits a count: units, clusters, memories,
masters, regions and clock domains are lists of any length.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Callable

import yaml

from atelier_spec.block import Problem
from atelier_spec.protocols import CORE_CLASSES, PROTOCOLS, SLOT_CLASSES

STANDARD = "atelier-machine/0.1"
SCHEMA = Path(__file__).resolve().parent / "data" / "schemas" / "machine-0.1.schema.json"

_UNITS = {"": 1, "B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30, "TiB": 1 << 40,
          "KB": 10**3, "MB": 10**6, "GB": 10**9}


def size(v) -> int:
    """Bytes from 4096, '128KiB', '2MiB', '64B'."""
    if isinstance(v, int):
        return v
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]i?B|B)?\s*", str(v))
    if not m:
        raise ValueError(f"size {v!r}")
    n = float(m.group(1)) * _UNITS[m.group(2) or ""]
    if n != int(n):
        raise ValueError(f"size {v!r} is not a whole number of bytes")
    return int(n)


def load(path: str | Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.suffix == ".json" else yaml.safe_load(p.read_text())


@cache
def _validator():
    import jsonschema
    return jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text()))


def check(m: dict, blocks: Callable[[str], dict | None] | None = None) -> list[Problem]:
    out: list[Problem] = []
    for e in sorted(_validator().iter_errors(m), key=lambda e: list(e.path)):
        out.append(Problem("error", ".".join(str(x) for x in e.path) or "(top)", e.message))
    if out:
        return out
    ids = _Ids(m)
    out += ids.problems
    out += _check_units(m, blocks)
    out += _check_clusters(m, ids)
    out += _check_memory(m, ids)
    out += _check_movement(m, ids)
    out += _check_interconnect(m, ids)
    out += _check_control(m, blocks)
    out += _check_address_map(m)
    out += _check_clocks(m, ids)
    out += _check_space(m)
    return out


class _Ids:
    """Every id a machine defines, and what it is. Ids are unique across units, clusters, memories."""

    def __init__(self, m: dict):
        self.problems: list[Problem] = []
        self.kind: dict[str, str] = {}
        for kind, names in (("unit", m.get("units", {})), ("cluster", m.get("clusters", {})),
                            ("memory", [x["name"] for x in m.get("memory", [])])):
            for n in names:
                if n in self.kind:
                    self.problems.append(Problem("error", f"{kind}:{n}", f"id also used by a {self.kind[n]}"))
                self.kind.setdefault(n, kind)

    def resolves(self, ref: str) -> bool:
        """`l2`, `engine.*.regs`, `sm_cluster.*`, `engine.3`: the first segment must be an id."""
        return ref.split(".")[0] in self.kind


def _unit_class(u: dict, blocks) -> str | None:
    if "class" in u:
        return u["class"]
    if blocks and (b := blocks(u["block"])):
        return b.get("slot_class")
    return None


def _check_units(m: dict, blocks) -> list[Problem]:
    out = []
    for uid, u in m["units"].items():
        where = f"units.{uid}"
        cls = u.get("class")
        if cls and cls not in SLOT_CLASSES:
            out.append(Problem("error", where, f"class {cls!r} is not a slot class"))
        if blocks and "block" in u:
            b = blocks(u["block"])
            if b is None:
                out.append(Problem("warning", where, f"block {u['block']!r} not found in the libraries given"))
            elif cls and b.get("slot_class") != cls:
                out.append(Problem("error", where, f"block {u['block']!r} is a {b.get('slot_class')}, "
                                                   f"the unit says {cls}"))
    return out


def _check_clusters(m: dict, ids: _Ids) -> list[Problem]:
    out = []
    for cid, c in m.get("clusters", {}).items():
        for x in c["contains"]:
            if ids.kind.get(x) not in ("unit", "memory"):
                out.append(Problem("error", f"clusters.{cid}.contains", f"{x!r} is not a unit or a memory"))
    return out


def _check_memory(m: dict, ids: _Ids) -> list[Problem]:
    out = []
    in_cluster = {x for c in m.get("clusters", {}).values() for x in c["contains"]}
    for mem in m["memory"]:
        where = f"memory.{mem['name']}"
        per = mem["per"]
        if per == "cluster":
            if mem["name"] not in in_cluster:
                out.append(Problem("error", where, "per: cluster, but no cluster contains it"))
        elif per != "chip" and ids.kind.get(per) not in ("unit", "cluster"):
            out.append(Problem("error", where, f"per: {per!r} is not chip, cluster, or a unit or cluster id"))
        if "size" in mem:
            try:
                n = size(mem["size"])
                if mem.get("banks") and n % mem["banks"]:
                    out.append(Problem("error", where, f"size {mem['size']} does not split into {mem['banks']} banks"))
            except ValueError as e:
                out.append(Problem("error", where, str(e)))
        elif mem["kind"] != "dram":
            out.append(Problem("error", where, "needs a size"))
        if mem["kind"] == "cache" and not mem.get("line"):
            out.append(Problem("warning", where, "a cache without a line size"))
    levels = sorted({mem["level"] for mem in m["memory"]})
    if levels and levels != list(range(levels[0], levels[-1] + 1)):
        out.append(Problem("warning", "memory", f"levels {levels} have a gap"))
    return out


def _check_movement(m: dict, ids: _Ids) -> list[Problem]:
    out = []
    for i, d in enumerate(m.get("movement", {}).get("dma", [])):
        where = f"movement.dma[{i}]"
        if ids.kind.get(d.get("engine")) != "unit":
            out.append(Problem("error", where, f"engine {d.get('engine')!r} is not a unit"))
        for end in ("from", "to"):
            for x in d.get(end, []) if isinstance(d.get(end), list) else [d.get(end)]:
                if ids.kind.get(x) != "memory":
                    out.append(Problem("error", where, f"{end}: {x!r} is not a memory"))
    for i, b in enumerate(m.get("movement", {}).get("bridges", [])):
        for end in ("from_bus", "to_bus"):
            if b.get(end) not in PROTOCOLS:
                out.append(Problem("error", f"movement.bridges[{i}]", f"{end}: {b.get(end)!r} is not a protocol"))
    return out


def _check_interconnect(m: dict, ids: _Ids) -> list[Problem]:
    out = []
    ic = m.get("interconnect")
    if not ic:
        return out
    if ic.get("protocol") and ic["protocol"] not in PROTOCOLS:
        out.append(Problem("error", "interconnect.protocol", f"{ic['protocol']!r} is not a protocol"))
    for side in ("masters", "slaves"):
        for ref in ic.get(side, []):
            if not ids.resolves(ref):
                out.append(Problem("error", f"interconnect.{side}", f"{ref!r} names nothing in the machine"))
    return out


def _check_control(m: dict, blocks) -> list[Problem]:
    c = m["control"]
    u = m["units"].get(c["core"])
    if u is None:
        return [Problem("error", "control.core", f"{c['core']!r} is not a unit")]
    cls = _unit_class(u, blocks)
    if cls is not None and cls not in CORE_CLASSES:
        return [Problem("error", "control.core", f"{c['core']!r} is a {cls}; a control core is one of "
                                                 f"{sorted(CORE_CLASSES)}")]
    if c["model"] == "application" and c.get("os") == "none":
        return [Problem("warning", "control.os", "an application processor usually runs an OS")]
    return []


def _check_address_map(m: dict) -> list[Problem]:
    out, spans = [], []
    for name, r in m.get("address_map", {}).items():
        try:
            spans.append((r["base"], r["base"] + size(r["size"]), name))
        except ValueError as e:
            out.append(Problem("error", f"address_map.{name}", str(e)))
    spans.sort()
    for (a0, a1, an), (b0, b1, bn) in zip(spans, spans[1:]):
        if b0 < a1:
            out.append(Problem("error", "address_map", f"{an} [{a0:#x}, {a1:#x}) overlaps {bn} at {b0:#x}"))
    return out


def _check_clocks(m: dict, ids: _Ids) -> list[Problem]:
    out, owner = [], {}
    for dom, c in m.get("clocks", {}).items():
        for x in c["units"]:
            if x not in ids.kind:
                out.append(Problem("error", f"clocks.{dom}", f"{x!r} names nothing in the machine"))
            elif x in owner:
                out.append(Problem("error", f"clocks.{dom}", f"{x!r} is also in clock domain {owner[x]}"))
            owner.setdefault(x, dom)
    return out


def _lookup(m: dict, path: str):
    """`units.tensor.params.R`, `memory.l2.size` (memory is a list addressed by name)."""
    cur = m
    for part in path.split("."):
        if isinstance(cur, list):
            cur = next((x for x in cur if isinstance(x, dict) and x.get("name") == part), None)
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
        if cur is None:
            return None
    return cur


def _check_space(m: dict) -> list[Problem]:
    return [Problem("error", f"space.{k}", "names nothing in the machine")
            for k in m.get("space", {}) if _lookup(m, k) is None]
