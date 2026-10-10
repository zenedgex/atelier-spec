"""Block contracts (atelier-block/1.0): load, migrate from 0.x, and check without the RTL.

    from atelier_spec import block
    spec = block.load("acme_vec.block.yaml")      # JSON or YAML; 0.x files are migrated on load
    problems = block.check(spec)                    # [] when the contract is consistent

This is the RTL-free part of level 0: the contract against the standard and against itself.
atelier-blocks adds the RTL part (every port of the top module in exactly one group, parameters
present), with the SystemVerilog front end.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path


from atelier_spec import evidence, expr, ops
from atelier_spec.protocols import (CORE_CLASSES, MIGRATE_CLASS, PROTOCOLS, ROLES, SLOT_CLASSES,
                                    TENSOR_KINDS)

STANDARD = "atelier-block/1.0"
SCHEMA = Path(__file__).resolve().parent / "data" / "schemas" / "block-1.0.schema.json"

# 0.x capability ops -> op-set names
MIGRATE_OP = {"matmul_tile": "matmul", "elt_add": "add", "elt_mul": "mul", "elt_max": "max",
              "pool_max": "maxpool", "map_lut": "lut", "im2col_load": "im2col", "row_norm": "rmsnorm",
              "row_softmax": "softmax"}


@dataclass(frozen=True)
class Problem:
    level: str          # "error" or "warning"
    where: str          # a path in the contract, e.g. "programming.commands[vmul].fields"
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.where}: {self.message}"


def errors(problems: list[Problem]) -> list[Problem]:
    return [p for p in problems if p.level == "error"]


# ------------------------------------------------------------------------------------------ load
def read(path: str | Path) -> dict:
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".json":
        return json.loads(text)
    import yaml                                 # only for YAML files: the codecs run without it
    return yaml.safe_load(text)


def load(path: str | Path) -> dict:
    """Read a contract and migrate it to 1.0 if it is older."""
    return migrate(read(path))


def migrate(spec: dict) -> dict:
    """A 0.x contract as 1.0. A 1.x contract is returned unchanged (a copy)."""
    s = copy.deepcopy(spec)
    std = str(s.get("standard", ""))
    if not re.fullmatch(r"atelier-block/0\.\d+", std):
        return s
    s["standard"] = STANDARD
    if s.get("slot_class") in MIGRATE_CLASS:
        new, implied = MIGRATE_CLASS[s["slot_class"]]
        s["slot_class"] = new
        for k, v in implied.items():
            s.setdefault(k, v)
    cls = s.get("slot_class", "")
    caps = s.get("capabilities")
    if isinstance(caps, list):
        keep = []
        for c in caps:
            op = c.get("op") if isinstance(c, dict) else None
            if op in MIGRATE_OP:
                c["op"] = MIGRATE_OP[op]
            if cls.startswith("memory.") and op == "store_rows":
                # 0.x described a memory's row access as a capability; 1.0 has a memory contract
                s.setdefault("memory", {}).update(access="rows", data=c.get("data"))
                continue
            if cls in CORE_CLASSES and op and not ops.known(op):
                # 0.x named a core's ISA as a capability; 1.0 has an ISA binding. The toolchain and the
                # boot address are not in 0.x files: the check reports them until they are added.
                s.setdefault("programming", {"model": "isa", "isa": op})
                s.setdefault("core", {})["isa"] = op
                continue
            keep.append(c)
        s["capabilities"] = keep
    s.setdefault("programming", {"model": "none"})
    return s


# ----------------------------------------------------------------------------------------- check
@cache
def _validator():
    import jsonschema
    return jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text()))


def check(spec: dict) -> list[Problem]:
    """Problems with a contract, without its RTL. Errors make it non-conformant at L0."""
    out: list[Problem] = []
    for e in sorted(_validator().iter_errors(spec), key=lambda e: list(e.path)):
        where = ".".join(str(x) for x in e.path) or "(top)"
        out.append(Problem("error", where, e.message))
    if out:
        return out                                   # the rest assumes the shape is right
    params = _param_names(spec)
    out += _check_class(spec)
    out += _check_groups(spec)
    out += _check_evidence(spec)
    out += _check_capabilities(spec)
    out += _check_programming(spec, params)
    return out


def _param_names(spec: dict) -> set[str]:
    p = spec.get("parameters", {})
    return set(p) if isinstance(p, (dict, list)) else set()


def _check_class(spec: dict) -> list[Problem]:
    out = []
    cls = spec["slot_class"]
    if cls not in SLOT_CLASSES:
        out.append(Problem("error", "slot_class", f"{cls!r} is not a slot class of {STANDARD}"))
    if cls == "compute.tensor" and spec.get("kind") not in TENSOR_KINDS:
        out.append(Problem("error", "kind", f"a compute.tensor block needs kind, one of {sorted(TENSOR_KINDS)}"))
    if cls in CORE_CLASSES and "boot_addr" not in spec.get("core", {}):
        out.append(Problem("error", "core.boot_addr", f"a {cls} block needs a core contract with its boot address "
                                                      f"(a parameter or a strap port)"))
    return out


def _roles(group: dict) -> dict[str, str] | None:
    ports = group.get("ports")
    if isinstance(ports, list):
        return {r: r for r in ports}
    if isinstance(ports, dict):
        return dict(ports)
    return None                                       # prefix/suffix groups resolve against the RTL


def _check_groups(spec: dict) -> list[Problem]:
    out, seen = [], {}
    names = [g["name"] for g in spec["port_groups"]]
    for n in {n for n in names if names.count(n) > 1}:
        out.append(Problem("error", f"port_groups[{n}]", "group name used twice"))
    for g in spec["port_groups"]:
        where = f"port_groups[{g['name']}]"
        proto = PROTOCOLS.get(g["protocol"])
        if proto is None:
            out.append(Problem("error", where, f"unknown protocol {g['protocol']!r}"))
            continue
        req, opt = proto
        if "role" in g and g["role"] not in ROLES:
            out.append(Problem("error", where, f"role {g['role']!r}"))
        roles = _roles(g)
        if roles is None:
            if "prefix" not in g and "suffix" not in g and req:
                out.append(Problem("error", where, "needs ports, or a prefix/suffix to find them in the RTL"))
            continue
        if missing := req - set(roles):
            out.append(Problem("error", where, f"{g['protocol']}: missing roles {sorted(missing)}"))
        if extra := set(roles) - req - opt:
            out.append(Problem("error", where, f"{g['protocol']}: roles {sorted(extra)} are not in the "
                                               f"protocol (list such ports as fields)"))
        for port in [*roles.values(), *g.get("fields", [])]:
            if port in seen:
                out.append(Problem("error", where, f"port {port} is also in group {seen[port]}"))
            else:
                seen[port] = g["name"]
    return out


def _check_evidence(spec: dict) -> list[Problem]:
    out = []
    for section in ("timing", "cost"):
        for key, v in evidence.leaves(spec.get(section, {})):
            if v.get("evidence") not in evidence.LABELS:
                out.append(Problem("error", f"{section}.{key}",
                                   f"evidence {v.get('evidence')!r} is not a label of {evidence.STANDARD}"))
    prog = spec.get("programming", {})
    for c in prog.get("commands", []):
        if "cycles" in c and c.get("evidence") not in evidence.LABELS:
            out.append(Problem("error", f"programming.commands[{c['name']}].evidence",
                               "cycles need an evidence label"))
    for k in prog.get("microkernels", []):
        if "cycles" in k and k.get("evidence") not in evidence.LABELS:
            out.append(Problem("error", f"programming.microkernels[{k['symbol']}].evidence",
                               "cycles need an evidence label"))
    return out


def _extra_ops(spec: dict) -> dict:
    return spec.get("programming", {}).get("ops", {}) or {}


def _op_ok(name: str, extra: dict) -> bool:
    return ops.known(name, extra)


def _check_capabilities(spec: dict) -> list[Problem]:
    out, extra = [], _extra_ops(spec)
    caps = spec.get("capabilities")
    items = caps if isinstance(caps, list) else (caps or {}).get("ops", []) if isinstance(caps, dict) else []
    for c in items:
        name = c.get("op") if isinstance(c, dict) else c
        if isinstance(name, str) and not _op_ok(name, extra):
            out.append(Problem("error", "capabilities", f"op {name!r} is not in the op set or the block's extension"))
    return out


def _check_programming(spec: dict, params: set[str]) -> list[Problem]:
    prog = spec.get("programming", {"model": "none"})
    model, cls = prog["model"], spec["slot_class"]
    out: list[Problem] = []
    if cls == "accelerator.op" and model != "command":
        out.append(Problem("error", "programming.model", "an accelerator.op block is programmed by commands"))
    if cls in CORE_CLASSES and model != "isa":
        out.append(Problem("error", "programming.model", f"a {cls} block is programmed by an ISA"))
    extra = prog.get("ops", {}) or {}
    for name in extra:
        if not ops.namespaced(name):
            out.append(Problem("error", f"programming.ops[{name}]",
                               "a block's own op needs a namespace, e.g. acme.fft256"))
        elif "reference" not in extra[name]:
            out.append(Problem("error", f"programming.ops[{name}]", "a new op needs a reference"))
    if "registers" in prog:
        out += _check_registers(prog)
    if model == "command":
        out += _check_commands(prog, params, extra)
    elif model == "isa":
        out += _check_isa(prog, params, extra)
    else:
        for k in ("commands", "intrinsics", "microkernels", "toolchain"):
            if k in prog:
                out.append(Problem("error", f"programming.{k}", f"not used with model {model!r}"))
    return out


def _check_registers(prog: dict) -> list[Problem]:
    """Names once, fields inside the word without overlap, lanes dividing their bits, and no two
    registers (arrays included) at one address."""
    out: list[Problem] = []
    word = prog.get("word", 64)
    regs = prog["registers"]
    names = [r["name"] for r in regs]
    for v in {v for v in names if names.count(v) > 1}:
        out.append(Problem("error", "programming.registers", f"name {v!r} used twice"))
    taken: dict[int, str] = {}
    for r in regs:
        where = f"programming.registers[{r['name']}]"
        used: dict[int, str] = {}
        for f in r["fields"]:
            lo, hi = f["bits"]
            if lo > hi or hi >= word:
                out.append(Problem("error", f"{where}.fields[{f['name']}]", f"bits [{lo}, {hi}] outside a {word}-bit word"))
                continue
            if (hi - lo + 1) % f.get("lanes", 1):
                out.append(Problem("error", f"{where}.fields[{f['name']}]", f"{f['lanes']} lanes do not divide bits [{lo}, {hi}]"))
            if f["kind"] == "flag" and hi != lo:
                out.append(Problem("error", f"{where}.fields[{f['name']}]", "a flag is one bit"))
            for b in range(lo, hi + 1):
                if b in used:
                    out.append(Problem("error", f"{where}.fields[{f['name']}]", f"bit {b} is also in {used[b]}"))
                    break
                used[b] = f["name"]
        for i in range(r.get("count", 1)):
            a = r["addr"] + i * r.get("stride", 1)
            if a in taken:
                out.append(Problem("error", where, f"address {a:#x} is also {taken[a]}'s"))
                break
            taken[a] = r["name"]
    return out


def _check_commands(prog: dict, params: set[str], extra: dict) -> list[Problem]:
    out: list[Problem] = []
    if "word" not in prog:
        return [Problem("error", "programming.word", "a command set needs its descriptor word size")]
    word = prog["word"]
    for k in ("doorbell", "completion"):
        if k not in prog:
            out.append(Problem("warning", f"programming.{k}", "not given: the runtime cannot drive the unit"))
    cmds = prog.get("commands", [])
    if not cmds:
        out.append(Problem("error", "programming.commands", "a command set needs at least one command"))
    for key in ("name", "opcode"):
        vals = [c[key] for c in cmds]
        for v in {v for v in vals if vals.count(v) > 1}:
            out.append(Problem("error", "programming.commands", f"{key} {v!r} used twice"))
    for c in cmds:
        where = f"programming.commands[{c['name']}]"
        fields = {f["name"]: f for f in c["fields"]}
        if len(fields) != len(c["fields"]):
            out.append(Problem("error", f"{where}.fields", "field name used twice"))
        used: dict[int, str] = {}
        for f in c["fields"]:
            lo, hi = f["bits"]
            if lo > hi:
                out.append(Problem("error", f"{where}.fields[{f['name']}]", f"bits [{lo}, {hi}]: low above high"))
                continue
            if hi >= word:
                out.append(Problem("error", f"{where}.fields[{f['name']}]",
                                   f"bits [{lo}, {hi}] do not fit a {word}-bit word"))
            for b in range(lo, hi + 1):
                if b in used:
                    out.append(Problem("error", f"{where}.fields[{f['name']}]", f"bit {b} is also in {used[b]}"))
                    break
                used[b] = f["name"]
        opf = [f for f in c["fields"] if f["kind"] == "opcode"]
        if len(opf) > 1:
            out.append(Problem("error", f"{where}.fields", "more than one opcode field"))
        elif opf:
            lo, hi = opf[0]["bits"]
            if c["opcode"] >= 1 << (hi - lo + 1):
                out.append(Problem("error", f"{where}.opcode", f"{c['opcode']:#x} does not fit bits [{lo}, {hi}]"))
        temps: set[str] = set()
        for i, s in enumerate(c["semantics"]):
            if not _op_ok(s["op"], extra):
                out.append(Problem("error", f"{where}.semantics[{i}]",
                                   f"op {s['op']!r} is not in the op set or the block's extension"))
            for name in s.get("in", []):
                if isinstance(name, str) and name not in fields and name not in temps:
                    out.append(Problem("error", f"{where}.semantics[{i}]",
                                       f"input {name!r} is neither a field nor an earlier result"))
            if "out" in s:
                temps.add(s["out"])
            for a, v in (s.get("attrs") or {}).items():
                if isinstance(v, str) and v in fields:
                    continue                                 # an attribute taken from a field
        env_names = set(fields) | params
        for k in ("constraints", "cycles"):
            texts = c.get(k, [])
            texts = texts if isinstance(texts, list) else [texts]
            for t in texts:
                if isinstance(t, (int, float)):
                    continue
                try:
                    unknown = expr.names(t) - env_names
                except expr.ExprError as e:
                    out.append(Problem("error", f"{where}.{k}", str(e)))
                    continue
                if unknown:
                    out.append(Problem("error", f"{where}.{k}",
                                       f"{t!r}: {sorted(unknown)} are neither fields nor parameters"))
    return out


def _check_isa(prog: dict, params: set[str], extra: dict) -> list[Problem]:
    out: list[Problem] = []
    for k in ("isa", "toolchain"):
        if k not in prog:
            out.append(Problem("error", f"programming.{k}", "an ISA binding needs it"))
    if "iss" not in prog.get("toolchain", {}):
        out.append(Problem("warning", "programming.toolchain.iss",
                           "no ISS: kernels can only be checked on the RTL"))
    if not prog.get("intrinsics") and not prog.get("microkernels"):
        out.append(Problem("warning", "programming", "no intrinsics or micro-kernels: the compiler can only "
                                                     "emit portable C for this core"))
    for i, it in enumerate(prog.get("intrinsics", [])):
        if not _op_ok(it["op"], extra):
            out.append(Problem("error", f"programming.intrinsics[{i}]", f"op {it['op']!r} is not in the op set"))
    for k in prog.get("microkernels", []):
        where = f"programming.microkernels[{k['symbol']}]"
        if not _op_ok(k["op"], extra):
            out.append(Problem("error", where, f"op {k['op']!r} is not in the op set"))
        if isinstance(k.get("cycles"), str):
            known = params | set(k.get("tile", {})) | set(re.findall(r"\b[A-Za-z_]\w*\b", k["signature"]))
            try:
                unknown = expr.names(k["cycles"]) - known
            except expr.ExprError as e:
                out.append(Problem("error", f"{where}.cycles", str(e)))
                continue
            if unknown:
                out.append(Problem("error", f"{where}.cycles", f"{sorted(unknown)} are not arguments or parameters"))
    return out
