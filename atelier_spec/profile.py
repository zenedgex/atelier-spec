"""Profiles (atelier-profile/0.1): what one run of a network on an accelerator measured, layer by layer.

    from atelier_spec import machine, profile
    p = profile.load("run.yaml")
    problems = profile.check(p, m=machine.load("gemmini.yaml"))   # m optional

A profile is evidence, not a model: cycles and counters as the accelerator's simulator, its silicon or
the customer's tools reported them, with the run's evidence label (rtl-sim, silicon, ...). Atelier's
own soc_top counters are written in the same form, so a design whose truth we know checks the analysis
of designs whose truth we don't (atelier-platform spec 20).

Per layer: `cycles` (required), `start`/`end` when the run's timeline is known (end - start = cycles),
`activity` per unit (`busy`: doing work; `stall`: has work but waits; the rest is idle, so busy + stall
<= cycles), `bytes` per memory or per DMA route (`from>to`), and any vendor `counters` as they come.
Layers may overlap in time (pipelined frames): a total is given in `total`, not summed.

With a machine, every unit, memory and route a profile names must be in it.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from atelier_spec import evidence
from atelier_spec.block import Problem
from atelier_spec.machine import load  # noqa: F401  (one loader for YAML and JSON)

STANDARD = "atelier-profile/0.1"
SCHEMA = Path(__file__).resolve().parent / "data" / "schemas" / "profile-0.1.schema.json"


@cache
def _validator():
    import jsonschema
    return jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text()))


def check(p: dict, m: dict | None = None) -> list[Problem]:
    out: list[Problem] = []
    for e in sorted(_validator().iter_errors(p), key=lambda e: list(e.path)):
        out.append(Problem("error", ".".join(str(x) for x in e.path) or "(top)", e.message))
    if out:
        return out
    run = p["run"]
    if run["evidence"] not in evidence.LABELS:
        out.append(Problem("error", "run.evidence", f"{run['evidence']!r} is not a label of {evidence.STANDARD}"))
    if m is not None and run["machine"]["name"] != m["name"]:
        out.append(Problem("error", "run.machine.name", f"{run['machine']['name']!r}, the machine is {m['name']!r}"))
    units = set(m["units"]) if m else None
    places = ({x["name"] for x in m.get("memory", [])} | {s["name"] for s in m.get("movement", {}).get("streams", [])}
              if m else None)
    seen = set()
    for i, layer in enumerate(p["layers"]):
        where = f"layers.{layer['name']}"
        if layer["name"] in seen:
            out.append(Problem("error", where, "layer name used twice"))
        seen.add(layer["name"])
        c = layer["cycles"]
        if "start" in layer and "end" in layer:
            if layer["end"] < layer["start"]:
                out.append(Problem("error", where, f"ends at {layer['end']}, before its start {layer['start']}"))
            elif layer["end"] - layer["start"] != c:
                out.append(Problem("error", where, f"end - start = {layer['end'] - layer['start']}, cycles = {c}"))
        for u, a in layer.get("activity", {}).items():
            if a.get("busy", 0) + a.get("stall", 0) > c:
                out.append(Problem("error", f"{where}.activity.{u}",
                                   f"busy {a.get('busy', 0)} + stall {a.get('stall', 0)} > the layer's {c} cycles"))
            if "start" in a and "end" in a and a["end"] < a["start"]:
                out.append(Problem("error", f"{where}.activity.{u}", "ends before it starts"))
        if units is not None:
            for u in list(layer.get("units", [])) + list(layer.get("activity", {})):
                if u.split(".")[0] not in units:
                    out.append(Problem("error", where, f"unit {u!r} is not in machine {m['name']!r}"))
            for k in layer.get("bytes", {}):
                for x in k.split(">"):
                    if x not in places:
                        out.append(Problem("error", f"{where}.bytes", f"{x!r} is not a memory or stream of "
                                                                      f"machine {m['name']!r}"))
    return out
