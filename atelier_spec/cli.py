"""atelier-spec: check block contracts and machine descriptions.

    atelier-spec block check FILE...            # 0.x files are migrated first
    atelier-spec block migrate FILE [-o OUT]    # write the 1.0 form
    atelier-spec machine check FILE... [--blocks DIR]...
    atelier-spec versions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atelier_spec import STANDARDS, block, machine


def _report(name: str, problems) -> bool:
    errs = block.errors(problems)
    print(f"{'ok  ' if not errs else 'FAIL'}  {name}")
    for p in problems:
        print(f"      {p}")
    return not errs


def _library(dirs: list[str]):
    """name -> contract, from block files (*.json, *.yaml) found under the directories given."""
    found: dict[str, dict] = {}
    for d in dirs:
        for f in sorted(Path(d).rglob("*")):
            if f.suffix in (".json", ".yaml", ".yml") and ("block" in f.name or f.parent.name != Path(d).name):
                try:
                    s = block.load(f)
                except Exception:          # noqa: BLE001 - not a block file
                    continue
                if isinstance(s, dict) and str(s.get("standard", "")).startswith("atelier-block/"):
                    found[s["name"]] = s
    return lambda name: found.get(name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="atelier-spec")
    sub = ap.add_subparsers(dest="what", required=True)
    b = sub.add_parser("block").add_subparsers(dest="cmd", required=True)
    bc = b.add_parser("check")
    bc.add_argument("files", nargs="+")
    bm = b.add_parser("migrate")
    bm.add_argument("file")
    bm.add_argument("-o", "--out")
    m = sub.add_parser("machine").add_subparsers(dest="cmd", required=True)
    mc = m.add_parser("check")
    mc.add_argument("files", nargs="+")
    mc.add_argument("--blocks", action="append", default=[], help="a directory of block contracts")
    sub.add_parser("versions")
    a = ap.parse_args(argv)

    if a.what == "versions":
        print("\n".join(STANDARDS))
        return 0
    if a.what == "block" and a.cmd == "migrate":
        text = json.dumps(block.load(a.file), indent=1) + "\n"
        Path(a.out).write_text(text) if a.out else sys.stdout.write(text)
        return 0
    ok = True
    if a.what == "block":
        for f in a.files:
            ok &= _report(f, block.check(block.load(f)))
    else:
        lookup = _library(a.blocks) if a.blocks else None
        for f in a.files:
            ok &= _report(f, machine.check(machine.load(f), lookup))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
