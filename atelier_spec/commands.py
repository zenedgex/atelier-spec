"""Command words from a block's command set (programming.commands), by its bit-field table alone.

    cs = commands.CommandSet(spec["programming"])
    word = cs.encode("compute", a_addr=3, w_addr=0, acc_addr=8, length=16, accumulate=1)
    cs.decode(word)          # ("compute", {"a_addr": 3, ...})

No block is named here: the compiler's Target/Cmd encoder (C++) is generated from the same table
and checked against this one.
"""

from __future__ import annotations


class CommandError(ValueError):
    pass


class CommandSet:
    def __init__(self, programming: dict):
        if programming.get("model") != "command":
            raise CommandError("not a command set (programming.model is not 'command')")
        self.word = programming["word"]
        self.commands = {c["name"]: c for c in programming["commands"]}

    @staticmethod
    def _opfield(c: dict) -> dict:
        return next(f for f in c["fields"] if f["kind"] == "opcode")

    def encode(self, name: str, **values) -> int:
        c = self.commands[name]
        word = 0
        names = set()
        for f in c["fields"]:
            lo, hi = f["bits"]
            width = hi - lo + 1
            names.add(f["name"])
            if f["kind"] == "opcode":
                v = c["opcode"]
            elif f["kind"] == "reserved":
                v = 0
            else:
                v = values.get(f["name"], 0)
                if f["kind"] == "enum" and isinstance(v, str):
                    v = f["values"][v]
                v = int(v)
            if not 0 <= v < 1 << width:
                raise CommandError(f"{name}.{f['name']} = {v} does not fit bits [{lo}, {hi}]")
            word |= v << lo
        unknown = set(values) - names
        if unknown:
            raise CommandError(f"{name} has no fields {sorted(unknown)}")
        return word

    def decode(self, word: int) -> tuple[str, dict]:
        for c in self.commands.values():
            lo, hi = self._opfield(c)["bits"]
            if word >> lo & ((1 << (hi - lo + 1)) - 1) == c["opcode"]:
                out = {}
                for f in c["fields"]:
                    if f["kind"] in ("opcode", "reserved"):
                        continue
                    flo, fhi = f["bits"]
                    out[f["name"]] = word >> flo & ((1 << (fhi - flo + 1)) - 1)
                return c["name"], out
        raise CommandError(f"no command has the opcode of {word:#x}")
