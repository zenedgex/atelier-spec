"""Program images (image/1, standards/image.md): records, DMA descriptors and the container.

    from atelier_spec import image, machine
    b = image.Binding(m)                                    # queue and space numbers from the machine
    recs = [image.Wait(b.queue("cim"), token=1, value=1),
            image.Dma(b.queue("dma", 0), image.Desc.copy(b.space("dram"), 0, b.space("actmem"), 0,
                                                         elem=8, src=[(64, 8)], dst=[(64, 8)])),
            image.Signal(b.queue("dma", 0), token=1), image.End()]
    words = image.encode(recs)
    image.decode(words) == recs
    blob = image.pack([words], image.machine_hash(m))

The compiler's C++ serializer is checked against this one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

MAGIC = 0x31495441          # "ATI1"
VERSION = 1
END, CMD, CFG, DMA, LAUNCH, WAIT, SIGNAL = range(7)
KINDS = {END: "END", CMD: "CMD", CFG: "CFG", DMA: "DMA", LAUNCH: "LAUNCH", WAIT: "WAIT", SIGNAL: "SIGNAL"}
MULTI = 1
COPY, FILL, INLINE, GATHER = range(4)
NONE = 0xFFFF_FFFF          # the source space of fill and inline
MAX_WORDS = 0xFFFF
M32 = 0xFFFF_FFFF


class ImageError(ValueError):
    pass


def _u64(v: int) -> list[int]:
    v &= (1 << 64) - 1
    return [v & M32, v >> 32]


def _s64(lo: int, hi: int) -> int:
    v = lo | hi << 32
    return v - (1 << 64) if v >> 63 else v


# ------------------------------------------------------------------ the DMA descriptor

@dataclass(frozen=True)
class Desc:
    """A copy between two spaces. dims: [(count, stride in bytes)], innermost first."""
    src_space: int
    dst_space: int
    elem: int
    src_base: int = 0
    dst_base: int = 0
    src: tuple = ()
    dst: tuple = ()
    mode: int = COPY
    value: int = 0
    data: bytes = b""

    @staticmethod
    def copy(src_space, src_base, dst_space, dst_base, elem, src, dst) -> "Desc":
        return Desc(src_space, dst_space, elem, src_base, dst_base, tuple(map(tuple, src)), tuple(map(tuple, dst)))

    @staticmethod
    def fill(dst_space, dst_base, elem, dst, value) -> "Desc":
        return Desc(NONE, dst_space, elem, 0, dst_base, (), tuple(map(tuple, dst)), FILL, value)

    @staticmethod
    def inline(dst_space, dst_base, elem, dst, data: bytes) -> "Desc":
        return Desc(NONE, dst_space, elem, 0, dst_base, (), tuple(map(tuple, dst)), INLINE, data=bytes(data))

    @property
    def elements(self) -> int:
        return prod(c for c, _ in self.dst)

    def check(self):
        if self.mode == GATHER:
            raise ImageError("gather is reserved")
        if not 1 <= self.elem <= 0xFFFF:
            raise ImageError(f"element bytes {self.elem}")
        if not 1 <= len(self.dst) <= 15 or len(self.src) > 15:
            raise ImageError("1 to 15 dimensions a side")
        if self.mode == COPY:
            if not self.src:
                raise ImageError("a copy needs source dimensions")
            if prod(c for c, _ in self.src) != self.elements:
                raise ImageError(f"{prod(c for c, _ in self.src)} source elements, {self.elements} destination")
        elif self.src:
            raise ImageError("fill and inline take no source dimensions")
        if self.mode == INLINE and len(self.data) != self.elements * self.elem:
            raise ImageError(f"inline data is {len(self.data)} bytes, the destination takes {self.elements * self.elem}")

    def words(self) -> list[int]:
        self.check()
        w = [self.src_space & M32, self.dst_space & M32,
             self.elem | self.mode << 16 | len(self.src) << 20 | len(self.dst) << 24,
             *_u64(self.src_base), *_u64(self.dst_base)]
        for c, s in (*self.src, *self.dst):
            w += [c & M32, *_u64(s)]
        if self.mode == FILL:
            w += _u64(self.value)
        elif self.mode == INLINE:
            d = self.data + bytes(-len(self.data) % 4)
            w += [int.from_bytes(d[i:i + 4], "little") for i in range(0, len(d), 4)]
        return w

    @staticmethod
    def from_words(w: list[int]) -> "Desc":
        elem, mode, ns, nd = w[2] & 0xFFFF, w[2] >> 16 & 0xF, w[2] >> 20 & 0xF, w[2] >> 24 & 0xF
        i, dims = 7, []
        for _ in range(ns + nd):
            dims.append((w[i], _s64(w[i + 1], w[i + 2])))
            i += 3
        src, dst = tuple(dims[:ns]), tuple(dims[ns:])
        d = Desc(w[0], w[1], elem, _s64(w[3], w[4]), _s64(w[5], w[6]), src, dst, mode)
        if mode == FILL:
            d = Desc(**{**d.__dict__, "value": w[i] | w[i + 1] << 32})
        elif mode == INLINE:
            n = prod(c for c, _ in dst) * elem
            data = b"".join(x.to_bytes(4, "little") for x in w[i:])[:n]
            d = Desc(**{**d.__dict__, "data": data})
        d.check()
        return d

    def offsets(self, side: str) -> list[int]:
        """The byte offsets of each element on one side, in iteration order (innermost fastest)."""
        base, dims = (self.src_base, self.src) if side == "src" else (self.dst_base, self.dst)
        offs = [base]
        for c, s in dims:
            offs = [o + k * s for k in range(c) for o in offs]
        return offs

    def footprint(self) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
        """(reads, writes): (space, first byte, end byte) ranges, merged."""
        def ranges(space, side):
            out = []
            for o in sorted(self.offsets(side)):
                if out and o <= out[-1][1]:
                    out[-1][1] = max(out[-1][1], o + self.elem)
                else:
                    out.append([o, o + self.elem])
            return [(space, a, b) for a, b in out]
        return (ranges(self.src_space, "src") if self.mode == COPY else []), ranges(self.dst_space, "dst")


# ------------------------------------------------------------------ records

@dataclass(frozen=True)
class Record:
    queues: tuple = ()

    kind = -1

    def payload(self) -> list[int]:
        return []


def _q(q) -> tuple:
    return tuple(q) if isinstance(q, (list, tuple)) else (q,)


@dataclass(frozen=True)
class Cmd(Record):
    words: tuple = ()
    kind = CMD

    def __init__(self, q, words):
        object.__setattr__(self, "queues", _q(q))
        object.__setattr__(self, "words", tuple(words))

    def payload(self):
        return list(self.words)


@dataclass(frozen=True)
class Cfg(Record):
    addr: int = 0
    data: tuple = ()
    kind = CFG

    def __init__(self, q, addr, data):
        object.__setattr__(self, "queues", _q(q))
        object.__setattr__(self, "addr", addr)
        object.__setattr__(self, "data", tuple(data))

    def payload(self):
        return [self.addr, *self.data]


@dataclass(frozen=True)
class Dma(Record):
    desc: Desc = field(default=None)
    kind = DMA

    def __init__(self, q, desc: Desc):
        object.__setattr__(self, "queues", _q(q))
        object.__setattr__(self, "desc", desc)

    def payload(self):
        return self.desc.words()


@dataclass(frozen=True)
class Launch(Record):
    kernel: int = 0
    args: tuple = ()
    kind = LAUNCH

    def __init__(self, q, kernel, args=()):
        object.__setattr__(self, "queues", _q(q))
        object.__setattr__(self, "kernel", kernel)
        object.__setattr__(self, "args", tuple(args))

    def payload(self):
        return [self.kernel, *self.args]


@dataclass(frozen=True)
class Wait(Record):
    token: int = 0
    value: int = 0
    kind = WAIT

    def __init__(self, q, token, value):
        object.__setattr__(self, "queues", _q(q))
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "value", value & M32)

    def payload(self):
        return [self.token, self.value]


@dataclass(frozen=True)
class Signal(Record):
    token: int = 0
    kind = SIGNAL

    def __init__(self, q, token):
        object.__setattr__(self, "queues", _q(q))
        object.__setattr__(self, "token", token)

    def payload(self):
        return [self.token]


@dataclass(frozen=True)
class End(Record):
    kind = END

    def __init__(self):
        object.__setattr__(self, "queues", ())


def encode(records: list[Record]) -> list[int]:
    out = []
    for r in records:
        if isinstance(r, End):
            out.append(END)
            continue
        if not r.queues:
            raise ImageError(f"{KINDS[r.kind]} names no queue")
        qw = [r.queues[0]] if len(r.queues) == 1 else [len(r.queues), *r.queues]
        p = r.payload()
        n = len(qw) + len(p)
        if n > MAX_WORDS:
            raise ImageError(f"a {KINDS[r.kind]} record of {n} words: split it (split_inline does for inline data)")
        out += [r.kind | (MULTI if len(r.queues) > 1 else 0) << 8 | n << 16, *[x & M32 for x in qw], *[x & M32 for x in p]]
    return out


def decode(words: list[int]) -> list[Record]:
    out, i = [], 0
    while i < len(words):
        h = words[i]
        kind, flags, n = h & 0xFF, h >> 8 & 0xFF, h >> 16
        if kind == END:
            out.append(End())
            i += 1
            continue
        body = words[i + 1:i + 1 + n]
        if len(body) != n:
            raise ImageError(f"record at word {i} runs past the stream")
        if flags & MULTI:
            qs, p = tuple(body[1:1 + body[0]]), body[1 + body[0]:]
        else:
            qs, p = (body[0],), body[1:]
        if kind == CMD:
            r = Cmd(qs, p)
        elif kind == CFG:
            r = Cfg(qs, p[0], p[1:])
        elif kind == DMA:
            r = Dma(qs, Desc.from_words(p))
        elif kind == LAUNCH:
            r = Launch(qs, p[0], p[1:])
        elif kind == WAIT:
            r = Wait(qs, p[0], p[1])
        elif kind == SIGNAL:
            r = Signal(qs, p[0])
        else:
            raise ImageError(f"unknown record kind {kind} at word {i}")
        out.append(r)
        i += 1 + n
    return out


def split_inline(q, space: int, base: int, data: bytes, elem: int = 8, max_words: int = MAX_WORDS) -> list[Dma]:
    """Inline data of any size as DMA records that each fit a record."""
    per = ((max_words - 16) * 4) // elem * elem
    return [Dma(q, Desc.inline(space, base + o, elem, [(len(c) // elem, elem)], c))
            for o in range(0, len(data), per) for c in [data[o:o + per]]]


# ------------------------------------------------------------------ the container and the machine

def _fnv(text: str) -> int:
    h = 0x811C9DC5
    for b in text.encode():
        h = (h ^ b) * 0x01000193 & M32
    return h


def machine_hash(m: dict) -> int:
    """FNV-1a 32 over the queue and space numbering: an image binds to it."""
    from atelier_spec import machine
    return _fnv(";".join(f"{q.unit}:{q.kind}:{q.channel}" for q in machine.queues(m)) + "|"
                + ";".join(machine.spaces(m)))


class Binding:
    """Queue and space numbers by name, from a machine description."""

    def __init__(self, m: dict):
        from atelier_spec import machine
        self.queues = machine.queues(m)
        self.spaces = machine.spaces(m)
        self._q = {(q.unit, q.channel): i for i, q in enumerate(self.queues)}
        self._s = {s: i for i, s in enumerate(self.spaces)}
        self.hash = machine_hash(m)

    def queue(self, unit: str, channel: int = 0) -> int:
        try:
            return self._q[unit, channel]
        except KeyError:
            raise ImageError(f"no queue for {unit} channel {channel}") from None

    def space(self, name: str) -> int:
        try:
            return self._s[name]
        except KeyError:
            raise ImageError(f"no space {name!r}") from None


def pack(streams: list[list[int]], mhash: int) -> bytes:
    head = [MAGIC, VERSION, mhash & M32, len(streams)]
    off = (4 + 3 * len(streams)) * 4
    for s in streams:
        head += [*_u64(off), len(s)]
        off += 4 * len(s)
    return b"".join(w.to_bytes(4, "little") for w in head + [x for s in streams for x in s])


def unpack(blob: bytes, mhash: int | None = None) -> list[list[int]]:
    w = [int.from_bytes(blob[i:i + 4], "little") for i in range(0, len(blob) - len(blob) % 4, 4)]
    if len(w) < 4 or w[0] != MAGIC:
        raise ImageError("not an image/1 (magic)")
    if w[1] != VERSION:
        raise ImageError(f"image version {w[1]}, this reads {VERSION}")
    if mhash is not None and w[2] != mhash & M32:
        raise ImageError(f"built for another machine (hash {w[2]:#010x}, this machine {mhash & M32:#010x})")
    out = []
    for k in range(w[3]):
        lo, hi, n = w[4 + 3 * k:7 + 3 * k]
        o = (lo | hi << 32) // 4
        out.append(w[o:o + n])
    return out
