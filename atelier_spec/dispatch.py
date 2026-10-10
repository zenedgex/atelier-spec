"""The reference dispatcher (image/1, standards/image.md §3, §5): queues, tokens, any timing.

    from atelier_spec import dispatch, image
    mem = dispatch.Memory({b.space("dram"): bytearray(1 << 20), b.space("actmem"): bytearray(1 << 16)})
    r = dispatch.run(image.decode(words), queues=len(b.queues), mem=mem,
                     workers={b.queue("cim"): engine_step}, depth=16, seed=3)
    r.cycles, r.races, r.deadlock

Every queue runs its records in order; only tokens order queues against each other. A worker's
step reads when it starts and writes when it ends, and each step takes its cycles plus a random
delay (`jitter`), so a racy image gives different results for different seeds. A race (two
unordered steps whose footprints overlap, one writing) is found exactly, whatever the timing, by
vector clocks: each queue's clock advances per step, SIGNAL publishes it into the token, WAIT joins
it. DMA records run here (copy, fill, inline); other kinds go to the `workers` the caller gives,
by queue: worker(record, mem) -> Step.
"""

from __future__ import annotations

import heapq
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from atelier_spec import image


class Memory:
    """The spaces by index: bytearrays (addressed) or lists (stream ports, elements in order)."""

    def __init__(self, spaces: dict[int, bytearray | list]):
        self.spaces = spaces

    def read(self, space: int, offsets: list[int], elem: int) -> list[bytes]:
        s = self.spaces[space]
        if isinstance(s, list):                       # a stream in: take the next elements
            out, s[:len(offsets)] = s[:len(offsets)], []
            if len(out) != len(offsets):
                raise image.ImageError(f"stream space {space} ran dry")
            return out
        return [bytes(s[o:o + elem]) for o in offsets]

    def write(self, space: int, offsets: list[int], elem: int, data: list[bytes]):
        s = self.spaces[space]
        if isinstance(s, list):
            s.extend(data)
            return
        for o, d in zip(offsets, data):
            if o < 0 or o + elem > len(s):
                raise image.ImageError(f"space {space}: bytes [{o}, {o + elem}) outside its {len(s)}")
            s[o:o + elem] = d


@dataclass
class Step:
    """What a record does: its cycles, its footprint (space, first, end), and the write it makes at
    its end (given what it read at its start)."""
    cycles: int
    reads: list = field(default_factory=list)
    writes: list = field(default_factory=list)
    start: Callable[[Memory], object] = lambda mem: None
    end: Callable[[Memory, object], None] = lambda mem, got: None


def dma_step(rec: image.Dma, mem: Memory, bytes_per_cycle: int = 8, setup: int = 4) -> Step:
    d = rec.desc
    reads, writes = d.footprint()
    for sp, _, _ in reads + writes:
        if sp not in mem.spaces:
            raise image.ImageError(f"no space {sp} in this memory")
    dst = d.offsets("dst")
    if d.mode == image.COPY:
        src = d.offsets("src")
        start = lambda m: m.read(d.src_space, src, d.elem)
    elif d.mode == image.FILL:
        v = (d.value & ((1 << 8 * d.elem) - 1)).to_bytes(d.elem, "little")
        start = lambda m: [v] * len(dst)
    else:
        start = lambda m: [d.data[i * d.elem:(i + 1) * d.elem] for i in range(len(dst))]
    if isinstance(mem.spaces.get(d.src_space), list):     # stream ends carry no footprint
        reads = []
    if isinstance(mem.spaces.get(d.dst_space), list):
        writes = []
    return Step(setup + -(-len(dst) * d.elem // bytes_per_cycle), reads, writes, start,
                lambda m, got: m.write(d.dst_space, dst, d.elem, got))


@dataclass
class Race:
    space: int
    first: tuple            # (queue, record index in the stream, wrote)
    second: tuple
    bytes: tuple            # the overlap [a, b)

    def __str__(self):
        return (f"space {self.space} bytes [{self.bytes[0]}, {self.bytes[1]}): record {self.first[1]} on queue "
                f"{self.first[0]} ({'write' if self.first[2] else 'read'}) and record {self.second[1]} on queue "
                f"{self.second[0]} ({'write' if self.second[2] else 'read'}) are not ordered by tokens")


@dataclass
class Result:
    cycles: int
    tokens: dict
    races: list
    deadlock: str = ""
    busy: dict = field(default_factory=dict)       # queue -> cycles spent on steps

    @property
    def ok(self) -> bool:
        return not self.races and not self.deadlock


def check_order(records: list[image.Record]) -> list[str]:
    """§5 Progress: a WAIT for token ≥ v comes after at least v SIGNALs of that token in the stream."""
    seen, out = {}, []
    for i, r in enumerate(records):
        if isinstance(r, image.Signal):
            seen[r.token] = seen.get(r.token, 0) + len(r.queues)
        elif isinstance(r, image.Wait) and seen.get(r.token, 0) < r.value:
            out.append(f"record {i}: WAIT token {r.token} >= {r.value}, but only {seen.get(r.token, 0)} SIGNALs "
                       f"come before it")
    return out


def _ge(a: int, b: int) -> bool:
    """a >= b modulo 2^32 (wrap-safe: within half the range)."""
    return ((a - b) & image.M32) < 1 << 31


def run(records: list[image.Record], queues: int, mem: Memory, workers: dict[int, Callable] | None = None,
        depth: int = 16, jitter: int = 0, seed: int = 0, dma: Callable = dma_step, check: bool = True) -> Result:
    """Run a record stream. `workers[q](record, mem) -> Step` for the non-DMA records of queue q."""
    workers = workers or {}
    rng = random.Random(seed)
    fifo = [deque() for _ in range(queues)]
    vc = [[0] * queues for _ in range(queues)]     # vc[q]: what queue q has seen of every queue
    tokens: dict[int, int] = {}
    tclock: dict[tuple[int, int], list[int]] = {}  # (token, value) -> the clock it carries
    log: dict[int, list] = {}                      # space -> [(a, b, queue, epoch, wrote, index)]
    races: list[Race] = []
    busy = {q: 0 for q in range(queues)}
    running: list = []                             # heap of (end time, queue, step, got)
    active = [False] * queues
    now, nxt, ended = 0, 0, False

    logged = [0, 1024]                             # entries in the log; the size that triggers pruning

    def prune():
        """Drop the accesses every other queue already knows happened: they can never race (exact)."""
        known = [min([vc[q][q2] for q in range(queues) if q != q2] or [0]) for q2 in range(queues)]
        n = 0
        for space in log:
            log[space] = [e for e in log[space] if e[3] > known[e[2]]]
            n += len(log[space])
        logged[0], logged[1] = n, max(1024, 2 * n)

    def access(q, idx, space, a, b, wrote):
        mine = vc[q]
        for (a2, b2, q2, ep, w2, i2) in log.setdefault(space, []):
            if q2 != q and (wrote or w2) and a < b2 and a2 < b and ep > mine[q2]:
                if check:
                    races.append(Race(space, (q2, i2, w2), (q, idx, wrote), (max(a, a2), min(b, b2))))
        log[space].append((a, b, q, mine[q], wrote, idx))
        logged[0] += 1
        if logged[0] > logged[1]:
            prune()

    def dispatch_some():
        nonlocal nxt, ended
        while nxt < len(records) and not ended:
            r = records[nxt]
            if isinstance(r, image.End):
                ended = True
                nxt += 1
                break
            for q in r.queues:
                if not 0 <= q < queues:
                    raise image.ImageError(f"record {nxt}: queue {q}, the machine has {queues}")
            if any(len(fifo[q]) >= depth for q in r.queues):
                return                              # a full queue stalls dispatch, in order
            for q in r.queues:
                fifo[q].append((nxt, r))
            nxt += 1

    def start_heads():
        progress = True
        while progress:
            progress = False
            for q in range(queues):
                while not active[q] and fifo[q]:
                    idx, r = fifo[q][0]
                    if isinstance(r, image.Wait):
                        if not _ge(tokens.get(r.token, 0), r.value):
                            break
                        c = tclock.get((r.token, r.value))
                        if c:
                            vc[q] = [max(x, y) for x, y in zip(vc[q], c)]
                        fifo[q].popleft()
                        progress = True
                    elif isinstance(r, image.Signal):
                        v = tokens.get(r.token, 0) + 1 & image.M32
                        tokens[r.token] = v
                        prev = tclock.get((r.token, v - 1 & image.M32), [0] * queues)
                        tclock[(r.token, v)] = [max(x, y) for x, y in zip(prev, vc[q])]
                        fifo[q].popleft()
                        progress = True
                    else:
                        if isinstance(r, image.Dma):
                            st = dma(r, mem)
                        elif q in workers:
                            st = workers[q](r, mem)
                        else:
                            raise image.ImageError(f"record {idx}: no worker for queue {q} "
                                                   f"({image.KINDS[r.kind]})")
                        vc[q][q] += 1
                        for sp, a, b in st.reads:
                            access(q, idx, sp, a, b, False)
                        for sp, a, b in st.writes:
                            access(q, idx, sp, a, b, True)
                        got = st.start(mem)
                        dur = st.cycles + (rng.randint(0, jitter) if jitter else 0)
                        busy[q] += dur
                        active[q] = True
                        heapq.heappush(running, (now + dur, q, idx, st, got))
                        progress = True
                dispatch_some()

    dispatch_some()
    start_heads()
    while running:
        now, q, idx, st, got = heapq.heappop(running)
        st.end(mem, got)
        fifo[q].popleft()
        active[q] = False
        dispatch_some()
        start_heads()
    stuck = [q for q in range(queues) if fifo[q]]
    deadlock = ""
    if stuck or nxt < len(records) and not ended:
        heads = []
        for q in stuck:
            idx, r = fifo[q][0]
            heads.append(f"queue {q} at record {idx} ({image.KINDS[r.kind]}"
                         + (f" token {r.token} >= {r.value}, it is {tokens.get(r.token, 0)}" if isinstance(r, image.Wait)
                            else "") + ")")
        deadlock = "deadlock: " + ("; ".join(heads) or f"dispatch stalled at record {nxt}")
    return Result(now, dict(tokens), races, deadlock, busy)
