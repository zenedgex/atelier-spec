"""image/1 (standards/image.md) and the reference dispatcher (atelier_spec/dispatch.py)."""

import pytest

from atelier_spec import dispatch, image, machine
from atelier_spec.image import Cfg, Cmd, Desc, Dma, End, Launch, Signal, Wait

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def test_every_record_round_trips():
    recs = [Cmd(0, [0x1234, 0xFFFF_FFFF]), Cmd((0, 3, 7), [5]), Cfg(1, 0x40, [1, 2, 3]), Launch(2, 9, [4, 5]),
            Wait(1, 3, 7), Signal((1, 2), 3),
            Dma(4, Desc.copy(2, 0x1_0000_0000, 0, 64, 8, [(4, 8), (3, -256)], [(12, 8)])),
            Dma(4, Desc.fill(0, 0, 2, [(5, 2)], 0xBEEF)),
            Dma(5, Desc.inline(0, 8, 1, [(3, 1)], b"abc")), End()]
    words = image.encode(recs)
    assert image.decode(words) == recs
    assert words[-1] == image.END and all(0 <= w <= image.M32 for w in words)


def test_descriptor_rules():
    with pytest.raises(image.ImageError, match="6 source elements, 4 destination"):
        Desc.copy(0, 0, 1, 0, 8, [(6, 8)], [(4, 8)]).words()
    with pytest.raises(image.ImageError, match="inline data is 2 bytes"):
        Desc.inline(0, 0, 1, [(3, 1)], b"ab").words()
    with pytest.raises(image.ImageError, match="gather is reserved"):
        Desc(0, 1, 8, dst=((1, 8),), src=((1, 8),), mode=image.GATHER).words()
    d = Desc.copy(0, 100, 1, 0, 4, [(2, 4), (2, 64)], [(4, 4)])
    assert d.offsets("src") == [100, 104, 164, 168]
    assert d.footprint() == ([(0, 100, 108), (0, 164, 172)], [(1, 0, 16)])


def test_big_inline_data_splits():
    data = bytes(range(256)) * 2048                       # 512 KiB: more than one record holds
    recs = image.split_inline(0, 1, 4096, data)
    assert len(recs) > 1 and b"".join(r.desc.data for r in recs) == data
    assert image.decode(image.encode(recs)) == recs


def test_container_binds_to_the_machine():
    m = machine.load(ROOT / "conformance/machine/good/cim.yaml")
    b = image.Binding(m)
    assert b.queue("cim") == 0 and b.queue("dma", 1) == 2 and b.space("dram") == 6
    s0 = image.encode([Signal(b.queue("dma", 0), 0), End()])
    blob = image.pack([s0, s0[:1]], b.hash)
    assert image.unpack(blob, b.hash) == [s0, s0[:1]]
    m2 = machine.load(ROOT / "conformance/machine/good/npu.yaml")
    with pytest.raises(image.ImageError, match="another machine"):
        image.unpack(blob, image.machine_hash(m2))


# ------------------------------------------------------------------ the dispatcher

IN, OUT, ACT = 0, 1, 2                        # spaces: DRAM in, DRAM out, ACT (two band buffers)
ENG, LD, SV = 0, 1, 2                         # queues: the engine, DMA in, DMA out
BAND = 64                                     # bytes a band


def engine(rec, mem):
    """A unit: CMD [buffer] adds 1 to each byte of a band buffer in place, 100 cycles."""
    buf = rec.words[0] * BAND
    return dispatch.Step(100, [(ACT, buf, buf + BAND)], [(ACT, buf, buf + BAND)],
                         lambda m: bytes(m.spaces[ACT][buf:buf + BAND]),
                         lambda m, got: m.spaces[ACT].__setitem__(slice(buf, buf + BAND), bytes(x + 1 & 255 for x in got)))


def bands(n: int, drop: str = "") -> list:
    """Double-buffered: load band k into buffer k % 2, compute, save; tokens 1 (loaded), 2 (computed),
    3 (saved). `drop` removes one wait, to make a race."""
    recs = []
    for k in range(n):
        buf = (k % 2) * BAND
        if k >= 2 and drop != "reuse":
            recs.append(Wait(LD, 3, k - 1))          # buffer k % 2 is free once band k-2 is saved
        recs += [Dma(LD, Desc.copy(IN, k * BAND, ACT, buf, 8, [(BAND // 8, 8)], [(BAND // 8, 8)])), Signal(LD, 1)]
        if drop != "load":
            recs.append(Wait(ENG, 1, k + 1))
        recs += [Cmd(ENG, [k % 2]), Signal(ENG, 2), Wait(SV, 2, k + 1),
                 Dma(SV, Desc.copy(ACT, buf, OUT, k * BAND, 8, [(BAND // 8, 8)], [(BAND // 8, 8)])), Signal(SV, 3)]
    return recs + [End()]


def memory(n):
    return dispatch.Memory({IN: bytearray(bytes(range(256)) * (n * BAND // 256 + 1)), OUT: bytearray(n * BAND),
                            ACT: bytearray(2 * BAND)})


def test_bands_overlap_and_give_one_result_for_any_timing():
    n = 8
    want = bytes(x + 1 & 255 for x in memory(n).spaces[IN][:n * BAND])
    assert dispatch.check_order(bands(n)) == []
    cycles = set()
    for seed in range(20):
        mem = memory(n)
        r = dispatch.run(bands(n), 3, mem, {ENG: engine}, jitter=50, seed=seed)
        assert r.ok, (r.races[:1], r.deadlock)
        assert bytes(mem.spaces[OUT]) == want
        cycles.add(r.cycles)
    assert len(cycles) > 1                                  # the timing varied; the result did not
    r = dispatch.run(bands(n), 3, memory(n), {ENG: engine})
    # compute-bound: every load and save but the first load and the last save ran under compute
    assert r.cycles == r.busy[ENG] + r.busy[LD] // n + r.busy[SV] // n < sum(r.busy.values())


@pytest.mark.parametrize("drop", ["load", "reuse"])
def test_a_missing_token_is_a_race_whatever_the_timing(drop):
    for seed in range(3):
        r = dispatch.run(bands(6, drop), 3, memory(6), {ENG: engine}, seed=seed, jitter=50)
        assert r.races and not r.deadlock
        assert "not ordered by tokens" in str(r.races[0])


def test_a_wait_before_its_signal_can_deadlock():
    recs = [Wait(0, 1, 1), Cmd(0, [0]), Signal(1, 1), End()]
    assert "only 0 SIGNALs" in dispatch.check_order(recs)[0]
    r = dispatch.run(recs, 2, memory(1), {0: engine, 1: engine}, depth=1)
    assert "deadlock" in r.deadlock and "token 1 >= 1, it is 0" in r.deadlock


def test_multicast_and_token_wrap():
    recs = [Signal((0, 1, 2), 5), Wait(3, 5, 3), Cmd(3, [0]), End()]
    r = dispatch.run(recs, 4, memory(1), {3: engine})
    assert r.ok and r.tokens[5] == 3
    assert dispatch._ge(2, 0xFFFF_FFFE) and not dispatch._ge(0xFFFF_FFFE, 2)
