"""The op set's references for convolution, pooling, the table activations, the row unit's reductions and RoPE:
each against a float or brute-force oracle within its stated rounding, and, where Sutra's ISA model is present,
bit for bit against the unit that implements it (the row unit's softmax and norm)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from atelier_spec import ops
from atelier_spec import ops_ref as r

RNG = np.random.default_rng(7)
SUTRA = Path(__file__).resolve().parents[2] / "atelier-architectures" / "sutra" / "python"


def isa():
    if SUTRA.exists() and str(SUTRA) not in sys.path:
        sys.path.insert(0, str(SUTRA))
    return pytest.importorskip("sutra.isa")


def i8(*shape):
    return RNG.integers(-128, 128, shape)


# ------------------------------------------------------------------ convolution and pooling

def conv_oracle(x, w, stride, pad, groups, pad_value):
    """Loops over every output, in float64 (exact for these sizes)."""
    t, l, b, rr = pad
    xp = np.pad(x, ((0, 0), (t, b), (l, rr), (0, 0)), constant_values=pad_value).astype(np.float64)
    kh, kw, ci, co = w.shape
    n, h, wd, c = xp.shape
    oh, ow = (h - kh) // stride[0] + 1, (wd - kw) // stride[1] + 1
    y = np.zeros((n, oh, ow, co))
    go = co // groups
    for o in range(co):
        g = o // go
        for oy in range(oh):
            for ox in range(ow):
                win = xp[:, oy * stride[0]:oy * stride[0] + kh, ox * stride[1]:ox * stride[1] + kw, g * ci:(g + 1) * ci]
                y[:, oy, ox, o] = (win * w[:, :, :, o]).sum(axis=(1, 2, 3))
    return y.astype(np.int64)


@pytest.mark.parametrize("groups,c,co", [(1, 8, 16), (8, 8, 8), (2, 8, 4)], ids=["dense", "depthwise", "grouped"])
@pytest.mark.parametrize("stride,pad", [((1, 1), (1, 1, 1, 1)), ((2, 2), (0, 1, 1, 0))])
def test_conv2d_equals_the_loops(groups, c, co, stride, pad):
    x, w = i8(2, 7, 9, c), i8(3, 3, c // groups, co)
    want = conv_oracle(x, w, stride, pad, groups, pad_value=-5)
    got = r.conv2d(x, w, stride=stride, pad=pad, groups=groups, pad_value=-5)
    assert np.array_equal(got, want)
    assert np.array_equal(ops.reference("conv2d")(x, w, stride=stride, pad=pad, groups=groups, pad_value=-5), want)


def test_conv2d_refuses_overflow_and_bad_groups():
    with pytest.raises(OverflowError):
        r.conv2d(np.full((1, 3, 3, 4), 127), np.full((3, 3, 4, 1), 127), fmt="int16")
    with pytest.raises(ValueError, match="groups"):
        r.conv2d(i8(1, 4, 4, 6), i8(3, 3, 6, 4), groups=4)


def test_im2col_times_weights_is_the_convolution():
    x, w = i8(1, 6, 5, 8), i8(3, 3, 8, 4)
    cols = r.im2col(x, (3, 3), (2, 1), pad=(1, 1, 1, 1), pad_value=3)
    y = r.conv2d(x, w, stride=(2, 1), pad=(1, 1, 1, 1), pad_value=3)
    assert np.array_equal((cols @ w.reshape(-1, 4)).reshape(y.shape), y)


def test_maxpool_padding_never_wins():
    x = RNG.integers(-128, -100, (1, 5, 5, 3))           # all negative: a zero pad would win
    y = r.maxpool(x, (3, 3), (1, 1), pad=(1, 1, 1, 1))
    for oy in range(5):
        for ox in range(5):
            win = x[0, max(0, oy - 1):oy + 2, max(0, ox - 1):ox + 2]
            assert np.array_equal(y[0, oy, ox], win.max(axis=(0, 1)))


def test_upsample_nearest():
    x = i8(1, 2, 3, 4)
    y = r.upsample(x, 2)
    assert y.shape == (1, 4, 6, 4)
    assert np.array_equal(y[0, 3, 5], x[0, 1, 2]) and np.array_equal(y[0, 2, 4], x[0, 1, 2])


# ------------------------------------------------------------------ table activations

def test_silu_and_gelu_tables_follow_qdq_rounding():
    s_in, z_in, s_out, z_out = 0.05, 3, 0.03, -10
    t = r.silu_table(s_in, z_in, s_out, z_out)
    q = np.arange(-128, 128)
    v = (q - z_in) * s_in
    want = np.clip(np.round(v / (1 + np.exp(-v)) / s_out) + z_out, -128, 127)
    assert np.array_equal(t, want)
    x = i8(100)
    assert np.array_equal(r.silu(x, t), t[x + 128])
    g = r.gelu_table(s_in, z_in, s_out, z_out)
    gt = r.gelu_table(s_in, z_in, s_out, z_out, approximate="tanh")
    assert np.abs(g - gt).max() <= 1                     # the tanh form within one step at these scales
    assert np.array_equal(ops.reference("gelu")(x, g), g[x + 128])


# ------------------------------------------------------------------ the row unit's reductions

def test_softmax_is_q7_and_close_to_the_float():
    x = i8(64, 40)
    y = r.softmax(x)
    assert y.min() >= 0 and y.max() <= 127
    f = np.exp((x - x.max(axis=-1, keepdims=True)) / r.EXP_DIV)
    f = 128 * f / f.sum(axis=-1, keepdims=True)
    assert np.abs(y - f).max() <= 1.5                    # the tables' rounding: within one and a half steps
    assert np.all(np.abs(y.sum(axis=-1) - 128) <= x.shape[-1] // 2 + 1)


def test_softmax_shift_scales_the_scores():
    x = i8(8, 16)
    gap = x.max(axis=-1, keepdims=True) - x              # shift divides each score's distance below the row's max
    assert np.array_equal(r.softmax(x, shift=1), r.softmax(-(gap >> 1)))


def test_rmsnorm_is_x_g_over_twice_the_rms():
    x, g = i8(32, 48), RNG.integers(1, 128, 48)
    y = r.rmsnorm(x, g)
    var = np.maximum(((x * x).sum(-1) * r.norm_nr(48)) >> 24, 1)
    f = np.clip(x * g / (2 * np.sqrt(var))[:, None], -128, 127)
    assert np.abs(y - f).max() <= 1 + 0.005 * np.abs(f).max()   # the 1/128 mantissa table: within half a percent


def test_layernorm_centres_then_norms_then_biases():
    x, g, b = i8(16, 32) + 40, RNG.integers(1, 64, 32), RNG.integers(-20, 20, 32)
    y = r.layernorm(x, g, b)
    nr = r.norm_nr(32)
    mean = (x.sum(-1, keepdims=True) * nr + (1 << 23)) >> 24
    c = x - mean
    var = np.maximum(((c * c).sum(-1) * nr) >> 24, 1)
    f = np.clip(c * g / (2 * np.sqrt(var))[:, None] + b, -128, 127)
    assert np.abs(y - f).max() <= 1 + 0.005 * np.abs(f).max()
    assert np.array_equal(r.layernorm(x - mean, g, np.zeros(32, int)), r.rmsnorm(x - mean, g))   # a centred row


def test_softmax_and_rmsnorm_are_the_row_unit_bit_for_bit():
    s = isa()
    for _ in range(20):
        x = i8(RNG.integers(1, 9), RNG.integers(2, 200))
        assert np.array_equal(r.softmax(x), s.row_softmax(x))
        g = RNG.integers(-128, 128, x.shape[-1])
        assert np.array_equal(r.rmsnorm(x, g), s.row_norm(x, g, s.norm_nr(x.shape[-1])))
    assert np.array_equal(r.RSQRT, s.RSQRT) and np.array_equal(r.EXP, s.EXP) and np.array_equal(r.RECIP, s.RECIP)


def test_msb_is_exact_at_every_power_of_two():
    v = np.array([1 << k for k in range(40)] + [(1 << k) - 1 for k in range(2, 40)], dtype=np.int64)
    want = np.array([k for k in range(40)] + [k - 1 for k in range(2, 40)])
    assert np.array_equal(r._msb(v), want)


# ------------------------------------------------------------------ RoPE

@pytest.mark.parametrize("layout", ["halves", "interleaved"])
def test_rope_rotates_pairs(layout):
    x, pos = i8(3, 10, 16), np.arange(5, 15)
    y = r.rope(x, pos, layout=layout)
    d = 16
    ang = pos[:, None] * 10000.0 ** (-np.arange(0, d, 2) / d)
    a, b = (x[..., :d // 2], x[..., d // 2:]) if layout == "halves" else (x[..., 0::2], x[..., 1::2])
    fa, fb = a * np.cos(ang) - b * np.sin(ang), a * np.sin(ang) + b * np.cos(ang)
    ya, yb = (y[..., :d // 2], y[..., d // 2:]) if layout == "halves" else (y[..., 0::2], y[..., 1::2])
    assert np.abs(ya - np.clip(fa, -128, 127)).max() <= 1 and np.abs(yb - np.clip(fb, -128, 127)).max() <= 1
    assert np.array_equal(r.rope(x, np.zeros(10, int), layout=layout), x)   # position 0: no rotation


def test_every_op_has_a_reference():
    for name, entry in ops.opset()["ops"].items():
        assert entry.get("reference") == "ref", f"{name} is still pending"
        assert callable(ops.reference(name))
