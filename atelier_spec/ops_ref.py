"""Bit-exact NumPy references of the op set. Every function computes in int64 and saturates to `fmt`.

Overflow of int64 is an error, never a wrap: inputs are checked against their formats by the caller.
"""

from __future__ import annotations

import numpy as np

from atelier_spec.ops import fmt_range


def _sat(x, fmt):
    lo, hi = fmt_range(fmt)
    return np.clip(np.asarray(x, dtype=np.int64), lo, hi)


def _i(x):
    return np.asarray(x, dtype=np.int64)


def matmul(a, b, fmt="int32"):
    y = _i(a) @ _i(b)
    lo, hi = fmt_range(fmt)
    if y.min(initial=0) < lo or y.max(initial=0) > hi:
        raise OverflowError(f"matmul result does not fit {fmt}")   # matmul is exact: no saturation
    return y


def add(a, b, fmt="int32"):
    return _sat(_i(a) + _i(b), fmt)


def sub(a, b, fmt="int32"):
    return _sat(_i(a) - _i(b), fmt)


def mul(a, b, fmt="int32"):
    return _sat(_i(a) * _i(b), fmt)


def max(a, b, fmt="int32"):  # noqa: A001 - the op's name
    return _sat(np.maximum(_i(a), _i(b)), fmt)


def min(a, b, fmt="int32"):  # noqa: A001
    return _sat(np.minimum(_i(a), _i(b)), fmt)


def relu(x, fmt="int32"):
    return _sat(np.maximum(_i(x), 0), fmt)


def round_shift(x, shift: int, mode: str = "half_up"):
    """x / 2**shift rounded: half_up (ties toward +inf), half_even, or floor."""
    x = _i(x)
    if shift == 0:
        return x
    q = x >> shift                                   # floor
    if mode == "floor":
        return q
    rem = x - (q << shift)
    half = 1 << (shift - 1)
    if mode == "half_up":
        return q + (rem >= half)
    if mode == "half_even":
        return q + ((rem > half) | ((rem == half) & (q & 1 == 1)))
    raise ValueError(f"unknown rounding {mode!r}")


def requant(x, multiplier=1, shift=0, zero_point=0, round="half_up", fmt="int8", axis=-1):  # noqa: A002
    """y = sat(round_shift(x * multiplier, shift, round) + zero_point); per-channel arrays along `axis`."""
    x = _i(x)
    def chan(v):
        v = _i(v)
        if v.ndim == 0:
            return v
        shape = [1] * x.ndim
        shape[axis] = -1
        return v.reshape(shape)
    m, s, z = chan(multiplier), np.asarray(shift), chan(zero_point)
    if s.ndim == 0:
        y = round_shift(x * m, int(s), round)
    else:
        y = np.stack([round_shift(xc, int(sc), round)
                      for xc, sc in zip(np.moveaxis(x * m, axis, 0), s)], axis=0)
        y = np.moveaxis(y, 0, axis)
    return _sat(y + z, fmt)


def lut(x, table, in_fmt="int8", fmt="int8"):
    lo, hi = fmt_range(in_fmt)
    table = _i(table)
    if len(table) != hi - lo + 1:
        raise ValueError(f"table has {len(table)} entries, {in_fmt} needs {hi - lo + 1}")
    return _sat(table[_i(x) - lo], fmt)


def transpose(x, perm):
    return np.transpose(x, perm)


def concat(xs, axis=0):
    return np.concatenate(xs, axis=axis)


def gather(x, idx, axis=0):
    return np.take(x, idx, axis=axis)


# ------------------------------------------------------------------ convolution and pooling (NHWC, HWIO)

def _pads(pad):
    t, l, b, r = (pad, pad, pad, pad) if np.isscalar(pad) else tuple(pad)
    return int(t), int(l), int(b), int(r)


def _pad_nhwc(x, pad, value):
    t, l, b, r = _pads(pad)
    return np.pad(_i(x), ((0, 0), (t, b), (l, r), (0, 0)), constant_values=value)


def _windows(x, kernel, stride):
    """[N, OH, OW, KH, KW, C] views of a padded NHWC map."""
    kh, kw = kernel
    sh, sw = stride
    n, h, w, c = x.shape
    oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
    if oh < 1 or ow < 1:
        raise ValueError(f"kernel {kernel} does not fit a {h}x{w} map")
    s = x.strides
    return np.lib.stride_tricks.as_strided(x, (n, oh, ow, kh, kw, c), (s[0], s[1] * sh, s[2] * sw, s[1], s[2], s[3]),
                                           writeable=False)


def conv2d(x, w, kernel=None, stride=(1, 1), pad=(0, 0, 0, 0), groups=1, pad_value=0, fmt="int32"):
    """y[n, oy, ox, o] = sum over (kh, kw, c in o's group) of x_padded * w: exact, refusing a result that does not
    fit `fmt` (as matmul). x NHWC, w HWIO with I = C / groups; groups == C is depthwise. pad_value: what the
    padding holds (a quantized input's zero point)."""
    x, w = _i(x), _i(w)
    kh, kw, ci, co = w.shape
    if kernel is not None and tuple(kernel) != (kh, kw):
        raise ValueError(f"kernel {kernel} but weights {kh}x{kw}")
    c = x.shape[3]
    if c % groups or co % groups or ci != c // groups:
        raise ValueError(f"{c} channels, {co} outputs, weights {ci} inputs: not {groups} groups")
    win = _windows(_pad_nhwc(x, pad, pad_value), (kh, kw), stride)
    g_in, g_out = c // groups, co // groups
    y = np.concatenate([np.einsum("nyxhwc,hwco->nyxo", win[..., g * g_in:(g + 1) * g_in],
                                  w[:, :, :, g * g_out:(g + 1) * g_out]) for g in range(groups)], axis=3)
    lo, hi = fmt_range(fmt)
    if y.min(initial=0) < lo or y.max(initial=0) > hi:
        raise OverflowError(f"conv2d result does not fit {fmt}")
    return y


def maxpool(x, kernel, stride, pad=(0, 0, 0, 0), fmt="int8"):
    """The maximum over each window; the padding never wins (it holds the format's minimum)."""
    lo, _ = fmt_range(fmt)
    return _sat(_windows(_pad_nhwc(x, pad, lo), tuple(kernel), tuple(stride)).max(axis=(3, 4)), fmt)


def upsample(x, scale, mode="nearest"):
    """Nearest-neighbour upsampling of an NHWC map by an integer scale."""
    if mode != "nearest":
        raise ValueError(f"upsample mode {mode!r}: nearest only")
    return np.repeat(np.repeat(_i(x), scale, axis=1), scale, axis=2)


def im2col(x, kernel, stride, pad=(0, 0, 0, 0), pad_value=0):
    """[N * OH * OW, KH * KW * C]: each output pixel's window in (kh, kw, c) order, row-major over pixels."""
    win = _windows(_pad_nhwc(x, pad, pad_value), tuple(kernel), tuple(stride))
    n, oh, ow, kh, kw, c = win.shape
    return win.reshape(n * oh * ow, kh * kw * c).copy()


# ------------------------------------------------------------------ table activations

def _table(f, in_scale, in_zp, out_scale, out_zp, in_fmt="int8", fmt="int8"):
    """The table of y = f(x) between two quantized formats (ONNX QDQ: x = (q - zp) s, q' = round_half_even(y / s')
    + zp', saturated): one entry per input value, computed in float64."""
    lo, hi = fmt_range(in_fmt)
    q = np.arange(lo, hi + 1, dtype=np.float64)
    y = f((q - in_zp) * in_scale)
    return _sat(np.round(y / out_scale) + out_zp, fmt)


def silu_table(in_scale, in_zp, out_scale, out_zp, in_fmt="int8", fmt="int8"):
    return _table(lambda v: v / (1 + np.exp(-v)), in_scale, in_zp, out_scale, out_zp, in_fmt, fmt)


def gelu_table(in_scale, in_zp, out_scale, out_zp, in_fmt="int8", fmt="int8", approximate="none"):
    """GELU's table: exact (erf) or the tanh approximation, as the model's own op says."""
    if approximate == "tanh":
        f = lambda v: 0.5 * v * (1 + np.tanh(np.sqrt(2 / np.pi) * (v + 0.044715 * v ** 3)))  # noqa: E731
    else:
        from math import erf
        f = np.vectorize(lambda v: 0.5 * v * (1 + erf(v / np.sqrt(2))))
    return _table(f, in_scale, in_zp, out_scale, out_zp, in_fmt, fmt)


def silu(x, table, in_fmt="int8", fmt="int8"):
    """SiLU as a table (silu_table: the model's quantization fixes it); y = table[x]."""
    return lut(x, table, in_fmt, fmt)


def gelu(x, table, in_fmt="int8", fmt="int8"):
    """GELU as a table (gelu_table); y = table[x]."""
    return lut(x, table, in_fmt, fmt)


# ------------------------------------------------------------------ reductions along the last axis (the row unit)

# The row unit's tables, constants of this standard (the RTL's row_tables.svh and Sutra's isa.py hold the same):
# RSQRT[odd][i] = round(2^15 / sqrt((1 + (i + 1/2)/128) 2^odd)), EXP[d] = round(32767 exp(-d/8)),
# RECIP[i] = round(2^16 / (1 + (i + 1/2)/256)); rounding half to even, in float64.
NORM_OUT = 32
EXP_DIV = 8
RSQRT = np.array([[round(32768 / np.sqrt((1 + (i + .5) / 128) * 2 ** odd)) for i in range(128)] for odd in (0, 1)],
                 dtype=np.int64)
EXP = np.array([round(32767 * np.exp(-d / EXP_DIV)) for d in range(256)], dtype=np.int64)
RECIP = np.array([round(65536 / (1 + (i + .5) / 256)) for i in range(256)], dtype=np.int64)


def _msb(v):
    """Index of the highest set bit of positive integers (exact, no float)."""
    v = _i(v)
    e = np.zeros(v.shape, np.int64)
    t = np.maximum(v, 1)
    for b in (32, 16, 8, 4, 2, 1):
        m = t >= (np.int64(1) << b)
        e = np.where(m, e + b, e)
        t = np.where(m, t >> b, t)
    return e


def norm_nr(n: int) -> int:
    """The norm's mean multiplier for rows of n elements: mean = (sum * nr) >> 24."""
    return round((1 << 24) / n)


def _rsqrt_rows(c, var):
    """(rn, sh): c * rn >> sh is c / sqrt(var) in the norm's output scale (rn from RSQRT, by exponent and mantissa)."""
    v = np.maximum(var, 1)
    e = _msb(v)
    idx = ((v << 7) >> e) - 128
    return RSQRT[e & 1, idx] * NORM_OUT, 21 + (e >> 1)


def rmsnorm(x, g, fmt="int8"):
    """RMS norm on rows x [..., n] with gains g [n] (INT8): the row unit's NORM, exact.
    var = (sum x^2 * nr) >> 24; y = sat(round_half_up(x g rn / 2^sh))."""
    x = _i(x)
    rn, sh = _rsqrt_rows(x, ((x * x).sum(axis=-1) * norm_nr(x.shape[-1])) >> 24)
    sh = sh[..., None]
    return _sat((x * _i(g) * rn[..., None] + (np.int64(1) << (sh - 1))) >> sh, fmt)


def layernorm(x, g, b, fmt="int8"):
    """Layer norm on rows x [..., n] with gains g and biases b [n]: the rows centred on their mean
    (mean = round_half_up(sum x * nr / 2^24)), then the norm's tables on the centred values, then the bias.
    y = sat(round_half_up(c g rn / 2^sh) + b)."""
    x = _i(x)
    nr = norm_nr(x.shape[-1])
    mean = (x.sum(axis=-1, keepdims=True) * nr + (1 << 23)) >> 24
    c = x - mean
    rn, sh = _rsqrt_rows(c, ((c * c).sum(axis=-1) * nr) >> 24)
    sh = sh[..., None]
    return _sat(((c * _i(g) * rn[..., None] + (np.int64(1) << (sh - 1))) >> sh) + _i(b), fmt)


def softmax(x, shift: int = 0, fmt="uint7"):
    """Softmax on rows x [..., n] (INT8 scores): the row unit's SOFTMAX, Q7 probabilities, exact.
    d = min((max - x) >> shift, 255); e = EXP[d]; y = round(e * RECIP[mantissa of sum e] * 128 / 2^(16 + exponent))."""
    s = _i(x)
    e = EXP[np.minimum((s.max(axis=-1, keepdims=True) - s) >> shift, 255)]
    tot = e.sum(axis=-1, keepdims=True)
    b = _msb(tot)
    rr = RECIP[((tot << 8) >> b) - 256]
    return _sat((e * rr * 128 + (np.int64(1) << (15 + b))) >> (16 + b), fmt)


# ------------------------------------------------------------------ rotary position embedding

ROPE_FRAC = 14                                       # Q14: cos 1 is exact (2^14), as the integer LLM model has


def rope_tables(positions, dim: int, theta: float = 10000.0):
    """(cos, sin) [len(positions), dim / 2] in Q14: round_half_even(2^14 trig(p theta^(-2i/dim))), in float64."""
    p = np.asarray(positions, dtype=np.float64)[:, None]
    inv = theta ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)
    ang = p * inv[None, :]
    return np.round(np.cos(ang) * (1 << ROPE_FRAC)).astype(np.int64), np.round(np.sin(ang) * (1 << ROPE_FRAC)).astype(np.int64)


def rope(x, positions, theta: float = 10000.0, layout: str = "halves", fmt="int8"):
    """Rotary embedding on x [..., seq, dim] at positions [seq]: pairs (halves: i and i + dim/2, as GPT-NeoX and
    Qwen; interleaved: 2i and 2i + 1) rotated by Q14 tables; y = sat(round_half_up(rotated / 2^14))."""
    x = _i(x)
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError(f"rope needs an even dimension, not {dim}")
    cos, sin = rope_tables(positions, dim, theta)
    if layout == "halves":
        a, b = x[..., :dim // 2], x[..., dim // 2:]
    elif layout == "interleaved":
        a, b = x[..., 0::2], x[..., 1::2]
    else:
        raise ValueError(f"rope layout {layout!r}: halves or interleaved")
    ya = round_shift(a * cos - b * sin, ROPE_FRAC)
    yb = round_shift(a * sin + b * cos, ROPE_FRAC)
    y = np.empty_like(x)
    if layout == "halves":
        y[..., :dim // 2], y[..., dim // 2:] = ya, yb
    else:
        y[..., 0::2], y[..., 1::2] = ya, yb
    return _sat(y, fmt)
