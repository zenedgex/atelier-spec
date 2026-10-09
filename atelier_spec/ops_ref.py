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
