# Atelier Op Set

Version 0.1 (draft), 2026-10-09.
- Data: `atelier_spec/data/opset-0.1.yaml`
- References: `atelier_spec/ops_ref.py`

The op set is the shared vocabulary of the compiler, the blocks and verification:
- a unit or accelerator declares the ops it implements **by name**;
- the compiler matches graph ops to units by those names;
- verification checks every unit against the op's reference.

## Rules

- **Integer semantics are exact.** Computation is in int64. Results are saturated to the output format, except `matmul`, which is exact and refuses a result that doesn't fit its format.
- **Rounding is named.** `round_shift(x, s, mode)` divides by 2^s with `floor`, `half_up` (ties toward +∞) or `half_even`.
- **`requant`:** `y = sat(round_shift(x * multiplier, shift, round) + zero_point)`. Multiplier and zero point may be per channel along an axis.
- **Formats:** int4, int8, int16, int32, int48, int64, uint7 (Q7), uint8.
- **Versioning:** every op carries its own `version`. A change to its semantics bumps it, and blocks declare the version they implement.
- **Customer ops:** a block MAY add ops in its own namespace (`acme.fft256`), each with a reference. The compiler side is a native plugin.

## Status of the references

| Status | Ops |
|---|---|
| **Written** (`reference: ref`) | every op: matmul, conv2d, add, sub, mul, max, min, relu, requant, lut (exp_lut), silu, gelu, softmax, rmsnorm, layernorm, rope, transpose, concat, gather, maxpool, upsample, im2col |
| **Pending** | none (2026-10-13) |

A block MAY declare a pending op. Its L1 check waits for the reference.
`ops.reference(name)` raises `NotImplementedError` for a pending op, so nothing silently passes.

The references for softmax and rmsnorm are the row unit's: its tables (`RSQRT`, `EXP`, `RECIP`) are
constants of this standard, and the references equal Sutra's `isa.row_softmax` and `isa.row_norm` bit for
bit (`tests/test_ops_ref.py`). Layer norm uses the same tables on rows centred on their mean; it defines
what a unit implementing it must match.

SiLU and GELU are tables the model's quantization fixes: `silu_table` and `gelu_table` build them from the
input and output scales and zero points (QDQ, rounding half to even). Convolution pads with `pad_value`, the
input's zero point; pooling pads with the format's minimum, so the padding never wins.
