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
| **Written** (`reference: ref`) | matmul, add, sub, mul, max, min, relu, requant, lut (exp_lut), transpose, concat, gather |
| **Pending** | conv2d, silu, gelu, softmax, rmsnorm, rope, maxpool, upsample, im2col (from Sutra's `isa.py`); layernorm |

A block MAY declare a pending op. Its L1 check waits for the reference.
`ops.reference(name)` raises `NotImplementedError` for a pending op, so nothing silently passes.

The references for softmax, rmsnorm and the table activations come from Sutra's `isa.py`
(`row_softmax`, `row_norm`, the post-op tables). They will match today's RTL bit for bit, and that
match is their acceptance test.
