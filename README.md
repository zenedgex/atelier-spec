# atelier-spec

The Atelier standards: the formats every other Atelier repository reads and writes. If two tools agree on these files, they work together, and that includes a customer's own tools.

| Standard | Version | Prose | Machine-readable | Checker |
|---|---|---|---|---|
| Block contract | `atelier-block/1.0` (draft) | [standards/block.md](standards/block.md) | [schema](atelier_spec/data/schemas/block-1.0.schema.json) | `atelier-spec block check` |
| Machine description | `atelier-machine/0.1` (draft) | [standards/machine.md](standards/machine.md) | [schema](atelier_spec/data/schemas/machine-0.1.schema.json) | `atelier-spec machine check` |
| Op set | `atelier-ops/0.1` (draft) | [standards/ops.md](standards/ops.md) | [opset-0.1.yaml](atelier_spec/data/opset-0.1.yaml) | references in `atelier_spec/ops_ref.py` |
| Evidence labels | `atelier-evidence/1.0` (draft) | [standards/evidence.md](standards/evidence.md) | | `atelier_spec.evidence` |
| Program image | not yet written | [standards/image.md](standards/image.md) | | |

## Use

```bash
uv venv && uv pip install -e '.[dev]'
.venv/bin/atelier-spec block check conformance/block/good/acme_vec.block.yaml
.venv/bin/atelier-spec block migrate old/block.json            # 0.x -> 1.0
.venv/bin/atelier-spec machine check conformance/machine/good/*.yaml --blocks path/to/blocks
.venv/bin/pytest -q
```

```python
from atelier_spec import block, machine
problems = block.check(block.load("acme_vec.block.yaml"))   # [] when consistent
problems = machine.check(machine.load("npu.yaml"), blocks=lookup)
```

## Conformance suite

- `conformance/block/good` and `conformance/machine/good` must pass.
- Each file in `conformance/*/bad` must fail with the error named on its first line (`# expect: ...`).
- `tools/make_conformance.py` regenerates the bad cases from the good ones.
- Another implementation of these standards is conformant when it agrees on every file.

## Status

Drafts, versioned as above.
- **Available:** block 1.0 and machine 0.1, with schemas, checkers and conformance suites; the op set 0.1, with bit-exact references for 12 ops; evidence 1.0.
- **Next:** the program image 1.0, and the remaining op references (softmax, rmsnorm, rope, table activations, pooling).

Issues and proposals are welcome.

## Licence

Apache-2.0. See [LICENSE](LICENSE).

Atelier and Sutra are products of ZenEdgeX (https://zenedgex.in, hello@zenedgex.in).
