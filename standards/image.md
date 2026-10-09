# Atelier Program Image

Status: **not yet written**. It is drafted together with the native compiler's command target.

## Inputs to the draft

- Today's format: the runtime image of the codesign-engine package (`codesign_engine/compiler/runtime_image.py`). It becomes the `commands` section.
- The planned sections: header, `commands` per command-driven unit, `kernels` per ISA unit, `weights`, `constants`, `buffers`, `relocs`, `checks`.
- Requirements:
  - portable across control cores: little-endian, no core-specific code outside `kernels`;
  - 32-bit counts and 64-bit offsets;
  - per-cluster streams indexed for parallel dispatch, with no limit on their number.

## Relation to IREE

On Linux, host and RTOS targets the program is an IREE `.vmfb`, and this image's sections are the
payload of its HAL executables. On bare-metal microcontrollers the minimal runtime reads this image
directly. Both come from the same compiler output.
