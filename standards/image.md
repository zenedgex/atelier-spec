# Atelier Program Image

Version image/1 (draft), 2026-10-10.
- Reference codec: `atelier_spec/image.py`
- Reference dispatcher: `atelier_spec/dispatch.py`
- Design: atelier-platform spec 16 (dispatch and data movement)

A program image is what a job runs: one **record stream** per dispatcher, which routes records to the machine's queues (`machine.queues`). The machine description fixes the numbering:
- queue *i* is `machine.queues(m)[i]`;
- space *j* is `machine.spaces(m)[j]`.

The image carries a hash of both numberings, so the runtime refuses an image built for another machine.

image/0 is today's Sutra streamer format (records CFG=1, CMD=2, WR=3, SYNC=4, IMG=5, OUT=6, SEL=7, SAVE=8, END=0, in `cmd_stream.sv` and the compiler's `SerializeImage`). It stays readable until the dispatcher replaces `cmd_stream.sv`.

## 1. Words

- All fields are 32-bit little-endian words.
- 64-bit values (addresses, strides, fill values) take two words, low first.
- Counts are 32-bit; addresses and strides are 64-bit (strides signed).
- Nothing in the format limits the number of queues, tokens, spaces, streams or records (R15).

## 2. Container

| Word | Field |
|---|---|
| 0 | magic `0x31495441` ("ATI1") |
| 1 | version, 1 |
| 2 | machine hash: FNV-1a 32 over the queue and space numbering (`image.machine_hash`) |
| 3 | n, the number of record streams |
| 4 + 3k | stream k: byte offset from the image start (64 bits, two words), then its length in words |

Then the streams, each word-aligned.
- Stream k is dispatcher k's: one per chip, or one per cluster instance when `dispatch.per` names a cluster.
- Identical clusters may share one stream's offset (R15.3).

Later sections (kernels for LAUNCH, weights and constants for DMA from the image, relocations, checks) are added as further offset tables after the streams. The version stays 1 because they are additions.

## 3. Records

```
header:  kind [7:0] | flags [15:8] | words [31:16]     words: the words that follow the header
queue:   q                                              (flags.MULTI = 0)
         n, q0 .. q(n-1)                                (flags.MULTI = 1: the same record to n queues)
payload: by kind
```

Every kind except END carries a queue field. Flags bit 0 is MULTI; the other bits are reserved and zero. A record of more than 65,535 words is split into several records; the encoder does this for inline data.

| Kind | Code | Payload | Queue runs it as |
|---|---|---|---|
| END | 0 | (no queue field) | the dispatcher: the job is done once every queue has drained |
| CMD | 1 | the unit's command words (a 64-bit command: low, high) | a command; done when the unit reports it done |
| CFG | 2 | address, then configuration data (the unit's word width, low first) | writes at address, address + 1, ... |
| DMA | 3 | a descriptor (§4) | a copy; done when its last write is accepted. A unit's own queue MAY take inline DMA records into that unit's private memories (its block declares which), run in order with its commands: data the unit's next commands need, with no token between them |
| LAUNCH | 4 | kernel index, then argument words | a kernel launch on an ISA unit; done when the kernel returns |
| WAIT | 5 | token, value | holds this queue until token ≥ value (modulo 2³²) |
| SIGNAL | 6 | token | when every earlier record of this queue is done, adds 1 to the token |

**Tokens:**
- start at 0 when the dispatcher starts;
- are numbered 0 .. `dispatch.tokens` − 1;
- token 0 is the job token by convention: END waits for nothing else, but the runtime may wait on it.

**Order:**
- each queue runs its records in order;
- across queues, only tokens order anything;
- a record sent to several queues (MULTI) runs once in each.

## 4. DMA descriptor

| Word | Field |
|---|---|
| 0 | source space (`0xFFFFFFFF` for fill and inline) |
| 1 | destination space |
| 2 | element bytes [15:0], mode [19:16], source dims [23:20], destination dims [27:24] |
| 3, 4 | source base (bytes) |
| 5, 6 | destination base (bytes) |
| then | source dims, innermost first: count, stride (two words), three words each |
| then | destination dims, the same |
| then | by mode: fill: the value (two words); inline: the data, padded to whole words |

**Modes:**

| Mode | Code | Meaning |
|---|---|---|
| copy | 0 | the source's elements, in iteration order (innermost dimension fastest), to the destination's, in its iteration order |
| fill | 1 | every destination element is the value's low element bytes |
| inline | 2 | the data in the record is the source, in order |
| gather | 3 | reserved (paged KV caches, embeddings) |

**Rules:**
- **Element counts match:** the product of the source counts equals the product of the destination counts. For fill and inline there are no source dims.
- **Dimensions** are 1 to 15 per side; the DMA block's `DIMS` parameter says how many it runs natively. The compiler splits deeper ones into several descriptors.
- **A stream port** (an address-less space) ignores its base and strides; its elements come or go in order.
- **The route** (source space → destination space) must be one of the machine's `movement.dma` routes for the DMA engine owning the queue.

**The footprint** of a descriptor is the set of byte ranges it reads and writes per space. The reference dispatcher's race check uses it (§5), and so does the compiler's static check.

## 5. What a correct image guarantees

- **Determinism:** the result is the same for every timing the rules allow. Any queue may be delayed by any amount at any record.
- **No races:** two records on different queues whose footprints overlap, at least one of them writing, are ordered by a chain of SIGNAL → WAIT.
  - The reference dispatcher checks this with vector clocks while it runs, under random delays.
  - The compiler refuses an image that fails it.
- **Progress:** in the stream, a WAIT for token ≥ v comes after at least v SIGNALs of that token.
  - The dispatcher routes records in stream order and stalls when a queue is full. With this rule, a WAIT's signals are always already dispatched, so full queues can never deadlock. Without it, one can.
  - The reference checks the rule (`dispatch.check_order`) and reports any deadlock with the queues and tokens involved.

## 6. Relation to IREE

On Linux, host and RTOS targets the program is an IREE `.vmfb`, and this image's streams are the payload of its HAL executables. Tokens are the device-level form of IREE's stream timepoints. On bare-metal microcontrollers the minimal runtime reads this image directly. Both come from the same compiler output.
