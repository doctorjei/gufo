# RFC: I4R128 — rotated 4-bit expert codes as a GGUF tensor type

Status: implemented in core (CPU reference + sizing + loader acceptance); GPU
kernels not yet written. Written 2026-11-08 against main `7e8f75a8`.

Provenance: the `jtsylve/hgn-spec` 1.1.0 (MIT/CC-BY) `i4r` store (code 23),
`STORAGE.md` sections "The 128-point rotation", "Rotation vectors", "i4r".
This document adapts that store to a Gufo GGUF tensor type and is the
Gufo-side authority for the byte layout.

Contents:

- [Type identity](#type-identity) and [block layout](#block-layout).
- [Tensor layout](#tensor-layout).
- [The stored numbers are rotated](#the-stored-numbers-are-rotated), with the
  five-stage matvec and the four rotation stages per layer.
- [Rotation vectors](#rotation-vectors) and [rotation marker](#rotation-marker).
- [Transcode is lossless](#transcode-is-lossless).
- [What is implemented here](#what-is-implemented-here) and
  [what the GPU path still needs](#what-the-gpu-path-still-needs).

## Type identity

| Property | Value |
| --- | --- |
| Type name | `I4R128` |
| GGUF type id | **64** (`GgmlType::kI4R128`) |
| Block size | 128 values |
| Block bytes | **66** = 64 B codes + 2 B `f16` scale |
| Bits per value | 4.125 (66/128) |
| Code domain | offset binary `c = nibble − 8` ∈ [−8, 7] |
| Scale | IEEE-754 binary16, little-endian, one per block |

**Type-id choice.** Upstream `GGML_TYPE_COUNT` is 44 (ids 0..43 used or
reserved), and Gufo's `GgmlType` enum defined no private range. 64 sits above
that with a 20-id buffer (44..63) for upstream growth, and inside every width
field: GGUF stores the type as `uint32`, Gufo's enum is `uint16`. The id
appears only in the tensor-info field and in the sizing switch, never inside a
payload, so if upstream ever claims 64 a renumber is a two-line change.

## Block layout

A block covers 128 consecutive values along the **input** dimension (`ne0`),
the contiguous axis in GGUF.

```
byte  0 .. 63   : 128 x 4-bit offset-binary codes, GGUF nibble order
byte 64 .. 65   : f16 scale d (little-endian)
```

**Nibble order (GGUF convention, as in Q4_0 / Q4_K):** for block byte `b`
(0 ≤ b < 64), the low nibble is value `b` and the high nibble is value
`64 + b`. This deliberately differs from HGN `i4r`'s sequential-pair order
(byte `j` holds values `2j`, `2j+1`); the transcoder permutes nibbles so
kernels unpack with the standard Q4_0 pattern.

Decode of one block (the stored number `M`, before rotation):

```
for v in 0..127:
    nib = (code_byte[v & 63] >> (4 * (v >= 64))) & 0xF
    M[v] = (nib - 8) * d
```

Exact in f32; `d` may be any f16 value including 0.

**Implementation trap.** In C++ `byte & 0x0FU` promotes to *unsigned* int, so
`(byte & 0x0FU) - 8` wraps to ~2^32 rather than going negative. Cast to `int`
before subtracting. `i4r128_reference_test` asserts the full code domain
precisely to keep this from recurring.

## Tensor layout

Quantization groups run along `ne0` (input). For `[ne0, ne1, ne2]` in GGUF
order (`ne0` fastest):

- row stride = `(ne0 / 128) * 66` bytes; `ne0` must be a multiple of 128.
- data order: expert-major (`ne2`), then output row (`ne1`), then the
  `ne0/128` blocks of that row, in order.
- total bytes = `ne0 * ne1 * ne2 * 66 / 128`.
- tensor data start follows the file's `general.alignment` (32 in the base
  model); the type itself requires no extra alignment.

Constraint `ne0 % 128 == 0` is enforced by the loader: `EncodedSizeBytes`
returns 0 for a misaligned count, which the reader treats as unsized and
rejects unless an extension owns the layout. In the Flash-Next trunk, gate/up
have `ne0 = 2560` (20 blocks/row) and down has `ne0 = 640` (5 blocks/row).

## The stored numbers are rotated

`I4R128` holds `M`, the matrix in the 128-point rotated basis, exactly as
HGN `i4r`. The true weight `W` a dense matmul would multiply is

```
W = diag(sv) · Hn · M · Hn · diag(su)        Hn = H128 / sqrt(128)
```

with `H128` the Sylvester Hadamard matrix, `h(i,j) = (−1)^popcount(i&j)`,
`0 ≤ i,j < 128`. `Hn` is symmetric and orthonormal, so `Hn · Hn = I` — the
transform is its own inverse, which is what lets one kernel serve both sides.

The products are block-diagonal: `Hn · M` transforms blocks of 128 rows
(output side), `M · Hn` transforms blocks of 128 columns (input side). Each
expert matrix is rotated independently using the layer-shared vectors below.

Kernel form of `y = W·x`, with `W` never materialised:

1. `x ← x * su`
2. rotate each 128-block of `x` with `Hn`
3. `y ← M · x`
4. rotate each 128-block of `y` with `Hn`
5. `y ← y * sv`

**Four rotation stages per layer, not one per matmul.** Gate and up share both
the input vector and `su`, so their input rotation is computed once. The
stages are:

| Stage | What rotates | Width | Blocks |
| --- | --- | ---: | ---: |
| 1 | shared gate/up input | 2560 | 20 |
| 2 | fused gate/up output | 1280 | 10 |
| 3 | down input | 640 | 5 |
| 4 | down output | 2560 | 20 |

## Rotation vectors

Stored as plain `F16` GGUF tensors, one per role per layer, shared by all 512
experts of that layer (the HGN `i4r` sharing rule):

| GGUF tensor | f16 count | Replaces (HGN v2 name) |
| --- | ---: | --- |
| `blk.N.ffn_experts_rot.su` | 2560 | `layers.N.mlp.experts.su` |
| `blk.N.ffn_gate_exps_rot.sv` | 640 | `layers.N.mlp.experts.gate_up_proj.svh[0:640]` |
| `blk.N.ffn_up_exps_rot.sv` | 640 | `layers.N.mlp.experts.gate_up_proj.svh[640:1280]` |
| `blk.N.ffn_down_exps_rot.su` | 640 | `layers.N.mlp.experts.down_proj.suh` |
| `blk.N.ffn_down_exps_rot.sv` | 2560 | `layers.N.mlp.experts.down_proj.svh` |

Pairing rule — how a loader finds the vectors for a rotated tensor:

| Rotated tensor | `su` | `sv` |
| --- | --- | --- |
| `blk.N.ffn_gate_exps.weight` | `blk.N.ffn_experts_rot.su` | `blk.N.ffn_gate_exps_rot.sv` |
| `blk.N.ffn_up_exps.weight`   | `blk.N.ffn_experts_rot.su` | `blk.N.ffn_up_exps_rot.sv` |
| `blk.N.ffn_down_exps.weight` | `blk.N.ffn_down_exps_rot.su` | `blk.N.ffn_down_exps_rot.sv` |

In the source checkpoint every `su` and every `i4r` `sv` holds only ±1 (the
writer's choice, not a format requirement). The format accepts any f16, so
loaders must not assume ±1.

## Rotation marker

GGUF v3 per-tensor KV metadata is not writable with the pinned `gguf-py`, so
the marker is global KV (upgrade path: per-tensor `rot` keys once supported):

```
qwen4exp.experts.rotation        STRING  "hadamard128"
qwen4exp.experts.rotation_spec   STRING  "hgn-spec 1.1.0 STORAGE.md#i4r"
qwen4exp.experts.rotation_scale  STRING  "1/sqrt(128) per side, Sylvester"
```

A loader treats every `I4R128` tensor as rotated with the paired vectors iff
`qwen4exp.experts.rotation == "hadamard128"`. Absence, or any other value,
is a hard error — never silently treat rotated codes as plain. Silently
skipping the rotation produces plausible-looking weights and badly wrong
output.

## Transcode is lossless

HGN `i4r` → GGUF `I4R128` is a pure re-layout. Per expert matrix, per row of
`K` values:

1. HGN code plane: row `R` at `R*K/2`, byte `j` = value `2j` (low) | value
   `2j+1` (high). Unpack to sequential 4-bit values.
2. Re-pack each 128-value group in GGUF nibble order.
3. HGN scale plane: scale `g` of row `R` at `N*K/2 + 2*(R*K/128 + g)`; copy
   the f16 bits unchanged into block bytes 64..65.

No requantization, no rounding, no rescaling: codes and scale bits are
preserved exactly. Gate/up split: HGN fused rows `0..639` → gate,
`640..1279` → up (block-aligned at 640 = 5×128).

Because the transcode preserves bytes, the GGUF `I4R128` total must equal the
HGN `i4r` total. It does: 62.285 GB over the 48 trunk layers both ways. That
identity is asserted in `i4r128_reference_test`.

## What is implemented here

Core only, all CPU-testable:

- `GgmlType::kI4R128 = 64` and its `ToString` case
  (`src/core/gguf_reader.hpp`).
- `block_i4r128` (66 bytes, `#pragma pack`, `static_assert`ed) and
  `QuantizedBlockElements` → 128 (`src/core/quant/ggml_dequant.hpp`).
- Sizing: `QuantizedRowBytes` / `EncodedSizeBytes` cases
  (`src/core/quant/ggml_dequant.cpp`). This **is** the loader allowlist —
  `gguf_reader.cpp` accepts a type exactly when `EncodedSizeBytes` returns
  non-zero and `ne0 % block == 0`, so no separate allowlist edit was needed.
- `DequantizeI4R128` / `DotProductI4R128`, plus the `DequantizeRow` case.
- `src/core/quant/i4r_rotation.{hpp,cpp}`: `Fwht128`, `RotateBlock128`,
  `RotateBlocks`, `RotatedGemv`, and the `LayerRotationStages` description.
- Unified dispatch: `IsSupported` / `Dequantize` / `Dot` route `I4R128`
  (`src/core/quant/ggml_gemm.cpp`).
- `tests/core/i4r128_reference_test.cpp`, labels `cpu;core;quant`, 675
  checks: layout and sizing, the full 16-code offset-binary domain, GGUF
  nibble order, zero-scale, multi-block rows with independent scales, FWHT
  ordering against the explicit Sylvester definition, `Hn·Hn = I`, norm
  preservation, block tiling, the staged-matvec-vs-explicit-dense-`W`
  identity at three shapes, and dispatch routing.

`Dequantize`/`Dot` on `I4R128` yield the **rotated stored numbers**. Any
caller that needs the true weight must go through the rotation stages; the
headers say so at each entry point.

## What the GPU path still needs

Not written, and deliberately so — there is no ROCm toolchain in the box this
was authored in, and uncompiled HIP that nobody has run is a liability rather
than a head start.

1. MMVQ/MMQ kernels for `I4R128` under
   `src/models/qwen38_flash_next/kernels/rocm/mmq/`, following the existing
   `mmvq` / `qfn_mmq` structure.
2. The four rotation stages as kernels, with stage 1 hoisted out of the expert
   loop since all 512 experts of a layer share `su` and the input.
   `ds4_rocm_indexer.hip.hpp` (DeepSeek V4 Flash) is the in-tree precedent
   for a Walsh-Hadamard transform in a kernel.
3. Rotation-vector upload: the five f16 tensors per layer must reach the
   device alongside the expert weights, and the pairing table above has to be
   resolved at load time.
4. A GPU-side test that checks the kernel against `RotatedGemv` on the same
   inputs. The CPU reference is the oracle; do not re-derive expectations.

Cannot be decoded at load time: 62 GB of experts would roughly 4× the upload.
The kernels have to consume the rotated codes directly.

## What is NOT changed

Only the three routed-expert tensors per trunk layer change
(`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` → `I4R128`, plus the five
f16 vectors per layer). Shared experts, attention, hyper-connections, PLE,
embeddings, `lm_head`, MTP and every other tensor are copied byte-identical
from `UD-Q4_K_XL`. Any measured speed or quality delta is attributable to
the expert scheme alone.
