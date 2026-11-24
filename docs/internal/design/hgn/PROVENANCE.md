# Provenance of the halogen-format work

This branch runs halogen's published Qwen3.8 Flash-Next v2 weights in Gufo:
the routed i4r experts, the dense ht trellis matrices, q6g64 and the rest. The
kernel techniques come from Peonist, LLC's **halogen-flash** inference engine.
They were reimplemented here through a clean-room process, so that no halogen
source code, binary content or trace reached the implementation.

## Sources the implementation used

1. **The public HGN format specification**: hgn-spec v1.1.0,
   <https://github.com/jtsylve/hgn-spec> (MIT OR CC-BY-4.0). The storage
   formats, the decoders in `src/core/quant/hgn_decode.cpp` and
   `tools/hgn2gguf/` come from it. The decoders are checked against its
   `TEST-VECTORS.md`.
2. **Two functional specs written for this work** under the clean-room
   process below. They are published at
   <https://github.com/doctorjei/i4r-expert-decode-halogen> (the i4r spec as
   `README.md`, the ht spec as `ht-dense-decode.md`):
   - the **i4r decode spec** (draft 1 plus addendum 1): batched routed-expert
     decode for i4r experts, implemented in
     `src/core/quant/i4r_fused_kernels.hip.cpp`;
   - the **ht dense decode spec** (draft 3): matrix-vector products for 4-bit
     ht weights, implemented in `src/core/quant/ht_dense_kernels.hip.cpp`.
3. **Gufo upstream.**

The q6g64 kernel (`src/core/quant/q6g64_dense_kernels.hip.cpp`) was designed
from the public format alone.

### What the specs did and did not cover

- **From the specs:** the batched routed-expert decode (three launches per
  MoE layer, device-side selection, last-arriving workgroup finishing, the
  nibble-to-f16 construction, the shared expert folded into the expert
  launches) and the ht dense decode (grouped launches, per-workgroup input
  rotation, band reduction with the output rotation, whole-tile loads with
  look-ahead, the arithmetic codebook). The ht spec's small-batch figures
  also motivated the separate verify-batch split plan and the 5- and
  6-token builds in `ht_dense_kernels.hip.cpp`; the plan itself was chosen
  by measurement.
- **Designed on this side, from public sources only:** sharing an expert's
  weights across the tokens of an MTP verify batch (the `*ShareKernel`
  launches); all of prefill for these formats, i.e. grouping a chunk's slots
  by expert, the matrix-core (WMMA) grouped expert kernels, and decoding ht
  and q6g64 matrices to f16 for a GEMM (`src/core/quant/hgn_prefill*`); the
  q6g64 kernels; and the hgn2gguf converter. The specs cover decode only.

## The clean-room process

- **Analysis:** a separate analysis agent studied halogen-flash 0.19 under
  its end-user licence. Section 3 of the licence permits reverse engineering
  and welcomes interoperability research. The agent wrote functional specs:
  prose, tables of numbers and data-flow descriptions only, with no code,
  pseudo-code mirroring the subject, disassembly, identifiers, strings or hex.
- **Gate:** a separate gatekeeper reviewed each spec until it held no copied
  material. Only then, with the project director's approval, did it release
  the spec to the implementation side.
- **Implementation:** the implementer worked only from the released specs and
  public sources. It never had access to the analysis side's material, or to
  halogen binaries, images or traces.

halogen's licence asks that derived source or kernels not be redistributed as
one's own work. Hence this note: the techniques are halogen's, and the code
is an independent implementation from the descriptions above.

## Weights

The published checkpoint files (`peonist-ai/halogen-qwen3.8-flash-next`) are
derivatives of Qwen3.8-Flash-Next. Their publisher states that the base
model's licence (Qwen Community License 1.0) governs them. `tools/hgn2gguf`
converts them losslessly; no weights are part of this repository.

"halogen" is a trademark of Peonist, LLC. This project is not affiliated with
or endorsed by Peonist.
