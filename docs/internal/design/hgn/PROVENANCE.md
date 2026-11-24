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
   process below:
   - the **i4r decode spec** (draft 1 plus addendum 1): batched routed-expert
     decode for i4r experts, implemented in
     `src/core/quant/i4r_fused_kernels.hip.cpp`;
   - the **ht dense decode spec** (draft 3): matrix-vector products for 4-bit
     ht weights, implemented in `src/core/quant/ht_dense_kernels.hip.cpp`.
3. **Gufo upstream.**

The q6g64 kernel (`src/core/quant/q6g64_dense_kernels.hip.cpp`) was designed
from the public format alone.

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
