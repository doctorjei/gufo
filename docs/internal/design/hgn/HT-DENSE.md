# halogen v2 weights in Gufo: loading and the ht kernel

## Sources

- **Format:** the public hgn-spec v1.1.0 (MIT OR CC-BY-4.0), `STORAGE.md` and `TENSORS.md`.
- **Execution technique:** Peonist's halogen-flash engine, through the clean-room *ht dense decode spec*, a functional description released by the clean-room gate. No halogen code, binary or trace was read by the implementer. See [PROVENANCE.md](PROVENANCE.md).

## Files

| File | What |
|---|---|
| `tools/hgn2gguf/` | Lossless v2 HGN → GGUF converter, its type table (`TYPES.md`), the spec test vectors and the verifier |
| `src/core/quant/hgn_decode.{hpp,cpp}` | Size rules and CPU decoders for the new types: `HT4`, `HT3`, `Q6G64`, `FP8G`, `Q4C`, `Q8G64` |
| `src/core/quant/ht_dense.hpp`, `ht_dense_kernels.hip.cpp` | The in-place `HT4` matrix-vector kernel |
| `tests/core/hgn_decode_test.cpp` | CPU decoders against the spec's test vectors |
| `tests/core/ht_dense_test.hip` | GPU kernel against the CPU decoder, plus timing at the real shapes |

## Loading (`weights.cpp`, `device_model.cpp`, `engine.cpp`)

- **`HT4` dense matrices are kept in place.** This covers the attention and linear-attention projections, the indexer, the shared expert and the output head. The binder attaches the F16 vectors `<name>_rot.su` and `<name>_rot.sv`, and the upload widens them to F32.
  - Setting `GUFO_HT_DECODE_AT_LOAD=1` decodes them to F16 at load instead. That is the reference path for checking the kernel.
- **Other HGN stores are decoded on the host at bind time.** `Q6G64` (hyper-connections), `Q4C` (embedding, `ssm_alpha`/`ssm_beta`, shared-expert gate) and `Q8G64` (MTP dense) are decoded into the F16 or F32 format the role accepts, then uploaded from host buffers (`TensorRef::kHostShard`).
  - Stacked expert tensors in an HGN store, which means the MTP head's `Q4C` experts, are refused for now.
- **The fp8 n-gram table** (`FP8G`) is read row by row like the other table formats, with its global scale.
  - It can live in a separate GGUF that the trunk names in `qwen4exp.ple.table_file`. The engine opens that file from the trunk's directory.

## The ht kernel

The structure follows the functional spec.

**Launches and work split**
- There is one launch per matrix call, for up to 8 token rows. Larger batches loop in chunks of 8.
- Each workgroup has 256 lanes and owns one 128-row band plus a slice of the input dimension in 128-column blocks.
  - The host picks the split per matrix to give about 640 workgroups.
  - Example splits: `attn_qkv` 7, `ssm_out` 24, the output head 1.

**Input staging**
- Each workgroup stages its own slice: x·su, a 128-point Hadamard transform in registers (one wave per block), ×1/√128, rounded to f16 into shared memory.
- So the input rotation is repeated per workgroup instead of running as a launch of its own.

**Decoding**
- Lane l of a wave loads word l and word l−1 (mod 32) of each of the 8 row tiles of one column tile. Each such load instruction reads one whole 128-byte tile. The next column tile's words are loaded before the current one is decoded.
- The two words hold the 16-bit states of the lane's eight values: tile row l mod 16, columns 8(l/16)..+7.
- The codebook is computed in registers. For each value: multiply by `0x83DCD12D`, byte-sum with `v_sad_u8`, form `0x6400 | a`, then one packed f16 FMA with the spec constants.

**Arithmetic and reduction**
- Products use `v_dot2_f32_f16` into f32, with 8 rows × T tokens of accumulators per lane.
- Lanes l and l+16 share a row. The 8 waves add through shared memory.
- With a split input, the last workgroup of each band adds the partial sums in a fixed order, so the result is deterministic.
- The output side applies a Hadamard transform, ×1/√128 and sv, and writes F32.

**Not yet done (from the spec)**
- Grouping the projections that share an input into one launch. Today `qkv` and `gate`, or `q`, `k` and `v`, are separate launches.
- bf16 output.
- A fused SiLU input stage for the shared expert at large verify batches.
- MTP's draft-vocabulary table copies `lm_head` rows, which `HT4` cannot give. That needs its own handling before MTP runs with an `HT4` head.
