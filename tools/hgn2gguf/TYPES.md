# GGUF types for halogen's published stores

`hgn2gguf.py` writes the halogen Flash-Next v2 checkpoint as a GGUF that keeps
every quantized payload byte for byte. The stores have no upstream GGML type,
so they get new type ids, next to Gufo's `I4R128` (64). All ids sit far above
upstream `GGML_TYPE_COUNT`. Each one appears only in the tensor-info type
field and in a loader's sizing switch, so renumbering is cheap.

The byte layouts are the ones in the public hgn-spec v1.1.0 `STORAGE.md`
(MIT OR CC-BY-4.0). This file only fixes how they sit in GGUF.

| Id | Name | hgn-spec store | Payload | Size for GGUF dims `[K, N]` |
|---:|---|---|---|---|
| 64 | `I4R128` | `i4r` (re-laid out) | see `docs/internal/design/i4r128/RFC.md` | `K·N·66/128` |
| 65 | `HT4` | `ht`, variant `0x1208` | unchanged | `N·K/2` |
| 66 | `HT3` | `ht`, variant `0x1206` | unchanged (not in v2; ht43 experts) | `N·K·3/8` |
| 67 | `Q6G64` | `q6g64`, variant 64 | unchanged | `N·pad(13K/16, 16)` |
| 68 | `FP8G` | `fp8g` | unchanged, including its padding and trailing `f32` scale | `pad(160N, 64) + 4` (K = 160) |
| 69 | `Q4C` | `q4c`, variant 2 (planar, learned codebook) | unchanged | `64 + pad(N·K/2, 64) + N·pad(K/16, 16)` |
| 70 | `Q8G64` | `q8g64` | unchanged (MTP head, not yet written) | `N·(K + K/16)` |

`N` is the product of every GGUF dimension after the first (`ne1·ne2·…`).
A 1-D tensor of one of these types has `N = 1`.

## Conventions

- **Alignment.** The file sets `general.alignment = 64`, so every payload starts
  at a 64-byte boundary, as in HGN. `Q6G64` and `Q8G64` need 16 bytes, and
  `Q4C` places its planes at 64-byte offsets from the payload start.
- **Rotated matrices** (`HT4`, `HT3`, `I4R128`) hold `M`, the stored numbers in
  the 128-point Hadamard basis:
  `W = diag(sv) · Hn · M · Hn · diag(su)`, `Hn = H128/√128` (STORAGE.md, "The
  128-point rotation").
  - The vectors are F16 tensors. For a dense tensor `X.weight`, they are
    `X_rot.su` (K values) and `X_rot.sv` (N values).
  - For the experts, the names are as in the I4R128 RFC: `blk.N.ffn_experts_rot.su`, `blk.N.ffn_{gate,up,down}_exps_rot.sv` and `blk.N.ffn_down_exps_rot.su`.
  - Markers: `qwen4exp.dense.rotation = "hadamard128"` and `qwen4exp.experts.rotation = "hadamard128"`.
  - A loader must refuse a rotated type when its marker is missing.
- **`ht` has no scales.** Each row's magnitude is in `sv`, so `sv` is not
  ±1. The `su` vectors and the `i4r` `sv` vectors are ±1 in the published
  file, but a loader must not assume that.
- **`FP8G`** is the per-layer n-gram embedding table (`per_layer_token_embd`).
  `value = fp8(code) × scale`, using the spec's fp8 encoding (bias 7,
  `0x7F` = 480). The scale is the `f32` at byte offset `pad(160N, 64)` of the
  payload.
- **`Q4C`**: `value = scale(row, k/32) × codebook[code]`. The codebook is the
  first 64 bytes of the payload (16 `f32` levels). Codes use the HGN nibble
  order: byte `j` holds value `2j` in its low nibble and `2j+1` in its high one.

## What the converter changes, and what it does not

- **Unchanged bytes:** every `ht`, `q6g64`, `q4c`, `fp8g` and rotation-vector
  payload, except for whole-unit moves:
  - DeltaNet value heads go from HGN's grouped order back to GGUF's tiled
    order, GGUF head `g` taking HGN head `h` with `g = h/3 + 16·(h mod 3)`.
    That moves 128-row bands of `attn_qkv` (rows 4096 and up) and
    `attn_gate`, 128-column blocks of `ssm_out`, rows of `ssm_alpha` and
    `ssm_beta`, and the matching `sv` and `su` entries.
  - `index_qk_proj` is split into `indexer.q_proj` (bands 0–3) and
    `indexer.k_proj` (band 4).
- **Re-laid out, losslessly:** `i4r` → `I4R128`. Nibbles go into GGUF order,
  and the gate and up halves become separate tensors.
- **Converted values:**
  - bf16 norm gains become F32 with +1 (GGUF convention).
  - `A_log` becomes `ssm_a = −exp(A_log)` in F32.
  - Other bf16 tensors become F32 exactly, except `ple_key` and `ple_value`,
    which stay BF16 bytes.
- **Not taken from v2:** all metadata (tokenizer, hyper-parameters, PLE
  arrays) comes from the stock UD-Q4_K_XL GGUF. The converter checks that v2's
  three PLE index vectors equal the stock metadata arrays.
- **Not written yet:** the MTP head (v2's `mtp.*`, stored as `q4c`, `q8g64`
  and bf16) as Gufo's MTP sidecar.
