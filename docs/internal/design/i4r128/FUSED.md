# I4R128 batched routed-expert decode

Source: `src/core/quant/i4r_fused_kernels.hip.cpp`, `src/core/quant/i4r_fused.hpp`.
Provenance: Peonist's halogen-flash technique, reimplemented clean-room from
the *i4r decode spec* (see [../hgn/PROVENANCE.md](../hgn/PROVENANCE.md)).
Wired in `Executor::MoeExperts` (rotated layers with `rot_split`) and
`DeviceModel::Upload` (load-time repack).

## Launches per MoE layer

| Step | Launches | Notes |
|---|---|---|
| Router logits + top-k | stock (dense + `RouterTopK`) | ids and weights stay on the device |
| Shared expert gate/up + SwiGLU | stock fused Q8_0 GEMV (+ input quantize) | main stream, before gate+up |
| Routed gate+up, all selected experts, plus shared down | 1 | second workgroup range = shared down |
| Routed down, all selected experts, combine, output rotation, gated shared add | 1 | writes the layer output |

No host round trip: the expert kernels read `s_.ids` / `s_.weights` on the
device, so decode is graph-capturable again (the unbatched path had to
force eager decode).

## Weight layout

Each expert tensor is repacked once at load: codes region
(`[experts][rows][blocks][64 B]`) then scales region
(`[experts][rows][blocks]` f16). Same size as the GGUF 66-byte blocks; done
in place through one temporary. The Q8_0 shared down projection is repacked
the same way (32 B codes, f16 scale per block).

## Work split

- 256-lane workgroups (8 waves of 32), 64 output rows each, 4 lanes per row.
  Lane `q` of a row loads bytes `16q..16q+15` of every 64-byte block: low
  nibbles are values `16q..`, high nibbles `64+16q..` of that block. All of a
  lane's 16-byte code loads are issued before any arithmetic.
- Gate+up: `slots x (2 ff / 64)` workgroups; the four workgroups of one
  (slot, 128-row tile) are adjacent (gate/up x two 64-row halves).
- Down: `tokens x (hidden/128) x used x 2` workgroups; the `2 x used`
  workgroups of one output tile are adjacent.
- Shared down: `tokens x (hidden/128) x (shared_ff/128)` workgroups, 128 rows
  x 128 inputs each, two lanes per row.

## Arithmetic

- Nibble to f16 exactly: `code | 0x6400` is `1024 + code`; one packed f16
  subtraction of 1032 gives `code - 8`. Pairs come out as (byte0, byte2) and
  (byte1, byte3), so the input is stored in LDS permuted `[x0, x2, x1, x3]` per
  group of four and each pair is one aligned half2.
- `v_dot2_f32_f16` into f32; the block's f16 scale multiplies the lane's
  block sum once.
- Input precision: by default the f16 input is split `x = hi + lo` (both f16)
  and each code pair is dotted with both, giving ~22-bit input precision for
  one extra dot2. `plain_f16 = 1` uses the single f16 input (the spec's
  variant); see the benchmark for its error and speed.

## Finishing (last arriver)

Every workgroup writes f32 partials, fences, and bumps an arrival counter;
the last one for a tile resets it and finishes the tile in one wave:

- gate+up: `g = sv_gate * H(gate)`, `u = sv_up * H(up)`,
  `d_in = H(su_down * silu(g) * u)`, written over the gate partials;
- down: `sum_e w_e * part_e`, then `H`, `* sv_down` once on the sum, plus
  `sigmoid(gate) * shared_out`, written to the layer output;
- shared down: sum of the slice partials into `shared_out`.

`H` is the orthonormal 128-point Walsh-Hadamard transform in registers: each
lane holds `tile[l + 32 j]`, strides 1..16 use `__shfl_xor`, 32 and 64 stay in
the lane.

## Test seams

- `rot_per_slot_for_test_`: the down finisher also leaves each slot's
  unweighted rotated output in `s_.down_e` (the unbatched path's layout), so
  `i4r_executor_parity_test` keeps its per-slot dense comparisons.
- `i4r_fused_bench`: real-geometry cross-check against the unbatched
  dispatch, shared-range check against a host double reference, and timing.
