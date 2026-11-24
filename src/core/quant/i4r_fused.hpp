#ifndef GUFO_CORE_QUANT_I4R_FUSED_HPP_
#define GUFO_CORE_QUANT_I4R_FUSED_HPP_

// Batched I4R128 routed-expert decode: two expert launches per MoE layer
// (gate+up for all selected experts; down for all selected experts with the
// weighted combine). Defined in i4r_fused_kernels.hip.cpp.
//
// Weight layout: every expert tensor passed here has been repacked by
// i4r_repack_split into a codes region ([experts][rows][blocks][64 B]) then a
// scales region ([experts][rows][blocks] f16), same total size as the GGUF
// interleaved 66-byte blocks.
//
// Expert ids and weights are read on the device. Nothing here synchronises
// with the host, so both launches are graph-capturable.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

extern "C" {

struct I4rGateUpArgs {
  const std::uint8_t* gate;  ///< repacked [n_experts x ff x hidden]
  const std::uint8_t* up;    ///< repacked [n_experts x ff x hidden]
  std::size_t n_experts;
  const float* x;            ///< [n_tokens x hidden], unrotated layer input
  const std::int32_t* ids;   ///< [n_slots] device expert ids
  const float* su_experts;   ///< [hidden]
  const float* sv_gate;      ///< [ff]
  const float* sv_up;        ///< [ff]
  const float* su_down;      ///< [ff]
  float* gate_part;  ///< [n_slots x ff]; on exit holds the down input
  float* up_part;    ///< [n_slots x ff]; scratch
  unsigned* counters;  ///< [n_slots x ff/128], zero on entry and on exit
  int hidden;
  int ff;
  int used;  ///< experts per token
  /// 0: input staged as an f16 hi/lo split (default, ~f32 accuracy).
  /// 1: plain f16 input (one dot2 per code pair).
  int plain_f16;
  /// Optional second workgroup range (addendum 1): the shared expert's down
  /// projection, Q8_0 repacked by i4r_repack_q8_split, [hidden x shared_ff].
  /// Split into 128-row output tiles x 128-value input slices; the last
  /// slice to arrive at a tile sums the partials into shared_out. Null
  /// shared_down disables the range.
  const std::uint8_t* shared_down;
  const float* shared_in;   ///< [n_tokens x shared_ff] SwiGLU output
  float* shared_part;       ///< [n_tokens x shared_ff/128 x hidden] scratch
  float* shared_out;        ///< [n_tokens x hidden]
  unsigned* shared_counters;  ///< [n_tokens x hidden/128], zero in and out
  int shared_ff;
  int n_tokens;
  /// Optional prefill grouping (i4r_group_slots): when group_offsets is set,
  /// the launch runs one set of workgroups per expert over its slots.
  const std::int32_t* group_offsets;  ///< [n_experts + 1]
  const std::int32_t* group_slots;    ///< [n_slots], slots ordered by expert
  unsigned* group_counters;           ///< [n_experts x ff/128], zero in and out
};

struct I4rDownArgs {
  const std::uint8_t* down;  ///< repacked [n_experts x hidden x ff]
  std::size_t n_experts;
  const float* down_in;      ///< [n_slots x ff] (gate+up's gate_part)
  const std::int32_t* ids;   ///< [n_slots]
  const float* weights;      ///< [n_slots] combine weights
  const float* sv_down;      ///< [hidden]
  const float* shared_out;   ///< [n_tokens x hidden] or null
  const float* shared_gate;  ///< pre-sigmoid gate, token t at t*stride
  std::uint32_t shared_gate_stride;
  float* down_part;          ///< [n_slots x hidden] scratch
  float* out;                ///< [n_tokens x hidden]
  unsigned* counters;        ///< [n_tokens x hidden/128], zero on entry/exit
  int hidden;
  int ff;
  int used;
  /// Test seam: also leave each slot's unweighted rotated output in
  /// down_part (the [slots x hidden] layout of the unfused path).
  int per_slot_out;
  int plain_f16;  ///< as in I4rGateUpArgs
  const std::int32_t* group_offsets;  ///< as in I4rGateUpArgs (same grouping)
  const std::int32_t* group_slots;
};

/// Launches gate+up for n_slots = n_tokens * used slots. False when the
/// geometry has no compiled instantiation (hidden/128 not in the list).
bool i4r_fused_gate_up(const I4rGateUpArgs* args, int n_slots,
                       hipStream_t stream);

/// Launches down + combine for n_tokens tokens.
bool i4r_fused_down(const I4rDownArgs* args, int n_tokens, hipStream_t stream);

/// Groups a batch's slots by expert: offsets[e]..offsets[e+1] index the
/// slots of expert e in `slots`. cursor is [n_experts] scratch.
bool i4r_group_slots(const std::int32_t* ids, int n_slots, int n_experts,
                     std::int32_t* offsets, std::int32_t* cursor, std::int32_t* slots,
                     hipStream_t stream);

/// Repacks n_blocks Q8_0 blocks (34 bytes: f16 scale + 32 int8) into a codes
/// region (32 bytes per block) then a scales region (f16 per block).
bool i4r_repack_q8_split(const std::uint8_t* src, std::uint8_t* dst,
                         std::size_t n_blocks, hipStream_t stream);

/// Repacks n_blocks interleaved 66-byte blocks at src into the split layout
/// at dst (distinct buffers of n_blocks * 66 bytes).
bool i4r_repack_split(const std::uint8_t* src, std::uint8_t* dst,
                      std::size_t n_blocks, hipStream_t stream);

}  // extern "C"

/// Batches that use the grouped (prefill) launches: wider than the verify
/// batches the shared-expert kernels cover.
[[nodiscard]] constexpr bool I4rGroupedBatch(std::uint32_t n_tokens,
                                             std::uint32_t used) noexcept {
  return n_tokens > 32 || n_tokens * used > 256;
}

/// Whether the fused kernels have an instantiation for this geometry.
[[nodiscard]] constexpr bool I4rFusedSupports(std::uint32_t hidden,
                                              std::uint32_t ff) noexcept {
  if (hidden % 128 != 0 || ff % 128 != 0) return false;
  const std::uint32_t nh = hidden / 128, nf = ff / 128;
  const bool h = nh == 1 || nh == 2 || nh == 4 || nh == 5 || nh == 8 ||
                 nh == 16 || nh == 20 || nh == 32;
  const bool f = nf == 1 || nf == 2 || nf == 4 || nf == 5 || nf == 8 ||
                 nf == 10 || nf == 16;
  return h && f;
}

#endif  // GUFO_CORE_QUANT_I4R_FUSED_HPP_
