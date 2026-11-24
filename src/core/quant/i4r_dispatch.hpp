#ifndef GUFO_CORE_QUANT_I4R_DISPATCH_HPP_
#define GUFO_CORE_QUANT_I4R_DISPATCH_HPP_

#include <cstddef>
#include <cstdint>

#include "src/core/quant/i4r_rotation.hpp"

// Host-side orchestration for the rotated (I4R128) routed-expert path.
//
// The five rotation stages of i4r_rotation.hpp wrap a plain matmul on the
// stored codes M. This file sequences those stages across a whole trunk layer
// -- which buffer, which scale, which expert -- without naming any particular
// accelerator. The GPU wires the primitives below to the i4r_launch_*
// kernels; the CPU test wires them to the reference rotate/decode so the
// stage order, scale pairing, expert-slot mapping and block counts can be
// checked against RotatedGemv without a device.
//
// This is dispatch option (a): one single-token launch per (token, expert)
// slot. It is correct but launch-heavy; batched per-expert dispatch is a
// later step. No speed is claimed here.

namespace gufo::quant {

/// The primitives a rotated-expert dispatch runs on. Implementations own the
/// memory movement; the dispatch itself allocates nothing beyond the scratch
/// the caller hands it.
struct RotatedOps {
  virtual ~RotatedOps() = default;

  /// out[i] = Hn( (pre ? in[i]*pre[i] : in[i]) ) * (post ? post[i] : 1),
  /// applied block-wise over `count` elements. `count` must be a positive
  /// multiple of kI4rBlock. `pre`/`post` (when non-null) have `count`
  /// elements aligned with `in`.
  virtual void Rotate(const float* in, float* out, const float* pre_scale,
                      const float* post_scale, std::size_t count) = 0;

  /// y[0..n_out) = M * x for one expert's stored matrix. `M` points at the
  /// expert's raw I4R128 bytes (row stride n_blocks_in * 66). `x` has
  /// n_blocks_in * kI4rBlock elements, `y` has n_out.
  virtual void Matvec(const std::uint8_t* M, const float* x, float* y,
                      std::size_t n_out, std::size_t n_blocks_in) = 0;

  /// Y[t][o] = (M * X[t])[o] for one expert, `tokens` token rows. X is
  /// token-major [tokens x (n_blocks_in*kI4rBlock)], Y is [tokens x n_out].
  /// Used by the prefill phase; tokens >= 1.
  virtual void Gemm(const std::uint8_t* M, const float* X, float* Y,
                    std::size_t n_out, std::size_t n_in, int tokens) = 0;
};

/// The five per-layer rotation scales, widened to F32 by the loader. Each
/// points at a vector of the width named in the comment.
struct RotatedScales {
  const float* su_experts{nullptr};  ///< [hidden] shared gate/up input scale
  const float* sv_gate{nullptr};     ///< [ff] gate output scale
  const float* sv_up{nullptr};       ///< [ff] up output scale
  const float* su_down{nullptr};     ///< [ff] down input scale
  const float* sv_down{nullptr};     ///< [hidden] down output scale
};

/// Byte stride between consecutive experts of a raw I4R128 projection whose
/// output is `n_out` rows and whose input is `n_in` columns.
[[nodiscard]] constexpr std::size_t I4rExpertStride(std::size_t n_out,
                                                    std::size_t n_in) noexcept {
  return n_out * (n_in / kI4rBlock) * 66u;
}

/// Stage 1, hoisted once per layer: xr[t] = Hn(x[t] * su_experts) for each
/// of the `n_tokens` token rows. Every expert of a token reuses that token's
/// rotated row, which is the whole point of hoisting the input rotation.
/// `x`/`xr` are [n_tokens x hidden]; `hidden` must be a multiple of
/// kI4rBlock.
void RotateLayerInput(RotatedOps& ops, const float* x, const RotatedScales& s,
                      std::size_t hidden, std::size_t n_tokens, float* xr);

/// The gate/up side of one layer. For slot `sl` (token `t = sl / used`,
/// expert `e = ids[sl]`):
///
///   gate_out[sl] = sv_gate * Hn( M_gate[e] * xr[t] )
///   up_out[sl]   = sv_up   * Hn( M_up[e]   * xr[t] )
///
/// `gate`/`up` are the raw I4R128 expert tensors (expert stride
/// I4rExpertStride(ff, hidden)). `ids` has `n_slots` entries. The caller
/// applies SwiGLU to gate_out/up_out before the down side. `prefill` selects
/// the GEMM primitive over the matvec.
///
/// `scratch_gate`/`scratch_up` are one [ff] buffer each: the matmul writes
/// there and stage 2 rotates from that temp into the output, so no Rotate
/// call ever aliases its input with its output. See the alias contract at the
/// bottom of this header.
void RotatedGateUp(RotatedOps& ops, const std::uint8_t* gate,
                   const std::uint8_t* up, const RotatedScales& s,
                   const float* xr, const std::int32_t* ids, std::size_t hidden,
                   std::size_t ff, std::size_t n_slots, std::uint32_t used,
                   bool prefill, float* gate_out, float* up_out,
                   float* scratch_gate, float* scratch_up);

/// The down side of one layer. For slot `sl` (expert `e = ids[sl]`), with
/// `h` the SwiGLU-activated [n_slots x ff] input:
///
///   down_out[sl] = sv_down * Hn( M_down[e] * Hn(su_down * h[sl]) )
///
/// `scratch_in` is one [ff] buffer holding the stage-3 rotated down input;
/// `scratch_out` is one [hidden] buffer holding the matmul result before the
/// stage-4 rotate writes `down_out`. Both are reused per slot. `prefill`
/// selects the GEMM primitive over the matvec.
void RotatedDown(RotatedOps& ops, const std::uint8_t* down,
                 const RotatedScales& s, const float* h,
                 const std::int32_t* ids, std::size_t hidden, std::size_t ff,
                 std::size_t n_slots, std::uint32_t used, bool prefill,
                 float* down_out, float* scratch_in, float* scratch_out);

/// Float count of the rotated scratch above for a layer of `hidden` / `ff`
/// running `n_tokens` token rows: the [n_tokens x hidden] stage-1 output,
/// the gate and up pre-rotation temps plus the down-input rotation (three
/// [ff] buffers) and the down pre-rotation buffer ([hidden]). The temps are
/// reused across the slot loop, so they do not scale with the slot count.
[[nodiscard]] constexpr std::size_t RotatedScratchFloats(std::size_t hidden,
                                                         std::size_t ff,
                                                         std::size_t n_tokens) {
  return n_tokens * hidden + 3 * ff + hidden;
}

/// @name Contracts the caller must honour
///
/// Alias. Every Rotate issued here writes a buffer distinct from its input.
/// The accepted i4r_rotate_kernel declares both pointers `__restrict__`,
/// which promises the compiler they do not alias; calling it in place would
/// void that promise on the device. A CPU stand-in that copies through its
/// own temporary cannot establish the device contract, so the dispatch is
/// out-of-place by construction rather than by convention.
///
/// Expert ids. `ids` is read on the HOST: option (a) picks each slot's
/// expert weight pointer in a host loop. The caller must therefore hand the
/// dispatch an `ids` array that is (i) host-readable and (ii) already
/// reflects a COMPLETED router pass. The device-side `s_.ids` written by
/// RouterTopK satisfies neither on its own: it may live in hipMalloc memory
/// and the router is only queued, not finished. Executor::MoeExperts
/// downloads the ids to pinned host memory and synchronizes before
/// dispatching, and refuses to run while a graph capture is active, because
/// host-selected expert offsets would otherwise be frozen into the captured
/// kernel arguments and could never change on replay.
/// @{
/// @}

}  // namespace gufo::quant

#endif  // GUFO_CORE_QUANT_I4R_DISPATCH_HPP_
