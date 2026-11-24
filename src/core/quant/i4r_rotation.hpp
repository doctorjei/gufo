#ifndef GUFO_CORE_QUANT_I4R_ROTATION_HPP_
#define GUFO_CORE_QUANT_I4R_ROTATION_HPP_

#include <cstddef>

namespace gufo::quant {

// Rotation math for I4R128 (rotated 4-bit expert codes).
//
// An I4R128 tensor stores M, the weight matrix expressed in a 128-point
// Hadamard-rotated basis. The true weight a dense matmul would multiply is
//
//     W = diag(sv) * Hn * M * Hn * diag(su)        Hn = H128 / sqrt(128)
//
// with H128 the Sylvester-Hadamard matrix, h(i, j) = (-1)^popcount(i & j).
// Hn is symmetric and orthonormal, so Hn * Hn = I: the transform is its own
// inverse, which is what lets the same kernel serve both rotation sides.
//
// The products are block-diagonal: `M * Hn` transforms blocks of 128 along
// the input axis, `Hn * M` transforms blocks of 128 along the output axis.
// Never materialise W -- the point of the format is that W is not stored.
//
// Instead the matvec y = W x runs as five stages, which is the form both the
// CPU reference here and the GPU kernels implement:
//
//   1. x  <- x * su                     (scale input)
//   2. x  <- Hn applied to each 128-block of x
//   3. y  <- M * x                      (plain matmul on decoded codes)
//   4. y  <- Hn applied to each 128-block of y
//   5. y  <- y * sv                     (scale output)
//
// su/sv are per-layer f16 tensors shared by every expert of the layer, so
// stage 1/2 of a layer's input is computed once and reused by all experts.

/// Elements per rotation block. Also the I4R128 quantization group size.
inline constexpr std::size_t kI4rBlock = 128;

/// 1 / sqrt(128), the orthonormal scaling of Hn.
inline constexpr float kI4rInvSqrtBlock = 0.08838834764831845F;

/// In-place unnormalised Walsh-Hadamard transform of `kI4rBlock` floats using
/// the Sylvester (natural / Hadamard) ordering, i.e. multiplies by H128 with
/// h(i, j) = (-1)^popcount(i & j). The butterfly schedule below is exactly
/// that recursive construction, so no bit-reversal or reordering step is
/// needed. Not normalised: multiply by kI4rInvSqrtBlock for Hn.
void Fwht128(float* v) noexcept;

/// Applies the orthonormal Hn to one 128-element block, in place.
void RotateBlock128(float* v) noexcept;

/// Applies RotateBlock128 independently to each consecutive block of
/// `kI4rBlock` elements in `data[0 .. count)`. `count` must be a multiple of
/// `kI4rBlock`.
void RotateBlocks(float* data, std::size_t count) noexcept;

/// True when `count` can be tiled by whole rotation blocks.
[[nodiscard]] constexpr bool IsRotationAligned(std::size_t count) noexcept {
  return count != 0 && count % kI4rBlock == 0;
}

/// CPU reference for one rotated expert matvec, the five stages above.
///
///   y = sv * Hn( M * Hn( su * x ) )
///
/// `m` is the decoded stored matrix, row-major [n_out x n_in]. `x` has n_in
/// elements, `su` n_in, `sv` n_out, `y` n_out. Every axis must be a multiple
/// of kI4rBlock. Allocates one input-sized scratch buffer; intended as the
/// oracle for the GPU kernels and for the CPU fallback path.
void RotatedGemv(const float* m, std::size_t n_out, std::size_t n_in,
                 const float* x, const float* su, const float* sv, float* y);

/// The four rotation stages one trunk layer performs on its routed-expert path.
///
/// Gate and up share both the input vector and `su`, so their input rotation is
/// a single stage; their outputs rotate as one fused 1280-wide vector. Down
/// then rotates its own input and output. That is four stages per layer, not
/// one per matmul.
struct LayerRotationStages {
  std::size_t
      expert_input;  ///< stage 1: shared gate/up input (2560 = 20 blocks)
  std::size_t
      gate_up_output;      ///< stage 2: fused gate/up output (1280 = 10 blocks)
  std::size_t down_input;  ///< stage 3: down input (640 = 5 blocks)
  std::size_t down_output;  ///< stage 4: down output (2560 = 20 blocks)
};

/// The stage widths for the Flash-Next trunk layer shape, for cross-checking
/// against the model config at startup.
[[nodiscard]] constexpr LayerRotationStages
Qwen38FlashNextLayerStages() noexcept {
  return LayerRotationStages{2560, 1280, 640, 2560};
}

}  // namespace gufo::quant

#endif  // GUFO_CORE_QUANT_I4R_ROTATION_HPP_
