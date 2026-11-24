#ifndef GUFO_SRC_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_GDN_GEOMETRY_HPP_
#define GUFO_SRC_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_GDN_GEOMETRY_HPP_

// Host-only geometry contract for the Gated Delta Net (GDN) ROCm path.
//
// The GDN recurrence kernels carry the head dimension as a COMPILE-TIME
// constant (kGdnDim in kernels.hip.cpp). GatedDeltaNet's runtime `d` --
// the model's ssm_head_dim -- is never passed to the kernel, so a caller
// whose runtime head dimension differs from the compile-time one makes the
// kernel index state with d = kGdnDim against a buffer the host sized from
// ssm_head_dim. For a reduced head_dim the kernel addresses more than the
// host reserved: at head_dim 64 it reads and writes four times the state
// the host allocated, running past the end of every linear layer's state
// buffer. The batch entry point is narrower still: it takes no head-dimension
// argument at all and derives its channel stride from kGdnDim, so a
// reduced-geometry config silently mis-sizes every GDN buffer it touches.
//
// Production refuses reduced geometry at load time (Config::FromGguf), but
// the test seam (ModelWeights::BindWithConfig) copies a caller-supplied
// config verbatim, so the load-time gate does not cover it. The checks here
// close that gap at the executor and at the kernel wrapper.
//
// This header is deliberately free of HIP and of any model dependency so the
// predicate and the sizing arithmetic can be unit-tested on a CPU-only host
// (tests/core/gdn_geometry_test.cpp). kernels.hip.cpp ties kGdnDim to
// kGdnSupportedHeadDim with a static_assert, so the two cannot drift apart.

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next::rocm {

/// The only GDN head dimension the ROCm kernels are compiled to support.
/// Mirrors kGdnDim in kernels.hip.cpp; a static_assert there pins it.
inline constexpr std::uint32_t kGdnSupportedHeadDim = 128;

/// True when `head_dim` can be routed to the ROCm GDN path at all. Any
/// other value is reduced geometry the kernels cannot express and must be
/// rejected before any launch.
[[nodiscard]] constexpr bool GdnHeadDimSupported(std::uint32_t head_dim) noexcept {
  return head_dim == kGdnSupportedHeadDim;
}

/// State floats the host allocates per linear layer, from the runtime head
/// dimension: v_heads * head_dim^2 (executor.cpp CreateSession).
[[nodiscard]] constexpr std::size_t GdnHostStateFloats(std::uint32_t v_heads,
                                                      std::uint32_t head_dim) {
  return static_cast<std::size_t>(v_heads) * head_dim * head_dim;
}

/// State floats the kernel addresses per linear layer, using its
/// compile-time d: v_heads * kGdnSupportedHeadDim^2.
[[nodiscard]] constexpr std::size_t GdnKernelStateFloats(std::uint32_t v_heads) {
  return static_cast<std::size_t>(v_heads) * kGdnSupportedHeadDim *
         kGdnSupportedHeadDim;
}

/// True when the host sizing covers what the kernel addresses. Stated
/// separately from GdnHeadDimSupported so a caller can report the shortfall
/// magnitude (GdnStateShortfallFloats) rather than only the verdict.
[[nodiscard]] constexpr bool GdnStateSizedForKernel(std::uint32_t v_heads,
                                                   std::uint32_t head_dim) {
  return GdnHostStateFloats(v_heads, head_dim) >= GdnKernelStateFloats(v_heads);
}

/// Floats by which the kernel's addressing exceeds the host allocation
/// (0 when the geometry is supported).
[[nodiscard]] constexpr std::size_t GdnStateShortfallFloats(std::uint32_t v_heads,
                                                           std::uint32_t head_dim) {
  const std::size_t need = GdnKernelStateFloats(v_heads);
  const std::size_t have = GdnHostStateFloats(v_heads, head_dim);
  return need > have ? need - have : 0;
}

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_SRC_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_GDN_GEOMETRY_HPP_
