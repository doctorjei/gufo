#ifndef GUFO_SRC_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_BACKEND_GEOMETRY_HPP_
#define GUFO_SRC_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_BACKEND_GEOMETRY_HPP_

// Host-only backend geometry contract for the Qwen3.8-Flash-Next ROCm path.
//
// This header is the single host-side authority for WHICH model geometries
// the ROCm backend can execute, and for the ordering rule every caller
// must honour: geometry is validated BEFORE the first device operation.
//
// Why this exists (point 85 offline review findings 1-3):
//  * The session-level GDN guard in Executor::CreateSession runs only
//    after Executor::Create has already done quantized-tier init, stream
//    and event creation and BLAS setup, and after DeviceModel::Upload has
//    allocated and copied weights. It correctly protects the session's own
//    state allocations, but it cannot deliver "no device allocation or
//    use before rejection". The gate here is the step that runs before
//    upload on the entry path.
//  * The base attention kernels support only specific geometries. The
//    fallback Attention() returns WITHOUT writing output when
//    head_dim != 256 or (heads / kv_heads) % 4 != 0, and the WMMA path
//    admits exactly 24 query / 2 KV heads at head_dim 256 with ratio 4.
//    A model whose attention falls outside both contracts silently skips
//    attention in the executor fallback instead of reporting an error;
//    it must be rejected before upload.
//
// Scope: GeometryScope::kGdnSeamOnly validates the dimensions a GDN-seam
// caller depends on (GDN head width and state sizing) and nothing else.
// GeometryScope::kFullForward additionally requires every full-attention
// layer's geometry to be supported by the fallback or WMMA path. The
// tiny test fixture is a SEAM fixture: its attention geometry (head_dim
// 64, 2 query / 1 KV head, ratio 2) is supported by NEITHER path, so it
// must never be presented as full-Forward coverage --
// ValidateBackendGeometry(kFullForward) rejects it by name, and the
// driver asserts that rejection.
//
// HIP-free by construction: plain arithmetic and std::string only, so the
// REAL validator and the REAL entry gate are unit-testable on a CPU-only
// host with trap/counter backends (tests/core/gdn_geometry_test.cpp). The
// HIP translation units call the same functions; source review confirms
// the wrappers invoke them, but CPU stubs alone do not compile or prove
// device-side wiring.

#include <cstdint>
#include <string>

#include "src/models/qwen38_flash_next/config.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/gdn_geometry.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// What the caller intends to execute. Decides which dimensions are
/// load-bearing, and therefore which must be supported before the first
/// device operation.
enum class GeometryScope {
  /// The caller will exercise only the GDN (linear-layer) path. GDN
  /// geometry is load-bearing; attention geometry is not executed and is
  /// not validated.
  kGdnSeamOnly,
  /// The caller will run full Forward, including full-attention layers.
  /// Every backend-dependent dimension must be supported.
  kFullForward,
};

/// The full-attention contract of the base ROCm backend. Values are tied
/// to the kernel constants by static_asserts in kernels.hip.cpp so the
/// contract cannot drift from the code it describes.
namespace attn_contract {

/// The fallback Attention() kernel is written for 256-wide heads only
/// (kernels.hip.cpp: `if (d != 256) return;` -- it returns WITHOUT
/// producing output for any other width).
inline constexpr std::uint32_t kFallbackHeadDim = 256;
/// The fallback launches only when (heads / kv_heads) % 4 == 0; other
/// groupings return without output.
inline constexpr std::uint32_t kFallbackGroupMultiple = 4;
/// The WMMA causal-attention path admits exactly this geometry
/// (kernels.hip.cpp kWmmaQueryHeads / kWmmaKvHeads / kWmmaHeadDim /
/// kWmmaRatio).
inline constexpr std::uint32_t kWmmaQueryHeads = 24;
inline constexpr std::uint32_t kWmmaKvHeads = 2;
inline constexpr std::uint32_t kWmmaHeadDim = 256;
inline constexpr std::uint32_t kWmmaRatio = 4;

}  // namespace attn_contract

/// True when the fallback Attention() kernel would actually run for this
/// geometry rather than returning without output. The compress ratio does
/// not constrain the fallback: a ratio other than 4 simply disables the
/// sparse-list path and the kernel runs dense.
///
/// REVIEW-V4 FINDING 7: a model with ZERO query heads is not a supported
/// attention geometry. Zero passes both remainder tests (0 % kv == 0 and
/// (0/kv) % 4 == 0), so without an explicit positive-head condition an
/// attention-bearing config with heads == 0 would validate and continue
/// toward device categories instead of being rejected first. The
/// positive-head condition is required here, not left to the callers.
[[nodiscard]] constexpr bool AttentionFallbackSupported(std::uint32_t head_dim,
                                                       std::uint32_t heads,
                                                       std::uint32_t kv_heads) {
  return head_dim == attn_contract::kFallbackHeadDim && heads > 0 &&
         kv_heads > 0 && heads % kv_heads == 0 &&
         (heads / kv_heads) % attn_contract::kFallbackGroupMultiple == 0;
}

/// True when the WMMA causal-attention path admits this geometry.
[[nodiscard]] constexpr bool AttentionWmmaSupported(std::uint32_t heads,
                                                   std::uint32_t kv_heads,
                                                   std::uint32_t head_dim,
                                                   std::uint32_t ratio) {
  return heads == attn_contract::kWmmaQueryHeads &&
         kv_heads == attn_contract::kWmmaKvHeads &&
         head_dim == attn_contract::kWmmaHeadDim &&
         ratio == attn_contract::kWmmaRatio;
}

/// True when some backend path can execute full attention for this
/// geometry.
[[nodiscard]] constexpr bool AttentionGeometrySupported(
    std::uint32_t head_dim, std::uint32_t heads, std::uint32_t kv_heads,
    std::uint32_t ratio) {
  return AttentionFallbackSupported(head_dim, heads, kv_heads) ||
         AttentionWmmaSupported(heads, kv_heads, head_dim, ratio);
}

/// True when the model has at least one GDN (linear-attention) layer.
[[nodiscard]] inline bool HasGdnLayer(const Config& c) {
  for (std::uint32_t il = 0; il < c.num_layers; ++il)
    if (c.IsLinearLayer(il)) return true;
  return false;
}

/// True when the model has at least one full-attention layer.
[[nodiscard]] inline bool HasFullAttentionLayer(const Config& c) {
  for (std::uint32_t il = 0; il < c.num_layers; ++il)
    if (!c.IsLinearLayer(il)) return true;
  return false;
}

/// The shared pre-upload validation. Returns true if and only if every
/// backend-dependent dimension that `scope` makes load-bearing is
/// supported. On failure, *err names the dimension and the supported
/// contract. This is the function the real entry path calls BEFORE any
/// device allocation, upload, stream, BLAS or launch, and the function
/// the CPU test calls directly -- one validator, no duplication.
[[nodiscard]] inline bool ValidateBackendGeometry(const Config& c,
                                                GeometryScope scope,
                                                std::string* err) {
  const bool has_gdn = HasGdnLayer(c);
  if (has_gdn && !GdnHeadDimSupported(c.ssm_head_dim)) {
    if (err != nullptr)
      *err = "GDN head_dim " + std::to_string(c.ssm_head_dim) +
             " is not supported by the ROCm kernels (compiled for " +
             std::to_string(kGdnSupportedHeadDim) + ")";
    return false;
  }
  if (has_gdn && !GdnStateSizedForKernel(c.ssm_num_v_heads, c.ssm_head_dim)) {
    if (err != nullptr)
      *err = "GDN host state sizing under-covers the kernel: shortfall " +
             std::to_string(GdnStateShortfallFloats(c.ssm_num_v_heads,
                                                  c.ssm_head_dim)) +
             " floats";
    return false;
  }
  if (scope == GeometryScope::kFullForward && HasFullAttentionLayer(c) &&
      !AttentionGeometrySupported(c.head_dim, c.num_heads, c.num_kv_heads,
                                  c.compress_ratio)) {
    if (err != nullptr)
      *err = "full-attention geometry (head_dim " + std::to_string(c.head_dim) +
             ", " + std::to_string(c.num_heads) + " query / " +
             std::to_string(c.num_kv_heads) + " KV heads, ratio " +
             std::to_string(c.compress_ratio) +
             ") is supported by neither the fallback (head_dim 256 and a "
             "query/KV multiple of 4) nor WMMA (24/2/256/4)";
    return false;
  }
  return true;
}

/// The entry seam shared by the real driver and the CPU test: validate,
/// and ONLY on success run the continuation. `ops` models the first work
/// the caller would do next; a CPU test injects a trap/counter backend
/// and asserts that a rejected geometry touches none of it and that an
/// accepted geometry reaches the continuation exactly once. Mutating
/// this wiring -- dropping the validation call, inverting it, or running
/// the continuation before validating -- must fail the CPU test; that is
/// the point of the seam. The real driver instantiates this with a marker
/// continuation and calls it before DeviceModel::Upload.
template <class DeviceOps>
[[nodiscard]] bool RunPreUploadGeometryGate(const Config& c, GeometryScope scope,
                                           DeviceOps& ops, std::string* err) {
  if (!ValidateBackendGeometry(c, scope, err)) {
    return false;  // no continuation, and therefore no device work
  }
  ops.AfterGeometryGate();  // deliberately harmless host continuation
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_SRC_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_BACKEND_GEOMETRY_HPP_
