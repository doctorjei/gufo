#ifndef GUFO_SRC_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_GDN_ENTRY_ORCHESTRATION_HPP_
#define GUFO_SRC_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_GDN_ENTRY_ORCHESTRATION_HPP_

// Shared, HIP-free ENTRY ORCHESTRATION for the Qwen3.8-Flash-Next ROCm path
// (point 85 geometry correction v3; offline review v2 findings 1, 3, 4).
//
// WHY THIS EXISTS
// v2 put the geometry gate in the right header but the wrong place in the
// sequence: the real driver still called hipGetDeviceProperties BEFORE the
// gate (finding 4), and the tested "continuation" was a bare marker while
// the real upload / Executor::Create / full Forward remained separate
// statements after the template returned (finding 3). The seam-only label
// therefore did not keep full Forward off the execution path (finding 1).
//
// v3 moves the ACTUAL entry orchestration behind this shared injectable
// boundary. The orchestration owns the whole ordered sequence:
//
//   1. ValidateBackendGeometry(scope)          -- host-only, FIRST
//   2. (seam only) restricted-op availability -- still host-only; if the
//        restricted operation cannot be expressed with the current APIs,
//        the entry STOPS here, before the first device operation, and
//        reports the missing API (see kMissingRestrictedApiMessage).
//   3. ops.DeviceQuery()                      -- the FIRST HIP call
//   4. ops.Allocate() / ops.Upload() / ops.StreamEvent() / ops.BlasInit()
//   5. the scope's tail:
//        seam  -> ops.RestrictedLiveness()    (never Forward, never attention)
//        MoE   -> ops.MoeRestrictedWork()    (per-layer dispatch seam, no Forward)
//        full  -> ops.FullForward()          (only reachable under kFullForward)
//
// The real driver instantiates this with ops that perform the real HIP /
// executor work. The CPU test instantiates the SAME templates with
// trap/counter ops and asserts, per case: rejection produces ZERO effects
// in every category and a named error; the valid restricted case reaches
// exactly one harmless host continuation while full-Forward and attention
// counters stay zero; and the first recorded effect is the device query,
// never a device step that ran before validation.
//
// SEAM INCAPABILITY (finding 1, hard form)
// The seam entries are templates whose ops interface exposes NO
// FullForward / attention capability at all, and a static_assert rejects
// any ops type that does. A seam wired to a full-forward-capable object
// does not compile. The only place FullForward is ever called is
// RunGatedFullEntry, reachable only under GeometryScope::kFullForward,
// which the validator gates on the full-attention contract.
//
// WHAT THIS DOES NOT PROVE
// CPU injection cannot by itself prove compiled HIP wiring: that the real
// ops methods call the real HIP APIs in the real order is a SOURCE-REVIEW
// obligation on the HIP translation units (the driver), not something a
// trap backend can establish. This header keeps that obligation explicit.

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "src/models/qwen38_flash_next/config.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/backend_geometry.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

/// The documented reason the GDN seam cannot proceed to device work with
/// the current executor API. Executor::Forward is the only public entry
/// that runs model layers, and it necessarily visits every layer --
/// including full-attention layers the seam-only contract excludes. There
/// is no GDN-only execution entry (no Executor::RunGdnLayer or
/// equivalent), so a genuinely restricted GDN device operation cannot
/// be expressed. Per the dispatch: stop the seam path before device work
/// and document the missing API.
inline constexpr const char* kMissingRestrictedApiMessage =
    "MISSING API: the executor exposes no GDN-only restricted execution "
    "entry; Forward is the only layer-running API and it necessarily "
    "enters full attention, so the seam-only path stops before the first "
    "device operation";

/// Outcome of a gated entry attempt. Every non-completion outcome means
/// NO device work happened on that path.
enum class EntryOutcome {
  kRejectedGeometry,       ///< validation failed before any effect
  kStoppedMissingApi,      ///< seam restricted op unavailable; stopped pre-device
  kNotApplicable,          ///< seam scope but the model has no GDN layer to exercise
  kNoDevice,              ///< validation passed; DeviceQuery reported no device
  kCompletedRestricted,   ///< seam tail ran (restricted liveness)
  kCompletedMoeRestricted,///< MoE seam tail ran (per-layer dispatch, no Forward)
  kCompletedFull,         ///< full-forward tail ran
};

/// True when `ops` exposes a full-Forward capability. Used to make the
/// seam entries REJECT such ops at compile time.
template <class T, class = void>
struct HasFullForward : std::false_type {};
template <class T>
struct HasFullForward<
    T, std::void_t<decltype(std::declval<T&>().FullForward())>>
    : std::true_type {};

/// True when `ops` exposes an attention-entry capability (same reasoning).
template <class T, class = void>
struct HasAttentionEntry : std::false_type {};
template <class T>
struct HasAttentionEntry<
    T, std::void_t<decltype(std::declval<T&>().AttentionEntry())>>
    : std::true_type {};

// ---------------------------------------------------------------------------
// Seam entry (GDN path). Incapable of full Forward by interface.
//
// SeamOps contract (exactly these members; NO FullForward, NO
// AttentionEntry -- enforced by static_assert below):
//   bool RestrictedOpAvailable()   // can the restricted GDN op be expressed?
//   bool DeviceQuery()             // first HIP call; false = no device
//   void Allocate()
//   void Upload()
//   void StreamEvent()
//   void BlasInit()
//   void RestrictedLiveness()      // the restricted continuation
// ---------------------------------------------------------------------------
template <class SeamOps>
[[nodiscard]] EntryOutcome RunGatedSeamEntry(const Config& c, SeamOps& ops,
                                            std::string* err) {
  static_assert(!HasFullForward<SeamOps>::value,
                "seam entry must not be wired to full-forward-capable ops");
  static_assert(!HasAttentionEntry<SeamOps>::value,
                "seam entry must not be wired to attention-capable ops");

  // 1. Validation FIRST -- host-only, before any device effect.
  if (!ValidateBackendGeometry(c, GeometryScope::kGdnSeamOnly, err))
    return EntryOutcome::kRejectedGeometry;

  // 2. Applicability: a seam with no GDN layer has nothing to exercise.
  if (!HasGdnLayer(c)) {
    if (err != nullptr)
      *err = "seam scope but the model has no GDN (linear) layer to exercise";
    return EntryOutcome::kNotApplicable;
  }

  // 3. Restricted-op availability BEFORE the first device operation.
  //    With the current API this is false for the real driver: see
  //    kMissingRestrictedApiMessage.
  if (!ops.RestrictedOpAvailable()) {
    if (err != nullptr) *err = kMissingRestrictedApiMessage;
    return EntryOutcome::kStoppedMissingApi;
  }

  // 4. Device sequence -- every step gated behind validation.
  if (!ops.DeviceQuery()) {
    if (err != nullptr) *err = "no HIP device available";
    return EntryOutcome::kNoDevice;
  }
  ops.Allocate();
  ops.Upload();
  ops.StreamEvent();
  ops.BlasInit();

  // 5. Restricted tail. NOT Forward. NOT attention.
  ops.RestrictedLiveness();
  return EntryOutcome::kCompletedRestricted;
}

// ---------------------------------------------------------------------------
// MoE rotated-expert seam entry (the v7 i4r seam). Also incapable of full
// Forward by interface. Its restricted tail is the per-layer rotated
// dispatch (Executor::MoeExperts through the test access seam), which
// does not enter Forward or attention.
//
// MoeSeamOps contract: same device members as SeamOps, plus
//   void MoeRestrictedWork()
// and NO FullForward / AttentionEntry.
// ---------------------------------------------------------------------------
template <class MoeSeamOps>
[[nodiscard]] EntryOutcome RunGatedMoeSeamEntry(const Config& c,
                                               MoeSeamOps& ops,
                                               std::string* err) {
  static_assert(!HasFullForward<MoeSeamOps>::value,
                "MoE seam entry must not be wired to full-forward-capable ops");
  static_assert(!HasAttentionEntry<MoeSeamOps>::value,
                "MoE seam entry must not be wired to attention-capable ops");

  // The fixture is GDN-bearing; entry discipline validates its GDN
  // geometry before ANY device work even when the tail is the MoE seam
  // (defense in depth: a config that cannot pass GDN validation must not
  // reach the device at all). Attention is not load-bearing here.
  if (!ValidateBackendGeometry(c, GeometryScope::kGdnSeamOnly, err))
    return EntryOutcome::kRejectedGeometry;

  if (!ops.DeviceQuery()) {
    if (err != nullptr) *err = "no HIP device available";
    return EntryOutcome::kNoDevice;
  }
  ops.Allocate();
  ops.Upload();
  ops.StreamEvent();
  ops.BlasInit();
  ops.MoeRestrictedWork();
  return EntryOutcome::kCompletedMoeRestricted;
}

// ---------------------------------------------------------------------------
// Full-forward entry. The ONLY place FullForward is ever called, and only
// under kFullForward scope, which the validator gates on the
// full-attention contract.
//
// FullOps contract: the device members plus void FullForward().
// ---------------------------------------------------------------------------
template <class FullOps>
[[nodiscard]] EntryOutcome RunGatedFullEntry(const Config& c, FullOps& ops,
                                            std::string* err) {
  if (!ValidateBackendGeometry(c, GeometryScope::kFullForward, err))
    return EntryOutcome::kRejectedGeometry;

  if (!ops.DeviceQuery()) {
    if (err != nullptr) *err = "no HIP device available";
    return EntryOutcome::kNoDevice;
  }
  ops.Allocate();
  ops.Upload();
  ops.StreamEvent();
  ops.BlasInit();
  ops.FullForward();
  return EntryOutcome::kCompletedFull;
}

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_SRC_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_GDN_ENTRY_ORCHESTRATION_HPP_
