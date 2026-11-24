// CPU-only regression for the Point 85 GDN geometry defect and the v4
// shared entry orchestration (GDN seam + full-forward + MoE seam).
//
// v6 changes (fixture adoption decision, 2026-11-11):
//  * SECTION E pins BOTH adopted fixture identities (SHA-256 + size) and
//    ties the pre-launch rejection regression to the RETAINED frozen 64-wide
//    artifact's real header geometry (head 64 / conv 512 / value 256), not
//    a coincidental synthetic width. The adopted 128-wide positive is
//    asserted accepted; the frozen 64-wide is asserted rejected BEFORE any
//    device work on both the GDN seam and the MoE seam. See
//    NUMERICAL-REFERENCE-v6.md for every pinned number.
//
// v4 changes (offline review v3):
//  * Finding 1: the 24-query / 2-KV / 256-wide / ratio-2 case is NOT a
//    rejection. The fallback accepts it (group 12 % 4 == 0, ratio-
//    independent); it is now an ACCEPTANCE case (ENTRY_ATTN_WMMA_RATIO_FB_*).
//  * Finding 4: the REAL device-reaching MoE orchestration
//    (RunGatedMoeSeamEntry) now has its own trap backend and full
//    invalid-width / positive / order / scope matrix (section D).
//  * Classifier compatibility: every check ID matches
//    [A-Z][A-Z0-9_]{0,63} (underscores, no hyphens) so the project's result
//    classifier can consume them.
//  * Result protocol: the run ends with exactly one line
//      P85_RESULT {"schema":1,"completed":true,"checks":N,"failed":[...],"errors":[]}
//    and one line P85_IDS ["...",...] listing every check ID executed (the
//    known-ID universe for the coordinator). Nothing is written to stderr.
//
// Every check carries a stable case ID. A failing check prints
// "ASSERT-FAILED <ID>" for humans AND includes the ID in the P85_RESULT
// failed list, so the mutation controls can require the EXPECTED named
// assertion to fail -- a kill is a named behavioral assertion failing
// after a fresh successful build, never a build error.
//
// No HIP, no device, no model run -- pure host code, per the offline
// dispatch constraint.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/config.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/backend_geometry.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/gdn_entry_orchestration.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/gdn_geometry.hpp"
#include "tests/models/qwen38_flash_next/i4r_tiny_fixture.hpp"

namespace {

using gufo::models::qwen38_flash_next::rocm::GdnHeadDimSupported;
using gufo::models::qwen38_flash_next::rocm::GdnHostStateFloats;
using gufo::models::qwen38_flash_next::rocm::GdnKernelStateFloats;
using gufo::models::qwen38_flash_next::rocm::GdnStateShortfallFloats;
using gufo::models::qwen38_flash_next::rocm::GdnStateSizedForKernel;
using gufo::models::qwen38_flash_next::rocm::kGdnSupportedHeadDim;

namespace rocm = gufo::models::qwen38_flash_next::rocm;
using Config = gufo::models::qwen38_flash_next::Config;

int g_checks = 0;
int g_failures = 0;
std::vector<std::string> g_failed_ids;  // exact failed IDs (classifier input)
std::vector<std::string> g_all_ids;     // every executed ID (known universe)

// Named check. On failure records the stable ID for both the human line
// and the machine-readable P85_RESULT summary.
void CheckId(const char* id, bool ok, const std::string& what) {
  ++g_checks;
  g_all_ids.push_back(id);
  if (!ok) {
    ++g_failures;
    g_failed_ids.push_back(id);
    std::printf("  ASSERT-FAILED %s: %s\n", id, what.c_str());
  }
}

// The six widths that must be rejected, plus the one that must be accepted.
const std::vector<std::uint32_t>& BadWidths() {
  static const std::vector<std::uint32_t> kBad{0, 64, 96, 127, 129, 256};
  return kBad;
}

// ===========================================================================
// SECTION A -- helper truth-table coverage (v2, retained). NOT entry-path
// coverage: these call the predicate/sizing helpers directly.
// ===========================================================================

void CheckSupportedWidths() {
  CheckId("HELP_WIDTH_128", GdnHeadDimSupported(128), "head_dim 128 supported");
  CheckId("HELP_WIDTH_CONST", kGdnSupportedHeadDim == 128,
          "kGdnSupportedHeadDim is 128");
  CheckId("HELP_WIDTH_64", !GdnHeadDimSupported(64), "head_dim 64 rejected");
  CheckId("HELP_WIDTH_0", !GdnHeadDimSupported(0), "head_dim 0 rejected");
  CheckId("HELP_WIDTH_96", !GdnHeadDimSupported(96), "head_dim 96 rejected");
  CheckId("HELP_WIDTH_256", !GdnHeadDimSupported(256), "head_dim 256 rejected");
  CheckId("HELP_WIDTH_127", !GdnHeadDimSupported(127), "head_dim 127 rejected");
  CheckId("HELP_WIDTH_129", !GdnHeadDimSupported(129), "head_dim 129 rejected");
}

void CheckFaultCaseNumbers() {
  constexpr std::uint32_t kVHeads = 4;
  const std::size_t host64 = GdnHostStateFloats(kVHeads, 64);
  const std::size_t need = GdnKernelStateFloats(kVHeads);
  CheckId("HELP_FAULT_HOST64", host64 == 16384,
          "host state at 64-wide is 16,384 floats");
  CheckId("HELP_FAULT_NEED", need == 65536, "kernel addresses 65,536 floats");
  CheckId("HELP_FAULT_RATIO", need == 4 * host64, "kernel needs exactly 4x");
  CheckId("HELP_FAULT_SIZED", !GdnStateSizedForKernel(kVHeads, 64),
          "64-wide not sized for kernel");
  CheckId("HELP_FAULT_SHORTFALL",
          GdnStateShortfallFloats(kVHeads, 64) == 49152,
          "shortfall 49,152 floats = 192 KiB");
}

void CheckSupportedSizing() {
  CheckId("HELP_SIZE_TINY", GdnStateSizedForKernel(4, 128),
          "tiny 4-head 128-wide sized");
  CheckId("HELP_SIZE_TINY_EXACT",
          GdnHostStateFloats(4, 128) == GdnKernelStateFloats(4),
          "tiny 128-wide sizing exact");
  CheckId("HELP_SIZE_PROD", GdnStateSizedForKernel(48, 128),
          "production 48-head 128-wide sized");
  CheckId("HELP_SIZE_PROD_EXACT",
          GdnHostStateFloats(48, 128) == GdnKernelStateFloats(48),
          "production 128-wide sizing exact (786,432 floats)");
  CheckId("HELP_SIZE_PROD_SHORT", GdnStateShortfallFloats(48, 128) == 0,
          "production shortfall zero");
}

void CheckFixtureGeometry() {
  const Config c = gufo::test_fixture::TinyConfig();
  CheckId("HELP_FIX_HEAD", c.ssm_head_dim == 128, "fixture ssm_head_dim 128");
  CheckId("HELP_FIX_SUP", GdnHeadDimSupported(c.ssm_head_dim),
          "fixture head supported");
  CheckId("HELP_FIX_VALDIM", c.SsmValueDim() == 512, "SsmValueDim 512");
  CheckId("HELP_FIX_CONV", c.SsmConvChannels() == 1024, "SsmConvChannels 1024");
  CheckId("HELP_FIX_SIZE",
          GdnHostStateFloats(c.ssm_num_v_heads, c.ssm_head_dim) ==
              GdnKernelStateFloats(c.ssm_num_v_heads),
          "fixture state sizing matches kernel");
}

void CheckNegativeControl() {
  Config c = gufo::test_fixture::TinyConfig();
  c.ssm_head_dim = 64;
  CheckId("HELP_NEG_HEAD", !GdnHeadDimSupported(c.ssm_head_dim),
          "negative control 64-wide rejected");
  CheckId("HELP_NEG_SIZE", !GdnStateSizedForKernel(c.ssm_num_v_heads, 64),
          "negative control under-sized");
  CheckId("HELP_NEG_CONV", c.SsmConvChannels() == 512,
          "negative control 64-wide channels 512");
}

// ===========================================================================
// SECTION B -- validator + attention contract (v2, retained). Direct
// validator calls; the entry-path equivalents are in sections C/D.
// ===========================================================================

void CheckAttentionContract() {
  CheckId("ATT_FB_4Q1KV", rocm::AttentionFallbackSupported(256, 4, 1),
          "fallback 256 4q/1kv");
  CheckId("ATT_FB_8Q2KV", rocm::AttentionFallbackSupported(256, 8, 2),
          "fallback 256 8q/2kv");
  CheckId("ATT_FB_R12", rocm::AttentionFallbackSupported(256, 12, 1),
          "fallback 256 ratio-12");
  CheckId("ATT_FB_64", !rocm::AttentionFallbackSupported(64, 2, 1),
          "fallback rejects 64-wide");
  CheckId("ATT_FB_R2", !rocm::AttentionFallbackSupported(256, 2, 1),
          "fallback rejects ratio-2 group");
  CheckId("ATT_FB_R3", !rocm::AttentionFallbackSupported(256, 6, 2),
          "fallback rejects ratio-3 group");
  CheckId("ATT_FB_ZERO_KV", !rocm::AttentionFallbackSupported(256, 4, 0),
          "fallback rejects zero kv heads (before division)");
  // REVIEW-V4 FINDING 7: zero QUERY heads must be rejected by the
  // fallback itself, not merely by the division guard. In v4 the only
  // zero-head negative here was zero KV; (256, 0, 1) passed every
  // v4 condition because 0 % 1 == 0 and (0/1) % 4 == 0.
  CheckId("ATT_FB_ZERO_Q", !rocm::AttentionFallbackSupported(256, 0, 1),
          "fallback rejects zero query heads (positive-head condition)");
  CheckId("ATT_FB_ZERO_Q_ANY_KV",
          !rocm::AttentionFallbackSupported(256, 0, 4),
          "fallback rejects zero query heads at any nonzero kv");
  CheckId("ATT_WMMA_OK", rocm::AttentionWmmaSupported(24, 2, 256, 4),
          "WMMA 24/2/256/4");
  CheckId("ATT_WMMA_RATIO", !rocm::AttentionWmmaSupported(24, 2, 256, 2),
          "WMMA wrong ratio rejected");
  CheckId("ATT_WMMA_HEADS", !rocm::AttentionWmmaSupported(4, 1, 256, 4),
          "WMMA wrong heads rejected");
  CheckId("ATT_COMBINED_FIX", !rocm::AttentionGeometrySupported(64, 2, 1, 2),
          "fixture attention: neither path");
  CheckId("ATT_COMBINED_FB", rocm::AttentionGeometrySupported(256, 4, 1, 2),
          "fallback-supported passes combined at any ratio");
  // Zero query heads: neither the fallback (positive-head condition) nor
  // WMMA (heads must equal 24) admits it, so the combined predicate
  // rejects it for every ratio.
  CheckId("ATT_COMBINED_ZERO_Q",
          !rocm::AttentionGeometrySupported(256, 0, 1, 4),
          "combined rejects zero query heads at any ratio");
  // Width-vs-grouping isolation (review-v3 finding 8 support): the
  // fallback's width condition and its group condition are checked
  // independently so a width-only mutation flips exactly the width-
  // dependent positives, and a group-only mutation flips exactly the
  // group-dependent negatives.
  CheckId("ATT_FB_WIDTH_ONLY", !rocm::AttentionFallbackSupported(128, 4, 1),
          "fallback rejects wrong width even with good group");
  CheckId("ATT_FB_GROUP_ONLY", !rocm::AttentionFallbackSupported(256, 2, 1),
          "fallback rejects wrong group even at right width");
}

// ===========================================================================
// SECTION C -- GDN seam + full-forward ENTRY coverage through the SHARED
// orchestration with trap/counter backends (v3 core, IDs renamed v4).
//
// SeamTrapOps models the seam entry's ops interface: it has NO
// FullForward / AttentionEntry METHODS in the seam contract (the
// orchestration static_asserts their absence -- the seam is incapable of
// full Forward by interface). The full_forward / attention_entry
// COUNTERS exist and stay inert at zero in unmuted code; that inertness
// is exactly what the seam-positive assertions check, and a mutant that
// breaches the seam makes them move.
// ===========================================================================

struct SeamTrapOps {
  bool restricted_available = true;
  int device_query = 0;
  int allocate = 0;
  int upload = 0;
  int stream_event = 0;
  int blas_init = 0;
  int restricted_liveness = 0;
  // Inert counters: no method increments them in unmuted code. A seam
  // breach (mutation d) that wires forward/attention through the seam
  // makes them move; the seam-positive checks require them zero.
  int full_forward = 0;
  int attention_entry = 0;
  std::vector<std::string> order;

  bool RestrictedOpAvailable() { return restricted_available; }
  bool DeviceQuery() {
    ++device_query;
    order.push_back("DeviceQuery");
    return true;
  }
  void Allocate() {
    ++allocate;
    order.push_back("Allocate");
  }
  void Upload() {
    ++upload;
    order.push_back("Upload");
  }
  void StreamEvent() {
    ++stream_event;
    order.push_back("StreamEvent");
  }
  void BlasInit() {
    ++blas_init;
    order.push_back("BlasInit");
  }
  void RestrictedLiveness() {
    ++restricted_liveness;
    order.push_back("RestrictedLiveness");
  }
  int DeviceEffects() const {
    return device_query + allocate + upload + stream_event + blas_init +
           restricted_liveness;
  }
  int AllEffects() const {
    return DeviceEffects() + full_forward + attention_entry;
  }
};

// Full-forward trap: FullForward() increments BOTH full_forward and
// attention_entry, modeling that a real Forward visits the attention
// layer. Used for the full-scope positive (host validator positive only).
struct FullTrapOps {
  int device_query = 0;
  int allocate = 0;
  int upload = 0;
  int stream_event = 0;
  int blas_init = 0;
  int full_forward = 0;
  int attention_entry = 0;
  std::vector<std::string> order;

  bool DeviceQuery() {
    ++device_query;
    order.push_back("DeviceQuery");
    return true;
  }
  void Allocate() {
    ++allocate;
    order.push_back("Allocate");
  }
  void Upload() {
    ++upload;
    order.push_back("Upload");
  }
  void StreamEvent() {
    ++stream_event;
    order.push_back("StreamEvent");
  }
  void BlasInit() {
    ++blas_init;
    order.push_back("BlasInit");
  }
  void FullForward() {
    ++full_forward;
    ++attention_entry;
    order.push_back("FullForward");
  }
  int AllEffects() const {
    return device_query + allocate + upload + stream_event + blas_init +
           full_forward + attention_entry;
  }
};

Config WithGdnWidth(std::uint32_t w) {
  Config c = gufo::test_fixture::TinyConfig();
  c.ssm_head_dim = w;
  return c;
}

// All six rejected widths through the REAL seam entry: named GDN error,
// zero effects in EVERY category, empty order (finding 5).
void CheckEntryRejectsAllSixWidthsSeam() {
  for (std::uint32_t w : BadWidths()) {
    SeamTrapOps ops;
    std::string err;
    const rocm::EntryOutcome out =
        rocm::RunGatedSeamEntry(WithGdnWidth(w), ops, &err);
    const std::string id = "ENTRY_SEAM_REJECT_W" + std::to_string(w);
    CheckId(id.c_str(), out == rocm::EntryOutcome::kRejectedGeometry,
            "seam rejects width " + std::to_string(w));
    CheckId((id + "_ERR").c_str(),
            err.find("GDN head_dim " + std::to_string(w)) != std::string::npos,
            "seam rejection names the width: " + err);
    CheckId((id + "_ZERO").c_str(), ops.AllEffects() == 0,
            "seam rejection has ZERO effects in every category");
    CheckId((id + "_ORDER").c_str(), ops.order.empty(),
            "seam rejection performed no ordered device step");
  }
}

// All six rejected widths through the REAL full-forward entry too.
void CheckEntryRejectsAllSixWidthsFull() {
  for (std::uint32_t w : BadWidths()) {
    FullTrapOps ops;
    std::string err;
    const rocm::EntryOutcome out =
        rocm::RunGatedFullEntry(WithGdnWidth(w), ops, &err);
    const std::string id = "ENTRY_FULL_REJECT_W" + std::to_string(w);
    CheckId(id.c_str(), out == rocm::EntryOutcome::kRejectedGeometry,
            "full entry rejects width " + std::to_string(w) + " (GDN first)");
    CheckId((id + "_ZERO").c_str(), ops.AllEffects() == 0,
            "full rejection has ZERO effects (no query, no Forward)");
  }
}

// Positive width 128 through the seam entry with the restricted op wired:
// exactly one restricted liveness, device categories once each, and
// full-Forward/attention ABSENT. First ordered step is the device query
// (validation ran before it -- structural).
void CheckEntryPositiveSeam() {
  SeamTrapOps ops;
  std::string err;
  const rocm::EntryOutcome out =
      rocm::RunGatedSeamEntry(WithGdnWidth(128), ops, &err);
  CheckId("ENTRY_SEAM_POS_COMPLETE",
          out == rocm::EntryOutcome::kCompletedRestricted,
          "seam 128 completes restricted work");
  CheckId("ENTRY_SEAM_POS_LIVENESS", ops.restricted_liveness == 1,
          "exactly one restricted continuation");
  CheckId("ENTRY_SEAM_POS_QUERY", ops.device_query == 1, "one device query");
  CheckId("ENTRY_SEAM_POS_UPLOAD", ops.upload == 1, "one upload");
  CheckId("ENTRY_SEAM_POS_NOFWD", ops.full_forward == 0,
          "full-Forward ABSENT through the seam");
  CheckId("ENTRY_SEAM_POS_NOATTN", ops.attention_entry == 0,
          "attention entry ABSENT through the seam");
  // REVIEW-V4 FINDING 8: assert EVERY device category exactly once and
  // the COMPLETE ordered sequence. v4 asserted only query/upload counts
  // plus order[0]/order.back(); deleting one Allocate/StreamEvent/
  // BlasInit call, or swapping two middle categories, left every v4
  // seam-positive assertion satisfied. The full-order equality below
  // pins the exact sequence the orchestration documents.
  CheckId("ENTRY_SEAM_POS_ALLOC", ops.allocate == 1, "one allocate");
  CheckId("ENTRY_SEAM_POS_STREAM", ops.stream_event == 1,
          "one stream/event creation");
  CheckId("ENTRY_SEAM_POS_BLAS", ops.blas_init == 1, "one blas init");
  const std::vector<std::string> kSeamOrder{
      "DeviceQuery", "Allocate", "Upload", "StreamEvent", "BlasInit",
      "RestrictedLiveness"};
  CheckId("ENTRY_SEAM_POS_ORDER_FULL", ops.order == kSeamOrder,
          "complete order is exactly query->allocate->upload->stream->"
          "blas->restricted-liveness");
  CheckId("ENTRY_SEAM_POS_ORDER_LEN", ops.order.size() == kSeamOrder.size(),
          "no extra or missing ordered device step");
}

// Seam with the restricted op UNAVAILABLE (the real driver's case): stops
// with the missing-API outcome BEFORE the first device operation.
void CheckEntrySeamMissingApi() {
  SeamTrapOps ops;
  ops.restricted_available = false;
  std::string err;
  const rocm::EntryOutcome out =
      rocm::RunGatedSeamEntry(WithGdnWidth(128), ops, &err);
  CheckId("ENTRY_SEAM_MISSING_API",
          out == rocm::EntryOutcome::kStoppedMissingApi,
          "seam stops with missing-restricted-API");
  CheckId("ENTRY_SEAM_MISSING_MSG", err == rocm::kMissingRestrictedApiMessage,
          "missing-API message reported: " + err);
  CheckId("ENTRY_SEAM_MISSING_ZERO", ops.AllEffects() == 0,
          "missing-API stop performed ZERO device work");
}

// Attention-only model (no GDN layers): the seam has nothing to exercise
// and stops NOT-APPLICABLE before device work (intentional applicability).
void CheckEntryAttentionOnlyApplicability() {
  Config c = gufo::test_fixture::TinyConfig();
  c.full_attention_interval = 1;  // every layer is full attention
  c.ssm_head_dim = 64;           // irrelevant: no GDN layers
  SeamTrapOps ops;
  std::string err;
  const rocm::EntryOutcome out = rocm::RunGatedSeamEntry(c, ops, &err);
  CheckId("ENTRY_ATTN_ONLY_NOTAPPLIC",
          out == rocm::EntryOutcome::kNotApplicable,
          "attention-only model: seam NOT APPLICABLE");
  CheckId("ENTRY_ATTN_ONLY_ZERO", ops.AllEffects() == 0,
          "attention-only seam stop has ZERO device work");
}

// An unsupported full-attention config is rejected at the full entry with
// zero effects (finding 1, machine-side). The fixture itself now carries the
// supported FF-GEOM256 geometry, so set the prior tiny geometry explicitly:
// 64-wide heads, 2 query / 1 KV head, ratio 2, which neither device path
// supports.
void CheckEntryFullRejectsUnsupportedAttention() {
  Config c = gufo::test_fixture::TinyConfig();
  c.num_heads = 2;
  c.num_kv_heads = 1;
  c.head_dim = 64;
  c.compress_ratio = 2;
  FullTrapOps ops;
  std::string err;
  const rocm::EntryOutcome out = rocm::RunGatedFullEntry(c, ops, &err);
  CheckId("ENTRY_FULL_REJ_ATTN",
          out == rocm::EntryOutcome::kRejectedGeometry,
          "full entry rejects the fixture's unsupported attention");
  CheckId("ENTRY_FULL_REJ_ATTN_MSG",
          err.find("full-attention") != std::string::npos,
          "rejection names the attention geometry: " + err);
  CheckId("ENTRY_FULL_REJ_ATTN_ZERO", ops.AllEffects() == 0,
          "full-attention rejection has ZERO effects");
}

// A fallback-supported full config reaches the full-forward tail. This is
// a HOST VALIDATOR positive only -- NOT Forward numerical coverage.
void CheckEntryFullPositiveHostValidator() {
  Config c = gufo::test_fixture::TinyConfig();
  c.head_dim = 256;
  c.num_heads = 4;
  c.num_kv_heads = 1;  // fallback: 256-wide, (4/1)%4==0
  FullTrapOps ops;
  std::string err;
  const rocm::EntryOutcome out = rocm::RunGatedFullEntry(c, ops, &err);
  CheckId("ENTRY_FULL_POS_COMPLETE",
          out == rocm::EntryOutcome::kCompletedFull,
          "fallback-supported full config completes (host validator positive)");
  CheckId("ENTRY_FULL_POS_FWD", ops.full_forward == 1,
          "full-forward tail reached once");
  CheckId("ENTRY_FULL_POS_ATTN", ops.attention_entry == 1,
          "attention entry modeled as visited by Forward");
  // REVIEW-V4 FINDING 8: every device category once + complete order.
  CheckId("ENTRY_FULL_POS_QUERY", ops.device_query == 1, "one device query");
  CheckId("ENTRY_FULL_POS_ALLOC", ops.allocate == 1, "one allocate");
  CheckId("ENTRY_FULL_POS_UPLOAD", ops.upload == 1, "one upload");
  CheckId("ENTRY_FULL_POS_STREAM", ops.stream_event == 1,
          "one stream/event creation");
  CheckId("ENTRY_FULL_POS_BLAS", ops.blas_init == 1, "one blas init");
  const std::vector<std::string> kFullOrder{
      "DeviceQuery", "Allocate", "Upload", "StreamEvent", "BlasInit",
      "FullForward"};
  CheckId("ENTRY_FULL_POS_ORDER_FULL", ops.order == kFullOrder,
          "complete order is exactly query->allocate->upload->stream->"
          "blas->full-forward");
  CheckId("ENTRY_FULL_POS_ORDER_LEN", ops.order.size() == kFullOrder.size(),
          "no extra or missing ordered device step");
}

// Invalid attention groupings / zero-head cases rejected before any
// arithmetic requiring nonzero divisors.
//
// REVIEW-V3 FINDING 1: the v3 list wrongly included 24q/2kv/ratio-2 at
// 256-wide as a rejection case. It is NOT one: the fallback accepts the
// 24/2 group (12 is a multiple of 4) and the fallback does not consult
// compress_ratio at all (a non-4 ratio merely disables the sparse-list
// path; the kernel runs dense). WMMA's refusal of ratio != 4 does not
// negate the fallback's acceptance. That geometry is now tested as an
// ACCEPTANCE case in CheckEntryFullAcceptsFallbackAtWmmaHeadsWrongRatio;
// the fallback validator is NOT weakened to keep the old negative.
void CheckEntryInvalidAttentionGroups() {
  struct Case {
    std::uint32_t heads, kv, ratio;
    const char* id;
  };
  const std::vector<Case> cases{
      {4, 0, 4, "ENTRY_ATTN_ZERO_KV"},  // division by zero guarded
      {2, 1, 2, "ENTRY_ATTN_R2"},      // ratio 2, group 2, not mult of 4
      {6, 2, 3, "ENTRY_ATTN_R3"},      // ratio 3, group 3
      // REVIEW-V4 FINDING 7 real-entry negative: an attention-bearing
      // config with ZERO query heads must be rejected by the full entry
      // BEFORE any device category. v4's zero-head matrix covered zero
      // KV only; zero query heads validated and would have continued.
      {0, 1, 4, "ENTRY_ATTN_ZERO_Q"},  // zero query heads rejected
  };
  for (const Case& c : cases) {
    Config cfg = gufo::test_fixture::TinyConfig();
    cfg.full_attention_interval = 1;  // ensure a full-attention layer exists
    cfg.head_dim = 256;
    cfg.num_heads = c.heads;
    cfg.num_kv_heads = c.kv;
    cfg.compress_ratio = c.ratio;
    FullTrapOps ops;
    std::string err;
    const rocm::EntryOutcome out = rocm::RunGatedFullEntry(cfg, ops, &err);
    CheckId(c.id, out == rocm::EntryOutcome::kRejectedGeometry,
            "invalid attention grouping rejected (heads=" +
                std::to_string(c.heads) + " kv=" + std::to_string(c.kv) +
                " ratio=" + std::to_string(c.ratio) + ")");
    CheckId((std::string(c.id) + "_ZERO").c_str(), ops.AllEffects() == 0,
            "invalid attention rejection has ZERO effects");
  }
}

// REVIEW-V3 FINDING 1 -- corrected contract. 24 query / 2 KV heads at
// 256-wide with compress_ratio 2: WMMA refuses it (ratio != 4), but the
// FALLBACK accepts it -- the query/KV group is 12, a multiple of 4, and
// the fallback is ratio-independent. The full entry must therefore
// COMPLETE through the fallback dense path, not reject. This replaces the
// v3 case that wrongly expected rejection (and whose -ZERO assertion was
// consequently false).
void CheckEntryFullAcceptsFallbackAtWmmaHeadsWrongRatio() {
  Config cfg = gufo::test_fixture::TinyConfig();
  cfg.full_attention_interval = 1;
  cfg.head_dim = 256;
  cfg.num_heads = 24;
  cfg.num_kv_heads = 2;
  cfg.compress_ratio = 2;  // WMMA-invalid, fallback-irrelevant
  FullTrapOps ops;
  std::string err;
  const rocm::EntryOutcome out = rocm::RunGatedFullEntry(cfg, ops, &err);
  CheckId("ENTRY_ATTN_WMMA_RATIO_FB_ACCEPT",
          out == rocm::EntryOutcome::kCompletedFull,
          "24/2/256/ratio-2 ACCEPTED via fallback (group 12 % 4 == 0)");
  CheckId("ENTRY_ATTN_WMMA_RATIO_FB_FWD", ops.full_forward == 1,
          "full-forward tail reached exactly once");
  CheckId("ENTRY_ATTN_WMMA_RATIO_FB_ATTN", ops.attention_entry == 1,
          "attention visited exactly once through the fallback");
}

// ===========================================================================
// SECTION D -- MoE seam ENTRY coverage (review-v3 finding 4).
//
// RunGatedMoeSeamEntry is the orchestration the REAL driver uses to
// reach the device: it validates GDN geometry first, then queries the
// device, uploads, creates the executor and runs the per-layer rotated-
// expert dispatch. v3 tested only the GDN seam / full templates and
// pointed its MoE mutations at the wrong template. This section exercises
// the MoE orchestration itself with a trap backend: the full invalid-
// width matrix, the positive, the no-device stop, and the scope rule
// (GDN geometry is load-bearing only when GDN layers exist; attention
// geometry is not load-bearing for the MoE seam).
// ===========================================================================

struct MoeTrapOps {
  bool device_present = true;
  int device_query = 0;
  int allocate = 0;
  int upload = 0;
  int stream_event = 0;
  int blas_init = 0;
  int moe_work = 0;
  std::vector<std::string> order;

  bool DeviceQuery() {
    ++device_query;
    order.push_back("DeviceQuery");
    return device_present;
  }
  void Allocate() {
    ++allocate;
    order.push_back("Allocate");
  }
  void Upload() {
    ++upload;
    order.push_back("Upload");
  }
  void StreamEvent() {
    ++stream_event;
    order.push_back("StreamEvent");
  }
  void BlasInit() {
    ++blas_init;
    order.push_back("BlasInit");
  }
  void MoeRestrictedWork() {
    ++moe_work;
    order.push_back("MoeRestrictedWork");
  }
  int EffectsAfterQuery() const {
    return allocate + upload + stream_event + blas_init + moe_work;
  }
  int AllEffects() const { return device_query + EffectsAfterQuery(); }
};

// All six rejected GDN widths through the REAL MoE seam entry: the
// entry validates GDN geometry BEFORE any device work, so every rejected
// width must produce a named error, zero effects in every category and
// an empty order -- exactly the discipline the GDN seam enforces, now
// asserted on the template that actually reaches the device.
void CheckEntryRejectsAllSixWidthsMoe() {
  for (std::uint32_t w : BadWidths()) {
    MoeTrapOps ops;
    std::string err;
    const rocm::EntryOutcome out =
        rocm::RunGatedMoeSeamEntry(WithGdnWidth(w), ops, &err);
    const std::string id = "ENTRY_MOE_REJECT_W" + std::to_string(w);
    CheckId(id.c_str(), out == rocm::EntryOutcome::kRejectedGeometry,
            "MoE seam rejects width " + std::to_string(w));
    CheckId((id + "_ERR").c_str(),
            err.find("GDN head_dim " + std::to_string(w)) != std::string::npos,
            "MoE seam rejection names the width: " + err);
    CheckId((id + "_ZERO").c_str(), ops.AllEffects() == 0,
            "MoE seam rejection has ZERO effects in every category");
    CheckId((id + "_ORDER").c_str(), ops.order.empty(),
            "MoE seam rejection performed no ordered device step");
  }
}

// Positive width 128 through the MoE seam entry: completes the
// restricted MoE work, device categories once each, first ordered step
// is the device query, tail is the MoE dispatch.
void CheckEntryPositiveMoe() {
  MoeTrapOps ops;
  std::string err;
  const rocm::EntryOutcome out =
      rocm::RunGatedMoeSeamEntry(WithGdnWidth(128), ops, &err);
  CheckId("ENTRY_MOE_POS_COMPLETE",
          out == rocm::EntryOutcome::kCompletedMoeRestricted,
          "MoE seam 128 completes restricted MoE work");
  CheckId("ENTRY_MOE_POS_WORK", ops.moe_work == 1,
          "exactly one restricted MoE continuation");
  CheckId("ENTRY_MOE_POS_QUERY", ops.device_query == 1, "one device query");
  CheckId("ENTRY_MOE_POS_UPLOAD", ops.upload == 1, "one upload");
  // REVIEW-V4 FINDING 8: every category once + complete order (see the
  // seam-positive note). v4's first/last-only checks could not see a
  // dropped Allocate/StreamEvent/BlasInit or a middle-category swap.
  CheckId("ENTRY_MOE_POS_ALLOC", ops.allocate == 1, "one allocate");
  CheckId("ENTRY_MOE_POS_STREAM", ops.stream_event == 1,
          "one stream/event creation");
  CheckId("ENTRY_MOE_POS_BLAS", ops.blas_init == 1, "one blas init");
  const std::vector<std::string> kMoeOrder{
      "DeviceQuery", "Allocate", "Upload", "StreamEvent", "BlasInit",
      "MoeRestrictedWork"};
  CheckId("ENTRY_MOE_POS_ORDER_FULL", ops.order == kMoeOrder,
          "complete order is exactly query->allocate->upload->stream->"
          "blas->moe-restricted-work");
  CheckId("ENTRY_MOE_POS_ORDER_LEN", ops.order.size() == kMoeOrder.size(),
          "no extra or missing ordered device step");
}

// No device: validation passes, DeviceQuery reports absence, NOTHING
// after the query runs.
void CheckEntryMoeNoDevice() {
  MoeTrapOps ops;
  ops.device_present = false;
  std::string err;
  const rocm::EntryOutcome out =
      rocm::RunGatedMoeSeamEntry(WithGdnWidth(128), ops, &err);
  CheckId("ENTRY_MOE_NODEVICE_OUTCOME", out == rocm::EntryOutcome::kNoDevice,
          "MoE seam reports no-device");
  CheckId("ENTRY_MOE_NODEVICE_NOEFFECTS",
          ops.device_query == 1 && ops.EffectsAfterQuery() == 0,
          "nothing after the query ran without a device");
}

// Scope rule: the MoE seam gates on GDN geometry only when GDN layers
// exist. An attention-only model (no GDN layers) with a bad GDN width
// passes validation trivially (nothing GDN to validate) and proceeds to
// the MoE tail -- attention geometry is NOT load-bearing for the MoE
// seam. This pins the documented scope semantics machine-side.
void CheckEntryMoeScopeAttentionOnly() {
  Config c = gufo::test_fixture::TinyConfig();
  c.full_attention_interval = 1;  // every layer is full attention
  c.ssm_head_dim = 64;           // bad width, but no GDN layers exist
  MoeTrapOps ops;
  std::string err;
  const rocm::EntryOutcome out = rocm::RunGatedMoeSeamEntry(c, ops, &err);
  CheckId("ENTRY_MOE_SCOPE_ATTNONLY_PROCEEDS",
          out == rocm::EntryOutcome::kCompletedMoeRestricted,
          "attention-only model: MoE seam proceeds (no GDN to gate)");
  CheckId("ENTRY_MOE_SCOPE_ATTNONLY_WORK", ops.moe_work == 1,
          "MoE tail ran once under the trivially-passed GDN gate");
}

// ===========================================================================
// SECTION E -- FIXTURE ADOPTION REGRESSION (v6).
//
// Fixture adoption decision (2026-11-11):
//   * POSITIVE (restricted GDN parity): the verified 128-wide fixture.
//       SHA-256 8ed1955c428ec4607f44334d8bb9d82e7d244b1f8dc6a02b72e60002fc286bed
//       size    10,700,928 B
//   * REJECTION (pre-launch regression input): the frozen 64-wide fixture.
//       SHA-256 a00a6bbc5b6600103579fbbe156e9ef728ac354b594571fbd9c1d08969a2d0ff
//       size    7,529,856 B
//
// The identities are pinned here so the rejection regression is traceable to
// the retained artifact, not a coincidental synthetic width. The dims below
// are read from the two fixtures' GGUF tensor headers (NUMERICAL-REFERENCE-
// v6.md): the config helpers must derive geometry matching each header, the
// REAL gate must reject the frozen-64 geometry BEFORE any device work (zero
// effects, empty order) on BOTH the GDN seam and the MoE seam, and the
// adopted-128 geometry must be accepted. The SHA/size constants are not
// recomputed by this CPU test (no GGUF I/O by design); they are the pinned
// reference the wrapper's step-0 fixture-identity gate checks against disk.
// ===========================================================================

// Adopted positive fixture identity (restricted GDN parity).
constexpr std::uint64_t kPositiveFixtureSizeBytes = 10700928;
constexpr const char* kPositiveFixtureSha256 =
    "8ed1955c428ec4607f44334d8bb9d82e7d244b1f8dc6a02b72e60002fc286bed";
// Retained rejection-regression fixture identity (frozen 64-wide).
constexpr std::uint64_t kRejectionFixtureSizeBytes = 7529856;
constexpr const char* kRejectionFixtureSha256 =
    "a00a6bbc5b6600103579fbbe156e9ef728ac354b594571fbd9c1d08969a2d0ff";

// The frozen 64-wide fixture's header-derived geometry must match the
// negative-control config, and the REAL gate must reject it before launch.
void CheckFrozen64RejectionRegression() {
  Config c = gufo::test_fixture::TinyConfig();
  c.ssm_head_dim = 64;  // the frozen fixture's head dim (ssm_norm.weight [64])
  CheckId("ADOPT_REJ_HEAD64", c.ssm_head_dim == 64,
          "rejection fixture head dim 64");
  CheckId("ADOPT_REJ_CONV512", c.SsmConvChannels() == 512,
          "rejection fixture conv channels 512 (matches frozen ssm_conv1d [4,512])");
  CheckId("ADOPT_REJ_VALDIM256", c.SsmValueDim() == 256,
          "rejection fixture value dim 256 (matches frozen ssm_out [256,256])");
  CheckId("ADOPT_REJ_UNSUPPORTED", !GdnHeadDimSupported(c.ssm_head_dim),
          "rejection fixture width unsupported by the GDN gate");
  // GDN seam: named rejection, ZERO device effects, empty order.
  SeamTrapOps sops;
  std::string serr;
  const rocm::EntryOutcome sout = rocm::RunGatedSeamEntry(c, sops, &serr);
  CheckId("ADOPT_REJ_SEAM_REJECT",
          sout == rocm::EntryOutcome::kRejectedGeometry,
          "GDN seam rejects the frozen-64 fixture geometry");
  CheckId("ADOPT_REJ_SEAM_NAMED",
          serr.find("GDN head_dim 64") != std::string::npos,
          "GDN seam rejection names head_dim 64: " + serr);
  CheckId("ADOPT_REJ_SEAM_ZERO", sops.AllEffects() == 0,
          "frozen-64 seam rejection has ZERO device effects");
  CheckId("ADOPT_REJ_SEAM_ORDER", sops.order.empty(),
          "frozen-64 seam rejection performed no ordered device step");
  // MoE seam (the template that reaches the device): same discipline.
  MoeTrapOps mops;
  std::string merr;
  const rocm::EntryOutcome mout = rocm::RunGatedMoeSeamEntry(c, mops, &merr);
  CheckId("ADOPT_REJ_MOE_REJECT",
          mout == rocm::EntryOutcome::kRejectedGeometry,
          "MoE seam rejects the frozen-64 fixture geometry");
  CheckId("ADOPT_REJ_MOE_NAMED",
          merr.find("GDN head_dim 64") != std::string::npos,
          "MoE seam rejection names head_dim 64: " + merr);
  CheckId("ADOPT_REJ_MOE_ZERO", mops.AllEffects() == 0,
          "frozen-64 MoE seam rejection has ZERO device effects");
  CheckId("ADOPT_REJ_MOE_ORDER", mops.order.empty(),
          "frozen-64 MoE seam rejection performed no ordered device step");
}

// The adopted 128-wide positive fixture's header-derived geometry must match
// the positive config and be ACCEPTED by the real gate (restricted seam).
void CheckAdopted128Positive() {
  Config c = gufo::test_fixture::TinyConfig();  // already 128-wide
  CheckId("ADOPT_POS_HEAD128", c.ssm_head_dim == 128,
          "positive fixture head dim 128");
  CheckId("ADOPT_POS_CONV1024", c.SsmConvChannels() == 1024,
          "positive fixture conv channels 1024 (matches positive ssm_conv1d [4,1024])");
  CheckId("ADOPT_POS_VALDIM512", c.SsmValueDim() == 512,
          "positive fixture value dim 512 (matches positive ssm_out [512,256])");
  CheckId("ADOPT_POS_SUPPORTED", GdnHeadDimSupported(c.ssm_head_dim),
          "positive fixture width supported by the GDN gate");
  SeamTrapOps ops;
  std::string err;
  const rocm::EntryOutcome out = rocm::RunGatedSeamEntry(c, ops, &err);
  CheckId("ADOPT_POS_SEAM_COMPLETE",
          out == rocm::EntryOutcome::kCompletedRestricted,
          "GDN seam completes restricted work on the adopted-128 fixture");
  CheckId("ADOPT_POS_SEAM_LIVENESS", ops.restricted_liveness == 1,
          "exactly one restricted continuation on the adopted positive");
  CheckId("ADOPT_POS_SEAM_NOFWD", ops.full_forward == 0,
          "adopted positive stays restricted: no full-Forward through the seam");
}

}  // namespace

int main() {
  std::printf("GDN geometry + v4 shared entry orchestration regression (CPU-only)\n");
  // Section A: helper truth table (NOT entry-path coverage).
  CheckSupportedWidths();
  CheckFaultCaseNumbers();
  CheckSupportedSizing();
  CheckFixtureGeometry();
  CheckNegativeControl();
  // Section B: validator + attention contract.
  CheckAttentionContract();
  // Section C: GDN seam + full-forward entry-path coverage.
  CheckEntryRejectsAllSixWidthsSeam();
  CheckEntryRejectsAllSixWidthsFull();
  CheckEntryPositiveSeam();
  CheckEntrySeamMissingApi();
  CheckEntryAttentionOnlyApplicability();
  CheckEntryFullRejectsUnsupportedAttention();
  CheckEntryFullPositiveHostValidator();
  CheckEntryInvalidAttentionGroups();
  CheckEntryFullAcceptsFallbackAtWmmaHeadsWrongRatio();
  // Section D: MoE seam entry-path coverage (finding 4).
  CheckEntryRejectsAllSixWidthsMoe();
  CheckEntryPositiveMoe();
  CheckEntryMoeNoDevice();
  CheckEntryMoeScopeAttentionOnly();
  // Section E: fixture adoption regression (v6) -- frozen-64 rejection tied
  // to the retained artifact's real geometry; adopted-128 positive accepted.
  CheckFrozen64RejectionRegression();
  CheckAdopted128Positive();

  // Machine-readable summaries (exactly one of each, last lines of stdout,
  // nothing on stderr). P85_IDS = known-ID universe from this run;
  // P85_RESULT = the classifier-contract result line.
  std::printf("P85_IDS [");
  for (std::size_t i = 0; i < g_all_ids.size(); ++i) {
    std::printf("%s\"%s\"", i == 0 ? "" : ",", g_all_ids[i].c_str());
  }
  std::printf("]\n");
  std::printf("P85_RESULT {\"schema\":1,\"completed\":true,\"checks\":%d,"
              "\"failed\":[",
              g_checks);
  for (std::size_t i = 0; i < g_failed_ids.size(); ++i) {
    std::printf("%s\"%s\"", i == 0 ? "" : ",", g_failed_ids[i].c_str());
  }
  std::printf("],\"errors\":[]}\n");
  std::printf("%s: %d checks, %d failure(s)\n",
              g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
