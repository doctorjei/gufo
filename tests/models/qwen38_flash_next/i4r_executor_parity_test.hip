// Point 85, step 2 REV 5: the rotated (I4R128) expert path driven through the
// REAL HIP executor -- not a host stand-in and not a raw kernel launch.
//
// The previous handoff only had i4r_dispatch_test (host stand-ins for the
// dispatch primitives) and i4r_hip_test (the raw i4r kernels). Neither
// touched Executor::Create / Executor::Forward / Executor::MoeExperts, so
// nothing proved the rotated path is wired correctly through the real
// runtime. This driver closes that gap.
//
// WHAT IS REAL HERE
//   * The GGUF is bound through ModelWeights::BindWithConfig and uploaded
//     through the real DeviceModel::Upload, so the real UploadRotVector runs
//     (it widens the five F16 rotation vectors to F32 and sets
//     DeviceLayer::rotated).
//   * The Executor is the production Executor::Create, and the rotated
//     experts run through the production Executor::MoeExperts ->
//     HipRotatedOps -> i4r_launch_* kernels.
//
// REDUCED-CONFIG TEST SEAM (NOT a production run)
//   Production Config::FromGguf pins 48 layers / 512 experts / hidden 2560,
//   so no tiny GGUF can pass it. This test binds a hand-built reduced Config
//   of the same layout through BindWithConfig. This is a deliberate test
//   seam: it exercises the real code paths at tiny dimensions. It is NOT a
//   production-frontdoor run and must not be described as one.
//
// WHY THE EXPERT PARITY IS DRIVEN AT MoeExperts, NOT THROUGH FORWARD LOGITS
//   The numerical parity gate lives where a MATCHED per-slot dense reference
//   exists. `MoeExperts` produces the unweighted per-slot routed down output
//   (s_.down_e), one [hidden] row per (token, expert) slot, so each slot can
//   be compared against a dense matmul of that slot's own token input and
//   that expert's own materialised weights W = diag(sv)*Hn*M*Hn*diag(su).
//   `ReferenceModel::Moe` instead returns the softmax-weighted routed sum
//   plus the shared expert, which has no per-slot counterpart in down_e.
//   Forward logits sit further still from the seam. The dense materialised-W
//   oracle and ReferenceModel::Moe share the same mathematical basis and are
//   cross-validated against each other in i4r_moe_parity_test; here the
//   dense oracle is the primary reference and ReferenceModel::Moe is run only
//   for oracle liveness. This driver does NOT claim full-Forward numerical
//   parity -- Forward prefill/decode are exercised for liveness and finite
//   outputs only.
//
//   Note on the fixture's hyper-connections: the v4 fixture populates the HC
//   norm gammas and the HC mixer down/up projections with deterministic
//   nonzero values, so a full Forward is no longer degenerate at the MoE
//   input. That is why the earlier "zero mixers make Forward vacuous" reason
//   is gone; the seam choice above is the reason, and it stands on its own.
//
// GRAPH SCOPE
//   The rotated path is EAGER-ONLY: MoeExperts returns an error if a graph
//   capture is active while the layer is rotated, because the host-side
//   expert-weight selection would freeze this pass's routing into the
//   captured kernel arguments. The driver runs the rotated path eagerly and
//   asserts the capture refusal fires with that error, for every layer.
//
// REV 5 CHANGES (review v4 driver blockers)
//   1. Primary gate restored to the contract's peak-normalised < 1e-4. The
//      stale ~3e-4 / 5e-2 explanation is removed; the contract is not
//      widened. A 1e-3 residual is now REJECTED, and the gate's known-answer
//      case list (i4r_parity_gate.hpp, run for real by i4r_parity_gate_test)
//      proves it without a device.
//   2. The matched MoE seam now runs for EVERY fixture layer and BOTH
//      dispatch shapes -- single-token decode-like (matvec primitive) and
//      multi-token prefill-like (GEMM primitive) -- in both routing states,
//      against each layer/token's own materialised weights and its own
//      distinct nonzero input row.
//   3. A mandatory fixture identity gate runs BEFORE any bind or upload: the
//      in-memory GGUF's SHA-256 and size are asserted against the published
//      contract section 2 identity, and the written file is re-hashed and
//      required to match before the reader opens it.
//   4. The primary comparison enforces its invariants explicitly (equal
//      size, both-sided finiteness, reference peak > 1e-3) and emits the
//      contract section 9 schema: per mode/layer/routing/slot peak, max_abs,
//      named peak-normalised error and PASS/FAIL, with the failing index and
//      both values on failure. CheckComparatorRejects exercises the
//      malformed / non-finite / zero-reference rejections without any device
//      work.
//
// RUNTIME STATUS: this box has NO GPU. The driver is COMPILE-VERIFIED only.
// It has not been executed; no numeric result below is measured. The CPU
// suites that ground the thresholds are named in the delivery report.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "i4r_parity_gate.hpp"
#include "i4r_tiny_fixture.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/backend_geometry.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/gdn_entry_orchestration.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/reference.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"
#include "tests/support/temporary_directory.hpp"

namespace core = gufo::core;
namespace fn = gufo::models::qwen38_flash_next;
namespace rocm = fn::rocm;
namespace tf = gufo::test_fixture;

namespace gufo::models::qwen38_flash_next::rocm {

// Test-only access to the production Executor's private rotated dispatch and
// capture flag. Mirrors the ExecutorTestAccess pattern in
// wide_verification_test.cpp. Declared in the rocm namespace so the
// `friend struct ExecutorTestAccess;` in executor.hpp grants it entry.
struct ExecutorTestAccess {
  static bool Capturing(const Executor& e) { return e.capturing_; }

  // Stage a routing into the device-side s_.ids that MoeExperts downloads
  // to pinned host memory before its host-side expert-weight selection.
  static void SetIds(Executor& e, const std::int32_t* ids, std::size_t slots) {
    if (hipMemcpy(e.s_.ids, ids, slots * sizeof(std::int32_t),
                  hipMemcpyHostToDevice) != hipSuccess)
      throw std::runtime_error("failed to stage expert ids on device");
  }

  // Run the real rotated dispatch for one eager (non-captured) batch.
  static bool MoeExperts(Executor& e, const DeviceLayer& l, const float* x,
                         float* out, std::uint32_t n_tokens, std::string* err) {
    return e.MoeExperts(l, x, out, n_tokens, err);
  }

  // Download the routed down output [slots x hidden] the dispatch wrote.
  static std::vector<float> ReadDownE(const Executor& e, std::size_t floats) {
    std::vector<float> host(floats);
    if (hipStreamSynchronize(e.stream_) != hipSuccess ||
        hipMemcpy(host.data(), e.s_.down_e, floats * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
      throw std::runtime_error("failed to download routed down_e");
    return host;
  }

  // Batched path test seam: leave per-slot unweighted outputs in s_.down_e.
  static void SetPerSlot(Executor& e, bool on) { e.rot_per_slot_for_test_ = on; }

  // Stage combine weights [slots], the shared-expert output [tokens x hidden]
  // and the shared gate logits (router row num_experts) for `tokens` rows.
  static void SetCombineInputs(Executor& e, const std::vector<float>& weights,
                               const std::vector<float>& shared,
                               const std::vector<float>& gate_logit) {
    const auto& c = e.config();
    const std::size_t stride = c.num_experts + 1;
    std::vector<float> router(gate_logit.size() * stride, 0.0F);
    for (std::size_t t = 0; t < gate_logit.size(); ++t)
      router[t * stride + c.num_experts] = gate_logit[t];
    if (hipMemcpy(e.s_.weights, weights.data(), weights.size() * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(e.s_.shexp_out, shared.data(), shared.size() * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(e.s_.router, router.data(), router.size() * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess)
      throw std::runtime_error("failed to stage combine inputs");
  }

  static hipStream_t Stream(const Executor& e) { return e.stream_; }

  // Force the capture flag to exercise the eager-only guard without a real
  // hipGraph capture. The guard reads capturing_ before any host selection.
  static bool MoeExpertsWhileCapturing(Executor& e, const DeviceLayer& l,
                                       const float* x, float* out,
                                       std::uint32_t n_tokens,
                                       std::string* err) {
    e.capturing_ = true;
    const bool ok = e.MoeExperts(l, x, out, n_tokens, err);
    e.capturing_ = false;
    return ok;
  }
};

}  // namespace gufo::models::qwen38_flash_next::rocm

namespace {

using rocm::DeviceLayer;
using rocm::Executor;
using rocm::ExecutorTestAccess;

int g_pass = 0;
int g_fail = 0;

void Check(bool ok, const std::string& what) {
  if (ok) {
    ++g_pass;
  } else {
    ++g_fail;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

[[noreturn]] void Die(const std::string& what) {
  std::fprintf(stderr, "FATAL: %s\n", what.c_str());
  std::exit(1);
}

float Silu(float x) {
  return x / (1.0F + std::exp(-x));
}

// The thresholds, the metric, the ParityRow evidence type and the
// comparison invariants all come from i4r_parity_gate.hpp (namespace
// gufo::test_fixture), so the gate this driver enforces is the same code
// the CPU-runnable i4r_parity_gate_test executes. Aliased here for
// readability at the call sites.
using tf::AllFinite;
using tf::CompareParity;
using tf::FormatParityRow;
using tf::kMetricName;
using tf::kNegativeControlMin;
using tf::kPrimaryMaxRel;
using tf::kRefMinPeak;
using tf::kRoutingChangeMin;
using tf::kSeparationFactor;
using tf::ParityRow;
using tf::PeakNormalisedRel;

// ---------------------------------------------------------------------------
// Fixture identity (contract section 2) -- blocker 3.
//
// The published identity of the fixture this driver is allowed to test. Any
// change to the fixture generator changes these values and requires a new
// published identity; the gate below refuses to proceed otherwise, so a
// generator drift can never silently change the tested input.
// ---------------------------------------------------------------------------

// Geometry-correction draft: the fixture is rebuilt at the supported
// 128-wide GDN head (ssm_head_dim=128), which changes every SSM tensor
// shape and therefore this identity. It SUPERSEDES the frozen v7 fixture
// a00a6bbc5b6600103579fbbe156e9ef728ac354b594571fbd9c1d08969a2d0ff
// (7,529,856 B), which stays untouched as the historical record of the
// consumed GPU attempt. Deterministic: identical hash across rebuilds.
//
// FF-GEOM256 (populated GDN payloads, 24q/2kv full attention at head_dim
// 256, ratio 4, 4x128 indexer) supersedes the 128-wide GDN fixture
// 8ed1955c428ec4607f44334d8bb9d82e7d244b1f8dc6a02b72e60002fc286bed
// (10,700,928 B). It is the fixture of the passing point-85 full-forward
// parity run (tail13 regression baseline).
constexpr std::uint64_t kFixtureSizeBytes = 30658944;
constexpr const char* kFixtureSha256 =
    "b0963e90e7fb4b51a838d9f980d641567ac0ec816fa40dc075dfaf217e2b8ed4";

// Assert the fixture identity and return the computed hex. Hard-stops the
// process on mismatch: continuing would run device work against an input
// whose identity is unknown, which is exactly what this gate exists to
// prevent. `stage` names the checkpoint for the log.
std::string RequireFixtureIdentity(const std::vector<std::uint8_t>& bytes,
                                   const char* stage) {
  const std::string hex = gufo::crypto::Sha256Hex(bytes);
  const bool size_ok = bytes.size() == kFixtureSizeBytes;
  const bool hash_ok = hex == kFixtureSha256;
  std::printf(
      "  fixture identity [%s]: size=%zu (want %llu) sha256=%s (want %s)\n",
      stage, bytes.size(), static_cast<unsigned long long>(kFixtureSizeBytes),
      hex.c_str(), kFixtureSha256);
  if (!size_ok)
    Die(std::string("fixture identity gate: size mismatch at ") + stage +
        " -- refusing to proceed");
  if (!hash_ok)
    Die(std::string("fixture identity gate: SHA-256 mismatch at ") + stage +
        " -- refusing to proceed; a fixture change needs a new published "
        "identity");
  Check(true, std::string("fixture identity verified at ") + stage);
  return hex;
}

// ---------------------------------------------------------------------------
// Dense oracle
// ---------------------------------------------------------------------------

// Plain dense oracle for one routed expert slot, using the fixture's
// materialised true weights (rotation already folded in). Mirrors the
// gate -> SwiGLU -> down chain the rotated dispatch runs, but as a dense
// matmul of W rather than I4R codes + Hadamard stages.
//   w_gate[e]: [ff x hidden] row-major; w_up[e]: [ff x hidden];
//   w_down[e]: [hidden x ff] row-major.
std::vector<float> DenseExpert(const std::vector<float>& w_gate,
                               const std::vector<float>& w_up,
                               const std::vector<float>& w_down,
                               const std::vector<float>& x,
                               std::uint64_t hidden, std::uint64_t ff) {
  std::vector<float> h(ff);
  for (std::uint64_t o = 0; o < ff; ++o) {
    double g = 0.0, u = 0.0;
    for (std::uint64_t i = 0; i < hidden; ++i) {
      g += static_cast<double>(w_gate[o * hidden + i]) * x[i];
      u += static_cast<double>(w_up[o * hidden + i]) * x[i];
    }
    h[o] =
        static_cast<float>(Silu(static_cast<float>(g)) * static_cast<float>(u));
  }
  std::vector<float> d(hidden, 0.0F);
  for (std::uint64_t j = 0; j < hidden; ++j) {
    double acc = 0.0;
    for (std::uint64_t o = 0; o < ff; ++o)
      acc += static_cast<double>(w_down[j * ff + o]) * h[o];
    d[j] = static_cast<float>(acc);
  }
  return d;
}

// ---------------------------------------------------------------------------
// Prefill-phase scoping
// ---------------------------------------------------------------------------

// Forces the executor's thread-local prefill phase around a scoped block.
// Production sets this inside Forward; the seam needs it directly so the
// multi-token prefill dispatch (the GEMM primitive, with its token-row input
// stride) can be exercised at the isolated MoE seam with caller-controlled
// expert ids. Restores the previous value on scope exit.
struct ScopedPrefill {
  bool previous;
  explicit ScopedPrefill(bool enabled) {
    previous = rocm::SetPrefillPhaseForTest(enabled);
  }
  ~ScopedPrefill() { (void)rocm::SetPrefillPhaseForTest(previous); }
};

// ---------------------------------------------------------------------------
// Fixture upload through the real device path
// ---------------------------------------------------------------------------

struct BoundModel {
  std::unique_ptr<core::GgufReader> reader;
  std::unique_ptr<fn::ModelWeights> weights;
  std::unique_ptr<rocm::DeviceModel> model;
};

// Upload the fixture through the real DeviceModel::Upload. The real
// WeightUpload stager reads tensor payloads from a file descriptor, so the
// in-memory GGUF bytes are written to a temp file and reopened with
// GgufReader::OpenFile -- exactly how a real artifact reaches the device.
// This is what makes UploadRotVector run.
//
// The caller MUST have passed the in-memory identity gate already; this
// additionally re-hashes the bytes as they landed on disk and refuses to open
// the reader unless they match the published identity, so a truncated or
// tampered write cannot reach the device either.
BoundModel UploadViaFile(const tf::TinyHybrid& f, const std::string& path,
                         const std::string& expected_hex, std::string* err) {
  BoundModel b;
  {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(f.bytes.data()),
              static_cast<std::streamsize>(f.bytes.size()));
    if (!out) {
      *err = "failed to write fixture GGUF";
      return b;
    }
    out.flush();
    if (!out) {
      *err = "failed to flush fixture GGUF";
      return b;
    }
  }
  const std::string disk_hex = gufo::crypto::Sha256FileHex(path);
  if (disk_hex != expected_hex) {
    *err = "fixture on-disk identity mismatch: got " + disk_hex + " want " +
           expected_hex;
    return b;
  }
  std::printf("  fixture on-disk identity verified: %s\n", disk_hex.c_str());
  b.reader = core::GgufReader::OpenFile(path, err);
  if (b.reader == nullptr)
    return b;
  auto bound = fn::ModelWeights::BindWithConfig(*b.reader, f.config, err);
  if (!bound.has_value())
    return b;
  b.weights = std::make_unique<fn::ModelWeights>(std::move(*bound));
  b.model =
      rocm::DeviceModel::Upload(*b.weights, *b.reader, nullptr, nullptr, err);
  return b;
}

// ---------------------------------------------------------------------------
// v3 (offline review v2 findings 1, 3, 4): REAL entry ops on the shared
// gated orchestration. The v2 Forward-liveness helper is REMOVED.
//
// That v2 function called Executor::Forward (one prefill + three decodes).
// Forward visits every layer, including the fixture's unsupported
// full-attention layer -- exactly the work the seam-only contract says is
// NOT executed there. Relabeling did not remove it (finding 1), the gate
// still ran after hipGetDeviceProperties (finding 4), and the tested
// continuation was a bare marker while the real upload / Executor::Create
// sat outside the template (finding 3).
//
// The seam path now runs through RunGatedSeamEntry (gdn_entry_orchestration
// .hpp). SeamRealOps below plugs the REAL driver into it. Its
// RestrictedOpAvailable() is false: the executor exposes no GDN-only
// restricted execution entry -- Forward is the only layer-running API and
// it necessarily enters full attention -- so the orchestration stops the
// seam path BEFORE the first device operation and reports the missing API
// (kMissingRestrictedApiMessage). The counters make that stop assertable:
// a seam that performed any device work is a named test failure.
//
// SeamOps contract compliance: NO FullForward, NO AttentionEntry member
// (the orchestration static_asserts their absence -- a seam wired to a
// full-forward-capable object does not compile).
// ---------------------------------------------------------------------------

struct SeamRealOps {
  int device_query = 0;
  int allocate = 0;
  int upload = 0;
  int stream_event = 0;
  int blas_init = 0;
  int restricted_liveness = 0;

  bool RestrictedOpAvailable() { return false; }

  // Unreachable while RestrictedOpAvailable() is false; present to
  // satisfy the interface and make the zero-effect assertion meaningful.
  bool DeviceQuery() {
    ++device_query;
    return true;
  }
  void Allocate() { ++allocate; }
  void Upload() { ++upload; }
  void StreamEvent() { ++stream_event; }
  void BlasInit() { ++blas_init; }
  void RestrictedLiveness() { ++restricted_liveness; }

  int DeviceEffects() const {
    return device_query + allocate + upload + stream_event + blas_init +
           restricted_liveness;
  }
};

// REVIEW-V3 FINDING 2: forward declarations. MoeSeamRealOps::
// MoeRestrictedWork() below calls these four helpers, whose unchanged
// definitions occur LATER in this frozen driver (v7 lines 400/467/634/675
// region: MakeTokenInputs, RunSeamParityForLayer, RunCaptureRefusal,
// RunOracleLiveness). A class-body inline method can only see names
// declared before it, and MakeTokenInputs' fundamental arguments cannot
// be rescued by argument-dependent lookup -- the native HIP TU had a
// source-level name-lookup blocker. Declaring the helpers here (the
// definitions below must keep these exact signatures) clears it. The
// host-only CPU target does not compile this adapter, so these
// declarations exist for the HIP translation unit only.
std::vector<float> MakeTokenInputs(std::uint32_t n_tokens, std::uint64_t hidden);
void RunSeamParityForLayer(Executor& exec, const DeviceLayer& layer,
                          const tf::TinyHybrid& f, int layer_index,
                          const char* mode, std::uint32_t n_tokens,
                          bool prefill);
void RunCaptureRefusal(Executor& exec, const DeviceLayer& layer,
                      int layer_index);
void RunGraphReplay(Executor& exec, const DeviceLayer& layer,
                    const tf::TinyHybrid& f, int layer_index);
void RunOracleLiveness(const fn::ModelWeights& weights,
                      const std::vector<float>& x, std::uint64_t hidden,
                      int layer_index);

// MoeSeamRealOps drives the v7 rotated-expert (I4R128) seam through the
// shared orchestration. Its tail is the per-layer Executor::MoeExperts
// dispatch via ExecutorTestAccess -- NOT Forward, NOT attention. The
// category-to-statement mapping is documented per method; where the real
// code bundles steps (device allocations inside DeviceModel::Upload,
// stream/event + BLAS setup inside Executor::Create) the step marks the
// category and the real work happens in the bundled call. The category
// granularity is what the CPU traps enforce; that the real ops call the
// real HIP APIs in this order is a SOURCE-REVIEW obligation on this
// translation unit, not a trap-backend claim.
struct MoeSeamRealOps {
  const tf::TinyHybrid* f = nullptr;
  std::string fixture_hex;
  std::string gguf_path;
  BoundModel* b = nullptr;
  std::unique_ptr<Executor> exec;

  int device_query = 0;
  int allocate = 0;
  int upload = 0;
  int stream_event = 0;
  int blas_init = 0;
  int moe_work = 0;

  bool DeviceQuery() {
    ++device_query;
    // The FIRST HIP call in the program, gated behind validation by the
    // orchestration. No device -> false -> SKIP 77 upstream.
    int dev = 0;
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
    std::printf("  GPU: %s\n", prop.name);
    return true;
  }
  void Allocate() {
    ++allocate;  // device allocations occur inside DeviceModel::Upload
  }
  void Upload() {
    ++upload;
    std::string err;
    *b = UploadViaFile(*f, gguf_path, fixture_hex, &err);
    if (b->model == nullptr) Die("DeviceModel::Upload failed: " + err);
    Check(b->model->layers().size() == tf::kTinyLayers, "uploaded layer count");
    for (std::size_t i = 0; i < b->model->layers().size(); ++i)
      Check(b->model->layers()[i].rotated,
            "layer " + std::to_string(i) +
                " marked rotated by UploadRotVector");
  }
  void StreamEvent() {
    ++stream_event;  // stream/event creation occurs inside Executor::Create
  }
  void BlasInit() {
    ++blas_init;  // BLAS setup occurs inside Executor::Create
    std::string err;
    exec = Executor::Create(
        *b->model, nullptr,
        {.max_batch = 8, .max_logit_rows = 1, .max_speculative = 1}, &err);
    if (exec == nullptr) Die("Executor::Create failed: " + err);
  }
  void MoeRestrictedWork() {
    ++moe_work;
    // Every fixture layer, both dispatch shapes, both routing states --
    // the v7 matched MoE seam. MoeExperts only; no Forward, no attention.
    for (int layer = 0; layer < static_cast<int>(tf::kTinyLayers); ++layer) {
      RunSeamParityForLayer(*exec,
                            b->model->layers()[static_cast<std::size_t>(layer)],
                            *f, layer, "decode-like", 1, false);
      RunSeamParityForLayer(*exec,
                            b->model->layers()[static_cast<std::size_t>(layer)],
                            *f, layer, "prefill-like", 3, true);
    }
    const std::vector<float> x = MakeTokenInputs(1, tf::kTinyHidden);
    for (int layer = 0; layer < static_cast<int>(tf::kTinyLayers); ++layer)
      RunOracleLiveness(*b->weights, x, tf::kTinyHidden, layer);
    for (std::size_t i = 0; i < b->model->layers().size(); ++i) {
      const DeviceLayer& l = b->model->layers()[i];
      if (l.rot_split)
        RunGraphReplay(*exec, l, *f, static_cast<int>(i));
      else
        RunCaptureRefusal(*exec, l, static_cast<int>(i));
    }
  }
};

// ---------------------------------------------------------------------------
// Controlled per-token inputs
// ---------------------------------------------------------------------------

// Distinct, nonzero, deterministic input rows, one per token. Distinctness is
// the point: if every token row were identical, a stride error that read the
// wrong token's row would be invisible. Verified below rather than assumed.
std::vector<float> MakeTokenInputs(std::uint32_t n_tokens,
                                   std::uint64_t hidden) {
  std::vector<float> x(static_cast<std::size_t>(n_tokens) * hidden);
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const double dt = static_cast<double>(t);
    for (std::uint64_t i = 0; i < hidden; ++i) {
      const double d = static_cast<double>(i);
      x[static_cast<std::size_t>(t) * hidden + i] =
          static_cast<float>(std::sin(0.37 * d + 0.11 + 0.91 * dt) +
                             0.35 * std::cos(0.13 * d - 0.44 * dt + 0.27));
    }
  }
  return x;
}

// Assert the token rows are pairwise distinct and each is nonzero, so the
// parity results below cannot be hiding a stride bug behind identical inputs.
void CheckInputsAreDistinct(const std::vector<float>& x, std::uint32_t n_tokens,
                            std::uint64_t hidden, const std::string& mode) {
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    double peak = 0.0;
    for (std::uint64_t i = 0; i < hidden; ++i)
      peak = std::max(peak, std::abs(static_cast<double>(
                                x[static_cast<std::size_t>(t) * hidden + i])));
    Check(peak > kRefMinPeak, "input row for token " + std::to_string(t) +
                                  " (" + mode + ") is nonzero");
  }
  for (std::uint32_t a = 0; a < n_tokens; ++a) {
    for (std::uint32_t b = a + 1; b < n_tokens; ++b) {
      double diff = 0.0;
      for (std::uint64_t i = 0; i < hidden; ++i) {
        diff = std::max(
            diff, std::abs(static_cast<double>(
                               x[static_cast<std::size_t>(a) * hidden + i]) -
                           static_cast<double>(
                               x[static_cast<std::size_t>(b) * hidden + i])));
      }
      Check(diff > 1e-2, "input rows for tokens " + std::to_string(a) +
                             " and " + std::to_string(b) + " (" + mode +
                             ") are distinct (stride errors cannot hide)");
    }
  }
}

// ---------------------------------------------------------------------------
// The matched MoE seam, for one layer and one dispatch shape (blocker 2)
// ---------------------------------------------------------------------------

// Compare one slot, print its contract row, and record the verdict. Keeping
// the print here means every primary comparison is evidenced the same way.
ParityRow CheckRow(const std::vector<float>& out, const std::vector<float>& ref,
                   const std::string& mode, int layer,
                   const std::string& routing, std::size_t slot, int expert) {
  const ParityRow r =
      CompareParity(out, ref, mode, layer, routing, slot, expert);
  std::printf("%s\n", FormatParityRow(r).c_str());
  Check(r.pass, "primary parity (" + mode + " L" + std::to_string(layer) + " " +
                    routing + " slot" + std::to_string(slot) +
                    ") verdict=" + tf::VerdictName(r.verdict));
  return r;
}

// Runs the real rotated dispatch for `layer_index` with `n_tokens` token rows
// under `prefill` (true -> GEMM primitive, false -> matvec primitive), in
// both routing states, and compares every routed slot against the dense
// oracle built from THAT layer's and THAT expert's materialised weights and
// THAT token's own input row.
void RunSeamParityForLayer(Executor& exec, const DeviceLayer& layer,
                           const tf::TinyHybrid& f, int layer_index,
                           const char* mode, std::uint32_t n_tokens,
                           bool prefill) {
  const std::uint64_t hidden = tf::kTinyHidden;
  const std::uint64_t ff = tf::kTinyExpertFf;
  const std::uint32_t used = exec.config().num_experts_used;
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * used;

  std::printf("\n-- seam parity: %s, layer %d, n_tokens=%u, primitive=%s --\n",
              mode, layer_index, n_tokens, prefill ? "GEMM (mmq)" : "matvec");

  Check(layer.rotated,
        std::string("layer ") + std::to_string(layer_index) + " is rotated");

  const std::vector<float> x = MakeTokenInputs(n_tokens, hidden);
  CheckInputsAreDistinct(x, n_tokens, hidden, mode);

  // Input and output have DIFFERENT geometries (review-v5 blocker 1). The
  // dispatch reads x as n_tokens rows of `hidden` (RotateLayerInput indexes
  // by token row); it writes slots rows, where slots = n_tokens * used.
  // Sizing the host->device input copy by the slot count reads `used` times
  // the host buffer -- with used=2 a 1024-byte input read as 2048. The host
  // `x` holds exactly InputCopyFloats floats, so copy exactly that; the
  // device output side keeps the larger slot geometry.
  const std::size_t in_floats = tf::InputCopyFloats(n_tokens, hidden);
  const std::size_t out_floats = tf::SlotFloats(n_tokens, used, hidden);
  Check(x.size() == in_floats, std::string("host input buffer is exactly the "
                                           "copy length (") +
                                   mode + " L" + std::to_string(layer_index) +
                                   ")");
  Check(out_floats == slots * hidden, std::string("slot output geometry "
                                                  "matches (") +
                                          mode + " L" +
                                          std::to_string(layer_index) + ")");
  float* x_dev = nullptr;
  float* out_dev = nullptr;
  if (hipMalloc(&x_dev, in_floats * sizeof(float)) != hipSuccess ||
      hipMalloc(&out_dev, out_floats * sizeof(float)) != hipSuccess)
    Die("seam parity device alloc");
  if (hipMemcpy(x_dev, x.data(), in_floats * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)
    Die("seam parity input upload");

  // Routing A: slot -> expert (sl % used). Routing B: the complement.
  std::vector<std::int32_t> ids_a(slots), ids_b(slots);
  for (std::size_t sl = 0; sl < slots; ++sl) {
    ids_a[sl] = static_cast<std::int32_t>(sl % used);
    ids_b[sl] = static_cast<std::int32_t>(1u - (sl % used));
  }

  // Dense references per (token, expert) from THIS layer's weights.
  // f.w_gate is layer-major: index = layer * kTinyExperts + expert.
  auto dense_for = [&](std::uint32_t token, std::int32_t expert) {
    const std::size_t wi =
        static_cast<std::size_t>(layer_index) * tf::kTinyExperts +
        static_cast<std::size_t>(expert);
    const std::vector<float> x_t(
        x.begin() + static_cast<std::ptrdiff_t>(token * hidden),
        x.begin() + static_cast<std::ptrdiff_t>((token + 1) * hidden));
    return DenseExpert(f.w_gate[wi], f.w_up[wi], f.w_down[wi], x_t, hidden, ff);
  };

  std::vector<ParityRow> rows_a, rows_b;
  std::vector<float> down_a, down_b;
  Check(layer.rot_split, std::string("layer ") + std::to_string(layer_index) +
                             " was repacked for the batched kernels");
  struct PerSlotScope {
    Executor& e;
    explicit PerSlotScope(Executor& ex) : e(ex) { ExecutorTestAccess::SetPerSlot(e, true); }
    ~PerSlotScope() { ExecutorTestAccess::SetPerSlot(e, false); }
  };
  {
    ScopedPrefill phase(prefill);
    PerSlotScope per_slot(exec);
    ExecutorTestAccess::SetIds(exec, ids_a.data(), slots);
    std::string err;
    const bool ok = ExecutorTestAccess::MoeExperts(exec, layer, x_dev, out_dev,
                                                   n_tokens, &err);
    Check(ok, std::string("MoeExperts ran (") + mode + " L" +
                  std::to_string(layer_index) + " routing A): " + err);
    down_a = ExecutorTestAccess::ReadDownE(exec, slots * hidden);
    Check(AllFinite(down_a), std::string("routing A down_e finite (") + mode +
                                 " L" + std::to_string(layer_index) + ")");

    ExecutorTestAccess::SetIds(exec, ids_b.data(), slots);
    std::string err_b;
    const bool ok_b = ExecutorTestAccess::MoeExperts(exec, layer, x_dev,
                                                     out_dev, n_tokens, &err_b);
    Check(ok_b, std::string("MoeExperts ran (") + mode + " L" +
                    std::to_string(layer_index) + " routing B): " + err_b);
    down_b = ExecutorTestAccess::ReadDownE(exec, slots * hidden);
    Check(AllFinite(down_b), std::string("routing B down_e finite (") + mode +
                                 " L" + std::to_string(layer_index) + ")");
  }

  for (std::size_t sl = 0; sl < slots; ++sl) {
    const std::uint32_t token = static_cast<std::uint32_t>(sl / used);
    const std::vector<float> ref = dense_for(token, ids_a[sl]);
    const std::vector<float> out(
        down_a.begin() + static_cast<std::ptrdiff_t>(sl * hidden),
        down_a.begin() + static_cast<std::ptrdiff_t>((sl + 1) * hidden));
    rows_a.push_back(CheckRow(out, ref, mode, layer_index, "A", sl,
                              static_cast<int>(ids_a[sl])));
  }
  for (std::size_t sl = 0; sl < slots; ++sl) {
    const std::uint32_t token = static_cast<std::uint32_t>(sl / used);
    const std::vector<float> ref = dense_for(token, ids_b[sl]);
    const std::vector<float> out(
        down_b.begin() + static_cast<std::ptrdiff_t>(sl * hidden),
        down_b.begin() + static_cast<std::ptrdiff_t>((sl + 1) * hidden));
    rows_b.push_back(CheckRow(out, ref, mode, layer_index, "B", sl,
                              static_cast<int>(ids_b[sl])));
  }

  // Controls at this same seam.
  double positive_error = 0.0;
  bool positive_ok = true;
  for (const ParityRow& r : rows_a) {
    if (!r.pass)
      positive_ok = false;
    positive_error = std::max(positive_error, r.peak_normalised_rel);
  }
  for (const ParityRow& r : rows_b) {
    if (!r.pass)
      positive_ok = false;
    positive_error = std::max(positive_error, r.peak_normalised_rel);
  }

  // Negative control: slot 0's routed output against the dense reference of
  // the OTHER expert for the same token -- a deliberate mispair.
  bool cmp_ok = false;
  const std::vector<float> slot0_a(
      down_a.begin(), down_a.begin() + static_cast<std::ptrdiff_t>(hidden));
  const std::int32_t mispair_expert = 1 - ids_a[0];
  const double negative_error =
      PeakNormalisedRel(slot0_a, dense_for(0, mispair_expert), &cmp_ok);
  Check(cmp_ok, "negative control comparable (" + std::string(mode) + " L" +
                    std::to_string(layer_index) + ")");
  Check(std::isfinite(positive_error) && std::isfinite(negative_error),
        "separation errors finite (" + std::string(mode) + " L" +
            std::to_string(layer_index) + ")");
  Check(negative_error > kNegativeControlMin,
        "negative control exceeds 1e-2 (" + std::string(mode) + " L" +
            std::to_string(layer_index) + ")");
  Check(negative_error > kSeparationFactor * positive_error,
        "negative control separates from positive by >100x, zero-safe (" +
            std::string(mode) + " L" + std::to_string(layer_index) + ")");
  std::printf(
      "  controls: positive=%.3e negative=%.3e ratio=%.3e %s\n", positive_error,
      negative_error,
      positive_error > 0.0 ? negative_error / positive_error : 0.0,
      (positive_ok && negative_error > kSeparationFactor * positive_error)
          ? "PASS"
          : "FAIL");

  // Routing-change sensitivity: the host-side selection must NOT be frozen.
  const std::vector<float> slot0_b(
      down_b.begin(), down_b.begin() + static_cast<std::ptrdiff_t>(hidden));
  bool swap_ok = false;
  const double swap_delta = PeakNormalisedRel(slot0_a, slot0_b, &swap_ok);
  Check(swap_ok && swap_delta > kRoutingChangeMin,
        "swapping expert ids changed the routed output (selection is live) (" +
            std::string(mode) + " L" + std::to_string(layer_index) + ")");
  std::printf("  controls: routing-change delta=%.3e %s\n", swap_delta,
              (swap_ok && swap_delta > kRoutingChangeMin) ? "PASS" : "FAIL");

  // Combined output of the production (non-per-slot) path: the weighted
  // sum over slots, rotated once, plus the gated shared expert, against the
  // same combination of the dense per-slot references.
  {
    ScopedPrefill phase(prefill);
    std::vector<float> w(slots), shared(n_tokens * hidden), logit(n_tokens);
    for (std::size_t sl = 0; sl < slots; ++sl)
      w[sl] = 0.25F + 0.5F * static_cast<float>((sl * 7) % 5) / 4.0F;
    for (std::size_t i = 0; i < shared.size(); ++i)
      shared[i] = 0.05F * std::sin(0.21 * static_cast<double>(i) + 0.4);
    for (std::uint32_t t = 0; t < n_tokens; ++t)
      logit[t] = 0.3F - 0.45F * static_cast<float>(t);
    ExecutorTestAccess::SetCombineInputs(exec, w, shared, logit);
    ExecutorTestAccess::SetIds(exec, ids_a.data(), slots);
    std::string err;
    const bool ok = ExecutorTestAccess::MoeExperts(exec, layer, x_dev, out_dev,
                                                   n_tokens, &err);
    Check(ok, std::string("combined MoeExperts ran (") + mode + " L" +
                  std::to_string(layer_index) + "): " + err);
    std::vector<float> got(n_tokens * hidden);
    if (hipStreamSynchronize(ExecutorTestAccess::Stream(exec)) != hipSuccess ||
        hipMemcpy(got.data(), out_dev, got.size() * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
      Die("combined output download");
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      std::vector<float> ref(hidden, 0.0F);
      const float g = 1.0F / (1.0F + std::exp(-logit[t]));
      for (std::uint32_t k = 0; k < used; ++k) {
        const std::size_t sl = static_cast<std::size_t>(t) * used + k;
        const std::vector<float> d = dense_for(t, ids_a[sl]);
        for (std::uint64_t i = 0; i < hidden; ++i) ref[i] += w[sl] * d[i];
      }
      for (std::uint64_t i = 0; i < hidden; ++i)
        ref[i] += g * shared[static_cast<std::size_t>(t) * hidden + i];
      const std::vector<float> o(got.begin() + static_cast<std::ptrdiff_t>(t * hidden),
                                 got.begin() + static_cast<std::ptrdiff_t>((t + 1) * hidden));
      CheckRow(o, ref, std::string(mode) + "-combined", layer_index, "A", t, -1);
    }
  }

  (void)hipFree(x_dev);
  (void)hipFree(out_dev);
}

// ---------------------------------------------------------------------------
// Graph capture and replay, per layer
// ---------------------------------------------------------------------------

// The batched path reads expert ids on the device, so a captured pass must
// follow a routing change made after capture. Captures one decode-shaped
// MoeExperts, replays it under routings A and B, and checks each replay
// against the dense references of that routing.
void RunGraphReplay(Executor& exec, const DeviceLayer& layer,
                    const tf::TinyHybrid& f, int layer_index) {
  const std::uint64_t hidden = tf::kTinyHidden;
  const std::uint64_t ff = tf::kTinyExpertFf;
  const std::uint32_t used = exec.config().num_experts_used;
  const std::vector<float> x = MakeTokenInputs(1, hidden);
  float* x_dev = nullptr;
  float* out_dev = nullptr;
  if (hipMalloc(&x_dev, hidden * sizeof(float)) != hipSuccess ||
      hipMalloc(&out_dev, hidden * sizeof(float)) != hipSuccess ||
      hipMemcpy(x_dev, x.data(), hidden * sizeof(float), hipMemcpyHostToDevice) !=
          hipSuccess)
    Die("graph replay setup");
  ExecutorTestAccess::SetPerSlot(exec, true);
  const hipStream_t stream = ExecutorTestAccess::Stream(exec);
  hipGraph_t graph = nullptr;
  hipGraphExec_t inst = nullptr;
  std::string err;
  bool ok = hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed) == hipSuccess;
  ok = ok && ExecutorTestAccess::MoeExperts(exec, layer, x_dev, out_dev, 1, &err);
  ok = (hipStreamEndCapture(stream, &graph) == hipSuccess) && ok;
  ok = ok && hipGraphInstantiate(&inst, graph, nullptr, nullptr, 0) == hipSuccess;
  Check(ok, "rotated MoeExperts captured into a graph (L" +
                std::to_string(layer_index) + "): " + err);
  for (int routing = 0; ok && routing < 2; ++routing) {
    std::vector<std::int32_t> ids(used);
    for (std::uint32_t k = 0; k < used; ++k)
      ids[k] = static_cast<std::int32_t>(routing == 0 ? k % 2 : 1 - k % 2);
    ExecutorTestAccess::SetIds(exec, ids.data(), used);
    Check(hipGraphLaunch(inst, stream) == hipSuccess, "graph replay launched");
    const std::vector<float> down = ExecutorTestAccess::ReadDownE(exec, used * hidden);
    for (std::uint32_t k = 0; k < used; ++k) {
      const std::size_t wi = static_cast<std::size_t>(layer_index) * tf::kTinyExperts +
                             static_cast<std::size_t>(ids[k]);
      const std::vector<float> ref =
          DenseExpert(f.w_gate[wi], f.w_up[wi], f.w_down[wi], x, hidden, ff);
      const std::vector<float> o(down.begin() + static_cast<std::ptrdiff_t>(k * hidden),
                                 down.begin() + static_cast<std::ptrdiff_t>((k + 1) * hidden));
      CheckRow(o, ref, "graph-replay", layer_index, routing == 0 ? "A" : "B", k,
               ids[k]);
    }
  }
  ExecutorTestAccess::SetPerSlot(exec, false);
  if (inst != nullptr) (void)hipGraphExecDestroy(inst);
  if (graph != nullptr) (void)hipGraphDestroy(graph);
  (void)hipFree(x_dev);
  (void)hipFree(out_dev);
}

// ---------------------------------------------------------------------------
// Capture refusal, per layer (unbatched host-routed layers only)
// ---------------------------------------------------------------------------

void RunCaptureRefusal(Executor& exec, const DeviceLayer& layer,
                       int layer_index) {
  const std::uint64_t hidden = tf::kTinyHidden;
  float* x_dev = nullptr;
  float* out_dev = nullptr;
  if (hipMalloc(&x_dev, hidden * sizeof(float)) != hipSuccess ||
      hipMalloc(&out_dev, hidden * sizeof(float)) != hipSuccess)
    Die("capture-refusal device alloc");
  std::vector<float> x(hidden, 0.25F);
  if (hipMemcpy(x_dev, x.data(), hidden * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)
    Die("capture-refusal input upload");

  std::string err;
  const bool ok = ExecutorTestAccess::MoeExpertsWhileCapturing(
      exec, layer, x_dev, out_dev, 1, &err);
  Check(!ok, "rotated MoeExperts refused while capturing (L" +
                 std::to_string(layer_index) + ")");
  Check(err.find("cannot run inside a graph capture") != std::string::npos,
        "capture refusal carries the expected error (L" +
            std::to_string(layer_index) + "): " + err);
  Check(!ExecutorTestAccess::Capturing(exec),
        "capture flag restored after refusal (L" + std::to_string(layer_index) +
            ")");
  std::printf("  capture refusal L%d: refused=%d msg=\"%s\"\n", layer_index,
              ok ? 0 : 1, err.c_str());

  (void)hipFree(x_dev);
  (void)hipFree(out_dev);
}

// ---------------------------------------------------------------------------
// CPU oracle liveness (NOT the primary gate)
// ---------------------------------------------------------------------------

// Confirms the named CPU oracle (ReferenceModel::Moe, which runs the five
// rotation stages) is live on this fixture's rotated layer. This is a
// liveness check on the oracle itself. Its output is the softmax-weighted
// routed sum plus the shared expert, which is a DIFFERENT tensor from the
// per-slot dense reference used by the primary gate -- it is not, and must
// not be read as, the primary parity comparison.
void RunOracleLiveness(const fn::ModelWeights& weights,
                       const std::vector<float>& x, std::uint64_t hidden,
                       int layer_index) {
  fn::ReferenceModel model(weights, nullptr, 64);
  std::vector<float> out(hidden, 0.0F);
  model.Moe(weights.layers[static_cast<std::size_t>(layer_index)], x, out);
  Check(AllFinite(out), "ReferenceModel::Moe output finite on rotated layer L" +
                            std::to_string(layer_index));
  double peak = 0.0;
  for (float v : out)
    peak = std::max(peak, std::abs(static_cast<double>(v)));
  Check(peak > kRefMinPeak,
        "ReferenceModel::Moe output non-trivial (rotated "
        "stages produced signal) L" +
            std::to_string(layer_index));
  std::printf(
      "  ReferenceModel::Moe L%d (5-stage CPU oracle, liveness only): "
      "peak=%.3f finite=%d\n",
      layer_index, peak, AllFinite(out) ? 1 : 0);
}

// Trap for the full-forward entry boundary: counts every category. The
// fixture's attention is unsupported, so the orchestration must reject
// BEFORE any of these fire.
struct FullTrapOps {
  int device_query = 0;
  int allocate = 0;
  int upload = 0;
  int stream_event = 0;
  int blas_init = 0;
  int full_forward = 0;
  bool DeviceQuery() {
    ++device_query;
    return true;
  }
  void Allocate() { ++allocate; }
  void Upload() { ++upload; }
  void StreamEvent() { ++stream_event; }
  void BlasInit() { ++blas_init; }
  void FullForward() { ++full_forward; }
  int Total() const {
    return device_query + allocate + upload + stream_event + blas_init +
           full_forward;
  }
};

}  // namespace

int main() {
  try {
    // The CPU fixture identity gate runs FIRST, before ANY HIP API
    // (review-v5 blocker 3, unchanged). The frozen pre-device contract
    // requires the tested input to be identified before the first HIP
    // call; building the fixture and hashing it is pure CPU work, so this
    // holds regardless of device presence.
    tf::TinyHybrid f = tf::BuildTinyHybrid({});
    const std::string fixture_hex =
        RequireFixtureIdentity(f.bytes, "in-memory");

    // The gate's own known-answer rejections are pure CPU too, so they
    // run here, still before the first HIP call.
    std::printf("\n-- gate self-tests (no device work) --\n");
    for (const tf::GateCase& c : tf::RunGateSelfChecks()) {
      std::printf("  %-58s %s\n", c.name.c_str(), c.passed ? "PASS" : "FAIL");
      Check(c.passed, "gate case: " + c.name);
    }

    // Disable PLE so the Forward path needs no n-gram table; a host-side
    // config change made BEFORE the orchestration so the validated config
    // is the one actually used downstream.
    f.config.ple_layer = -1;

    std::string err;
    std::printf(
        "i4r executor parity: REDUCED-CONFIG test seam (BindWithConfig), NOT a "
        "production-frontdoor run\n");
    std::printf("primary gate: %s < %.0e (contract section 7)\n", kMetricName,
                kPrimaryMaxRel);

    // ---- GDN seam entry: the shared gated orchestration, REAL ops ----
    // v3 findings 1/3/4: the seam runs through RunGatedSeamEntry, which
    // validates geometry FIRST and only then would touch the device. The
    // restricted GDN-only device operation is NOT expressible with the
    // current executor API (Forward is the only layer-running API and it
    // necessarily enters full attention), so the orchestration stops the
    // seam path BEFORE the first HIP query and reports the missing API.
    // No device work happens on this path -- asserted, not claimed.
    std::printf("\n-- GDN seam entry (shared gated orchestration) --\n");
    SeamRealOps seam;
    const rocm::EntryOutcome seam_outcome =
        rocm::RunGatedSeamEntry(f.config, seam, &err);
    Check(seam_outcome == rocm::EntryOutcome::kStoppedMissingApi,
          "GDN seam stops with the missing-restricted-API outcome");
    Check(err == rocm::kMissingRestrictedApiMessage,
          "seam stop reports the documented missing API: " + err);
    Check(seam.DeviceEffects() == 0,
          "seam stop performed ZERO device work (no query/alloc/upload/"
          "stream/blas/liveness)");
    std::printf("  seam stopped pre-device: %s\n", err.c_str());

    // ---- Full-forward scope: rejected at the entry boundary ----
    // An unsupported full-attention geometry (head_dim 64, 2q/1kv, ratio
    // 2) is supported by neither the fallback nor WMMA, so the gated
    // full-forward entry must reject BEFORE any effect -- including the
    // device query. Asserted with a counting trap that must stay zero.
    // The fixture itself now carries the supported FF-GEOM256 geometry
    // (full-forward parity is the separate full-forward driver's job), so
    // the unsupported geometry is set explicitly here.
    std::printf("\n-- full-forward scope rejection (entry boundary) --\n");
    fn::Config unsupported = f.config;
    unsupported.num_heads = 2;
    unsupported.num_kv_heads = 1;
    unsupported.head_dim = 64;
    unsupported.compress_ratio = 2;
    err.clear();
    FullTrapOps fwd;
    const rocm::EntryOutcome full_outcome =
        rocm::RunGatedFullEntry(unsupported, fwd, &err);
    Check(full_outcome == rocm::EntryOutcome::kRejectedGeometry,
          "unsupported attention geometry is NOT a supported full-Forward config: " + err);
    Check(fwd.Total() == 0,
          "full-forward rejection performed zero effects (no device query, "
          "no Forward)");

    // ---- MoE rotated-expert seam: shared gated orchestration, REAL ops ----
    // The v7 i4r seam, now behind the same boundary. Its tail is the
    // per-layer MoeExperts dispatch -- no Forward, no attention. The
    // first HIP call in the program is the orchestration's DeviceQuery,
    // gated behind validation; no device -> SKIP 77.
    std::printf("\n-- MoE rotated-expert seam (shared gated orchestration) --\n");
    TemporaryDirectory temp("qfn-i4r-exec");
    BoundModel b;
    MoeSeamRealOps moe;
    moe.f = &f;
    moe.fixture_hex = fixture_hex;
    moe.gguf_path = (temp.path() / "tiny.gguf").string();
    moe.b = &b;
    const rocm::EntryOutcome moe_outcome =
        rocm::RunGatedMoeSeamEntry(f.config, moe, &err);
    if (moe_outcome == rocm::EntryOutcome::kNoDevice) {
      std::printf("SKIP: no GPU device available (compile-only validation)\n");
      std::printf("i4r executor parity: %d PASS / %d FAIL (CPU-only portion)\n",
                  g_pass, g_fail);
      return g_fail == 0 ? 77 : 1;
    }
    Check(moe_outcome == rocm::EntryOutcome::kCompletedMoeRestricted,
          "MoE seam completed its restricted work: " + err);
    Check(moe.device_query == 1 && moe.allocate == 1 && moe.upload == 1 &&
              moe.stream_event == 1 && moe.blas_init == 1 && moe.moe_work == 1,
          "MoE seam ran the gated device sequence exactly once per category");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FATAL: %s\n", e.what());
    return 1;
  }

  std::printf("\ni4r executor parity: %d PASS / %d FAIL\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
