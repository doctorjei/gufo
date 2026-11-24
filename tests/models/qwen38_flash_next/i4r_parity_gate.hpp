#ifndef GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_I4R_PARITY_GATE_HPP_
#define GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_I4R_PARITY_GATE_HPP_

// The parity gate for the rotated (I4R128) expert path: the thresholds, the
// metric, and the comparison invariants that the contract declares.
//
// This lives in a header rather than inside the HIP driver so the gate can be
// EXERCISED on a CPU-only box. The driver that needs a device reports SKIP
// there; the gate itself has no device dependency, so its known-answer cases
// (including the 1e-3 residual that a loosened gate would wrongly accept)
// run for real in i4r_parity_gate_test. One implementation, one set of
// numbers, no drift between what the driver enforces and what is tested.
//
// Everything here is pure: no I/O, no allocation policy, no globals beyond
// constexpr thresholds. Callers print and assert.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace gufo::test_fixture {

// ---------------------------------------------------------------------------
// Predeclared thresholds (PARITY-CONTRACT.md section 7). Agreed BEFORE the
// run; widening them after a result is not permitted.
// ---------------------------------------------------------------------------

/// Primary parity gate: peak-normalised max|out-ref| / peak|ref|.
inline constexpr double kPrimaryMaxRel = 1e-4;
/// Non-vacuity floor: a reference at or below this peak cannot establish
/// parity, so an all-zero reference cannot fake a pass.
inline constexpr double kRefMinPeak = 1e-3;
/// Routing-change sensitivity: changed routing must move the routed output
/// by more than this.
inline constexpr double kRoutingChangeMin = 1e-2;
/// Negative control (deliberate mispair) must exceed this.
inline constexpr double kNegativeControlMin = 1e-2;
/// Zero-safe separation: negative_error > 100 * positive_error.
inline constexpr double kSeparationFactor = 100.0;

/// The single metric name attached to every primary number. Contract section
/// 6: peak-normalised -- NOT the dispatch per-element metric and NOT
/// GPU44's max(1, |ref|) denominator.
inline constexpr const char* kMetricName =
    "peak_normalised_max_abs_over_peak_ref";

inline bool AllFinite(const std::vector<float>& v) {
  for (float f : v)
    if (!std::isfinite(f))
      return false;
  return true;
}

// Render a magnitude with scientific notation and enough significant digits
// that a small failing value survives the round trip into the raw evidence.
// std::to_string() gives six FRACTIONAL digits, which collapses 2.858e-07 to
// "0.000000" and can make two different failing residuals print identically
// -- erasing exactly the difference a reviewer needs to see. %.9e keeps nine
// significant digits across the whole float range, so 2.858e-07 prints as
// 2.858000000e-07 and stays distinguishable from, say, 3.1e-07.
inline std::string FmtSci(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.9e", v);
  return std::string(buf);
}

// ---------------------------------------------------------------------------
// Copy geometry (review-v5 blocker 1).
//
// The rotated dispatch reads its INPUT as n_tokens rows of `hidden` floats
// (RotateLayerInput indexes x by token row), but it WRITES slots rows where
// slots = n_tokens * num_experts_used. Those are different geometries and
// must not be conflated: using the slot count as the host->device copy length
// reads slots*hidden floats out of a host buffer that only holds
// n_tokens*hidden. With used=2 that is a 2x over-read of the caller's
// memory. These helpers are the single source of truth for the two sizes so
// the driver cannot mix them up, and the CPU copy-bound test pins them.
// ---------------------------------------------------------------------------

/// Floats the host input buffer holds -- and the ONLY amount that may be
/// copied from it.
inline std::size_t InputCopyFloats(std::uint32_t n_tokens,
                                   std::uint64_t hidden) {
  return static_cast<std::size_t>(n_tokens) * static_cast<std::size_t>(hidden);
}

/// Floats the slot/output side occupies (slots x hidden).
inline std::size_t SlotFloats(std::uint32_t n_tokens, std::uint32_t used,
                              std::uint64_t hidden) {
  return static_cast<std::size_t>(n_tokens) * static_cast<std::size_t>(used) *
         static_cast<std::size_t>(hidden);
}

/// Why a comparison did not pass. Ordered by the check that fired: the first
/// failing invariant decides the verdict, so a size mismatch is never
/// reported as a mere threshold miss.
enum class ParityVerdict {
  kPass,
  kSizeMismatch,
  kNonFinite,
  kVacuousRef,
  kOverThreshold,
};

inline const char* VerdictName(ParityVerdict v) {
  switch (v) {
    case ParityVerdict::kPass:
      return "PASS";
    case ParityVerdict::kSizeMismatch:
      return "FAIL_SIZE_MISMATCH";
    case ParityVerdict::kNonFinite:
      return "FAIL_NON_FINITE";
    case ParityVerdict::kVacuousRef:
      return "FAIL_VACUOUS_REFERENCE";
    case ParityVerdict::kOverThreshold:
      return "FAIL_OVER_THRESHOLD";
  }
  return "FAIL_UNKNOWN";
}

/// One comparison's full evidence, including the failing index and both
/// values so a failure is diagnosable from the log alone (contract section
/// 9: "Any assertion failure must print the failing index, both values, and
/// the metric name").
struct ParityRow {
  std::string mode;     // "decode-like" | "prefill-like" | test label
  int layer = -1;       // fixture layer index, -1 for gate self-tests
  std::string routing;  // "A" | "B" | "-"
  std::size_t slot = 0;
  int expert = -1;
  bool size_match = false;
  std::size_t out_size = 0;  ///< actual operand sizes, so a mismatch prints
  std::size_t ref_size = 0;  ///< the real lengths, not just "mismatch"
  bool finite_out = false;
  bool finite_ref = false;
  bool ref_nonvacuous = false;
  double peak_ref = 0.0;
  double max_abs = 0.0;
  double peak_normalised_rel = 0.0;
  std::size_t worst_index = 0;  ///< index of the max_abs element
  float worst_out = 0.0F;
  float worst_ref = 0.0F;
  std::size_t nonfinite_index = 0;  ///< first non-finite element, either side
  ParityVerdict verdict = ParityVerdict::kSizeMismatch;
  bool pass = false;
  const char* metric = kMetricName;
};

// Compare one routed slot against its own dense reference. Every invariant
// the contract declares is enforced here rather than assumed:
//   * equal, non-empty sizes -- no silent truncation or overrun,
//   * finiteness on BOTH sides -- a NaN cannot score as zero error,
//   * a non-vacuous reference (peak > 1e-3) -- an all-zero reference cannot
//     fake a pass,
//   * the metric is peak-normalised and named.
ParityRow CompareParity(const std::vector<float>& out,
                        const std::vector<float>& ref, const std::string& mode,
                        int layer, const std::string& routing, std::size_t slot,
                        int expert) {
  ParityRow r;
  r.mode = mode;
  r.layer = layer;
  r.routing = routing;
  r.slot = slot;
  r.expert = expert;

  r.out_size = out.size();
  r.ref_size = ref.size();
  r.size_match = out.size() == ref.size() && !out.empty();
  if (!r.size_match) {
    r.verdict = ParityVerdict::kSizeMismatch;
    return r;
  }

  r.finite_out = AllFinite(out);
  r.finite_ref = AllFinite(ref);
  if (!r.finite_out || !r.finite_ref) {
    for (std::size_t i = 0; i < out.size(); ++i) {
      if (!std::isfinite(out[i]) || !std::isfinite(ref[i])) {
        r.nonfinite_index = i;
        r.worst_out = out[i];
        r.worst_ref = ref[i];
        break;
      }
    }
    r.verdict = ParityVerdict::kNonFinite;
    return r;
  }

  for (std::size_t i = 0; i < out.size(); ++i) {
    const double ref_abs = std::abs(static_cast<double>(ref[i]));
    if (ref_abs > r.peak_ref)
      r.peak_ref = ref_abs;
    const double diff =
        std::abs(static_cast<double>(out[i]) - static_cast<double>(ref[i]));
    if (diff > r.max_abs) {
      r.max_abs = diff;
      r.worst_index = i;
      r.worst_out = out[i];
      r.worst_ref = ref[i];
    }
  }

  r.ref_nonvacuous = r.peak_ref > kRefMinPeak;
  if (!r.ref_nonvacuous) {
    r.verdict = ParityVerdict::kVacuousRef;
    return r;
  }

  r.peak_normalised_rel = r.max_abs / r.peak_ref;
  r.pass = r.peak_normalised_rel < kPrimaryMaxRel;
  r.verdict = r.pass ? ParityVerdict::kPass : ParityVerdict::kOverThreshold;
  return r;
}

// Render one row's evidence as a single line, plus a second diagnostic line
// when it failed. Identical formatting wherever the gate is reported. All
// fractional values go through FmtSci (scientific, 9 significant digits) so
// a small failing residual is preserved in the raw evidence rather than
// rounded to zero by std::to_string's six fractional digits.
inline std::string FormatParityRow(const ParityRow& r) {
  std::string line = "  [" + r.mode + " L" + std::to_string(r.layer) + " " +
                     r.routing + " slot" + std::to_string(r.slot) + " e" +
                     std::to_string(r.expert) + "] ";
  if (r.verdict == ParityVerdict::kSizeMismatch) {
    line += "FAIL_SIZE_MISMATCH out_len=" + std::to_string(r.out_size) +
            " ref_len=" + std::to_string(r.ref_size) + " metric=";
    line += r.metric;
    return line;
  }
  if (r.verdict == ParityVerdict::kNonFinite) {
    line += "FAIL_NON_FINITE at index " + std::to_string(r.nonfinite_index) +
            " out=" + FmtSci(static_cast<double>(r.worst_out)) +
            " ref=" + FmtSci(static_cast<double>(r.worst_ref)) + " metric=";
    line += r.metric;
    return line;
  }
  if (r.verdict == ParityVerdict::kVacuousRef) {
    line += "FAIL_VACUOUS_REFERENCE peak=" + FmtSci(r.peak_ref) +
            " <= 1e-03 metric=";
    line += r.metric;
    return line;
  }
  line += "peak=" + FmtSci(r.peak_ref) + " max_abs=" + FmtSci(r.max_abs) + " " +
          r.metric + "=" + FmtSci(r.peak_normalised_rel) + " " +
          VerdictName(r.verdict);
  if (!r.pass) {
    line += "\n      failing index " + std::to_string(r.worst_index) +
            " out=" + FmtSci(static_cast<double>(r.worst_out)) +
            " ref=" + FmtSci(static_cast<double>(r.worst_ref)) +
            " diff=" + FmtSci(r.max_abs) + " threshold=1e-04 metric=";
    line += r.metric;
  }
  return line;
}

// Peak-normalised ratio for the control lines, with the same guards folded
// in. `comparable` is false when the inputs cannot support a comparison
// (size mismatch, non-finite, or a reference below the non-vacuity floor),
// in which case the returned value is 0 and must not be read as a match.
inline double PeakNormalisedRel(const std::vector<float>& a,
                                const std::vector<float>& b, bool* comparable) {
  if (comparable == nullptr)
    return 0.0;
  *comparable = false;
  if (a.size() != b.size() || a.empty())
    return 0.0;
  if (!AllFinite(a) || !AllFinite(b))
    return 0.0;
  double peak = 0.0, max_abs = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    peak = std::max(peak, std::abs(static_cast<double>(b[i])));
    max_abs = std::max(max_abs, std::abs(static_cast<double>(a[i]) -
                                         static_cast<double>(b[i])));
  }
  if (peak <= kRefMinPeak)
    return 0.0;
  *comparable = true;
  return max_abs / peak;
}

// ---------------------------------------------------------------------------
// Gate self-tests: known-answer cases with no device dependency.
//
// The case list lives here once, so the driver and the CPU test assert on
// exactly the same thing. The important case is the 1e-3 witness: review
// v4 showed a 1e-3 peak-normalised residual passes a 5e-2 gate while
// failing the contract's 1e-4, so the gate must be proven to reject it.
// ---------------------------------------------------------------------------

struct GateCase {
  std::string name;
  bool passed;
};

inline std::vector<float> GateReference() {
  std::vector<float> ref(64, 0.0F);
  ref[0] = 1.0F;
  for (std::size_t i = 1; i < ref.size(); ++i)
    ref[i] = static_cast<float>(0.5 - 0.01 * static_cast<double>(i));
  return ref;
}

inline std::vector<GateCase> RunGateSelfChecks() {
  std::vector<GateCase> cases;
  const std::vector<float> ref = GateReference();

  // (a) The witness: a 1e-3 peak-normalised residual. Must FAIL 1e-4.
  std::vector<float> witness = ref;
  witness[7] = ref[7] + 1.0e-3F;
  const ParityRow w =
      CompareParity(witness, ref, "gate-witness", -1, "-", 0, 0);
  cases.push_back({"gate rejects a 1e-3 peak-normalised residual",
                   w.verdict == ParityVerdict::kOverThreshold && !w.pass});
  cases.push_back({"witness is above the 1e-4 primary threshold",
                   w.peak_normalised_rel > kPrimaryMaxRel});
  // Documents the gap that was closed: this same residual would have passed
  // the 5e-2 gate that v4 used.
  cases.push_back({"witness would have passed the removed 5e-2 gate",
                   w.peak_normalised_rel < 5e-2});

  // (b) A residual well inside 1e-4 must PASS.
  std::vector<float> good = ref;
  good[3] = ref[3] + 1.0e-7F;
  const ParityRow g = CompareParity(good, ref, "gate-positive", -1, "-", 0, 0);
  cases.push_back({"gate accepts a 1e-7 peak-normalised residual", g.pass});

  // (c) Size mismatch is a hard rejection, not a silent partial compare.
  std::vector<float> short_out = good;
  short_out.pop_back();
  const ParityRow s = CompareParity(short_out, ref, "gate-size", -1, "-", 0, 0);
  cases.push_back({"gate rejects a size mismatch",
                   !s.size_match && s.verdict == ParityVerdict::kSizeMismatch});
  cases.push_back({"gate rejects an empty pair",
                   CompareParity({}, {}, "gate-empty", -1, "-", 0, 0).verdict ==
                       ParityVerdict::kSizeMismatch});

  // (d) Non-finite on either side is rejected; a NaN cannot score as zero.
  const std::vector<std::pair<const char*, float>> poisons = {
      {"NaN", std::nanf("")},
      {"+Inf", std::numeric_limits<float>::infinity()},
      {"-Inf", -std::numeric_limits<float>::infinity()},
  };
  for (const auto& poison : poisons) {
    std::vector<float> out_bad = good;
    out_bad[5] = poison.second;
    const ParityRow ob =
        CompareParity(out_bad, ref, "gate-poison-out", -1, "-", 0, 0);
    cases.push_back(
        {std::string("gate rejects ") + poison.first + " on the output side",
         !ob.finite_out && ob.verdict == ParityVerdict::kNonFinite});

    std::vector<float> ref_bad = good;
    ref_bad[9] = poison.second;
    const ParityRow rb =
        CompareParity(good, ref_bad, "gate-poison-ref", -1, "-", 0, 0);
    cases.push_back(
        {std::string("gate rejects ") + poison.first + " on the reference side",
         !rb.finite_ref && rb.verdict == ParityVerdict::kNonFinite});
  }

  // (e) An all-zero reference is vacuous even with a zero difference.
  std::vector<float> zeros(ref.size(), 0.0F);
  const ParityRow z =
      CompareParity(zeros, zeros, "gate-zero-ref", -1, "-", 0, 0);
  cases.push_back(
      {"gate rejects a zero (vacuous) reference",
       !z.ref_nonvacuous && z.verdict == ParityVerdict::kVacuousRef});

  // (f) A reference below the non-vacuity floor is rejected too, even if it
  //     matches exactly.
  std::vector<float> tiny_ref(ref.size(), 1.0e-9F);
  const ParityRow t =
      CompareParity(tiny_ref, tiny_ref, "gate-tiny-ref", -1, "-", 0, 0);
  cases.push_back(
      {"gate rejects a reference below the 1e-3 floor",
       !t.ref_nonvacuous && t.verdict == ParityVerdict::kVacuousRef});

  // (g) Invariant precedence: a size mismatch outranks everything else.
  std::vector<float> nan_and_short = short_out;
  nan_and_short[0] = std::nanf("");
  const ParityRow ns =
      CompareParity(nan_and_short, ref, "gate-precedence", -1, "-", 0, 0);
  cases.push_back({"size mismatch outranks a non-finite value",
                   ns.verdict == ParityVerdict::kSizeMismatch});

  return cases;
}

}  // namespace gufo::test_fixture

#endif  // GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_I4R_PARITY_GATE_HPP_
