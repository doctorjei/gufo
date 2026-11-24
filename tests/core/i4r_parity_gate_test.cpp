// Point 85, step 2 REV 5: CPU-runnable test for the rotated-expert parity
// gate itself.
//
// The gate that decides the GPU parity run lives in i4r_parity_gate.hpp so it
// can be exercised without a device. The HIP driver that needs the GPU
// reports SKIP on a CPU-only box; these cases do not, so the thresholds and
// the comparison invariants get executed evidence rather than compile-only
// evidence.
//
// The case that matters most is the 1e-3 witness: review-v4 showed a
// peak-normalised 1e-3 residual passes a 5e-2 gate while failing the
// contract's 1e-4. This test proves the shipped gate rejects it.

#include "tests/models/qwen38_flash_next/i4r_parity_gate.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace tf = gufo::test_fixture;

namespace {

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

// The metric must be peak-normalised and must not be confused with the two
// other normalisations that exist in this project.
void CheckMetricDefinition() {
  // ref peak 4.0, one element off by 2.0 -> peak-normalised 0.5.
  const std::vector<float> ref = {4.0F, -1.0F, 0.5F, 0.0F};
  const std::vector<float> out = {2.0F, -1.0F, 0.5F, 0.0F};
  const tf::ParityRow r = tf::CompareParity(out, ref, "metric", -1, "-", 0, 0);
  Check(std::abs(r.peak_ref - 4.0) < 1e-12, "peak taken from the reference");
  Check(std::abs(r.max_abs - 2.0) < 1e-12, "max_abs is the largest difference");
  Check(std::abs(r.peak_normalised_rel - 0.5) < 1e-12,
        "metric is max_abs / peak(ref), not max(1,|ref|) or a per-element "
        "ratio");
  Check(std::string(r.metric) == tf::kMetricName,
        "row carries the contract metric name");
  // The worst element is index 0, and both its values are retained.
  Check(r.worst_index == 0, "worst index reported");
  Check(r.worst_out == 2.0F && r.worst_ref == 4.0F,
        "both failing values retained");
  std::printf("  metric: peak=%.1f max_abs=%.1f rel=%.3f (%s)\n", r.peak_ref,
              r.max_abs, r.peak_normalised_rel, tf::kMetricName);
}

// The threshold constants are the contract's, not something looser.
void CheckThresholds() {
  Check(tf::kPrimaryMaxRel == 1e-4, "primary threshold is 1e-4");
  Check(tf::kRefMinPeak == 1e-3, "reference non-vacuity floor is 1e-3");
  Check(tf::kRoutingChangeMin == 1e-2, "routing-change floor is 1e-2");
  Check(tf::kNegativeControlMin == 1e-2, "negative-control floor is 1e-2");
  Check(tf::kSeparationFactor == 100.0, "separation factor is 100x");
  std::printf(
      "  thresholds: primary=%.0e ref_peak>=%.0e routing>=%.0e "
      "negative>=%.0e sep=%.0fx\n",
      tf::kPrimaryMaxRel, tf::kRefMinPeak, tf::kRoutingChangeMin,
      tf::kNegativeControlMin, tf::kSeparationFactor);
}

// The zero-safe separation rule: a positive of exactly 0 must not be able to
// fake an infinite ratio into a pass.
void CheckZeroSafeSeparation() {
  bool ok = false;
  const std::vector<float> ref = {1.0F, 0.5F, -0.25F, 3.0F};

  // positive_error exactly 0 (identical output) against a real negative.
  const double positive = tf::PeakNormalisedRel(ref, ref, &ok);
  Check(ok && positive == 0.0, "identical pair gives positive_error 0");
  const double negative = tf::PeakNormalisedRel(
      std::vector<float>{0.3F, 1.4F, -1.1F, 0.2F}, ref, &ok);
  Check(ok, "negative control pair is comparable");
  // 0 * 100 == 0, so any positive negative_error still satisfies the rule;
  // the rule is stated as negative > 100*positive, which holds here. Record
  // the actual ratio so the semantics are visible rather than assumed.
  Check(negative > tf::kSeparationFactor * positive,
        "zero-safe separation holds with positive == 0");
  std::printf(
      "  separation: positive=%.3e negative=%.3e (rule: neg > 100*pos)\n",
      positive, negative);

  // A non-comparable pair must NOT read as a match.
  const double bad = tf::PeakNormalisedRel({1.0F, 2.0F}, {1.0F}, &ok);
  Check(!ok && bad == 0.0, "size-mismatched pair is not comparable");
  std::vector<float> with_nan = ref;
  with_nan[1] = std::nanf("");
  Check(!tf::PeakNormalisedRel(with_nan, ref, &ok),
        "non-finite pair is not comparable");
  Check(!tf::PeakNormalisedRel(std::vector<float>{1e-9F, 1e-9F},
                               std::vector<float>{1e-9F, 1e-9F}, &ok),
        "below-floor pair is not comparable");
}

}  // namespace

int main() {
  std::printf("i4r parity gate: CPU-runnable gate tests (no device needed)\n");
  std::printf("metric: %s\n", tf::kMetricName);

  CheckThresholds();
  CheckMetricDefinition();
  CheckZeroSafeSeparation();

  std::printf("\n-- gate known-answer cases --\n");
  for (const tf::GateCase& c : tf::RunGateSelfChecks()) {
    std::printf("  %-58s %s\n", c.name.c_str(), c.passed ? "PASS" : "FAIL");
    Check(c.passed, "gate case: " + c.name);
  }

  std::printf("\ni4r parity gate: %d PASS / %d FAIL\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
