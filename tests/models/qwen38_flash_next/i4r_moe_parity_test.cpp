// Point 85, step 1: the rotated expert path in the CPU reference model.
//
// The rotated storage path (I4R128 codes + the five rotation vectors, applied
// as the five stages of i4r_rotation.hpp) must produce the same MoE output
// as a plain dense matmul of the same mathematical weights,
// W = diag(sv) * Hn * M * Hn * diag(su), materialised by the fixture.
//
// A wrong pairing (gate fed the up vector, down fed the experts' su, a
// missing stage, a rotation applied on the wrong axis) changes the output by
// O(1), far outside the tolerance below.

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "i4r_tiny_fixture.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/reference.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace core = gufo::core;
namespace fn = gufo::models::qwen38_flash_next;
namespace tf = gufo::test_fixture;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<float> Concat(const std::vector<std::vector<float>>& per) {
  std::vector<float> all;
  for (const auto& w : per) {
    all.insert(all.end(), w.begin(), w.end());
  }
  return all;
}

/// Points a TensorRef at one layer's slice of a fixture-wide weight buffer.
/// The fixture appends experts in order, so layer `il` starts at
/// il * experts * rows * cols floats.
fn::TensorRef F32Ref(const std::vector<float>& data, std::uint64_t cols,
                     std::uint64_t rows, std::uint64_t experts,
                     std::uint64_t layer) {
  fn::TensorRef t;
  t.data = data.data() + layer * experts * rows * cols;
  t.type = core::GgmlType::kF32;
  t.cols = cols;
  t.rows = rows;
  t.experts = experts;
  t.name = "materialized";
  return t;
}

void CheckRotatedMatchesPlainDense() {
  const tf::TinyHybrid f = tf::BuildTinyHybrid({});
  std::string error;
  auto reader =
      core::GgufReader::OpenMemory(f.bytes.data(), f.bytes.size(), &error);
  Require(reader != nullptr, "fixture failed to open: " + error);
  auto w = fn::ModelWeights::BindWithConfig(*reader, f.config, &error);
  Require(w.has_value(), "fixture failed to bind: " + error);

  const std::vector<float> gate_all = Concat(f.w_gate);
  const std::vector<float> up_all = Concat(f.w_up);
  const std::vector<float> down_all = Concat(f.w_down);
  fn::ReferenceModel model(*w, nullptr, 64);
  std::vector<float> x(tf::kTinyHidden);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = std::sin(0.37 * static_cast<double>(i) + 0.11F);
  }

  for (std::uint32_t il = 0; il < tf::kTinyLayers; ++il) {
    fn::LayerWeights plain = w->layers[il];
    plain.rotation = fn::LayerRotation{};  // take the plain path
    plain.ffn_gate_exps = F32Ref(gate_all, tf::kTinyHidden, tf::kTinyExpertFf,
                                 tf::kTinyExperts, il);
    plain.ffn_up_exps = F32Ref(up_all, tf::kTinyHidden, tf::kTinyExpertFf,
                               tf::kTinyExperts, il);
    plain.ffn_down_exps = F32Ref(down_all, tf::kTinyExpertFf, tf::kTinyHidden,
                                 tf::kTinyExperts, il);

    std::vector<float> rotated_out(tf::kTinyHidden, 0.0F);
    std::vector<float> plain_out(tf::kTinyHidden, 0.0F);
    model.Moe(w->layers[il], x, rotated_out);
    model.Moe(plain, x, plain_out);

    double peak = 0.0, max_abs = 0.0;
    for (std::size_t i = 0; i < rotated_out.size(); ++i) {
      peak = std::max(peak, std::abs(static_cast<double>(plain_out[i])));
      max_abs = std::max(max_abs, std::abs(static_cast<double>(rotated_out[i]) -
                                           static_cast<double>(plain_out[i])));
      Require(std::isfinite(rotated_out[i]),
              "rotated path produced a non-finite value in layer " +
                  std::to_string(il));
      // Two-sided guard: the dense reference must be finite too, or peak /
      // max_abs can ignore a NaN on the reference side.
      Require(std::isfinite(plain_out[i]),
              "dense reference produced a non-finite value in layer " +
                  std::to_string(il));
    }
    Require(
        peak > 1e-3,
        "the plain reference output is trivially small; the test is vacuous");
    // The rotated path applies four float32 Hadamards (128-point butterflies)
    // around the expert matmuls; the dense reference folds the same transform
    // into W once, also in float32. The residual is the float32 rotation
    // chain, measured at ~3e-4 here. A wiring error is O(1) relative, so
    // 1e-3 still discriminates by three orders of magnitude (see the
    // negative control below).
    const double rel = max_abs / peak;
    Require(rel < 1e-3, "layer " + std::to_string(il) +
                            ": rotated path differs from plain dense by " +
                            std::to_string(rel) + " (max abs " +
                            std::to_string(max_abs) + ", peak " +
                            std::to_string(peak) + ")");
    std::printf("  layer %u: rotated vs dense max_rel=%.3e (peak %.3f) PASS\n",
                il, rel, peak);
  }
}

void CheckWrongPairingIsCaught() {
  // Negative control: swap the gate and up output vectors in the rotated
  // layer. The rotated path must then disagree with the dense reference,
  // proving the comparison above actually discriminates.
  const tf::TinyHybrid f = tf::BuildTinyHybrid({});
  std::string error;
  auto reader =
      core::GgufReader::OpenMemory(f.bytes.data(), f.bytes.size(), &error);
  Require(reader != nullptr, "fixture failed to open: " + error);
  auto w = fn::ModelWeights::BindWithConfig(*reader, f.config, &error);
  Require(w.has_value(), "fixture failed to bind: " + error);

  const std::vector<float> gate_all = Concat(f.w_gate);
  const std::vector<float> up_all = Concat(f.w_up);
  const std::vector<float> down_all = Concat(f.w_down);
  fn::ReferenceModel model(*w, nullptr, 64);
  std::vector<float> x(tf::kTinyHidden, 0.25F);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = std::cos(0.13 * static_cast<double>(i));
  }

  fn::LayerWeights swapped = w->layers[0];
  std::swap(swapped.rotation.sv_gate, swapped.rotation.sv_up);
  std::vector<float> swapped_out(tf::kTinyHidden, 0.0F);
  std::vector<float> plain_out(tf::kTinyHidden, 0.0F);
  model.Moe(swapped, x, swapped_out);
  fn::LayerWeights plain = w->layers[0];
  plain.rotation = fn::LayerRotation{};
  plain.ffn_gate_exps =
      F32Ref(gate_all, tf::kTinyHidden, tf::kTinyExpertFf, tf::kTinyExperts, 0);
  plain.ffn_up_exps =
      F32Ref(up_all, tf::kTinyHidden, tf::kTinyExpertFf, tf::kTinyExperts, 0);
  plain.ffn_down_exps =
      F32Ref(down_all, tf::kTinyExpertFf, tf::kTinyHidden, tf::kTinyExperts, 0);
  model.Moe(plain, x, plain_out);

  double peak = 0.0, max_abs = 0.0;
  for (std::size_t i = 0; i < plain_out.size(); ++i) {
    peak = std::max(peak, std::abs(static_cast<double>(plain_out[i])));
    max_abs = std::max(max_abs, std::abs(static_cast<double>(swapped_out[i]) -
                                         static_cast<double>(plain_out[i])));
  }
  Require(peak > 1e-3, "negative-control reference is trivially small");
  Require(max_abs / peak > 1e-2,
          "swapping the gate/up rotation vectors did not change the output; "
          "the parity check would not catch a mispairing");
  std::printf(
      "  negative control: mispaired vectors differ by %.3e (max rel) PASS\n",
      max_abs / peak);
}

}  // namespace

int main() {
  try {
    CheckRotatedMatchesPlainDense();
    CheckWrongPairingIsCaught();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
  std::puts("i4r moe parity: ALL PASS");
  return 0;
}
