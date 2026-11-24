// Point 85, step 1: the loader side of the I4R128 hybrid path.
//
// Covers the routed-expert allowlist, the rotation-vector pairing table, and
// the rotation marker contract (a rotated file must carry the marker; a
// loader must never read rotated codes as plain). Every case runs against an
// in-memory tiny GGUF built by i4r_tiny_fixture.hpp.

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#include "i4r_tiny_fixture.hpp"
#include "src/core/gguf_reader.hpp"
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

std::optional<fn::ModelWeights> Bind(const tf::TinyHybrid& f,
                                     std::string* error) {
  auto reader =
      core::GgufReader::OpenMemory(f.bytes.data(), f.bytes.size(), error);
  if (reader == nullptr) {
    return std::nullopt;
  }
  return fn::ModelWeights::BindWithConfig(*reader, f.config, error);
}

void ExpectBinds(const tf::TinyHybrid& f, const std::string& what) {
  std::string error;
  auto w = Bind(f, &error);
  if (!w.has_value()) {
    throw std::runtime_error(what +
                             ": expected bind to succeed, got: " + error);
  }
  return;
}

void ExpectFailsWith(const tf::TinyHybrid& f, const std::string& needle,
                     const std::string& what) {
  std::string error;
  auto w = Bind(f, &error);
  if (w.has_value()) {
    throw std::runtime_error(what + ": expected bind to fail with '" + needle +
                             "', but it succeeded");
  }
  if (error.find(needle) == std::string::npos) {
    throw std::runtime_error(what + ": expected error to contain '" + needle +
                             "', got: " + error);
  }
}

void CheckRotatedBinds() {
  const tf::TinyHybrid f = tf::BuildTinyHybrid({});
  std::string error;
  auto w = Bind(f, &error);
  if (!w.has_value()) {
    throw std::runtime_error("rotated hybrid failed to bind: " + error);
  }
  Require(w->layers.size() == tf::kTinyLayers, "wrong layer count");
  for (std::size_t il = 0; il < w->layers.size(); ++il) {
    const auto& l = w->layers[il];
    Require(l.rotation.rotated,
            "layer " + std::to_string(il) + " not marked rotated");
    Require(l.ffn_gate_exps.type == core::GgmlType::kI4R128,
            "gate expert type is not I4R128 (allowlist did not accept it)");
    Require(l.rotation.su_experts.cols == tf::kTinyHidden,
            "su_experts has the wrong length");
    Require(l.rotation.sv_gate.cols == tf::kTinyExpertFf,
            "sv_gate has the wrong length");
    Require(l.rotation.sv_up.cols == tf::kTinyExpertFf,
            "sv_up has the wrong length");
    Require(l.rotation.su_down.cols == tf::kTinyExpertFf,
            "su_down has the wrong length");
    Require(l.rotation.sv_down.cols == tf::kTinyHidden,
            "sv_down has the wrong length");
    for (const auto* ref :
         {&l.rotation.su_experts, &l.rotation.sv_gate, &l.rotation.sv_up,
          &l.rotation.su_down, &l.rotation.sv_down}) {
      Require(!ref->empty(),
              "layer " + std::to_string(il) + " is missing a rotation vector");
      Require(ref->type == core::GgmlType::kF16,
              "rotation vector is not F16 in layer " + std::to_string(il));
    }
    // The pairing table names each vector for the role it serves.
    Require(l.rotation.su_experts.name ==
                "blk." + std::to_string(il) + ".ffn_experts_rot.su",
            "gate/up input vector not paired to ffn_experts_rot.su");
    Require(l.rotation.sv_gate.name ==
                "blk." + std::to_string(il) + ".ffn_gate_exps_rot.sv",
            "gate output vector not paired to ffn_gate_exps_rot.sv");
    Require(l.rotation.sv_up.name ==
                "blk." + std::to_string(il) + ".ffn_up_exps_rot.sv",
            "up output vector not paired to ffn_up_exps_rot.sv");
    Require(l.rotation.su_down.name ==
                "blk." + std::to_string(il) + ".ffn_down_exps_rot.su",
            "down input vector not paired to ffn_down_exps_rot.su");
    Require(l.rotation.sv_down.name ==
                "blk." + std::to_string(il) + ".ffn_down_exps_rot.sv",
            "down output vector not paired to ffn_down_exps_rot.sv");
  }
  std::puts("  rotated hybrid binds, pairing table resolves");
}

void CheckPlainBinds() {
  tf::HybridVariant v;
  v.rotated = false;
  ExpectBinds(tf::BuildTinyHybrid(v), "plain Q4_K hybrid");
  std::string error;
  auto w = Bind(tf::BuildTinyHybrid(v), &error);
  Require(w.has_value() && !w->layers[0].rotation.rotated,
          "plain experts must not be marked rotated");
  std::puts("  plain (unrotated) hybrid binds unchanged");
}

void CheckMarkerEnforcement() {
  tf::HybridVariant no_marker;
  no_marker.marker = false;
  ExpectFailsWith(tf::BuildTinyHybrid(no_marker), "rotation marker",
                  "missing marker");

  tf::HybridVariant wrong_marker;
  wrong_marker.marker_value = "hadamard64";
  ExpectFailsWith(tf::BuildTinyHybrid(wrong_marker), "cannot decode",
                  "wrong marker value");
  std::puts("  missing or wrong rotation marker is a hard error");
}

void CheckMissingRotationVector() {
  const char* names[5] = {"ffn_experts_rot.su", "ffn_gate_exps_rot.sv",
                          "ffn_up_exps_rot.sv", "ffn_down_exps_rot.su",
                          "ffn_down_exps_rot.sv"};
  for (int i = 0; i < 5; ++i) {
    tf::HybridVariant v;
    v.drop_rot = i;
    ExpectFailsWith(tf::BuildTinyHybrid(v), names[i],
                    std::string("missing rotation vector ") + names[i]);
  }
  std::puts("  every rotation vector is required");
}

void CheckRotationVectorDtype() {
  tf::HybridVariant v;
  v.rot_as_f32 = true;
  ExpectFailsWith(tf::BuildTinyHybrid(v), "unsupported format",
                  "F32 rotation vector");
  std::puts("  rotation vectors must be F16");
}

void CheckMixedRotatedPlain() {
  tf::HybridVariant v;
  v.mixed = true;
  ExpectFailsWith(tf::BuildTinyHybrid(v), "mixes rotated", "mixed layer");
  std::puts("  a half-rotated layer is rejected");
}

}  // namespace

int main() {
  try {
    CheckRotatedBinds();
    CheckPlainBinds();
    CheckMarkerEnforcement();
    CheckMissingRotationVector();
    CheckRotationVectorDtype();
    CheckMixedRotatedPlain();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
  std::puts("i4r loader: ALL PASS");
  return 0;
}
