// Point 85, step 2: the host-side rotated-expert dispatch (option (a)).
//
// The dispatch in i4r_dispatch.hpp sequences the five rotation stages across
// a trunk layer -- buffer, scale, expert, block counts -- against an abstract
// RotatedOps. This test wires those ops to CPU stand-ins (reference rotate +
// raw I4R128 dot) and checks every slot against RotatedGemv, the existing
// CPU oracle for one full rotated matvec.
//
// What this pins down: stage order, the gate/up/down scale pairing, the
// slot -> (token, expert) mapping, the per-expert byte strides, and that the
// hoisted input rotation (xr reused by every expert) equals the per-expert
// full gemv. A wrong pairing or a mis-set stride moves the output by O(1).
//
// Review round 1 (step 2 review v1) added three structural gates here:
//   * no Rotate call may alias its input with its output (the device kernel
//     declares both __restrict__); CpuOps records any alias and the test
//     fails on it, so the contract is checked, not just documented.
//   * the comparator rejects non-finite values instead of scoring them as
//     zero error (std::max(worst, NaN) keeps `worst`).
//   * the negative control populates every reference slot and is paired
//     with a positive control, so "correct scales pass" is demonstrated in
//     the same run that shows "swapped scales fail".

#include "src/core/quant/i4r_dispatch.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../models/qwen38_flash_next/i4r_tiny_fixture.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/i4r_rotation.hpp"

namespace q = gufo::core;
namespace qr = gufo::quant;
namespace tf = gufo::test_fixture;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

// CPU stand-in for the GPU primitives. Rotate uses the reference RotateBlocks;
// Matvec dots the raw I4R128 row directly (the same decode the kernel does).
struct CpuOps : qr::RotatedOps {
  void Rotate(const float* in, float* out, const float* pre_scale,
              const float* post_scale, std::size_t count) override {
    Require(count != 0 && count % qr::kI4rBlock == 0,
            "rotate count misaligned");
    // The device kernel declares in/out __restrict__. A CPU stand-in that
    // routes through its own temporary would happily accept an aliased call
    // and hide a contract violation, so record it here and let the test fail.
    if (in == out) {
      saw_alias = true;
    }
    std::vector<float> tmp(count);
    for (std::size_t i = 0; i < count; ++i) {
      float v = in[i];
      if (pre_scale != nullptr) {
        v *= pre_scale[i];
      }
      tmp[i] = v;
    }
    qr::RotateBlocks(tmp.data(), count);
    for (std::size_t i = 0; i < count; ++i) {
      out[i] = tmp[i] * (post_scale != nullptr ? post_scale[i] : 1.0F);
    }
  }
  void Matvec(const std::uint8_t* M, const float* x, float* y,
              std::size_t n_out, std::size_t n_blocks_in) override {
    const std::size_t n_in = n_blocks_in * qr::kI4rBlock;
    const std::span<const float> xs(x, n_in);
    for (std::size_t o = 0; o < n_out; ++o) {
      y[o] = qr::DotProductI4R128(M + o * n_blocks_in * 66, xs, n_in);
    }
  }
  void Gemm(const std::uint8_t* M, const float* X, float* Y, std::size_t n_out,
            std::size_t n_in, int tokens) override {
    const std::size_t blocks_in = n_in / qr::kI4rBlock;
    for (int t = 0; t < tokens; ++t) {
      Matvec(M, X + static_cast<std::size_t>(t) * n_in,
             Y + static_cast<std::size_t>(t) * n_out, n_out, blocks_in);
    }
  }

  mutable bool saw_alias{false};
};

// Maximum relative error, with the finiteness gate the review requires.
// std::max(worst, NaN) returns `worst`, so without these checks a NaN or
// Inf output scores as zero error -- the worst possible failure mode for a
// comparator. Sizes must match and every value on both sides must be finite
// before any error is accumulated.
double MaxRel(const std::vector<float>& a, const std::vector<float>& b) {
  Require(a.size() == b.size(), "MaxRel: size mismatch (" +
                                    std::to_string(a.size()) + " vs " +
                                    std::to_string(b.size()) + ")");
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double av = static_cast<double>(a[i]);
    const double bv = static_cast<double>(b[i]);
    Require(std::isfinite(av),
            "MaxRel: non-finite output at index " + std::to_string(i));
    Require(std::isfinite(bv),
            "MaxRel: non-finite reference at index " + std::to_string(i));
    const double diff = std::abs(av - bv);
    const double denom = std::max(1e-6, std::max(std::abs(av), std::abs(bv)));
    worst = std::max(worst, diff / denom);
  }
  return worst;
}

// The comparator itself must reject injected non-finite values rather than
// report them as a clean match. Each case is built from two otherwise
// identical vectors so the only defect is the injected value.
void CheckComparatorRejectsNonFinite() {
  const std::size_t n = 64;
  std::vector<float> base(n, 1.5F);
  std::vector<float> same = base;
  // Sanity: the clean pair passes and scores zero.
  Require(MaxRel(base, same) == 0.0,
          "MaxRel: identical vectors scored nonzero");

  struct Case {
    const char* name;
    bool in_output;
    float poison;
  };
  const Case cases[] = {
      {"NaN in output", true, std::nanf("")},
      {"+Inf in output", true, HUGE_VALF},
      {"-Inf in output", true, -HUGE_VALF},
      {"NaN in reference", false, std::nanf("")},
      {"+Inf in reference", false, HUGE_VALF},
  };
  for (const auto& c : cases) {
    std::vector<float> a = base;
    std::vector<float> b = base;
    if (c.in_output) {
      a[n / 2] = c.poison;
    } else {
      b[n / 2] = c.poison;
    }
    bool threw = false;
    try {
      (void)MaxRel(a, b);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Require(threw, std::string("MaxRel accepted ") + c.name +
                       " instead of rejecting it");
    std::printf("comparator rejects %s: PASS\n", c.name);
  }

  // Size mismatch is a hard error too, not a short comparison.
  bool threw = false;
  try {
    (void)MaxRel(std::vector<float>(n), std::vector<float>(n - 1));
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Require(threw, "MaxRel accepted a size mismatch instead of rejecting it");
  std::printf("comparator rejects size mismatch: PASS\n");
}

// Runs one full layer through the dispatch and checks every slot against the
// per-expert RotatedGemv oracle, for a given prefill flag.
void CheckLayer(bool prefill) {
  const std::size_t hidden = 256;  // 2 blocks
  const std::size_t ff = 128;      // 1 block
  const std::size_t experts = 3;
  const std::uint32_t used = 2;
  const std::size_t n_tokens = 3;
  const std::size_t n_slots = n_tokens * used;

  tf::Lcg rng;
  auto rand_vec = [&](std::size_t n, float lo, float hi) {
    std::vector<float> v(n);
    for (auto& e : v) {
      e = lo + (hi - lo) * (rng.Uniform() * 0.5F + 0.5F);
    }
    return v;
  };

  // Stored rotated-basis matrices M per expert, packed to raw I4R128. The
  // oracle uses the DECODED bytes (what the kernel actually sees), so the
  // 4-bit quantization is common to both paths and cancels; what is left is
  // the dispatch wiring under test.
  std::vector<std::vector<float>> m_gate(experts), m_up(experts),
      m_down(experts);
  std::vector<std::vector<float>> d_gate(experts), d_up(experts),
      d_down(experts);
  std::vector<std::uint8_t> raw_gate, raw_up, raw_down;
  auto decode = [](const std::vector<std::uint8_t>& raw, std::size_t rows,
                   std::size_t cols) {
    std::vector<float> dec(rows * cols);
    for (std::size_t r = 0; r < rows; ++r) {
      qr::DequantizeI4R128(raw.data() + r * (cols / qr::kI4rBlock) * 66,
                           dec.data() + r * cols, cols);
    }
    return dec;
  };
  for (std::size_t e = 0; e < experts; ++e) {
    m_gate[e] = rand_vec(hidden * ff, -1.0F, 1.0F);
    m_up[e] = rand_vec(hidden * ff, -1.0F, 1.0F);
    m_down[e] = rand_vec(ff * hidden, -1.0F, 1.0F);
    auto g = tf::PackI4R128(m_gate[e], ff, hidden);
    auto u = tf::PackI4R128(m_up[e], ff, hidden);
    auto d = tf::PackI4R128(m_down[e], hidden, ff);
    d_gate[e] = decode(g, ff, hidden);
    d_up[e] = decode(u, ff, hidden);
    d_down[e] = decode(d, hidden, ff);
    raw_gate.insert(raw_gate.end(), g.begin(), g.end());
    raw_up.insert(raw_up.end(), u.begin(), u.end());
    raw_down.insert(raw_down.end(), d.begin(), d.end());
  }

  // Scales away from zero so a dropped scale is visible.
  qr::RotatedScales s;
  std::vector<float> su_e = rand_vec(hidden, 0.5F, 1.5F);
  std::vector<float> sv_g = rand_vec(ff, 0.5F, 1.5F);
  std::vector<float> sv_u = rand_vec(ff, 0.5F, 1.5F);
  std::vector<float> su_d = rand_vec(ff, 0.5F, 1.5F);
  std::vector<float> sv_d = rand_vec(hidden, 0.5F, 1.5F);
  s.su_experts = su_e.data();
  s.sv_gate = sv_g.data();
  s.sv_up = sv_u.data();
  s.su_down = su_d.data();
  s.sv_down = sv_d.data();

  std::vector<float> x = rand_vec(n_tokens * hidden, -1.0F, 1.0F);
  std::vector<std::int32_t> ids(n_slots);
  for (std::size_t sl = 0; sl < n_slots; ++sl) {
    ids[sl] = static_cast<std::int32_t>((sl * 2 + 1) % experts);
  }

  CpuOps ops;
  std::vector<float> xr(n_tokens * hidden);
  qr::RotateLayerInput(ops, x.data(), s, hidden, n_tokens, xr.data());

  std::vector<float> gate_out(n_slots * ff), up_out(n_slots * ff);
  std::vector<float> scratch_gate(ff), scratch_up(ff);
  qr::RotatedGateUp(ops, raw_gate.data(), raw_up.data(), s, xr.data(),
                    ids.data(), hidden, ff, n_slots, used, prefill,
                    gate_out.data(), up_out.data(), scratch_gate.data(),
                    scratch_up.data());

  // Oracle: full gemv from the raw x per slot.
  std::vector<float> ref_gate(n_slots * ff), ref_up(n_slots * ff);
  for (std::size_t sl = 0; sl < n_slots; ++sl) {
    const std::size_t t = sl / used;
    const std::size_t e = static_cast<std::size_t>(ids[sl]);
    qr::RotatedGemv(d_gate[e].data(), ff, hidden, x.data() + t * hidden,
                    su_e.data(), sv_g.data(), ref_gate.data() + sl * ff);
    qr::RotatedGemv(d_up[e].data(), ff, hidden, x.data() + t * hidden,
                    su_e.data(), sv_u.data(), ref_up.data() + sl * ff);
  }
  const double gate_err = MaxRel(gate_out, ref_gate);
  const double up_err = MaxRel(up_out, ref_up);
  Require(gate_err < 1e-4, "gate mismatch (prefill=" + std::to_string(prefill) +
                               ") rel=" + std::to_string(gate_err));
  Require(up_err < 1e-4, "up mismatch (prefill=" + std::to_string(prefill) +
                             ") rel=" + std::to_string(up_err));

  // Down side: feed an independent activated h (the swiglu formula is the
  // caller's; the dispatch only sees h).
  std::vector<float> h = rand_vec(n_slots * ff, -1.0F, 1.0F);
  std::vector<float> down_out(n_slots * hidden);
  std::vector<float> scratch_in(ff), scratch_out(hidden);
  qr::RotatedDown(ops, raw_down.data(), s, h.data(), ids.data(), hidden, ff,
                  n_slots, used, prefill, down_out.data(), scratch_in.data(),
                  scratch_out.data());
  std::vector<float> ref_down(n_slots * hidden);
  for (std::size_t sl = 0; sl < n_slots; ++sl) {
    const std::size_t e = static_cast<std::size_t>(ids[sl]);
    qr::RotatedGemv(d_down[e].data(), hidden, ff, h.data() + sl * ff,
                    su_d.data(), sv_d.data(), ref_down.data() + sl * hidden);
  }
  const double down_err = MaxRel(down_out, ref_down);
  Require(down_err < 1e-4, "down mismatch (prefill=" + std::to_string(prefill) +
                               ") rel=" + std::to_string(down_err));

  // The alias contract: no stage rotated in place. The CPU stand-in would
  // have worked either way, so assert it rather than rely on it passing.
  Require(!ops.saw_alias,
          "dispatch issued an in-place Rotate (violates the kernel's "
          "__restrict__ contract)");

  std::printf(
      "layer prefill=%d: gate %.2e up %.2e down %.2e (tol 1e-4) "
      "alias-free PASS\n",
      static_cast<int>(prefill), gate_err, up_err, down_err);
}

// The control pair the review asked for: with the CORRECT sv_gate the
// dispatch must match the oracle, and with a deliberately swapped sv_gate
// it must blow up. Both in one run, against a reference whose every slot is
// populated -- the previous version left the second half of ref_gate at zero,
// so the comparison scored ~1 error even with correct scales and the
// "negative" proved nothing.
void CheckControls() {
  const std::size_t hidden = 256, ff = 128, experts = 2;
  const std::uint32_t used = 2;
  const std::size_t n_tokens = 1, n_slots = n_tokens * used;
  tf::Lcg rng;
  auto rand_vec = [&](std::size_t n, float lo, float hi) {
    std::vector<float> v(n);
    for (auto& e : v) {
      e = lo + (hi - lo) * (rng.Uniform() * 0.5F + 0.5F);
    }
    return v;
  };
  std::vector<float> m_gate = rand_vec(hidden * ff, -1.0F, 1.0F);
  std::vector<std::uint8_t> raw_gate = tf::PackI4R128(m_gate, ff, hidden);
  std::vector<float> d_gate(hidden * ff);
  for (std::size_t r = 0; r < ff; ++r) {
    qr::DequantizeI4R128(raw_gate.data() + r * (hidden / qr::kI4rBlock) * 66,
                         d_gate.data() + r * hidden, hidden);
  }

  std::vector<float> su_e = rand_vec(hidden, 0.5F, 1.5F);
  std::vector<float> sv_g = rand_vec(ff, 0.5F, 1.5F);
  std::vector<float> sv_other = rand_vec(ff, 0.5F, 1.5F);
  std::vector<float> su_d = rand_vec(ff, 0.5F, 1.5F);
  std::vector<float> sv_d = rand_vec(hidden, 0.5F, 1.5F);
  std::vector<float> x = rand_vec(hidden, -1.0F, 1.0F);
  std::vector<std::int32_t> ids = {0, 0};

  // Reference: BOTH slots populated. Both slots route to expert 0 with the
  // same token input, so the two reference rows are identical, and neither
  // is left at zero.
  std::vector<float> ref_gate(n_slots * ff);
  for (std::size_t sl = 0; sl < n_slots; ++sl) {
    qr::RotatedGemv(d_gate.data(), ff, hidden, x.data(), su_e.data(),
                    sv_g.data(), ref_gate.data() + sl * ff);
  }
  for (std::size_t i = 0; i < ref_gate.size(); ++i) {
    Require(std::isfinite(ref_gate[i]),
            "control reference is non-finite at " + std::to_string(i));
  }
  // Guard against the vacuous case: an all-zero reference would make any
  // comparison meaningless.
  double ref_norm = 0.0;
  for (const float v : ref_gate) {
    ref_norm += static_cast<double>(v) * v;
  }
  Require(ref_norm > 1e-6, "control reference is (near) all zero");

  auto run_gate = [&](const std::vector<float>& sv_gate) {
    qr::RotatedScales s;
    s.su_experts = su_e.data();
    s.sv_gate = sv_gate.data();
    s.sv_up = sv_gate.data();
    s.su_down = su_d.data();
    s.sv_down = sv_d.data();
    CpuOps ops;
    std::vector<float> xr(hidden);
    qr::RotateLayerInput(ops, x.data(), s, hidden, n_tokens, xr.data());
    std::vector<float> gate_out(n_slots * ff), up_out(n_slots * ff);
    std::vector<float> scratch_gate(ff), scratch_up(ff);
    qr::RotatedGateUp(ops, raw_gate.data(), raw_gate.data(), s, xr.data(),
                      ids.data(), hidden, ff, n_slots, used, false,
                      gate_out.data(), up_out.data(), scratch_gate.data(),
                      scratch_up.data());
    Require(!ops.saw_alias, "control dispatch rotated in place");
    return gate_out;
  };

  // Positive: correct scale must match.
  const std::vector<float> ok_out = run_gate(sv_g);
  const double ok_err = MaxRel(ok_out, ref_gate);
  Require(ok_err < 1e-4,
          "positive control failed: correct sv_gate did not match the oracle, "
          "rel=" +
              std::to_string(ok_err));
  std::printf("positive control: correct sv_gate rel %.2e PASS\n", ok_err);

  // Negative: the swapped scale must be caught, by a wide margin.
  const std::vector<float> bad_out = run_gate(sv_other);
  const double bad_err = MaxRel(bad_out, ref_gate);
  Require(bad_err > 1e-2,
          "negative control too small: wrong scale not detected rel=" +
              std::to_string(bad_err));
  std::printf("negative control: swapped sv_gate rel %.2e PASS\n", bad_err);
  // The two must be far apart, or the tolerance sits in the middle of a
  // band the test cannot separate.
  Require(bad_err > 100.0 * std::max(ok_err, 1e-12),
          "positive and negative controls are not separated by 100x");
  std::printf("control separation: %.2e vs %.2e (>100x) PASS\n", ok_err,
              bad_err);
}

}  // namespace

int main() {
  try {
    CheckComparatorRejectsNonFinite();
    CheckLayer(/*prefill=*/false);
    CheckLayer(/*prefill=*/true);
    CheckControls();
  } catch (const std::exception& e) {
    std::printf("FAIL: %s\n", e.what());
    return 1;
  }
  std::printf("i4r_dispatch ALL PASS\n");
  return 0;
}
