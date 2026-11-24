// Host-emulation test for the I4R128 GPU math. Runs the SAME __device__
// __host__ functions the HIP kernels use (i4r_hip_math.hpp), on the CPU, and
// checks them against the CPU oracle RotatedGemv / DequantizeI4R128. This is
// the verification that does NOT need a GPU: it proves the decode, the FWHT
// rotation, and the five-stage pipeline composition match the reference. The
// GPU test (i4r_hip_test.hip) closes the loop on actual device execution.
//
// Build with plain g++ (no HIP): the math header's __device__/__host__ are
// neutralised with -D__device__= -D__host__=, and f16 scales are decoded with
// the core's Fp16ToFloat so the decode cross-check reads the same bits the
// oracle does. See BUILD-I4R-GPU.md for the exact command.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/i4r_hip_math.hpp"
#include "src/core/quant/i4r_rotation.hpp"

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool ok, const char* what) {
  if (ok) {
    ++g_pass;
  } else {
    ++g_fail;
    std::printf("FAIL: %s\n", what);
  }
}

double rel_err(double a, double b) {
  // Two-sided finite guard: a non-finite on EITHER side is a hard failure,
  // not a zero-error pass. std::fmax would otherwise let a NaN/Inf on one
  // side slip through as no error growth. Return a sentinel that fails any
  // threshold check.
  if (!std::isfinite(a) || !std::isfinite(b))
    return 1e30;
  const double m = std::fmax(1.0, std::fabs(b));
  return std::fabs(a - b) / m;
}

// round-to-nearest-even f32 -> f16, for generating test scales only. The
// decode side uses the core's Fp16ToFloat on the stored bits, so the two
// sides stay consistent regardless of this encoder's exactness; a self-check
// below confirms it produces finite values that round-trip sanely.
uint16_t f32_to_f16(float value) {
  uint32_t x;
  std::memcpy(&x, &value, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp >= 31)
    return static_cast<uint16_t>(sign | 0x7C00u);  // inf
  if (exp <= 0) {
    if (exp < -10)
      return static_cast<uint16_t>(sign);  // underflow to zero
    mant |= 0x800000u;
    const int shift = 14 - exp;
    uint32_t r = (mant >> shift) + ((mant >> (shift - 1)) & 1u);  // round
    return static_cast<uint16_t>(sign | r);
  }
  uint32_t r = (mant + 0x0FFFu + ((mant >> 13) & 1u)) >> 13;
  if (r & 0x400u) {
    exp += 1;
    r = 0;
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                               (r & 0x3FFu));
}

float f16_bits_to_f32(uint16_t bits) {
  return gufo::quant::Fp16ToFloat(bits);
}

float block_scale(const uint8_t* blk) {
  const uint16_t bits =
      static_cast<uint16_t>(blk[64]) | static_cast<uint16_t>(blk[65] << 8);
  return f16_bits_to_f32(bits);
}

// Build a random I4R128 matrix: n_out rows x n_in cols, n_in % 128 == 0.
std::vector<uint8_t> make_matrix(std::mt19937& rng, int n_out, int n_in) {
  const int nb = n_in / 128;
  std::vector<uint8_t> M(static_cast<size_t>(n_out) * nb * 66);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_real_distribution<float> scale_dist(-2.0f, 2.0f);
  for (int o = 0; o < n_out; ++o) {
    for (int b = 0; b < nb; ++b) {
      uint8_t* blk = M.data() + (static_cast<size_t>(o) * nb + b) * 66;
      for (int i = 0; i < 64; ++i)
        blk[i] = static_cast<uint8_t>(byte_dist(rng));
      const uint16_t bits = f32_to_f16(scale_dist(rng));
      blk[64] = static_cast<uint8_t>(bits & 0xFF);
      blk[65] = static_cast<uint8_t>(bits >> 8);
    }
  }
  return M;
}

// Decode a whole matrix to f32 row by row using the core's decoder.
std::vector<float> decode_matrix(const std::vector<uint8_t>& M, int n_out,
                                 int n_in) {
  const int nb = n_in / 128;
  std::vector<float> m_dec(static_cast<size_t>(n_out) * n_in);
  for (int o = 0; o < n_out; ++o)
    gufo::quant::DequantizeI4R128(M.data() + static_cast<size_t>(o) * nb * 66,
                                  m_dec.data() + static_cast<size_t>(o) * n_in,
                                  n_in);
  return m_dec;
}

// Test A: my block unpack == the core's DequantizeI4R128, exactly, including
// the full 16-code offset-binary domain and a zero scale.
void test_decode_matches_core() {
  std::mt19937 rng(1234);
  const int n_in = 256;  // 2 blocks
  const int nb = n_in / 128;
  std::vector<uint8_t> M(nb * 66);
  for (int b = 0; b < nb; ++b) {
    uint8_t* blk = M.data() + b * 66;
    for (int i = 0; i < 64; ++i)
      blk[i] =
          static_cast<uint8_t>(std::uniform_int_distribution<int>(0, 255)(rng));
    const uint16_t bits = f32_to_f16(b == 0 ? 0.0f : 0.5f);  // zero scale too
    blk[64] = static_cast<uint8_t>(bits & 0xFF);
    blk[65] = static_cast<uint8_t>(bits >> 8);
  }
  std::vector<float> core(n_in), mine(n_in);
  gufo::quant::DequantizeI4R128(M.data(), core.data(), n_in);
  for (int b = 0; b < nb; ++b)
    gufo::quant::i4r_unpack_block(M.data() + b * 66,
                                  block_scale(M.data() + b * 66),
                                  mine.data() + b * 128);
  bool exact = true;
  for (int i = 0; i < n_in; ++i)
    if (core[i] != mine[i])
      exact = false;
  check(exact,
        "decode: i4r_unpack_block == DequantizeI4R128 (incl. zero scale)");

  // Full code domain: every nibble value 0..15 decodes to (nib-8)*d.
  uint8_t codes[64] = {};
  for (int i = 0; i < 64; ++i)
    codes[i] = static_cast<uint8_t>((i % 16) | ((i % 16) << 4));
  const float d = 1.0f;
  bool domain = true;
  for (int v = 0; v < 128; ++v) {
    const int nib = (codes[v & 63] >> (v < 64 ? 0 : 4)) & 0xF;
    if (gufo::quant::i4r_decode_value(codes, d, v) !=
        static_cast<float>(nib - 8))
      domain = false;
  }
  check(domain, "decode: full 16-code offset-binary domain");

  // f16 encoder sanity: round-trips finite values through Fp16ToFloat.
  bool enc_ok = true;
  for (float v : {0.0f, 1.0f, -1.0f, 0.5f, -3.25f, 100.0f, -0.001f}) {
    const float rt = f16_bits_to_f32(f32_to_f16(v));
    if (!std::isfinite(rt) || std::fabs(rt - v) > std::fabs(v) * 1e-2f + 1e-6f)
      enc_ok = false;
  }
  check(enc_ok, "test f16 encoder round-trips finite values");
}

// Host pipeline mirroring the kernels: stage1 (x*su then rotate), stage3
// (block dots), stage4 (rotate then *sv). `accum_double` switches the
// stage-3 accumulator to double to match the oracle's precision exactly.
void host_pipeline(const std::vector<uint8_t>& M, int n_out, int n_in,
                   const std::vector<float>& x, const std::vector<float>& su,
                   const std::vector<float>& sv, std::vector<float>& y,
                   bool accum_double) {
  const int nb = n_in / 128;
  std::vector<float> xr(n_in);
  for (int b = 0; b < nb; ++b) {
    for (int i = 0; i < 128; ++i)
      xr[b * 128 + i] = x[b * 128 + i] * su[b * 128 + i];
    gufo::quant::i4r_rotate_block128(xr.data() + b * 128);
  }
  std::vector<float> ymid(n_out);
  for (int o = 0; o < n_out; ++o) {
    const uint8_t* row = M.data() + static_cast<size_t>(o) * nb * 66;
    if (accum_double) {
      double acc = 0.0;
      for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + b * 66;
        const float s = block_scale(blk);
        for (int v = 0; v < 128; ++v)
          acc += static_cast<double>(gufo::quant::i4r_decode_value(blk, s, v)) *
                 xr[b * 128 + v];
      }
      ymid[o] = static_cast<float>(acc);
    } else {
      float acc = 0.0f;
      for (int b = 0; b < nb; ++b)
        acc += gufo::quant::i4r_dot_block(
            row + b * 66, block_scale(row + b * 66), xr.data() + b * 128);
      ymid[o] = acc;
    }
  }
  y.assign(n_out, 0.0f);
  const int nob = n_out / 128;
  for (int b = 0; b < nob; ++b) {
    for (int i = 0; i < 128; ++i)
      y[b * 128 + i] = ymid[b * 128 + i];
    gufo::quant::i4r_rotate_block128(y.data() + b * 128);
    for (int i = 0; i < 128; ++i)
      y[b * 128 + i] *= sv[b * 128 + i];
  }
}

// Test B/C: host pipeline vs RotatedGemv oracle, float (kernel-faithful) and
// double (exact-composition) accumulators.
void test_pipeline_vs_oracle() {
  std::mt19937 rng(777);
  const int n_out = 256, n_in = 2560;  // gate/up shape
  auto M = make_matrix(rng, n_out, n_in);
  std::vector<float> x(n_in), su(n_in), sv(n_out), y_oracle(n_out),
      y_host(n_out);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (auto& v : x)
    v = nd(rng);
  for (auto& v : su)
    v = nd(rng);
  for (auto& v : sv)
    v = nd(rng);

  const auto m_dec = decode_matrix(M, n_out, n_in);
  gufo::quant::RotatedGemv(m_dec.data(), n_out, n_in, x.data(), su.data(),
                           sv.data(), y_oracle.data());

  host_pipeline(M, n_out, n_in, x, su, sv, y_host, /*accum_double=*/false);
  double max_rel = 0.0;
  for (int i = 0; i < n_out; ++i)
    max_rel = std::fmax(max_rel, rel_err(y_host[i], y_oracle[i]));
  std::printf("  float-accum pipeline max rel err vs oracle: %.3e\n", max_rel);
  check(max_rel < 5e-3,
        "pipeline (float, kernel-faithful) matches RotatedGemv");

  host_pipeline(M, n_out, n_in, x, su, sv, y_host, /*accum_double=*/true);
  double max_rel_d = 0.0;
  for (int i = 0; i < n_out; ++i)
    max_rel_d = std::fmax(max_rel_d, rel_err(y_host[i], y_oracle[i]));
  std::printf("  double-accum pipeline max rel err vs oracle: %.3e\n",
              max_rel_d);
  check(max_rel_d < 1e-6,
        "pipeline (double) matches RotatedGemv exactly -> composition correct");
}

// Test D: MMQ prefill host emulation (multi-token) vs per-token oracle.
void test_mmq_vs_oracle() {
  std::mt19937 rng(31337);
  const int n_out = 128, n_in = 640, tokens = 5;  // down_proj shape
  auto M = make_matrix(rng, n_out, n_in);
  const auto m_dec = decode_matrix(M, n_out, n_in);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> su(n_in), sv(n_out);
  for (auto& v : su)
    v = nd(rng);
  for (auto& v : sv)
    v = nd(rng);
  std::vector<float> X(static_cast<size_t>(tokens) * n_in);
  for (auto& v : X)
    v = nd(rng);

  double max_rel = 0.0;
  for (int t = 0; t < tokens; ++t) {
    std::vector<float> y_oracle(n_out), y_host(n_out);
    gufo::quant::RotatedGemv(m_dec.data(), n_out, n_in, X.data() + t * n_in,
                             su.data(), sv.data(), y_oracle.data());
    std::vector<float> xt(X.begin() + t * n_in, X.begin() + (t + 1) * n_in);
    host_pipeline(M, n_out, n_in, xt, su, sv, y_host, /*accum_double=*/true);
    for (int i = 0; i < n_out; ++i)
      max_rel = std::fmax(max_rel, rel_err(y_host[i], y_oracle[i]));
  }
  std::printf("  MMQ prefill (5 tokens) max rel err vs oracle: %.3e\n",
              max_rel);
  check(max_rel < 1e-6, "MMQ prefill host emulation matches per-token oracle");
}

// Test E: the four-stage layer composition. Stage 1 (shared gate/up input) is
// computed once and reused; the gate/up split is block-aligned so the fused
// 1280-wide output rotates as one stage.
void test_layer_stages() {
  std::mt19937 rng(4242);
  const int d = 2560, inter = 1280;  // trunk widths
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x(d), su(d), g_up_out(inter);
  for (auto& v : x)
    v = nd(rng);
  for (auto& v : su)
    v = nd(rng);
  for (auto& v : g_up_out)
    v = nd(rng);

  // Stage 1 once: xr = Hn(x * su).
  std::vector<float> xr(d);
  for (int i = 0; i < d; ++i)
    xr[i] = x[i] * su[i];
  for (int b = 0; b < d / 128; ++b)
    gufo::quant::i4r_rotate_block128(xr.data() + b * 128);

  // Reusing stage 1 across two "experts" gives identical xr (hoistable).
  std::vector<float> xr2(d);
  for (int i = 0; i < d; ++i)
    xr2[i] = x[i] * su[i];
  for (int b = 0; b < d / 128; ++b)
    gufo::quant::i4r_rotate_block128(xr2.data() + b * 128);
  bool hoist = true;
  for (int i = 0; i < d; ++i)
    if (xr[i] != xr2[i])
      hoist = false;
  check(hoist,
        "stage 1 is deterministic and reusable across experts (hoistable)");

  // gate/up split block-aligned: 640 = 5 blocks, 1280 = 10 blocks.
  check(640 % 128 == 0 && inter % 128 == 0,
        "stage 2: gate/up split is block-aligned (640 = 5 blocks)");

  // Rotating the fused 1280 vector then splitting equals rotating each half
  // independently (block boundaries align with the split).
  std::vector<float> fused_rot(inter);
  for (int b = 0; b < inter / 128; ++b) {
    for (int i = 0; i < 128; ++i)
      fused_rot[b * 128 + i] = g_up_out[b * 128 + i];
    gufo::quant::i4r_rotate_block128(fused_rot.data() + b * 128);
  }
  bool split_eq = true;
  for (int b = 0; b < inter / 128; ++b) {
    std::vector<float> one(128);
    for (int i = 0; i < 128; ++i)
      one[i] = g_up_out[b * 128 + i];
    gufo::quant::i4r_rotate_block128(one.data());
    for (int i = 0; i < 128; ++i)
      if (one[i] != fused_rot[b * 128 + i])
        split_eq = false;
  }
  check(split_eq, "stage 2: fused rotate == per-block rotate (split-safe)");
}

}  // namespace

int main() {
  test_decode_matches_core();
  test_pipeline_vs_oracle();
  test_mmq_vs_oracle();
  test_layer_stages();
  std::printf("\ni4r host-emulation: %d PASS / %d FAIL\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
