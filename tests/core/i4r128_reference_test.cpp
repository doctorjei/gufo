// CPU-reference tests for I4R128: block layout, nibble order, the 128-point
// Hadamard rotation, and the rotated matvec identity
//     y = sv * Hn( M * Hn( su * x ) )   ==   (diag(sv) Hn M Hn diag(su)) x
//
// These are the oracle the GPU MMQ/MMVQ kernels are checked against. Plain
// hand-rolled asserts, per repo convention; labels cpu;core;quant.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/ggml_gemm.hpp"
#include "src/core/quant/i4r_rotation.hpp"

namespace {

using gufo::quant::kI4rBlock;
using gufo::quant::kI4rInvSqrtBlock;

int g_checks = 0;
int g_failures = 0;

void Check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    std::cerr << "FAIL: " << what << '\n';
    ++g_failures;
  }
}

void CheckNear(float actual, float expected, const std::string& what,
               float tol = 1.0e-4F) {
  ++g_checks;
  if (!(std::abs(actual - expected) <= tol) || std::isnan(actual)) {
    std::cerr << "FAIL: " << what << " actual=" << actual
              << " expected=" << expected << " tol=" << tol << '\n';
    ++g_failures;
  }
}

int Popcount(std::uint32_t v) {
  int n = 0;
  while (v) {
    n += static_cast<int>(v & 1U);
    v >>= 1U;
  }
  return n;
}

// Explicit Sylvester-Hadamard entry, normalised: Hn[i][j] =
// (-1)^popcount(i&j)/sqrt(128).
float Hn(std::size_t i, std::size_t j) {
  return (Popcount(static_cast<std::uint32_t>(i & j)) % 2 == 0)
             ? kI4rInvSqrtBlock
             : -kI4rInvSqrtBlock;
}

std::uint16_t FloatToFp16(float f) {
  // Minimal round-to-nearest f32 -> f16 for test fixtures only.
  std::uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(float));
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  std::int32_t exp =
      static_cast<std::int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
  std::uint32_t frac = bits & 0x7FFFFFU;
  if (exp <= 0)
    return static_cast<std::uint16_t>(sign);
  if (exp >= 31)
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exp) << 10U) | (frac >> 13U));
}

// Packs 128 four-bit codes (each 0..15) plus an f16 scale into one 66-byte
// I4R128 block using the GGUF nibble convention: byte b low nibble = value b,
// high nibble = value 64 + b.
std::vector<std::uint8_t> PackBlock(const std::vector<std::uint8_t>& codes,
                                    float scale) {
  std::vector<std::uint8_t> blk(66, 0);
  for (std::size_t b = 0; b < 64; ++b) {
    blk[b] = static_cast<std::uint8_t>((codes[b] & 0x0FU) |
                                       ((codes[64 + b] & 0x0FU) << 4U));
  }
  const std::uint16_t d = FloatToFp16(scale);
  std::memcpy(blk.data() + 64, &d, sizeof(std::uint16_t));
  return blk;
}

// ---------------------------------------------------------------- layout ---

void TestLayout() {
  Check(sizeof(gufo::quant::block_i4r128) == 66, "block_i4r128 is 66 bytes");
  Check(
      gufo::quant::QuantizedBlockElements(gufo::core::GgmlType::kI4R128) == 128,
      "I4R128 block elements == 128");
  Check(gufo::core::ToString(gufo::core::GgmlType::kI4R128) == "I4R128",
        "ToString(I4R128)");
  Check(static_cast<std::uint16_t>(gufo::core::GgmlType::kI4R128) == 64,
        "I4R128 type id == 64");

  // 4.125 bpw exactly: 66/128.
  CheckNear(66.0F / 128.0F * 8.0F, 4.125F, "I4R128 bits per value == 4.125");

  // Row sizing on the real Flash-Next expert shapes.
  // gate/up: ne0 = 2560 (20 blocks/row); down: ne0 = 640 (5 blocks/row).
  Check(gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kI4R128, 2560) ==
            20 * 66,
        "row bytes for ne0=2560");
  Check(gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kI4R128, 640) ==
            5 * 66,
        "row bytes for ne0=640");
  // Misaligned ne0 must be rejected (0), not silently rounded.
  Check(
      gufo::quant::QuantizedRowBytes(gufo::core::GgmlType::kI4R128, 2500) == 0,
      "non-multiple-of-128 ne0 rejected");

  // Whole-tensor sizing for one expert each, and the full 512-expert layer.
  const auto t = gufo::core::GgmlType::kI4R128;
  const std::size_t gate_up = 2560U * 640U * 512U;  // gate or up, one layer
  const std::size_t down = 640U * 2560U * 512U;     // down, one layer
  Check(gufo::quant::EncodedSizeBytes(t, gate_up) == gate_up / 128 * 66,
        "encoded size, gate/up expert tensor");
  Check(gufo::quant::EncodedSizeBytes(t, down) == down / 128 * 66,
        "encoded size, down expert tensor");

  // Cross-check against the measured HGN v2 i4r store (62.285 GB over the 48
  // trunk layers). If this drifts, the layout contract has changed.
  const std::size_t all_experts = 48 * (2 * gate_up + down);
  const double gb = gufo::quant::EncodedSizeBytes(t, all_experts) / 1e9;
  CheckNear(static_cast<float>(gb), 62.285F,
            "48-layer expert total ~62.285 GB (matches HGN v2 i4r store)",
            0.01F);
}

// ------------------------------------------------------- decode/nibbles ---

void TestDecode() {
  // Full code domain with d = 1 decodes to offset binary -8..7.
  std::vector<std::uint8_t> codes(128);
  for (std::size_t i = 0; i < 128; ++i)
    codes[i] = static_cast<std::uint8_t>(i % 16);
  const auto blk = PackBlock(codes, 1.0F);
  std::vector<float> out(128, 999.0F);
  gufo::quant::DequantizeI4R128(blk.data(), out.data(), 128);
  for (std::size_t i = 0; i < 128; ++i) {
    CheckNear(out[i], static_cast<float>(static_cast<int>(codes[i]) - 8),
              "offset-binary decode at " + std::to_string(i));
  }

  // Nibble order is the GGUF convention, not HGN's sequential-pair order:
  // byte b low nibble -> value b, high nibble -> value 64 + b.
  std::vector<std::uint8_t> c2(128, 0);
  c2[3] = 5;   // value 3 lives in byte 3 low nibble
  c2[67] = 9;  // value 67 lives in byte 3 high nibble (3 + 64)
  const auto blk2 = PackBlock(c2, 2.0F);
  Check((blk2[3] & 0x0FU) == 5, "value 3 in byte 3 low nibble");
  Check((blk2[3] >> 4U) == 9, "value 67 in byte 3 high nibble");
  std::vector<float> o2(128, 0.0F);
  gufo::quant::DequantizeI4R128(blk2.data(), o2.data(), 128);
  CheckNear(o2[3], (5 - 8) * 2.0F, "decoded value 3");
  CheckNear(o2[67], (9 - 8) * 2.0F, "decoded value 67");

  // Zero scale decodes to all zeros (spec: d may be any f16 including 0).
  const auto blk0 = PackBlock(codes, 0.0F);
  std::vector<float> o0(128, 1.0F);
  gufo::quant::DequantizeI4R128(blk0.data(), o0.data(), 128);
  bool all_zero = true;
  for (float v : o0)
    all_zero = all_zero && (v == 0.0F);
  Check(all_zero, "zero scale decodes to all zeros");

  // Multi-block row: two blocks with different scales.
  auto row = PackBlock(codes, 1.0F);
  std::vector<std::uint8_t> codes_b(128, 15);
  auto second = PackBlock(codes_b, 4.0F);
  row.insert(row.end(), second.begin(), second.end());
  std::vector<float> o128(256, 0.0F);
  gufo::quant::DequantizeI4R128(row.data(), o128.data(), 256);
  CheckNear(o128[0], -8.0F, "block 0 first value");
  CheckNear(o128[128], (15 - 8) * 4.0F,
            "block 1 first value uses its own scale");

  // Dot product applies each block's own scale.
  // block 0: codes cycle 0..15 eight times -> sum(code-8) = 8*(120-128) = -64
  // block 1: all codes 15                 -> 128*(15-8) = 896
  // x == 1 everywhere, scales 1.0 and 4.0 -> -64*1 + 896*4 = 3520
  std::vector<float> x(256, 1.0F);
  const double expect = -64.0 * 1.0 + 896.0 * 4.0;
  CheckNear(
      gufo::quant::DotProductI4R128(row.data(), std::span<const float>(x), 256),
      static_cast<float>(expect), "dot uses per-block scale", 1.0e-2F);
}

// ------------------------------------------------------------- rotation ---

void TestRotation() {
  // H * e_0 = all ones (first column of the Sylvester matrix is +1).
  std::vector<float> v(kI4rBlock, 0.0F);
  v[0] = 1.0F;
  gufo::quant::Fwht128(v.data());
  bool ones = true;
  for (float e : v)
    ones = ones && (e == 1.0F);
  Check(ones, "H * e0 == all ones");

  // Ordering check against the explicit Sylvester definition for a non-trivial
  // vector: out[i] must equal sum_j (-1)^popcount(i&j) * in[j].
  std::vector<float> in(kI4rBlock);
  for (std::size_t j = 0; j < kI4rBlock; ++j)
    in[j] = static_cast<float>((j * 37U) % 23U) - 11.0F;
  std::vector<float> a = in, b = in;
  gufo::quant::Fwht128(a.data());
  for (std::size_t i = 0; i < kI4rBlock; ++i) {
    double acc = 0.0;
    for (std::size_t j = 0; j < kI4rBlock; ++j) {
      acc +=
          (Popcount(static_cast<std::uint32_t>(i & j)) % 2 == 0 ? 1.0 : -1.0) *
          in[j];
    }
    CheckNear(a[i], static_cast<float>(acc),
              "FWHT matches Sylvester h(i,j)=(-1)^popcount(i&j) at i=" +
                  std::to_string(i),
              1.0e-3F);
  }

  // Hn is its own inverse: rotating twice returns the input.
  std::vector<float> r = in;
  gufo::quant::RotateBlock128(r.data());
  gufo::quant::RotateBlock128(r.data());
  for (std::size_t i = 0; i < kI4rBlock; ++i)
    CheckNear(r[i], in[i], "Hn * Hn == I at " + std::to_string(i), 1.0e-4F);

  // Orthonormality: ||Hn x|| == ||x||.
  double n_in = 0.0, n_out = 0.0;
  for (float e : in)
    n_in += static_cast<double>(e) * e;
  std::vector<float> rot = in;
  gufo::quant::RotateBlock128(rot.data());
  for (float e : rot)
    n_out += static_cast<double>(e) * e;
  CheckNear(static_cast<float>(std::sqrt(n_out)),
            static_cast<float>(std::sqrt(n_in)), "Hn preserves the L2 norm",
            1.0e-4F);

  // RotateBlocks tiles independently and rejects nothing silently.
  std::vector<float> two(256);
  for (std::size_t i = 0; i < 256; ++i)
    two[i] = static_cast<float>(i % 13);
  std::vector<float> first_block(two.begin(), two.begin() + 128);
  std::vector<float> second_block(two.begin() + 128, two.end());
  std::vector<float> want_a = first_block, want_b = second_block;
  gufo::quant::RotateBlock128(want_a.data());
  gufo::quant::RotateBlock128(want_b.data());
  gufo::quant::RotateBlocks(two.data(), 256);
  for (std::size_t i = 0; i < 128; ++i) {
    CheckNear(two[i], want_a[i],
              "RotateBlocks block 0 at " + std::to_string(i));
    CheckNear(two[128 + i], want_b[i],
              "RotateBlocks block 1 at " + std::to_string(i));
  }

  Check(gufo::quant::IsRotationAligned(128) &&
            gufo::quant::IsRotationAligned(2560) &&
            !gufo::quant::IsRotationAligned(0) &&
            !gufo::quant::IsRotationAligned(2500),
        "IsRotationAligned accepts multiples of 128 only");

  const auto st = gufo::quant::Qwen38FlashNextLayerStages();
  Check(st.expert_input == 2560 && st.gate_up_output == 1280 &&
            st.down_input == 640 && st.down_output == 2560,
        "layer stage widths (4 stages) are block-aligned");
  Check(gufo::quant::IsRotationAligned(st.expert_input) &&
            gufo::quant::IsRotationAligned(st.gate_up_output) &&
            gufo::quant::IsRotationAligned(st.down_input) &&
            gufo::quant::IsRotationAligned(st.down_output),
        "all four layer stages are multiples of 128");
}

// Builds W = diag(sv) * LeftRot(Hn) * M * RightRot(Hn) * diag(su) explicitly,
// with no staging at all -- a plain double sum over both rotation indices --
// then compares W x against the staged RotatedGemv. The point of doing it this
// way is that the reference shares no code path with the staged implementation:
// in particular `su` sits inside W on the input side, so a staging error that
// applies the input scale in the wrong order cannot hide.
void TestRotatedGemvAgainstDense(std::size_t n_out, std::size_t n_in) {
  std::vector<float> m(n_out * n_in);
  for (std::size_t i = 0; i < m.size(); ++i)
    m[i] = static_cast<float>((i * 101U) % 17U) - 8.0F;
  std::vector<float> x(n_in), su(n_in), sv(n_out), y(n_out, 0.0F);
  for (std::size_t i = 0; i < n_in; ++i) {
    x[i] = static_cast<float>((i * 7U) % 11U) - 5.0F;
    su[i] = ((i % 3U) == 0U) ? -1.0F : 1.0F;
  }
  for (std::size_t i = 0; i < n_out; ++i)
    sv[i] = ((i % 5U) == 0U) ? 1.0F : -1.0F;

  std::vector<double> w(n_out * n_in, 0.0);
  for (std::size_t r = 0; r < n_out; ++r) {
    const std::size_t ri = r % kI4rBlock;
    const std::size_t rb = r / kI4rBlock;
    for (std::size_t c = 0; c < n_in; ++c) {
      const std::size_t ci = c % kI4rBlock;
      const std::size_t cb = c / kI4rBlock;
      double acc = 0.0;
      for (std::size_t k1 = 0; k1 < kI4rBlock; ++k1) {
        double inner = 0.0;
        for (std::size_t j = 0; j < kI4rBlock; ++j) {
          // (M * Hn)[row][c]: Hn indexed (j, ci); symmetric, so Hn(ci, j).
          inner += m[(rb * kI4rBlock + k1) * n_in + (cb * kI4rBlock + j)] *
                   Hn(ci, j);
        }
        acc += Hn(ri, k1) * inner;
      }
      w[r * n_in + c] = static_cast<double>(sv[r]) * acc * su[c];
    }
  }
  for (std::size_t r = 0; r < n_out; ++r) {
    double acc = 0.0;
    for (std::size_t c = 0; c < n_in; ++c)
      acc += w[r * n_in + c] * x[c];
    y[r] = static_cast<float>(acc);
  }

  std::vector<float> got(n_out, 0.0F);
  gufo::quant::RotatedGemv(m.data(), n_out, n_in, x.data(), su.data(),
                           sv.data(), got.data());

  double max_abs = 0.0, max_rel = 0.0;
  for (std::size_t r = 0; r < n_out; ++r) {
    const double d = std::abs(got[r] - y[r]);
    max_abs = std::max(max_abs, d);
    if (std::abs(y[r]) > 1.0e-6)
      max_rel = std::max(max_rel, d / std::abs(y[r]));
  }
  const std::string tag = std::to_string(n_out) + "x" + std::to_string(n_in);
  Check(max_abs < 1.0e-2, "staged gemv == dense W x, abs err " + tag + " = " +
                              std::to_string(max_abs));
  Check(max_rel < 1.0e-4, "staged gemv == dense W x, rel err " + tag + " = " +
                              std::to_string(max_rel));
}

void TestRotatedGemv() {
  TestRotatedGemvAgainstDense(128, 128);  // single block each side
  TestRotatedGemvAgainstDense(256, 128);  // multiple output blocks
  TestRotatedGemvAgainstDense(128, 256);  // multiple input blocks
}

// ----------------------------------------------------------- dispatch ----

void TestDispatch() {
  Check(gufo::quant::IsSupported(gufo::core::GgmlType::kI4R128),
        "unified dispatch supports I4R128");

  std::vector<std::uint8_t> codes(128);
  for (std::size_t i = 0; i < 128; ++i)
    codes[i] = static_cast<std::uint8_t>((i * 5) % 16);
  const auto blk = PackBlock(codes, 3.0F);

  std::vector<float> a(128, 0.0F), b(128, 0.0F);
  gufo::quant::Dequantize(gufo::core::GgmlType::kI4R128, blk.data(), a.data(),
                          128);
  gufo::quant::DequantizeI4R128(blk.data(), b.data(), 128);
  bool same = true;
  for (std::size_t i = 0; i < 128; ++i)
    same = same && (a[i] == b[i]);
  Check(same, "quant::Dequantize routes to DequantizeI4R128");

  Check(gufo::quant::DequantizeRow(gufo::core::GgmlType::kI4R128, blk.data(),
                                   a.data(), 128),
        "DequantizeRow accepts I4R128");
  Check(!gufo::quant::DequantizeRow(gufo::core::GgmlType::kI4R128, blk.data(),
                                    a.data(), 100),
        "DequantizeRow rejects a non-block-aligned k");

  std::vector<float> x(128, 2.0F);
  const float d1 = gufo::quant::Dot(gufo::core::GgmlType::kI4R128, blk.data(),
                                    std::span<const float>(x), 128);
  const float d2 =
      gufo::quant::DotProductI4R128(blk.data(), std::span<const float>(x), 128);
  Check(d1 == d2, "quant::Dot routes to DotProductI4R128");
}

}  // namespace

int main() {
  TestLayout();
  TestDecode();
  TestRotation();
  TestRotatedGemv();
  TestDispatch();

  std::cout << (g_failures == 0 ? "PASS" : "FAIL") << ": " << g_checks
            << " checks, " << g_failures << " failures\n";
  return g_failures == 0 ? 0 : 1;
}
