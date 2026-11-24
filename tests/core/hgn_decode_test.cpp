// Checks src/core/quant/hgn_decode against the public hgn-spec v1.1.0
// TEST-VECTORS.md (ht codebook, ht tiles at b = 3 and 4, the 128-point
// rotation, q6g64, fp8) and the store size rules.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/quant/hgn_decode.hpp"

namespace {

int failures = 0;

void Check(bool ok, const std::string& name) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", name.c_str());
  failures += ok ? 0 : 1;
}

std::uint16_t HalfBits(float v) {
  std::uint16_t h = 0;
  gufo::quant::FloatToHalfBits(&v, &h, 1);
  return h;
}

std::vector<std::uint8_t> GoldenWords(std::size_t n) {
  std::vector<std::uint8_t> out(4 * n);
  for (std::size_t w = 0; w < n; ++w) {
    const std::uint32_t v = 0x9E3779B9u * static_cast<std::uint32_t>(w + 1);
    std::memcpy(out.data() + 4 * w, &v, 4);
  }
  return out;
}

}  // namespace

int main() {
  using gufo::core::GgmlType;
  namespace q = gufo::quant;

  // ht codebook
  const struct {
    std::uint16_t state, bits;
  } cb[] = {{0x0000, 0xC2E8}, {0x0001, 0x3921}, {0x0002, 0xB72D},
            {0x1234, 0xB89E}, {0x8000, 0xBAF2}, {0xABCD, 0xBD42},
            {0xFFFF, 0xB936}};
  for (const auto& c : cb) {
    Check(HalfBits(q::HtCodebook(c.state)) == c.bits,
          "ht codebook " + std::to_string(c.state));
  }

  // ht tile, b = 4 and b = 3
  const auto words = GoldenWords(2048);
  std::uint16_t st[256];
  q::HtTileStates(words.data(), 4, st);
  const struct {
    int i;
    std::uint16_t s4, s3;
  } tiles[] = {{0, 0x7209, 0x4AC4},   {1, 0x209E, 0x5627},
               {2, 0x09E3, 0xB13C},   {7, 0x79B9, 0x3779},
               {8, 0x9B93, 0xBBCD},   {127, 0x9B90, 0xB4AC},
               {128, 0xB908, 0xA560}, {255, 0x3720, 0x6958}};
  for (const auto& t : tiles) {
    Check(st[t.i] == t.s4, "ht4 state " + std::to_string(t.i));
  }
  q::HtTileStates(words.data(), 3, st);
  for (const auto& t : tiles) {
    Check(st[t.i] == t.s3, "ht3 state " + std::to_string(t.i));
  }

  // The 128-point rotation of a [128, 128] ht tensor.
  std::vector<std::uint16_t> su(128), sv(128);
  for (int c = 0; c < 128; ++c) {
    su[c] = HalfBits((5 * c) % 7 < 3 ? -1.0f : 1.0f);
    const float v = static_cast<float>(8 + c % 8) / 512.0f;
    sv[c] = HalfBits(c % 3 == 0 ? -v : v);
  }
  std::vector<float> w(128 * 128);
  std::string error;
  Check(q::DecodeHgnToF32(GgmlType::kHT4, words.data(), 128, 128, su.data(),
                          sv.data(), w.data(), &error, 4),
        "ht decode runs");
  const struct {
    int r, c;
    double v;
  } rot[] = {{0, 0, -0.0005001630634069443}, {0, 1, -0.0008721128106117249},
             {1, 0, -0.021587662398815155},  {1, 1, 0.006169011816382408},
             {64, 64, -0.002170378342270851}, {127, 127, 0.003455530386418104}};
  for (const auto& r : rot) {
    const double got = w[r.r * 128 + r.c];
    Check(std::fabs(got - r.v) <= 1e-6 * std::fabs(r.v) + 1e-9,
          "rotation (" + std::to_string(r.r) + "," + std::to_string(r.c) +
              ")");
  }

  // q6g64: one row of 64; code of k is (37k + 11) mod 64, scale 1/32,
  // bias -1.
  const unsigned char q6row[64] = {
      0x0b, 0xa5, 0x4f, 0xe9, 0x83, 0x2d, 0xc7, 0x61, 0x0b, 0xa5, 0x4f, 0xe9,
      0x83, 0x2d, 0xc7, 0x61, 0x0b, 0xa5, 0x4f, 0xe9, 0x83, 0x2d, 0xc7, 0x61,
      0x0b, 0xa5, 0x4f, 0xe9, 0x83, 0x2d, 0xc7, 0x61, 0xdc, 0x21, 0xb7, 0xd8,
      0x21, 0x76, 0xc8, 0x2d, 0x76, 0x8b, 0x1d, 0x72, 0x8b, 0xdc, 0x62, 0x87,
      0x00, 0x28, 0x00, 0xbc, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float q6[64];
  Check(q::DecodeHgnToF32(GgmlType::kQ6G64, q6row, 64, 1, nullptr, nullptr,
                          q6, &error, 1),
        "q6g64 decode runs");
  bool q6ok = true;
  for (int k = 0; k < 64; ++k) {
    q6ok = q6ok && q6[k] == static_cast<float>((37 * k + 11) % 64) * 0.03125f -
                               1.0f;
  }
  Check(q6ok, "q6g64 row");

  // fp8
  Check(q::Fp8Value(0x01) == 0.001953125f && q::Fp8Value(0x38) == 1.0f &&
            q::Fp8Value(0x7E) == 448.0f && q::Fp8Value(0x7F) == 480.0f &&
            q::Fp8Value(0xB8) == -1.0f && q::Fp8Value(0xFF) == -480.0f,
        "fp8 codes");

  // size rules
  Check(q::HgnEncodedSizeBytes(GgmlType::kHT4, 2560u * 10240u, 2560) ==
            2560u * 10240u / 2,
        "size ht4");
  Check(q::HgnEncodedSizeBytes(GgmlType::kQ6G64, 320u * 10240u, 320) ==
            10240u * 272u,
        "size q6g64 padded rows");
  Check(q::HgnEncodedSizeBytes(GgmlType::kQ4C, 2560u * 48u, 2560) ==
            64u + 61440u + 48u * 160u,
        "size q4c");
  Check(q::HgnEncodedSizeBytes(GgmlType::kHT4, 100u * 128u, 100) == 0,
        "size ht4 rejects K % 128");

  std::printf("%s\n", failures == 0 ? "ALL PASS" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
