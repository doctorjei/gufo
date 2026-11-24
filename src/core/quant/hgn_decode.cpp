#include "src/core/quant/hgn_decode.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace gufo::quant {
namespace {

using core::GgmlType;

constexpr std::size_t Pad(std::size_t n, std::size_t a) noexcept {
  return (n + a - 1) / a * a;
}

float HalfToFloat(std::uint16_t bits) noexcept {
  _Float16 v;
  std::memcpy(&v, &bits, sizeof(v));
  return static_cast<float>(v);
}

const std::array<float, 65536>& Codebook() noexcept {
  // STORAGE.md "Codebook": x = s * 0x83DCD12D mod 2^32, a = byte sum of x,
  // T = (1774 a - 905216) / 2^18 rounded to f16. The numerator is below
  // 2^24, so the float division is exact before the f16 rounding.
  static const auto table = [] {
    std::array<float, 65536> t{};
    for (std::uint32_t s = 0; s < 65536; ++s) {
      const std::uint32_t x = s * 0x83DCD12Du;
      const std::uint32_t a =
          (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24);
      const float v =
          (1774.0f * static_cast<float>(a) - 905216.0f) / 262144.0f;
      t[s] = static_cast<float>(static_cast<_Float16>(v));
    }
    return t;
  }();
  return table;
}

void Fwht128(float* v) noexcept {
  for (unsigned h = 1; h < 128; h <<= 1) {
    for (unsigned i = 0; i < 128; i += 2 * h) {
      for (unsigned j = i; j < i + h; ++j) {
        const float a = v[j];
        const float b = v[j + h];
        v[j] = a + b;
        v[j + h] = a - b;
      }
    }
  }
}

/// One 128-row band of a 2-D ht tensor -> true weights (STORAGE.md "Tiles",
/// "The 128-point rotation").
/// `out` points at the band's first row (128 rows of `cols` floats).
void DecodeHtBand(const std::uint8_t* payload, std::uint64_t cols, unsigned b,
                  std::uint64_t band, const std::uint16_t* su,
                  const std::uint16_t* sv, float* out,
                  std::vector<float>& m) {
  const std::size_t K = cols;
  const std::size_t tile_bytes = 32u * b;
  const std::size_t col_tiles = K / 16;
  m.assign(128 * K, 0.0f);
  const auto& cb = Codebook();
  std::uint16_t states[256];
  for (std::size_t ct = 0; ct < col_tiles; ++ct) {
    for (std::size_t rt = 0; rt < 8; ++rt) {
      const std::uint8_t* tile =
          payload + ((band * col_tiles + ct) * 8 + rt) * tile_bytes;
      HtTileStates(tile, b, states);
      for (unsigned i = 0; i < 256; ++i) {
        const std::size_t tr = (i / 8) % 16;
        const std::size_t tc = 8 * (i / 128) + (i % 8);
        m[(rt * 16 + tr) * K + ct * 16 + tc] = cb[states[i]];
      }
    }
  }
  // M . Hn on each 128-value block of every row, then Hn . (...) across the
  // 128 rows; one 1/128 for the two 1/sqrt(128) factors.
  for (std::size_t r = 0; r < 128; ++r) {
    for (std::size_t g = 0; g < K / 128; ++g) {
      Fwht128(m.data() + r * K + g * 128);
    }
  }
  for (std::size_t h = 1; h < 128; h <<= 1) {
    for (std::size_t r = 0; r < 128; ++r) {
      if ((r & h) != 0) {
        continue;
      }
      float* x = m.data() + r * K;
      float* y = m.data() + (r | h) * K;
      for (std::size_t c = 0; c < K; ++c) {
        const float a = x[c];
        const float d = y[c];
        x[c] = a + d;
        y[c] = a - d;
      }
    }
  }
  std::vector<float> suf(K);
  for (std::size_t c = 0; c < K; ++c) {
    suf[c] = HalfToFloat(su[c]) / 128.0f;
  }
  for (std::size_t r = 0; r < 128; ++r) {
    const float s = HalfToFloat(sv[band * 128 + r]);
    const float* src = m.data() + r * K;
    float* dst = out + r * K;
    for (std::size_t c = 0; c < K; ++c) {
      dst[c] = src[c] * s * suf[c];
    }
  }
}

void QuantizeRowQ8_0(const float* x, std::size_t k, std::uint8_t* out) noexcept {
  for (std::size_t b = 0; b < k / 32; ++b) {
    const float* v = x + 32 * b;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[i]));
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    std::uint8_t* blk = out + 34 * b;
    const _Float16 dh = static_cast<_Float16>(d);
    std::memcpy(blk, &dh, 2);
    for (int i = 0; i < 32; ++i) {
      const int q = static_cast<int>(std::lrintf(v[i] * id));
      blk[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::clamp(q, -127, 127)));
    }
  }
}

void DecodeAffineRow(const std::uint8_t* row, std::size_t K, bool six,
                     float* out) noexcept {
  // q6g64: K/2 low-nibble bytes (value 2j low, 2j+1 high), K/4 bytes of
  // high bit pairs, then K/64 records (f16 scale, f16 bias).
  // q8g64: K code bytes, then K/64 records.
  const std::uint8_t* rec = row + (six ? 3 * K / 4 : K);
  for (std::size_t g = 0; g < K / 64; ++g) {
    std::uint16_t sb[2];
    std::memcpy(sb, rec + 4 * g, 4);
    const float scale = HalfToFloat(sb[0]);
    const float bias = HalfToFloat(sb[1]);
    for (std::size_t k = 64 * g; k < 64 * (g + 1); ++k) {
      unsigned c;
      if (six) {
        const unsigned lo = (row[k / 2] >> (4 * (k & 1))) & 0xFu;
        const unsigned hi = (row[K / 2 + k / 4] >> (2 * (k & 3))) & 0x3u;
        c = lo | (hi << 4);
      } else {
        c = row[k];
      }
      out[k] = static_cast<float>(c) * scale + bias;
    }
  }
}

template <typename Fn>
void ParallelFor(std::size_t n, unsigned threads, Fn&& fn) {
  if (threads == 0) {
    threads = std::max(1u, std::thread::hardware_concurrency());
  }
  threads = static_cast<unsigned>(std::min<std::size_t>(threads, n));
  if (threads <= 1) {
    for (std::size_t i = 0; i < n; ++i) {
      fn(i);
    }
    return;
  }
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      for (std::size_t i = t; i < n; i += threads) {
        fn(i);
      }
    });
  }
  for (auto& p : pool) {
    p.join();
  }
}

}  // namespace

std::size_t HgnEncodedSizeBytes(GgmlType type, std::size_t elements,
                                std::uint64_t cols) noexcept {
  if (cols == 0 || elements % cols != 0) {
    return 0;
  }
  const std::size_t K = cols;
  const std::size_t N = elements / cols;
  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max() / 4;
  if (N > kMax / std::max<std::size_t>(K, 1)) {
    return 0;
  }
  switch (type) {
    case GgmlType::kHT4:
      return K % 128 == 0 && N % 128 == 0 ? N * K / 2 : 0;
    case GgmlType::kHT3:
      return K % 128 == 0 && N % 128 == 0 ? N * K * 3 / 8 : 0;
    case GgmlType::kQ6G64:
      return K % 64 == 0 ? N * Pad(13 * K / 16, 16) : 0;
    case GgmlType::kQ8G64:
      return K % 64 == 0 ? N * (K + K / 16) : 0;
    case GgmlType::kQ4C:
      return K % 32 == 0 ? 64 + Pad(N * K / 2, 64) + N * Pad(K / 16, 16) : 0;
    case GgmlType::kFP8G:
      return K == 160 ? Pad(160 * N, 64) + 4 : 0;
    default:
      return 0;
  }
}

float Fp8Value(std::uint8_t code) noexcept {
  const unsigned e = (code >> 3) & 0xFu;
  const unsigned m = code & 7u;
  const float v = e == 0 ? static_cast<float>(m) / 512.0f
                         : (1.0f + static_cast<float>(m) / 8.0f) *
                               static_cast<float>(1u << e) / 128.0f;
  return (code & 0x80u) != 0 ? -v : v;
}

float HtCodebook(std::uint16_t state) noexcept { return Codebook()[state]; }

void HtTileStates(const std::uint8_t* tile, unsigned b,
                  std::uint16_t* states) noexcept {
  // STORAGE.md "Trellis decode": one circular bitstream of 256 b bits, each
  // u32 word read from its most significant bit. The state of value i is
  // bits b(i+1)-16 .. b(i+1)-1; each value shifts in b new bits.
  std::uint32_t words[32];
  std::memcpy(words, tile, 32u * b);
  const unsigned L = 256u * b;
  const auto bit = [&](unsigned n) noexcept -> std::uint32_t {
    n %= L;
    return (words[n >> 5] >> (31u - (n & 31u))) & 1u;
  };
  std::uint32_t s = 0;
  for (unsigned k = 0; k < 16; ++k) {
    s = (s << 1) | bit(L + b - 16 + k);
  }
  states[0] = static_cast<std::uint16_t>(s);
  for (unsigned i = 1; i < 256; ++i) {
    for (unsigned k = 0; k < b; ++k) {
      s = (s << 1) | bit(b * i + k);
    }
    states[i] = static_cast<std::uint16_t>(s & 0xFFFFu);
  }
}

void FloatToHalfBits(const float* src, std::uint16_t* dst,
                     std::size_t count) noexcept {
  for (std::size_t i = 0; i < count; ++i) {
    const _Float16 h = static_cast<_Float16>(src[i]);
    std::memcpy(dst + i, &h, sizeof(h));
  }
}

bool DecodeHgnToF32(GgmlType type, const void* payload, std::uint64_t cols,
                    std::uint64_t rows, const std::uint16_t* su,
                    const std::uint16_t* sv, float* out, std::string* error,
                    unsigned threads) noexcept {
  const auto fail = [&](const char* message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  const auto* p = static_cast<const std::uint8_t*>(payload);
  const std::size_t K = cols;
  const std::size_t N = rows;
  if (HgnEncodedSizeBytes(type, K * N, K) == 0) {
    return fail("HGN store shape breaks its size rule");
  }
  try {
    switch (type) {
      case GgmlType::kHT4:
      case GgmlType::kHT3: {
        if (su == nullptr || sv == nullptr) {
          return fail("ht tensor without rotation vectors");
        }
        const unsigned b = type == GgmlType::kHT4 ? 4 : 3;
        ParallelFor(N / 128, threads, [&](std::size_t band) {
          thread_local std::vector<float> m;
          DecodeHtBand(p, K, b, band, su, sv, out + band * 128 * K, m);
        });
        return true;
      }
      case GgmlType::kQ6G64:
      case GgmlType::kQ8G64: {
        const bool six = type == GgmlType::kQ6G64;
        const std::size_t row_bytes = six ? Pad(13 * K / 16, 16) : K + K / 16;
        ParallelFor(N, threads, [&](std::size_t r) {
          DecodeAffineRow(p + r * row_bytes, K, six, out + r * K);
        });
        return true;
      }
      case GgmlType::kQ4C: {
        float codebook[16];
        std::memcpy(codebook, p, sizeof(codebook));
        const std::size_t scales = 64 + Pad(N * K / 2, 64);
        const std::size_t scale_row = Pad(K / 16, 16);
        ParallelFor(N, threads, [&](std::size_t r) {
          const std::uint8_t* codes = p + 64 + r * K / 2;
          const std::uint8_t* srow = p + scales + r * scale_row;
          for (std::size_t g = 0; g < K / 32; ++g) {
            std::uint16_t bits;
            std::memcpy(&bits, srow + 2 * g, 2);
            const float scale = HalfToFloat(bits);
            for (std::size_t k = 32 * g; k < 32 * (g + 1); ++k) {
              const unsigned c = (codes[k / 2] >> (4 * (k & 1))) & 0xFu;
              out[r * K + k] = codebook[c] * scale;
            }
          }
        });
        return true;
      }
      default:
        return fail("not a decodable HGN store");
    }
  } catch (...) {
    return fail("HGN decode failed (allocation or thread start)");
  }
}

bool DecodeHgnToQ8_0(GgmlType type, const void* payload, std::uint64_t cols,
                     std::uint64_t rows, const std::uint16_t* su,
                     const std::uint16_t* sv, void* out, std::string* error,
                     unsigned threads) noexcept {
  const auto fail = [&](const char* message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  const auto* p = static_cast<const std::uint8_t*>(payload);
  auto* q = static_cast<std::uint8_t*>(out);
  const std::size_t K = cols;
  const std::size_t N = rows;
  const std::size_t qrow = K / 32 * 34;
  if (K % 32 != 0 || HgnEncodedSizeBytes(type, K * N, K) == 0) {
    return fail("HGN store shape breaks its size rule or Q8_0 blocks");
  }
  try {
    switch (type) {
      case GgmlType::kHT4:
      case GgmlType::kHT3: {
        if (su == nullptr || sv == nullptr) {
          return fail("ht tensor without rotation vectors");
        }
        const unsigned b = type == GgmlType::kHT4 ? 4 : 3;
        ParallelFor(N / 128, threads, [&](std::size_t band) {
          thread_local std::vector<float> m, rowsf;
          rowsf.resize(128 * K);
          DecodeHtBand(p, K, b, band, su, sv, rowsf.data(), m);
          for (std::size_t r = 0; r < 128; ++r)
            QuantizeRowQ8_0(rowsf.data() + r * K, K, q + (band * 128 + r) * qrow);
        });
        return true;
      }
      case GgmlType::kQ6G64:
      case GgmlType::kQ8G64: {
        const bool six = type == GgmlType::kQ6G64;
        const std::size_t row_bytes = six ? Pad(13 * K / 16, 16) : K + K / 16;
        ParallelFor(N, threads, [&](std::size_t r) {
          thread_local std::vector<float> row;
          row.resize(K);
          DecodeAffineRow(p + r * row_bytes, K, six, row.data());
          QuantizeRowQ8_0(row.data(), K, q + r * qrow);
        });
        return true;
      }
      case GgmlType::kQ4C: {
        float codebook[16];
        std::memcpy(codebook, p, sizeof(codebook));
        const std::size_t scales = 64 + Pad(N * K / 2, 64);
        const std::size_t scale_row = Pad(K / 16, 16);
        ParallelFor(N, threads, [&](std::size_t r) {
          thread_local std::vector<float> row;
          row.resize(K);
          const std::uint8_t* codes = p + 64 + r * K / 2;
          const std::uint8_t* srow = p + scales + r * scale_row;
          for (std::size_t g = 0; g < K / 32; ++g) {
            std::uint16_t bits;
            std::memcpy(&bits, srow + 2 * g, 2);
            const float scale = HalfToFloat(bits);
            for (std::size_t k = 32 * g; k < 32 * (g + 1); ++k)
              row[k] = codebook[(codes[k / 2] >> (4 * (k & 1))) & 0xFu] * scale;
          }
          QuantizeRowQ8_0(row.data(), K, q + r * qrow);
        });
        return true;
      }
      default:
        return fail("not a decodable HGN store");
    }
  } catch (...) {
    return fail("HGN decode failed (allocation or thread start)");
  }
}

}  // namespace gufo::quant
