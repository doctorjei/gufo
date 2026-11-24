// q6g64 matrix-vector kernel (see q6g64_dense.hpp).
//
// L lanes share a row (32 for long rows, 8 for short ones) and run over the
// whole row, so no input split or partial sums are needed. Each lane takes
// 64-value groups g = sub, sub + L, ... of its row: it loads the group's 32
// bytes of low nibbles, 16 bytes of high bit pairs and its 4-byte record,
// forms each code with shifts and masks, and accumulates sum(code * x) and
// sum(x) in f32; the group's scale and bias are applied once per group. The L
// lanes of a row are summed with cross-lane exchanges.
//
// One kernel for every token count, and each token's arithmetic is the same
// as for a single token (the same products, in the same order, and the same
// lane reduction), so a verify batch gives bit for bit the single-token
// results: speculative decoding then never changes greedy output.

#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/core/quant/q6g64_dense.hpp"

namespace gufo {
namespace quant {
namespace q6g64 {

constexpr int kThreads = 256;
constexpr int kWave = 32;

typedef std::uint32_t u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float HalfBits(std::uint32_t bits) {
  return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(bits)));
}

template <int T, int L>
__global__ __launch_bounds__(kThreads) void Q6g64RowKernel(Q6g64DenseArgs a) {
  constexpr int kRowsPerWg = kThreads / L;
  const int lane = threadIdx.x & (kWave - 1);
  const int sub = lane & (L - 1);
  const int row = blockIdx.x * kRowsPerWg + static_cast<int>(threadIdx.x) / L;
  const int K = a.cols;
  const int G = K / 64;
  const std::size_t S = (static_cast<std::size_t>(13) * K / 16 + 15) / 16 * 16;
  float acc[T];
#pragma unroll
  for (int t = 0; t < T; ++t) acc[t] = 0.0f;
  if (row < a.rows) {
    const std::uint8_t* p = a.w + static_cast<std::size_t>(row) * S;
    for (int g = sub; g < G; g += L) {
      const u32x4 lo0 = *reinterpret_cast<const u32x4*>(p + 32 * g);
      const u32x4 lo1 = *reinterpret_cast<const u32x4*>(p + 32 * g + 16);
      const u32x4 hi = *reinterpret_cast<const u32x4*>(p + K / 2 + 16 * g);
      const std::uint32_t rec = *reinterpret_cast<const std::uint32_t*>(p + 3 * K / 4 + 4 * g);
      const float scale = HalfBits(rec & 0xFFFFu);
      const float bias = HalfBits(rec >> 16);
      const std::uint32_t lw[8] = {lo0[0], lo0[1], lo0[2], lo0[3], lo1[0], lo1[1], lo1[2], lo1[3]};
      const std::uint32_t hw[4] = {hi[0], hi[1], hi[2], hi[3]};
#pragma unroll
      for (int t = 0; t < T; ++t) {
        if (t >= a.n_tokens) break;
        const float4* xv = reinterpret_cast<const float4*>(
            a.x + static_cast<std::size_t>(t) * a.x_stride + 64 * g);
        float dot = 0.0f, sx = 0.0f;
#pragma unroll
        for (int q = 0; q < 16; ++q) {
          const float4 x4 = xv[q];
          const float xs[4] = {x4.x, x4.y, x4.z, x4.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int m = 4 * q + e;  // value index in the group
            const std::uint32_t code = ((lw[m >> 3] >> (4 * (m & 7))) & 0xFu) |
                                       (((hw[m >> 4] >> (2 * (m & 15))) & 0x3u) << 4);
            dot += static_cast<float>(code) * xs[e];
            sx += xs[e];
          }
        }
        acc[t] += scale * dot + bias * sx;
      }
    }
  }
#pragma unroll
  for (int t = 0; t < T; ++t)
#pragma unroll
    for (int s = L / 2; s > 0; s >>= 1) acc[t] += __shfl_xor(acc[t], s, kWave);
  if (row < a.rows && sub == 0) {
#pragma unroll
    for (int t = 0; t < T; ++t)
      if (t < a.n_tokens) a.out[static_cast<std::size_t>(t) * a.out_stride + row] = acc[t];
  }
}

template <int T>
bool Launch(const Q6g64DenseArgs& a, hipStream_t stream) {
  if (a.cols / 64 >= 32) {
    constexpr int L = 32;
    Q6g64RowKernel<T, L><<<dim3((a.rows + kThreads / L - 1) / (kThreads / L)), kThreads, 0,
                           stream>>>(a);
  } else {
    constexpr int L = 8;
    Q6g64RowKernel<T, L><<<dim3((a.rows + kThreads / L - 1) / (kThreads / L)), kThreads, 0,
                           stream>>>(a);
  }
  return hipGetLastError() == hipSuccess;
}

}  // namespace q6g64
}  // namespace quant
}  // namespace gufo

extern "C" {

void q6g64_dense_scratch(int rows, int cols, std::size_t* partials, std::size_t* counters) {
  (void)rows;
  (void)cols;
  *partials = 0;  // no input split: no partial sums
  *counters = 0;
}

bool q6g64_dense_gemv(const Q6g64DenseArgs* args, hipStream_t stream) {
  using namespace gufo::quant::q6g64;
  const Q6g64DenseArgs& a = *args;
  if (a.rows <= 0 || a.cols <= 0 || a.cols % 64 != 0 || a.n_tokens < 1 || a.n_tokens > 8)
    return false;
  switch (a.n_tokens) {
    case 1: return Launch<1>(a, stream);
    case 2: return Launch<2>(a, stream);
    case 3:
    case 4: return Launch<4>(a, stream);
    default: return Launch<8>(a, stream);
  }
}

}  // extern "C"
