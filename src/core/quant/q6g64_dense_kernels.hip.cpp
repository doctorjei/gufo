// q6g64 matrix-vector kernel (see q6g64_dense.hpp).
//
// Workgroup: 256 lanes = 32 rows x 8 lanes per row, over a slice of SG
// 64-value groups of the input (SG a multiple of 8). Each lane takes groups
// sub, sub + 8, ... of the slice: it loads the group's 32 bytes of low
// nibbles, 16 bytes of high bit pairs and its 4-byte record, forms the 64
// codes once, then for every token reads the input straight from memory
// (the 4 rows of a wave read the same input, so it is served from cache)
// and accumulates sum(code * x) and sum(x) in f32; scale and bias are applied
// once per group. The 8 lanes of a row are summed with cross-lane exchanges.
// Matrices with few rows are split along the input so there are still about
// 256 workgroups; each writes its partial sums and the last to arrive adds
// them in slice order and writes the band.
// A single token uses a separate row kernel (Q6g64RowKernel) with more
// lanes per row and no split.

#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/core/quant/q6g64_dense.hpp"

namespace gufo {
namespace quant {
namespace q6g64 {

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kLanes = 8;                    // lanes per row
constexpr int kRows = kThreads / kLanes;     // 32 rows per workgroup
constexpr int kTargetWorkgroups = 256;
constexpr int kMaxTokens = 8;

typedef std::uint32_t u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float HalfBits(std::uint32_t bits) {
  return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(bits)));
}

__device__ __forceinline__ bool LastArrival(unsigned* counter, unsigned expected) {
  __shared__ unsigned last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned old = atomicAdd(counter, 1u);
    last = (old == expected - 1u) ? 1u : 0u;
    if (last) atomicExch(counter, 0u);
  }
  __syncthreads();
  const bool is_last = last != 0u;
  if (is_last) __threadfence();
  return is_last;
}

__device__ __forceinline__ float LoadPartial(const float* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

struct Plan {
  int splits;
  int sg;  // groups per slice
};

__host__ __device__ inline Plan PlanFor(int rows, int cols) {
  const int bands = (rows + kRows - 1) / kRows;
  const int G = cols / 64;
  const int max_splits = G / kLanes > 0 ? G / kLanes : 1;  // at least 8 groups a slice
  int splits = (kTargetWorkgroups + bands - 1) / bands;
  splits = splits < 1 ? 1 : (splits > max_splits ? max_splits : splits);
  int sg = (G + splits - 1) / splits;
  sg = (sg + kLanes - 1) / kLanes * kLanes;
  splits = (G + sg - 1) / sg;
  return {splits, sg};
}

template <int T>
__global__ __launch_bounds__(kThreads) void Q6g64Kernel(Q6g64DenseArgs a, int splits, int sg) {
  const int band = blockIdx.x / splits;
  const int split = blockIdx.x % splits;
  const int K = a.cols;
  const int G = K / 64;
  const int g0 = split * sg;
  const int g1 = min(G, g0 + sg);
  const int lane = threadIdx.x & (kWave - 1);
  const int sub = lane & (kLanes - 1);
  const int row = band * kRows + static_cast<int>(threadIdx.x) / kLanes;
  const std::size_t S = (static_cast<std::size_t>(13) * K / 16 + 15) / 16 * 16;
  float acc[T];
#pragma unroll
  for (int t = 0; t < T; ++t) acc[t] = 0.0f;
  if (row < a.rows) {
    const std::uint8_t* p = a.w + static_cast<std::size_t>(row) * S;
    for (int g = g0 + sub; g < g1; g += kLanes) {
      const u32x4 lo0 = *reinterpret_cast<const u32x4*>(p + 32 * g);
      const u32x4 lo1 = *reinterpret_cast<const u32x4*>(p + 32 * g + 16);
      const u32x4 hi = *reinterpret_cast<const u32x4*>(p + K / 2 + 16 * g);
      const std::uint32_t rec = *reinterpret_cast<const std::uint32_t*>(p + 3 * K / 4 + 4 * g);
      const float scale = HalfBits(rec & 0xFFFFu);
      const float bias = HalfBits(rec >> 16);
      const std::uint32_t lw[8] = {lo0[0], lo0[1], lo0[2], lo0[3], lo1[0], lo1[1], lo1[2], lo1[3]};
      const std::uint32_t hw[4] = {hi[0], hi[1], hi[2], hi[3]};
      float code[64];
#pragma unroll
      for (int m = 0; m < 64; ++m)
        code[m] = static_cast<float>(((lw[m >> 3] >> (4 * (m & 7))) & 0xFu) |
                                     (((hw[m >> 4] >> (2 * (m & 15))) & 0x3u) << 4));
#pragma unroll
      for (int t = 0; t < T; ++t) {
        if (t >= a.n_tokens) break;
        const float4* xv = reinterpret_cast<const float4*>(
            a.x + static_cast<std::size_t>(t) * a.x_stride + 64 * g);
        float dot = 0.0f, sx = 0.0f;
#pragma unroll
        for (int q = 0; q < 16; ++q) {
          const float4 x4 = xv[q];
          dot += code[4 * q] * x4.x + code[4 * q + 1] * x4.y + code[4 * q + 2] * x4.z +
                 code[4 * q + 3] * x4.w;
          sx += (x4.x + x4.y) + (x4.z + x4.w);
        }
        acc[t] += scale * dot + bias * sx;
      }
    }
  }
#pragma unroll
  for (int t = 0; t < T; ++t)
#pragma unroll
    for (int s = kLanes / 2; s > 0; s >>= 1) acc[t] += __shfl_xor(acc[t], s, kWave);
  const int r_local = static_cast<int>(threadIdx.x) / kLanes;
  if (splits == 1) {
    if (row < a.rows && sub == 0) {
#pragma unroll
      for (int t = 0; t < T; ++t)
        if (t < a.n_tokens) a.out[static_cast<std::size_t>(t) * a.out_stride + row] = acc[t];
    }
    return;
  }
  if (sub == 0) {
#pragma unroll
    for (int t = 0; t < T; ++t)
      a.partials[((static_cast<std::size_t>(band) * splits + split) * kMaxTokens + t) * kRows + r_local] =
          acc[t];
  }
  if (!LastArrival(a.counters + band, static_cast<unsigned>(splits))) return;
  for (int i = threadIdx.x; i < T * kRows; i += kThreads) {
    const int t = i / kRows, r = i % kRows;
    const int orow = band * kRows + r;
    if (t >= a.n_tokens || orow >= a.rows) continue;
    float s = 0.0f;
    for (int q = 0; q < splits; ++q)
      s += LoadPartial(a.partials + ((static_cast<std::size_t>(band) * splits + q) * kMaxTokens + t) * kRows + r);
    a.out[static_cast<std::size_t>(t) * a.out_stride + orow] = s;
  }
}

// One token: L lanes share a row (32 for long rows, 8 for short ones) and
// run over the whole row, so no split or partial sums are needed. This
// gives more lanes per row than the split kernel, which suits a single
// token better.
template <int L>
__global__ __launch_bounds__(kThreads) void Q6g64RowKernel(Q6g64DenseArgs a) {
  constexpr int kRowsPerWg = kThreads / L;
  const int lane = threadIdx.x & (kWave - 1);
  const int sub = lane & (L - 1);
  const int row = blockIdx.x * kRowsPerWg + static_cast<int>(threadIdx.x) / L;
  const int K = a.cols;
  const int G = K / 64;
  const std::size_t S = (static_cast<std::size_t>(13) * K / 16 + 15) / 16 * 16;
  float acc = 0.0f;
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
      const float4* xv = reinterpret_cast<const float4*>(a.x + 64 * g);
      float dot = 0.0f, sx = 0.0f;
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        const float4 x4 = xv[q];
        const float xs[4] = {x4.x, x4.y, x4.z, x4.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int m = 4 * q + e;
          const std::uint32_t code = ((lw[m >> 3] >> (4 * (m & 7))) & 0xFu) |
                                     (((hw[m >> 4] >> (2 * (m & 15))) & 0x3u) << 4);
          dot += static_cast<float>(code) * xs[e];
          sx += xs[e];
        }
      }
      acc += scale * dot + bias * sx;
    }
  }
#pragma unroll
  for (int s = L / 2; s > 0; s >>= 1) acc += __shfl_xor(acc, s, kWave);
  if (row < a.rows && sub == 0) a.out[row] = acc;
}

bool LaunchOneToken(const Q6g64DenseArgs& a, hipStream_t stream) {
  if (a.cols / 64 >= 32) {
    constexpr int L = 32;
    Q6g64RowKernel<L><<<dim3((a.rows + kThreads / L - 1) / (kThreads / L)), kThreads, 0, stream>>>(a);
  } else {
    constexpr int L = 8;
    Q6g64RowKernel<L><<<dim3((a.rows + kThreads / L - 1) / (kThreads / L)), kThreads, 0, stream>>>(a);
  }
  return hipGetLastError() == hipSuccess;
}

template <int T>
bool Launch(const Q6g64DenseArgs& a, const Plan& p, hipStream_t stream) {
  const int bands = (a.rows + kRows - 1) / kRows;
  Q6g64Kernel<T><<<dim3(static_cast<unsigned>(bands * p.splits)), kThreads, 0, stream>>>(
      a, p.splits, p.sg);
  return hipGetLastError() == hipSuccess;
}

}  // namespace q6g64
}  // namespace quant
}  // namespace gufo

extern "C" {

void q6g64_dense_scratch(int rows, int cols, std::size_t* partials, std::size_t* counters) {
  using namespace gufo::quant::q6g64;
  const Plan p = PlanFor(rows, cols);
  const int bands = (rows + kRows - 1) / kRows;
  *partials = p.splits > 1 ? static_cast<std::size_t>(bands) * p.splits * kMaxTokens * kRows : 0;
  *counters = p.splits > 1 ? static_cast<std::size_t>(bands) : 0;
}

bool q6g64_dense_gemv(const Q6g64DenseArgs* args, hipStream_t stream) {
  using namespace gufo::quant::q6g64;
  const Q6g64DenseArgs& a = *args;
  if (a.rows <= 0 || a.cols <= 0 || a.cols % 64 != 0 || a.n_tokens < 1 || a.n_tokens > 8)
    return false;
  const Plan p = PlanFor(a.rows, a.cols);
  std::size_t need_p = 0, need_c = 0;
  q6g64_dense_scratch(a.rows, a.cols, &need_p, &need_c);
  if (need_p > a.partial_capacity || need_c > a.counter_capacity ||
      (need_p > 0 && (a.partials == nullptr || a.counters == nullptr)))
    return false;
  switch (a.n_tokens) {
    case 1: return LaunchOneToken(a, stream);
    case 2: return Launch<2>(a, p, stream);
    case 3:
    case 4: return Launch<4>(a, p, stream);
    default: return Launch<8>(a, p, stream);
  }
}

}  // extern "C"
