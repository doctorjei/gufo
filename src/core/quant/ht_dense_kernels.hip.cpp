// Matrix-vector products for 4-bit ht weights, read in place (see
// ht_dense.hpp and docs/internal/design/hgn/HT-DENSE.md).
//
// Format (public hgn-spec v1.1.0, STORAGE.md "ht"): 16 x 16 tiles of 128
// bytes, 32 little-endian words read from the most significant bit as one
// circular bitstream; value i of a tile has the 16-bit state made of stream
// bits 4(i+1)-16 .. 4(i+1)-1 and sits at tile row (i/8) mod 16, tile column
// 8(i/128) + i mod 8. Its stored number is codebook 1 of the state. Tiles
// are ordered band (128 rows), column tile, row tile.
//
// Work split (clean-room ht dense decode spec; docs/internal/design/hgn/PROVENANCE.md):
//   * 256-lane workgroups (8 waves of 32). One workgroup owns one 128-row
//     band and a contiguous slice of the input, in 128-column blocks; the
//     host picks the split per matrix.
//   * The workgroup stages its slice: x * su, a 128-point Hadamard transform
//     in registers (one wave per block), 1/sqrt(128), rounded to f16 into
//     shared memory.
//   * Each wave takes column tiles of the slice in turn. Lane l loads word l
//     and word l-1 (mod 32) of each of the 8 row tiles of its column tile:
//     every load instruction covers one whole 128-byte tile. The two words
//     hold the lane's eight 16-bit states (values 8l..8l+7: tile row l mod 16,
//     columns 8(l/16)..+7). The next column tile's words are loaded before
//     the current one is decoded.
//   * The codebook is computed in registers: x = s * 0x83DCD12D, a = byte sum
//     of x (one v_sad_u8), then (1024 + a) as f16 bits (0x6400 | a) and one
//     packed f16 FMA with the spec constants gives two stored numbers.
//   * Packed f16 dot products accumulate into f32; each lane keeps 8 rows x
//     T tokens of sums. Lanes l and l+16 share a row (one cross-lane add),
//     the 8 waves add through shared memory.
//   * With a split input, workgroups write partial sums and the last one to
//     finish a band (arrival counter, reset by that workgroup) adds them in
//     a fixed order, applies the output Hadamard transform, 1/sqrt(128) and
//     sv, and writes the band. No floating-point atomics, no second launch.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>

#include "src/core/quant/ht_dense.hpp"

namespace gufo {
namespace quant {
namespace htdense {

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kTargetWorkgroups = 640;
constexpr int kMaxTokens = 8;
constexpr int kMaxGroup = HT_DENSE_MAX_GROUP;
constexpr float kInvSqrt128 = 0.08838834764831845f;

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

__device__ __forceinline__ half2_t AsHalf2(std::uint32_t bits) {
  return __builtin_bit_cast(half2_t, bits);
}

__device__ __forceinline__ float Dot2(half2_t a, half2_t b, float c) {
  return __builtin_amdgcn_fdot2(a, b, c, false);
}

// 128-point Hadamard transform on one wave; lane l holds v[j] = x[l + 32 j].
__device__ __forceinline__ void Fwht128Wave(float v[4]) {
  const int lane = threadIdx.x & (kWave - 1);
#pragma unroll
  for (int s = 1; s < kWave; s <<= 1) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float p = __shfl_xor(v[j], s, kWave);
      v[j] = (lane & s) ? (p - v[j]) : (v[j] + p);
    }
  }
  const float a = v[0] + v[1], b = v[0] - v[1];
  const float c = v[2] + v[3], d = v[2] - v[3];
  v[0] = (a + c) * kInvSqrt128;
  v[2] = (a - c) * kInvSqrt128;
  v[1] = (b + d) * kInvSqrt128;
  v[3] = (b - d) * kInvSqrt128;
}

// Two stored numbers from two 16-bit states (STORAGE.md "Codebook"): the
// single f16 FMA (1024 + a) * 0x1EEE + 0xC931 of the spec.
__device__ __forceinline__ half2_t Codebook2(std::uint32_t s0, std::uint32_t s1) {
  const std::uint32_t a0 = __builtin_amdgcn_sad_u8(s0 * 0x83DCD12Du, 0u, 0u);
  const std::uint32_t a1 = __builtin_amdgcn_sad_u8(s1 * 0x83DCD12Du, 0u, 0u);
  const half2_t h = AsHalf2((0x6400u | a0) | ((0x6400u | a1) << 16));
  return __builtin_elementwise_fma(h, AsHalf2(0x1EEE1EEEu), AsHalf2(0xC931C931u));
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

// One workgroup's work for matrix `a`: band and input slice from `block`.
template <int T>
__device__ __forceinline__ void HtBody(const HtDenseArgs& a, int splits, int per, int block) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int nblk = a.cols / 128;
  const int band = block / splits;
  const int split = block % splits;
  const int b0 = split * per;
  const int nb = min(nblk, b0 + per) - b0;
  const int lane = threadIdx.x & (kWave - 1);
  const int wave = threadIdx.x / kWave;
  const int xw = per * 128;  // staged values per token
  _Float16* xs = reinterpret_cast<_Float16*>(smem);
  const int region = max(T * xw * 2, T * 128 * 4);
  float* fin = reinterpret_cast<float*>(smem + region);

  // Stage the slice: (x * su) rotated, f16.
  for (int k = wave; k < nb; k += 8) {
    const int c0 = (b0 + k) * 128;
    float s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) s[j] = a.su[c0 + lane + 32 * j];
#pragma unroll
    for (int t = 0; t < T; ++t) {
      float v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        v[j] = t < a.n_tokens ? a.x[static_cast<std::size_t>(t) * a.x_stride + c0 + lane + 32 * j] * s[j]
                              : 0.0f;
      Fwht128Wave(v);
#pragma unroll
      for (int j = 0; j < 4; ++j) xs[t * xw + k * 128 + lane + 32 * j] = static_cast<_Float16>(v[j]);
    }
  }
  __syncthreads();

  float acc[8][T];
#pragma unroll
  for (int r = 0; r < 8; ++r)
#pragma unroll
    for (int t = 0; t < T; ++t) acc[r][t] = 0.0f;

  const std::size_t col_tiles = static_cast<std::size_t>(a.cols) / 16;
  const std::uint32_t* words = reinterpret_cast<const std::uint32_t*>(a.codes);
  const int prev_lane = (lane + kWave - 1) & (kWave - 1);
  const int ntile = nb * 8;
  std::uint32_t cur[8], prv[8], ncur[8], nprv[8];
  const auto load = [&](int ctl, std::uint32_t (&c)[8], std::uint32_t (&p)[8]) {
    const std::uint32_t* t0 =
        words + ((static_cast<std::size_t>(band) * col_tiles + b0 * 8 + ctl) * 8) * 32;
#pragma unroll
    for (int rt = 0; rt < 8; ++rt) {
      c[rt] = __builtin_nontemporal_load(t0 + rt * 32 + lane);
      p[rt] = t0[rt * 32 + prev_lane];
    }
  };
  int ctl = wave;
  if (ctl < ntile) load(ctl, cur, prv);
  for (; ctl < ntile; ctl += 8) {
    const bool more = ctl + 8 < ntile;
    if (more) load(ctl + 8, ncur, nprv);
    const int xo = ctl * 16 + 8 * (lane >> 4);
    half2_t xin[T][4];
#pragma unroll
    for (int t = 0; t < T; ++t) {
      const half2_t* p = reinterpret_cast<const half2_t*>(xs + t * xw + xo);
#pragma unroll
      for (int k = 0; k < 4; ++k) xin[t][k] = p[k];
    }
#pragma unroll
    for (int rt = 0; rt < 8; ++rt) {
      const std::uint64_t v = (static_cast<std::uint64_t>(prv[rt]) << 32) | cur[rt];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        // Value j = 2k, 2k+1 of the lane: state = v >> (28 - 4 j), 16 bits.
        const std::uint32_t s0 = static_cast<std::uint32_t>(v >> (28 - 8 * k)) & 0xFFFFu;
        const std::uint32_t s1 = static_cast<std::uint32_t>(v >> (24 - 8 * k)) & 0xFFFFu;
        const half2_t w2 = Codebook2(s0, s1);
#pragma unroll
        for (int t = 0; t < T; ++t) acc[rt][t] = Dot2(w2, xin[t][k], acc[rt][t]);
      }
    }
    if (more) {
#pragma unroll
      for (int rt = 0; rt < 8; ++rt) {
        cur[rt] = ncur[rt];
        prv[rt] = nprv[rt];
      }
    }
  }

  // Lanes l and l+16 share a row; then the 8 waves add through shared memory
  // (reusing the staging area).
#pragma unroll
  for (int r = 0; r < 8; ++r)
#pragma unroll
    for (int t = 0; t < T; ++t) acc[r][t] += __shfl_xor(acc[r][t], 16, kWave);
  // The 8 waves add into one [T][128] buffer in turn (fixed order, so the
  // result is deterministic); a per-wave buffer would cost 32 KiB of LDS at
  // 8 tokens and cap occupancy.
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem);
  for (int w = 0; w < 8; ++w) {
    if (wave == w && lane < 16) {
#pragma unroll
      for (int r = 0; r < 8; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) {
          float& slot = red[t * 128 + r * 16 + lane];
          slot = (w == 0) ? acc[r][t] : slot + acc[r][t];
        }
    }
    __syncthreads();
  }
  for (int i = threadIdx.x; i < T * 128; i += kThreads) {
    const int t = i / 128, r = i % 128;
    const float s = red[t * 128 + r];
    if (splits == 1)
      fin[i] = s;
    else
      a.partials[((static_cast<std::size_t>(band) * splits + split) * kMaxTokens + t) * 128 + r] = s;
  }
  if (splits > 1) {
    if (!LastArrival(a.counters + band, static_cast<unsigned>(splits))) return;
    for (int i = threadIdx.x; i < T * 128; i += kThreads) {
      const int t = i / 128, r = i % 128;
      float s = 0.0f;
      for (int p = 0; p < splits; ++p)
        s += LoadPartial(a.partials + ((static_cast<std::size_t>(band) * splits + p) * kMaxTokens + t) * 128 + r);
      fin[i] = s;
    }
  }
  __syncthreads();
  // Output side: Hadamard transform across the band's 128 rows, 1/sqrt(128),
  // sv. One wave per token.
  if (wave < T && wave < a.n_tokens) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = fin[wave * 128 + lane + 32 * j];
    Fwht128Wave(v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = band * 128 + lane + 32 * j;
      a.out[static_cast<std::size_t>(wave) * a.out_stride + row] = v[j] * a.sv[row];
    }
  }
}

// Up to kMaxGroup matrices that read the same input share one launch; each
// owns a contiguous range of workgroups, its own split and scratch.
struct GroupPlan {
  HtMatrix m[kMaxGroup];
  int splits[kMaxGroup];
  int per[kMaxGroup];
  int wg_begin[kMaxGroup + 1];
  int partial_off[kMaxGroup];
  int counter_off[kMaxGroup];
  int count;
  const float* x;
  int cols;
  int n_tokens;
  int x_stride;
  float* partials;
  unsigned* counters;
};

template <int T>
__global__ __launch_bounds__(kThreads) void HtGroupKernel(GroupPlan g) {
  int mi = 0;
  while (mi + 1 < g.count && static_cast<int>(blockIdx.x) >= g.wg_begin[mi + 1]) ++mi;
  const HtDenseArgs a{g.m[mi].codes, g.m[mi].su, g.m[mi].sv, g.x, g.m[mi].out,
                      g.partials + g.partial_off[mi], g.counters + g.counter_off[mi],
                      g.m[mi].rows, g.cols, g.n_tokens, g.x_stride, g.m[mi].out_stride};
  HtBody<T>(a, g.splits[mi], g.per[mi], static_cast<int>(blockIdx.x) - g.wg_begin[mi]);
}

struct Plan {
  int splits;
  int per;
};

Plan PlanFor(int rows, int cols) {
  const int bands = rows / 128;
  const int nblk = cols / 128;
  int splits = std::min(nblk, std::max(1, (kTargetWorkgroups + bands - 1) / bands));
  const int per = (nblk + splits - 1) / splits;
  splits = (nblk + per - 1) / per;
  return {splits, per};
}

}  // namespace htdense
}  // namespace quant
}  // namespace gufo

extern "C" {

int ht_dense_splits(int rows, int cols) {
  return gufo::quant::htdense::PlanFor(rows, cols).splits;
}

std::size_t ht_dense_partial_floats(int rows, int cols) {
  using namespace gufo::quant::htdense;
  const Plan p = PlanFor(rows, cols);
  return p.splits > 1 ? static_cast<std::size_t>(rows / 128) * p.splits * kMaxTokens * 128 : 0;
}

bool ht_dense_group(const HtGroupArgs* args, hipStream_t stream) {
  using namespace gufo::quant::htdense;
  const HtGroupArgs& a = *args;
  if (a.count < 1 || a.count > kMaxGroup || a.cols <= 0 || a.cols % 128 != 0 ||
      a.n_tokens < 1 || a.n_tokens > kMaxTokens)
    return false;
  GroupPlan g{};
  g.count = a.count;
  g.x = a.x;
  g.cols = a.cols;
  g.n_tokens = a.n_tokens;
  g.x_stride = a.x_stride;
  g.partials = a.partials;
  g.counters = a.counters;
  const int T = a.n_tokens == 1 ? 1 : a.n_tokens == 2 ? 2 : a.n_tokens <= 4 ? 4 : 8;
  std::size_t lds = 0, partials = 0, counters = 0;
  int wg = 0;
  for (int i = 0; i < a.count; ++i) {
    const HtMatrix& m = a.m[i];
    if (m.rows <= 0 || m.rows % 128 != 0 || m.codes == nullptr) return false;
    const Plan p = PlanFor(m.rows, a.cols);
    g.m[i] = m;
    g.splits[i] = p.splits;
    g.per[i] = p.per;
    g.wg_begin[i] = wg;
    g.partial_off[i] = static_cast<int>(partials);
    g.counter_off[i] = static_cast<int>(counters);
    wg += (m.rows / 128) * p.splits;
    if (p.splits > 1) partials += static_cast<std::size_t>(m.rows / 128) * p.splits * kMaxTokens * 128;
    counters += m.rows / 128;
    const int region = std::max(T * p.per * 128 * 2, T * 128 * 4);
    lds = std::max(lds, static_cast<std::size_t>(region) + T * 128 * 4);
  }
  g.wg_begin[a.count] = wg;
  if (lds > 64 * 1024 || partials > a.partial_capacity || counters > a.counter_capacity ||
      (partials > 0 && a.partials == nullptr) || a.counters == nullptr)
    return false;
  const dim3 grid(static_cast<unsigned>(wg));
  switch (T) {
    case 1: HtGroupKernel<1><<<grid, kThreads, lds, stream>>>(g); break;
    case 2: HtGroupKernel<2><<<grid, kThreads, lds, stream>>>(g); break;
    case 4: HtGroupKernel<4><<<grid, kThreads, lds, stream>>>(g); break;
    default: HtGroupKernel<8><<<grid, kThreads, lds, stream>>>(g); break;
  }
  return hipGetLastError() == hipSuccess;
}

void ht_dense_group_scratch(const int* rows, int count, int cols, std::size_t* partials,
                            std::size_t* counters) {
  using namespace gufo::quant::htdense;
  *partials = 0;
  *counters = 0;
  for (int i = 0; i < count; ++i) {
    *partials += ht_dense_partial_floats(rows[i], cols);
    *counters += static_cast<std::size_t>(rows[i] / 128);
  }
}

bool ht_dense_gemv(const HtDenseArgs* args, hipStream_t stream) {
  const HtDenseArgs& a = *args;
  HtGroupArgs g{};
  g.m[0] = HtMatrix{a.codes, a.su, a.sv, a.out, a.rows, a.out_stride};
  g.count = 1;
  g.x = a.x;
  g.cols = a.cols;
  g.n_tokens = a.n_tokens;
  g.x_stride = a.x_stride;
  g.partials = a.partials;
  g.counters = a.counters;
  g.partial_capacity = ht_dense_partial_floats(a.rows, a.cols);
  g.counter_capacity = static_cast<std::size_t>(a.rows / 128);
  return ht_dense_group(&g, stream);
}

}  // extern "C"
