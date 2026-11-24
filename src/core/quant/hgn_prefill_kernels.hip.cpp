// Wide (prefill) kernels for kHT4 and kQ6G64 weights (see hgn_prefill.hpp).

#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/core/quant/hgn_prefill.hpp"

namespace gufo {
namespace quant {
namespace hgn_prefill {

constexpr int kWave = 32;
constexpr float kInvSqrt128 = 0.08838834764831845f;

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef std::uint32_t u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ half2_t AsHalf2(std::uint32_t bits) {
  return __builtin_bit_cast(half2_t, bits);
}

// Two stored numbers from two 16-bit states (STORAGE.md "Codebook"): the
// single f16 FMA (1024 + a) * 0x1EEE + 0xC931, a = byte sum of s * 0x83DCD12D.
__device__ __forceinline__ half2_t Codebook2(std::uint32_t s0, std::uint32_t s1) {
  const std::uint32_t a0 = __builtin_amdgcn_sad_u8(s0 * 0x83DCD12Du, 0u, 0u);
  const std::uint32_t a1 = __builtin_amdgcn_sad_u8(s1 * 0x83DCD12Du, 0u, 0u);
  const half2_t h = AsHalf2((0x6400u | a0) | ((0x6400u | a1) << 16));
  return __builtin_elementwise_fma(h, AsHalf2(0x1EEE1EEEu), AsHalf2(0xC931C931u));
}

// 128-point Hadamard transform on one wave, scaled by 1/sqrt(128); lane l
// holds v[j] = x[l + 32 j]. Natural order in and out.
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

// One workgroup per (band, column tile); wave rt decodes row tile rt. Tile
// (band, ct, rt) is 32 words at ((band * col_tiles + ct) * 8 + rt) * 32. Lane l
// holds values 8 l .. 8 l + 7 of the tile: tile row l % 16, columns
// 8 (l / 16) .. + 7. Value j's 16-bit state is the lane's word joined below its
// predecessor word (the tile's last word for lane 0), shifted by 28 - 4 j.
__global__ __launch_bounds__(256) void HtDecodeF16Kernel(const std::uint32_t* __restrict__ words,
                                                         int cols,
                                                         _Float16* __restrict__ out) {
  const int col_tiles = cols / 16;
  const int band = blockIdx.x / col_tiles;
  const int ct = blockIdx.x % col_tiles;
  const int rt = threadIdx.x / kWave;
  const int lane = threadIdx.x & (kWave - 1);
  const std::uint32_t* t =
      words + ((static_cast<std::size_t>(band) * col_tiles + ct) * 8 + rt) * 32;
  const std::uint32_t cur = t[lane];
  const std::uint32_t prv = t[(lane + kWave - 1) & (kWave - 1)];
  const std::uint64_t v = (static_cast<std::uint64_t>(prv) << 32) | cur;
  std::uint32_t h[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const std::uint32_t s0 = static_cast<std::uint32_t>(v >> (28 - 8 * k)) & 0xFFFFu;
    const std::uint32_t s1 = static_cast<std::uint32_t>(v >> (24 - 8 * k)) & 0xFFFFu;
    h[k] = __builtin_bit_cast(std::uint32_t, Codebook2(s0, s1));
  }
  const std::size_t row = static_cast<std::size_t>(band) * 128 + rt * 16 + (lane & 15);
  _Float16* dst = out + row * cols + ct * 16 + 8 * (lane >> 4);
  *reinterpret_cast<u32x4*>(dst) = u32x4{h[0], h[1], h[2], h[3]};
}

// One wave per (token, 128-value block); 8 waves a workgroup.
__global__ __launch_bounds__(256) void HtRotateInputKernel(const float* __restrict__ x,
                                                           const float* __restrict__ su,
                                                           _Float16* __restrict__ out,
                                                           int n_blocks, int total) {
  const int w = blockIdx.x * 8 + threadIdx.x / kWave;
  if (w >= total) return;
  const int lane = threadIdx.x & (kWave - 1);
  const int t = w / n_blocks;
  const int b = w % n_blocks;
  const std::size_t cols = static_cast<std::size_t>(n_blocks) * 128;
  const float* xr = x + t * cols + b * 128;
  const float* s = su + b * 128;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = xr[lane + 32 * j] * s[lane + 32 * j];
  Fwht128Wave(v);
  _Float16* o = out + t * cols + b * 128;
#pragma unroll
  for (int j = 0; j < 4; ++j) o[lane + 32 * j] = static_cast<_Float16>(v[j]);
}

__global__ __launch_bounds__(256) void HtRotateOutputKernel(float* __restrict__ y,
                                                            const float* __restrict__ sv,
                                                            int n_bands, int total) {
  const int w = blockIdx.x * 8 + threadIdx.x / kWave;
  if (w >= total) return;
  const int lane = threadIdx.x & (kWave - 1);
  const int t = w / n_bands;
  const int b = w % n_bands;
  float* yr = y + static_cast<std::size_t>(t) * n_bands * 128 + b * 128;
  const float* s = sv + b * 128;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = yr[lane + 32 * j];
  Fwht128Wave(v);
#pragma unroll
  for (int j = 0; j < 4; ++j) yr[lane + 32 * j] = v[j] * s[lane + 32 * j];
}

__device__ __forceinline__ float HalfBits(std::uint32_t bits) {
  return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(bits)));
}

// One thread per (row, 64-value group); writes the group's 64 values.
__global__ __launch_bounds__(256) void Q6g64DecodeF16Kernel(const std::uint8_t* __restrict__ w,
                                                            int rows, int cols,
                                                            _Float16* __restrict__ out) {
  const int G = cols / 64;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * G) return;
  const std::size_t r = i / G;
  const int g = static_cast<int>(i % G);
  const std::size_t S = (static_cast<std::size_t>(13) * cols / 16 + 15) / 16 * 16;
  const std::uint8_t* p = w + r * S;
  const u32x4 lo0 = *reinterpret_cast<const u32x4*>(p + 32 * g);
  const u32x4 lo1 = *reinterpret_cast<const u32x4*>(p + 32 * g + 16);
  const u32x4 hi = *reinterpret_cast<const u32x4*>(p + cols / 2 + 16 * g);
  const std::uint32_t rec = *reinterpret_cast<const std::uint32_t*>(p + 3 * cols / 4 + 4 * g);
  const float scale = HalfBits(rec & 0xFFFFu);
  const float bias = HalfBits(rec >> 16);
  const std::uint32_t lw[8] = {lo0[0], lo0[1], lo0[2], lo0[3], lo1[0], lo1[1], lo1[2], lo1[3]};
  const std::uint32_t hw[4] = {hi[0], hi[1], hi[2], hi[3]};
  _Float16* dst = out + r * cols + 64 * g;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    std::uint32_t o[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      std::uint32_t pair = 0;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int m = 8 * q + 2 * e + h;
        const std::uint32_t code = ((lw[m >> 3] >> (4 * (m & 7))) & 0xFu) |
                                   (((hw[m >> 4] >> (2 * (m & 15))) & 0x3u) << 4);
        const _Float16 val = static_cast<_Float16>(static_cast<float>(code) * scale + bias);
        pair |= static_cast<std::uint32_t>(__builtin_bit_cast(std::uint16_t, val)) << (16 * h);
      }
      o[e] = pair;
    }
    *reinterpret_cast<u32x4*>(dst + 8 * q) = u32x4{o[0], o[1], o[2], o[3]};
  }
}

}  // namespace hgn_prefill
}  // namespace quant
}  // namespace gufo

extern "C" {

bool ht_decode_f16(const std::uint8_t* codes, int rows, int cols, void* out,
                   hipStream_t stream) {
  using namespace gufo::quant::hgn_prefill;
  if (rows <= 0 || cols <= 0 || rows % 128 != 0 || cols % 128 != 0) return false;
  const unsigned grid = static_cast<unsigned>(rows / 128) * static_cast<unsigned>(cols / 16);
  HtDecodeF16Kernel<<<grid, 256, 0, stream>>>(reinterpret_cast<const std::uint32_t*>(codes),
                                              cols, static_cast<_Float16*>(out));
  return hipGetLastError() == hipSuccess;
}

bool ht_rotate_input_f16(const float* x, const float* su, void* out, int n_tokens,
                         int cols, hipStream_t stream) {
  using namespace gufo::quant::hgn_prefill;
  if (n_tokens <= 0 || cols <= 0 || cols % 128 != 0) return false;
  const int total = n_tokens * (cols / 128);
  HtRotateInputKernel<<<static_cast<unsigned>((total + 7) / 8), 256, 0, stream>>>(
      x, su, static_cast<_Float16*>(out), cols / 128, total);
  return hipGetLastError() == hipSuccess;
}

bool ht_rotate_output(float* y, const float* sv, int n_tokens, int rows,
                      hipStream_t stream) {
  using namespace gufo::quant::hgn_prefill;
  if (n_tokens <= 0 || rows <= 0 || rows % 128 != 0) return false;
  const int total = n_tokens * (rows / 128);
  HtRotateOutputKernel<<<static_cast<unsigned>((total + 7) / 8), 256, 0, stream>>>(
      y, sv, rows / 128, total);
  return hipGetLastError() == hipSuccess;
}

bool q6g64_decode_f16(const std::uint8_t* w, int rows, int cols, void* out,
                      hipStream_t stream) {
  using namespace gufo::quant::hgn_prefill;
  if (rows <= 0 || cols <= 0 || cols % 64 != 0) return false;
  const std::size_t n = static_cast<std::size_t>(rows) * (cols / 64);
  Q6g64DecodeF16Kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
      w, rows, cols, static_cast<_Float16*>(out));
  return hipGetLastError() == hipSuccess;
}

}  // extern "C"
