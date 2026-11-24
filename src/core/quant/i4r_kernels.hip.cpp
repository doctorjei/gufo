// HIP kernels for the I4R128 rotated-expert path: the four FWHT rotation
// stages, MMVQ decode (single-token GEMV) and MMQ prefill (multi-token GEMM).
//
// Contract authority: docs/internal/design/i4r128/RFC.md. The pure math
// (decode, FWHT, block dot) lives in i4r_hip_math.hpp and is shared with the
// host-emulation test; these kernels add only memory movement around it.
//
// Stage plan per trunk layer (gate/up share su and the input, so stage 1 is
// computed once and reused by all 512 experts):
//   stage 1  x <- Hn( x * su )            width 2560, launched once per layer
//   stage 2  g <- Hn( g )                width 1280 (fused gate/up output)
//   stage 3  d_in <- Hn( d_in )         width 640
//   stage 4  y <- Hn( y ) * sv          width 2560
// The rotate kernel below covers all four: pass pre_scale for stage 1,
// post_scale for stage 4, neither for stages 2/3.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/core/quant/i4r_hip_math.hpp"

namespace gufo {
namespace quant {

// One block of 128 threads rotates one 128-block. Optional pre_scale is
// multiplied in before the butterfly (su, stage 1); optional post_scale after
// (sv, stage 4). The butterfly uses the ds4-style per-thread ownership over
// shared memory; the recurrence is i4r_fwht128's, so the result is the
// Sylvester Hn with no reordering.
__global__ void i4r_rotate_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  const float* __restrict__ pre_scale,
                                  const float* __restrict__ post_scale,
                                  int n_blocks) {
  const int blk = blockIdx.x;
  if (blk >= n_blocks)
    return;
  const int tid = threadIdx.x;  // 0..127
  __shared__ float s[I4R_BLOCK];

  float v = in[static_cast<int64_t>(blk) * I4R_BLOCK + tid];
  if (pre_scale != nullptr)
    v *= pre_scale[static_cast<int64_t>(blk) * I4R_BLOCK + tid];
  s[tid] = v;
  __syncthreads();

  for (unsigned stride = 1u; stride < I4R_BLOCK; stride <<= 1u) {
    if ((tid & stride) == 0u) {
      const unsigned base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
      const float a = s[base];
      const float b = s[base + stride];
      s[base] = a + b;
      s[base + stride] = a - b;
    }
    __syncthreads();
  }

  float r = s[tid] * I4R_INV_SQRT_BLOCK;
  if (post_scale != nullptr)
    r *= post_scale[static_cast<int64_t>(blk) * I4R_BLOCK + tid];
  out[static_cast<int64_t>(blk) * I4R_BLOCK + tid] = r;
}

// MMVQ decode: y[o] = dot(M_row_o, xr) for one token. M is I4R128 with row
// stride (n_blocks_in * 66) bytes; xr is the already-rotated input (output of
// stages 1/2). One block per output row; threads split the row's 128-blocks
// and block-reduce.
__global__ void i4r_mmvq_kernel(const uint8_t* __restrict__ M,
                                const float* __restrict__ xr,
                                float* __restrict__ y, int n_out,
                                int n_blocks_in) {
  const int o = blockIdx.x;
  if (o >= n_out)
    return;
  const int tid = threadIdx.x;  // 128 threads
  const uint8_t* row = M + static_cast<int64_t>(o) * n_blocks_in * 66;

  float acc = 0.0f;
  for (int b = tid; b < n_blocks_in; b += blockDim.x) {
    const uint8_t* blk = row + b * 66;
    const float scale =
        __half2float(*reinterpret_cast<const __half*>(blk + 64));
    acc += i4r_dot_block(blk, scale, xr + b * I4R_BLOCK);
  }

  __shared__ float red[I4R_BLOCK];
  red[tid] = acc;
  __syncthreads();
  for (unsigned s = I4R_BLOCK / 2u; s > 0u; s >>= 1u) {
    if (tid < static_cast<int>(s))
      red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    y[o] = red[0];
}

// MMQ prefill: Y[t][o] = dot(M_row_o, X[t]) for a tile of output rows x
// tokens. X is token-major [tokens x n_in] of already-rotated inputs; Y is
// [tokens x n_out]. blockDim = MMQ_ROWS * MMQ_TOK; each thread owns one
// (row, token) cell and walks the row's input blocks.
constexpr int kMmqRows = 8;
constexpr int kMmqTok = 16;

__global__ void i4r_mmq_kernel(const uint8_t* __restrict__ M,
                               const float* __restrict__ X,
                               float* __restrict__ Y, int n_out, int n_in,
                               int tokens) {
  const int n_blocks_in = n_in / I4R_BLOCK;
  const int row0 = blockIdx.x * kMmqRows;
  const int tok0 = blockIdx.y * kMmqTok;
  const int tid = threadIdx.x;
  const int rl = tid / kMmqTok;  // 0..kMmqRows-1
  const int tl = tid % kMmqTok;  // 0..kMmqTok-1
  const int o = row0 + rl;
  const int t = tok0 + tl;
  if (o >= n_out || t >= tokens)
    return;

  const uint8_t* row = M + static_cast<int64_t>(o) * n_blocks_in * 66;
  const float* xt = X + static_cast<int64_t>(t) * n_in;
  float acc = 0.0f;
  for (int b = 0; b < n_blocks_in; ++b) {
    const uint8_t* blk = row + b * 66;
    const float scale =
        __half2float(*reinterpret_cast<const __half*>(blk + 64));
    acc += i4r_dot_block(blk, scale, xt + b * I4R_BLOCK);
  }
  Y[static_cast<int64_t>(t) * n_out + o] = acc;
}

}  // namespace quant
}  // namespace gufo

// Host-callable launchers. The GPU test links these; they keep launch geometry
// in one place.
extern "C" {

void i4r_launch_rotate(const float* in, float* out, const float* pre_scale,
                       const float* post_scale, int n_blocks,
                       hipStream_t stream) {
  gufo::quant::i4r_rotate_kernel<<<n_blocks, 128, 0, stream>>>(
      in, out, pre_scale, post_scale, n_blocks);
}

void i4r_launch_mmvq(const uint8_t* M, const float* xr, float* y, int n_out,
                     int n_blocks_in, hipStream_t stream) {
  gufo::quant::i4r_mmvq_kernel<<<n_out, 128, 0, stream>>>(M, xr, y, n_out,
                                                          n_blocks_in);
}

void i4r_launch_mmq(const uint8_t* M, const float* X, float* Y, int n_out,
                    int n_in, int tokens, hipStream_t stream) {
  dim3 grid((n_out + gufo::quant::kMmqRows - 1) / gufo::quant::kMmqRows,
            (tokens + gufo::quant::kMmqTok - 1) / gufo::quant::kMmqTok);
  gufo::quant::i4r_mmq_kernel<<<
      grid, gufo::quant::kMmqRows * gufo::quant::kMmqTok, 0, stream>>>(
      M, X, Y, n_out, n_in, tokens);
}

}  // extern "C"
