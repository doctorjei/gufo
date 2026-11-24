// Batched I4R128 routed-expert kernels: three launches per MoE layer
// (router + top-k stay on the stock path; this file adds gate+up for all
// selected experts, and down for all selected experts with the combine).
//
// Provenance: the technique is Peonist's halogen-flash engine's, reimplemented
// clean-room from a functional spec of its i4r routed-expert decode (the "i4r
// decode spec", draft 1 + addendum 1) and the public hgn-spec v1.1.0; see
// docs/internal/design/hgn/PROVENANCE.md.
//
// Design summary (see docs/internal/design/i4r128/FUSED.md):
//   * Expert weights are repacked once at load into two regions per tensor:
//     all codes (rows x blocks x 64 B) then all f16 scales (rows x blocks).
//   * Row-only split: a 256-lane workgroup owns 64 output rows of one
//     selected expert; four adjacent lanes share a row and take one 16-byte
//     quarter of every 64-byte code block. All of a lane's code loads are
//     issued before any arithmetic.
//   * Nibbles become f16 exactly (code | 0x6400 is 1024 + code; one packed
//     subtraction of 1032 gives code - 8). Packed f16 pairs are dotted with
//     the f16 input into f32 (v_dot2_f32_f16); each 128-value block's scale
//     is applied once to the lane's block sum.
//   * The input is held in LDS as an f16 hi/lo split (x = hi + lo), so the
//     dot product sees ~22 bits of the f32 input for one extra dot2 per pair.
//     args.plain_f16 = 1 selects the plain f16 input (spec variant).
//   * Expert selection is read from device memory (ids / weights written by
//     the router's top-k). No host round trip; the launches are capturable.
//   * Finishing work runs in the last workgroup to arrive at a 128-row tile
//     (arrival counters, reset by that workgroup): Hadamard rotations in
//     registers within one wave, SwiGLU, the weighted combine over experts,
//     the output rotation once on the combined sum, and the gated shared
//     expert.
//   * Experts shared across a batch (MTP verify): the first slot in the batch
//     that selected an expert owns it. Its workgroups load the expert's codes
//     once and apply them to every token that selected it, one token at a
//     time through the same LDS buffer; the other slots' workgroups exit at
//     once. Arrival counts are unchanged, and each token's arithmetic is the
//     same as without sharing, so results are identical. Separate kernels
//     (GateUpShareKernel, DownShareKernel), used for batches of 2..32 tokens,
//     so the single-token and prefill kernels keep their register budget.
//   * Prefill chunks (more than 32 tokens): the slots are first grouped by
//     expert on the device (i4r_group_slots); GateUpGroupKernel and
//     DownGroupKernel give each expert its workgroups, which load its codes
//     once and run through all its slots with the same per-slot arithmetic.
//
// Rotation contract (unchanged from i4r_kernels.hip.cpp / the RFC):
//   xr   = Hn(su_experts * x)                  (fused into gate+up's staging)
//   g    = sv_gate * Hn(Mg xr),  u = sv_up * Hn(Mu xr)
//   d_in = Hn(su_down * silu(g) * u)
//   y_e  = sv_down * Hn(Md d_in)
//   out  = sum_e w_e y_e + sigmoid(gate) * shared
//        = sv_down * Hn(sum_e w_e Md d_in_e) + sigmoid(gate) * shared

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/core/quant/i4r_fused.hpp"

namespace gufo {
namespace quant {
namespace fused {

constexpr int kRowsPerWg = 64;
constexpr int kLanesPerRow = 4;
constexpr int kThreads = kRowsPerWg * kLanesPerRow;  // 256 = 8 waves of 32
constexpr int kWave = 32;
constexpr float kInvSqrt128 = 0.08838834764831845f;


typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef std::uint32_t u32x4 __attribute__((ext_vector_type(4)));
using GateUpArgs = ::I4rGateUpArgs;
using DownArgs = ::I4rDownArgs;

__device__ __forceinline__ float Dot2(half2_t a, half2_t b, float c) {
  return __builtin_amdgcn_fdot2(a, b, c, false);
}

__device__ __forceinline__ half2_t AsHalf2(std::uint32_t bits) {
  return __builtin_bit_cast(half2_t, bits);
}

// LDS position of input value v inside its group of four: the code word
// yields pairs (byte0, byte2) and (byte1, byte3), so the input is stored
// as [x0, x2, x1, x3] per group to make each pair one aligned half2.
__device__ __forceinline__ int Permute4(int v) {
  const int i = v & 3;
  return (v & ~3) | ((i & 1) << 1) | (i >> 1);
}

// 128-point orthonormal Walsh-Hadamard transform on one wave of 32 lanes.
// Lane l holds v[j] = tile[l + 32 j], j = 0..3. Strides 1..16 cross lanes,
// strides 32 and 64 stay inside the lane. Natural (Sylvester) order in and
// out; the stages commute, so the order matches i4r_fwht128.
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
  float a = v[0] + v[1], b = v[0] - v[1];
  float c = v[2] + v[3], d = v[2] - v[3];
  v[0] = a + c;
  v[2] = a - c;
  v[1] = b + d;
  v[3] = b - d;
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] *= kInvSqrt128;
}

__device__ __forceinline__ float Silu(float x) {
  return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float Sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// Stores one input value into the hi/lo LDS arrays at its permuted slot.
template <bool kSplitInput>
__device__ __forceinline__ void StageValue(_Float16* hi, _Float16* lo, int v,
                                           float x) {
  const int p = Permute4(v);
  const _Float16 h = static_cast<_Float16>(x);
  hi[p] = h;
  if constexpr (kSplitInput) lo[p] = static_cast<_Float16>(x - static_cast<float>(h));
}

// Dot of one 16-byte code quarter (32 codes) against its input. q is the
// quarter (0..3); the low nibbles are values 16q..16q+15 of the block, the
// high nibbles 64+16q..64+16q+15. xb points at the block's 128 permuted
// inputs (hi), xl at the lo half (when split).
template <bool kSplitInput>
__device__ __forceinline__ float DotQuarter(u32x4 codes, const _Float16* xh,
                                            const _Float16* xl, int q) {
  const std::uint32_t w[4] = {codes[0], codes[1], codes[2], codes[3]};
  const half2_t bias = AsHalf2(0x64086408u);  // (1032, 1032)
  const half2_t* lo_h = reinterpret_cast<const half2_t*>(xh + 16 * q);
  const half2_t* hi_h = reinterpret_cast<const half2_t*>(xh + 64 + 16 * q);
  const half2_t* lo_l = reinterpret_cast<const half2_t*>(xl + 16 * q);
  const half2_t* hi_l = reinterpret_cast<const half2_t*>(xl + 64 + 16 * q);
  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    // Word k holds bytes 4k..4k+3 of the quarter: values 4k..4k+3 (low
    // nibbles) and 64+4k.. (high nibbles), relative to the quarter base.
    const half2_t c02 = AsHalf2(((w[k] >> 0) & 0x000F000Fu) | 0x64006400u) - bias;
    const half2_t c13 = AsHalf2(((w[k] >> 8) & 0x000F000Fu) | 0x64006400u) - bias;
    const half2_t h02 = AsHalf2(((w[k] >> 4) & 0x000F000Fu) | 0x64006400u) - bias;
    const half2_t h13 = AsHalf2(((w[k] >> 12) & 0x000F000Fu) | 0x64006400u) - bias;
    acc = Dot2(c02, lo_h[2 * k], acc);
    acc = Dot2(c13, lo_h[2 * k + 1], acc);
    acc = Dot2(h02, hi_h[2 * k], acc);
    acc = Dot2(h13, hi_h[2 * k + 1], acc);
    if constexpr (kSplitInput) {
      acc = Dot2(c02, lo_l[2 * k], acc);
      acc = Dot2(c13, lo_l[2 * k + 1], acc);
      acc = Dot2(h02, hi_l[2 * k], acc);
      acc = Dot2(h13, hi_l[2 * k + 1], acc);
    }
  }
  return acc;
}

// Arrival: every lane has stored its partials. Returns true (uniform over
// the workgroup) in the last workgroup to arrive; that workgroup resets the
// counter for the next use and may then read every partial of the tile.
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

// Coherent read of a partial written by another workgroup (bypasses the
// non-coherent per-CU cache).
__device__ __forceinline__ float LoadPartial(const float* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// Most tokens a batch can share an expert across (one member per token).
constexpr int kMaxMembers = 32;
constexpr int kMaxScanSlots = 256;

// Whether a batch uses the expert-sharing kernels: verify-sized batches only.
// Single tokens and prefill chunks keep the plain kernels.
__host__ __device__ inline bool ShareExperts(int n_tokens, int n_slots) {
  return n_tokens > 1 && n_tokens <= kMaxMembers && n_slots <= kMaxScanSlots;
}

// Finds the slots of the batch that selected this slot's expert. Returns the
// member count (uniform over the workgroup): 0 when an earlier slot owns the
// expert (this workgroup has nothing to do), else the members in slot order,
// starting with this slot. Within a token the selected experts are distinct,
// so the members are in distinct tokens. Without sharing (one token, or a
// batch too large for the scan) every slot owns itself.
__device__ __forceinline__ int ExpertMembers(const std::int32_t* ids, int slot, int n_slots,
                                             int n_tokens, int* members) {
  __shared__ std::int32_t sid[kMaxScanSlots];
  __shared__ int count;
  if (!ShareExperts(n_tokens, n_slots)) {
    if (threadIdx.x == 0) members[0] = slot;
    __syncthreads();
    return 1;
  }
  for (int j = threadIdx.x; j < n_slots; j += blockDim.x) sid[j] = ids[j];
  __syncthreads();
  if (threadIdx.x == 0) {
    const std::int32_t e = sid[slot];
    int n = 0;
    for (int j = 0; j < n_slots; ++j) {
      if (sid[j] != e) continue;
      if (j < slot) { n = 0; break; }
      members[n++] = j;
    }
    count = n;
  }
  __syncthreads();
  return count;
}

// ---------------------------------------------------------------------------
// Shared-expert down (addendum 1), as the second workgroup range of the
// gate+up launch: Q8_0 [hidden x shared_ff], repacked into int8 codes then
// f16 scales per 32-value block. Workgroup = one 128-row output tile x one
// 128-value input slice; two lanes per row, 64 codes (4 x 16 B) each.
// ---------------------------------------------------------------------------
template <bool kSplitInput>
__device__ __forceinline__ void SharedDownRange(const GateUpArgs& a, int w,
                                                _Float16* xh, _Float16* xl) {
  const int tiles = a.hidden / 128;
  const int slices = a.shared_ff / 128;
  const int token = w / (tiles * slices);
  const int r = w % (tiles * slices);
  const int tile = r / slices;
  const int slice = r % slices;
  const int tid = threadIdx.x;
  const int row = tile * 128 + (tid >> 1);
  const int hq = tid & 1;
  const std::size_t rows = static_cast<std::size_t>(a.hidden);
  const std::size_t row_bytes = static_cast<std::size_t>(a.shared_ff);
  const std::uint8_t* codes = a.shared_down + row * row_bytes + slice * 128 + hq * 64;
  u32x4 c[4];
#pragma unroll
  for (int k = 0; k < 4; ++k)
    c[k] = __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(codes + 16 * k));
  const __half* scales = reinterpret_cast<const __half*>(a.shared_down + rows * row_bytes) +
                         row * (row_bytes / 32) + slice * 4 + hq * 2;
  const float* in = a.shared_in + static_cast<std::size_t>(token) * a.shared_ff + slice * 128;
  if (tid < 128) StageValue<kSplitInput>(xh, xl, tid, in[tid]);
  __syncthreads();
  const half2_t bias = AsHalf2(0x64806480u);  // (1152, 1152)
  const half2_t* ph = reinterpret_cast<const half2_t*>(xh + hq * 64);
  const half2_t* pl = reinterpret_cast<const half2_t*>(xl + hq * 64);
  float acc = 0.0f;
#pragma unroll
  for (int blk = 0; blk < 2; ++blk) {
    float bsum = 0.0f;
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      const u32x4 v = c[blk * 2 + kk];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const std::uint32_t wv = v[j] ^ 0x80808080u;  // int8 -> offset binary
        const half2_t c02 = AsHalf2((wv & 0x00FF00FFu) | 0x64006400u) - bias;
        const half2_t c13 = AsHalf2(((wv >> 8) & 0x00FF00FFu) | 0x64006400u) - bias;
        const int pidx = (blk * 32 + kk * 16 + 4 * j) / 2;
        bsum = Dot2(c02, ph[pidx], bsum);
        bsum = Dot2(c13, ph[pidx + 1], bsum);
        if constexpr (kSplitInput) {
          bsum = Dot2(c02, pl[pidx], bsum);
          bsum = Dot2(c13, pl[pidx + 1], bsum);
        }
      }
    }
    acc = fmaf(bsum, __half2float(scales[blk]), acc);
  }
  acc += __shfl_xor(acc, 1, kWave);
  float* part = a.shared_part + (static_cast<std::size_t>(token) * slices + slice) * a.hidden;
  if (hq == 0) part[row] = acc;
  if (!LastArrival(a.shared_counters + token * tiles + tile, static_cast<unsigned>(slices)))
    return;
  if (tid >= kWave) return;
  const int lane = tid;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = tile * 128 + lane + 32 * j;
    float sum = 0.0f;
    for (int sl = 0; sl < slices; ++sl)
      sum += LoadPartial(a.shared_part + (static_cast<std::size_t>(token) * slices + sl) * a.hidden + i);
    a.shared_out[static_cast<std::size_t>(token) * a.hidden + i] = sum;
  }
}

// ---------------------------------------------------------------------------
// Gate + up for every selected (token, slot), with the input rotation fused
// into the LDS staging and SwiGLU + down-input rotation in the finisher.
//
// grid.x = n_slots * (2 * ff / 64). Within a slot, workgroup r covers tile
// r / 4 (128 rows of ff); r % 4 picks gate/up (bit 1) and the 64-row half
// (bit 0), so the four arrivals of a tile are adjacent.
// ---------------------------------------------------------------------------
template <int NB, bool kSplitInput>
__global__ __launch_bounds__(kThreads) void GateUpKernel(GateUpArgs a) {
  constexpr int kIn = NB * 128;
  __shared__ __attribute__((aligned(16))) _Float16 xh[kIn];
  __shared__ __attribute__((aligned(16))) _Float16 xl[kSplitInput ? kIn : 8];

  const int tiles = a.ff / 128;
  const int per_slot = tiles * 4;
  const int routed_wgs = per_slot * a.n_tokens * a.used;
  if (static_cast<int>(blockIdx.x) >= routed_wgs) {
    SharedDownRange<kSplitInput>(a, blockIdx.x - routed_wgs, xh, xl);
    return;
  }
  const int slot = blockIdx.x / per_slot;
  const int r = blockIdx.x % per_slot;
  const int tile = r >> 2;
  const int which = (r >> 1) & 1;  // 0 gate, 1 up
  const int half = r & 1;
  const int token = slot / a.used;
  const int tid = threadIdx.x;
  const int lane_row = tid >> 2;  // 0..63
  const int q = tid & 3;
  const int row = tile * 128 + half * 64 + lane_row;

  const int expert = a.ids[slot];
  const std::uint8_t* base = which == 0 ? a.gate : a.up;
  const std::size_t row_codes = static_cast<std::size_t>(NB) * 64;
  const std::size_t exp_rows = static_cast<std::size_t>(a.ff);
  const std::uint8_t* codes_row =
      base + (static_cast<std::size_t>(expert) * exp_rows + row) * row_codes;
  const __half* scales_row =
      reinterpret_cast<const __half*>(base + a.n_experts * exp_rows * row_codes) +
      (static_cast<std::size_t>(expert) * exp_rows + row) * NB;

  // Issue this lane's code loads first; the staging below overlaps them.
  u32x4 c[NB];
#pragma unroll
  for (int b = 0; b < NB; ++b)
    c[b] = __builtin_nontemporal_load(
        reinterpret_cast<const u32x4*>(codes_row + b * 64 + q * 16));

  // Stage xr = Hn(su * x[token]) into LDS: wave w rotates blocks w, w+8, ...
  {
    const int wave = tid / kWave;
    const int lane = tid & (kWave - 1);
    const float* x = a.x + static_cast<std::size_t>(token) * kIn;
    for (int b = wave; b < NB; b += kThreads / kWave) {
      float v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int i = b * 128 + lane + 32 * j;
        v[j] = x[i] * a.su_experts[i];
      }
      Fwht128Wave(v);
#pragma unroll
      for (int j = 0; j < 4; ++j) StageValue<kSplitInput>(xh, xl, b * 128 + lane + 32 * j, v[j]);
    }
  }
  __syncthreads();

  float s[NB];
#pragma unroll
  for (int b = 0; b < NB; ++b) s[b] = __half2float(scales_row[b]);
  float acc = 0.0f;
#pragma unroll
  for (int b = 0; b < NB; ++b)
    acc = fmaf(DotQuarter<kSplitInput>(c[b], xh + b * 128, xl + b * 128, q), s[b], acc);
  acc += __shfl_xor(acc, 1, kWave);
  acc += __shfl_xor(acc, 2, kWave);

  float* part = (which == 0 ? a.gate_part : a.up_part) +
                static_cast<std::size_t>(slot) * a.ff;
  if (q == 0) part[row] = acc;

  if (!LastArrival(a.counters + slot * tiles + tile, 4u)) return;
  if (tid >= kWave) return;

  // Finisher (one wave): g, u rotated out of the matmul basis, SwiGLU,
  // then rotated into down's input basis. Written in place over the gate
  // partials of this tile (only this wave touches them now).
  const int lane = tid;
  float* gp = a.gate_part + static_cast<std::size_t>(slot) * a.ff + tile * 128;
  const float* up = a.up_part + static_cast<std::size_t>(slot) * a.ff + tile * 128;
  float g[4], u[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    g[j] = LoadPartial(gp + lane + 32 * j);
    u[j] = LoadPartial(up + lane + 32 * j);
  }
  Fwht128Wave(g);
  Fwht128Wave(u);
  float h[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = tile * 128 + lane + 32 * j;
    const float gv = g[j] * a.sv_gate[i];
    const float uv = u[j] * a.sv_up[i];
    h[j] = Silu(gv) * uv * a.su_down[i];
  }
  Fwht128Wave(h);
#pragma unroll
  for (int j = 0; j < 4; ++j) gp[lane + 32 * j] = h[j];
}

// ---------------------------------------------------------------------------
// Gate + up for a verify batch with experts shared across tokens (see the
// file comment); same layout and finisher as GateUpKernel. A separate kernel
// so the single-token and prefill path keeps its register budget.
// Gate + up for every selected (token, slot), with the input rotation fused
// into the LDS staging and SwiGLU + down-input rotation in the finisher.
//
// grid.x = n_slots * (2 * ff / 64). Within a slot, workgroup r covers tile
// r / 4 (128 rows of ff); r % 4 picks gate/up (bit 1) and the 64-row half
// (bit 0), so the four arrivals of a tile are adjacent.
// ---------------------------------------------------------------------------
// Gate + up of one expert's 64-row half tile for each member slot (slots
// that selected `expert`), one slot at a time through the same LDS buffer;
// the last of the tile's four workgroups to arrive at `counter` finishes the
// tile of every member. Shared by the verify (share) and prefill (grouped)
// kernels.
template <int NB, bool kSplitInput>
__device__ __forceinline__ void GateUpMembers(const GateUpArgs& a, int expert, int tile,
                                              int which, int half, const int* members,
                                              int n_mem, unsigned* counter, _Float16* xh,
                                              _Float16* xl) {
  constexpr int kIn = NB * 128;
  const int tid = threadIdx.x;
  const int lane_row = tid >> 2;  // 0..63
  const int q = tid & 3;
  const int row = tile * 128 + half * 64 + lane_row;
  const std::uint8_t* base = which == 0 ? a.gate : a.up;
  const std::size_t row_codes = static_cast<std::size_t>(NB) * 64;
  const std::size_t exp_rows = static_cast<std::size_t>(a.ff);
  const std::uint8_t* codes_row =
      base + (static_cast<std::size_t>(expert) * exp_rows + row) * row_codes;
  const __half* scales_row =
      reinterpret_cast<const __half*>(base + a.n_experts * exp_rows * row_codes) +
      (static_cast<std::size_t>(expert) * exp_rows + row) * NB;

  // Issue this lane's code loads first; the staging below overlaps them.
  u32x4 c[NB];
#pragma unroll
  for (int b = 0; b < NB; ++b)
    c[b] = __builtin_nontemporal_load(
        reinterpret_cast<const u32x4*>(codes_row + b * 64 + q * 16));
#pragma unroll 1
  for (int m = 0; m < n_mem; ++m) {
    const int ms = members[m];
    const int mt = ms / a.used;
    if (m > 0) __syncthreads();  // the previous token's dot is done with LDS
    // Keep the code unpacking inside the loop: unpacked once for all tokens
    // it would need far more registers than the packed codes.
#pragma unroll
    for (int b = 0; b < NB; ++b) asm volatile("" : "+v"(c[b]));
    // Stage xr = Hn(su * x[token]) into LDS: wave w rotates blocks w, w+8, ...
    {
      const int wave = tid / kWave;
      const int lane = tid & (kWave - 1);
      const float* x = a.x + static_cast<std::size_t>(mt) * kIn;
      for (int b = wave; b < NB; b += kThreads / kWave) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int i = b * 128 + lane + 32 * j;
          v[j] = x[i] * a.su_experts[i];
        }
        Fwht128Wave(v);
#pragma unroll
        for (int j = 0; j < 4; ++j) StageValue<kSplitInput>(xh, xl, b * 128 + lane + 32 * j, v[j]);
      }
    }
    __syncthreads();

    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < NB; ++b)
      acc = fmaf(DotQuarter<kSplitInput>(c[b], xh + b * 128, xl + b * 128, q),
                 __half2float(scales_row[b]), acc);
    acc += __shfl_xor(acc, 1, kWave);
    acc += __shfl_xor(acc, 2, kWave);

    float* part = (which == 0 ? a.gate_part : a.up_part) +
                  static_cast<std::size_t>(ms) * a.ff;
    if (q == 0) part[row] = acc;
  }

  // The owner slot's four workgroups arrive for every member: they all hold
  // the same member list, so the last of them finishes the tile of each.
  if (!LastArrival(counter, 4u)) return;
  if (tid >= kWave) return;

  // Finisher (one wave), per member slot: g, u rotated out of the matmul
  // basis, SwiGLU, then rotated into down's input basis. Written in place
  // over the gate partials of this tile (only this wave touches them now).
  const int lane = tid;
#pragma unroll 1
  for (int m = 0; m < n_mem; ++m) {
    const int ms = members[m];
    float* gp = a.gate_part + static_cast<std::size_t>(ms) * a.ff + tile * 128;
    const float* up = a.up_part + static_cast<std::size_t>(ms) * a.ff + tile * 128;
    float g[4], u[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      g[j] = LoadPartial(gp + lane + 32 * j);
      u[j] = LoadPartial(up + lane + 32 * j);
    }
    Fwht128Wave(g);
    Fwht128Wave(u);
    float h[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int i = tile * 128 + lane + 32 * j;
      const float gv = g[j] * a.sv_gate[i];
      const float uv = u[j] * a.sv_up[i];
      h[j] = Silu(gv) * uv * a.su_down[i];
    }
    Fwht128Wave(h);
#pragma unroll
    for (int j = 0; j < 4; ++j) gp[lane + 32 * j] = h[j];
  }
}

template <int NB, bool kSplitInput>
__global__ __launch_bounds__(kThreads) void GateUpShareKernel(GateUpArgs a) {
  constexpr int kIn = NB * 128;
  __shared__ __attribute__((aligned(16))) _Float16 xh[kIn];
  __shared__ __attribute__((aligned(16))) _Float16 xl[kSplitInput ? kIn : 8];

  const int tiles = a.ff / 128;
  const int per_slot = tiles * 4;
  const int routed_wgs = per_slot * a.n_tokens * a.used;
  if (static_cast<int>(blockIdx.x) >= routed_wgs) {
    SharedDownRange<kSplitInput>(a, blockIdx.x - routed_wgs, xh, xl);
    return;
  }
  const int slot = blockIdx.x / per_slot;
  const int r = blockIdx.x % per_slot;
  const int tile = r >> 2;
  const int which = (r >> 1) & 1;  // 0 gate, 1 up
  const int half = r & 1;
  __shared__ int members[kMaxMembers];
  const int n_mem = ExpertMembers(a.ids, slot, a.n_tokens * a.used, a.n_tokens, members);
  if (n_mem == 0) return;
  GateUpMembers<NB, kSplitInput>(a, a.ids[slot], tile, which, half, members, n_mem,
                                 a.counters + slot * tiles + tile, xh, xl);
}


// Gate + up for a prefill chunk, slots grouped by expert (i4r_group_slots):
// grid.x = n_experts * (2 * ff / 64), then the shared-expert range. Expert e's
// workgroups load its codes once and run through all its slots; experts no
// slot selected exit at once.
template <int NB, bool kSplitInput>
__global__ __launch_bounds__(kThreads) void GateUpGroupKernel(GateUpArgs a) {
  constexpr int kIn = NB * 128;
  __shared__ __attribute__((aligned(16))) _Float16 xh[kIn];
  __shared__ __attribute__((aligned(16))) _Float16 xl[kSplitInput ? kIn : 8];
  const int tiles = a.ff / 128;
  const int per_expert = tiles * 4;
  const int routed_wgs = per_expert * static_cast<int>(a.n_experts);
  if (static_cast<int>(blockIdx.x) >= routed_wgs) {
    SharedDownRange<kSplitInput>(a, blockIdx.x - routed_wgs, xh, xl);
    return;
  }
  const int e = blockIdx.x / per_expert;
  const int r = blockIdx.x % per_expert;
  const int off = a.group_offsets[e];
  const int n_mem = a.group_offsets[e + 1] - off;
  if (n_mem == 0) return;
  GateUpMembers<NB, kSplitInput>(a, e, r >> 2, (r >> 1) & 1, r & 1, a.group_slots + off, n_mem,
                                 a.group_counters + e * tiles + (r >> 2), xh, xl);
}

// Combine for one token's 128-row output tile (one wave): weighted sum of
// the token's slots, the output rotation once on the sum, the gated shared
// expert.
__device__ __forceinline__ void Combine(const DownArgs& a, int token, int tile) {
  const int lane = threadIdx.x;
  float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int e = 0; e < a.used; ++e) {
    const int sl = token * a.used + e;
    const float w = a.weights[sl];
    float* p = a.down_part + static_cast<std::size_t>(sl) * a.hidden + tile * 128;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[j] = LoadPartial(p + lane + 32 * j);
      sum[j] = fmaf(w, v[j], sum[j]);
    }
    if (a.per_slot_out) {
      // Test seam: the unweighted, fully rotated output of this slot, in
      // the [slots x hidden] layout of the unfused path.
      Fwht128Wave(v);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int i = tile * 128 + lane + 32 * j;
        p[lane + 32 * j] = v[j] * a.sv_down[i];
      }
    }
  }
  Fwht128Wave(sum);
  const float gate = a.shared_out != nullptr
                         ? Sigmoid(a.shared_gate[static_cast<std::size_t>(token) *
                                                 a.shared_gate_stride])
                         : 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = tile * 128 + lane + 32 * j;
    float o = sum[j] * a.sv_down[i];
    if (a.shared_out != nullptr)
      o = fmaf(gate, a.shared_out[static_cast<std::size_t>(token) * a.hidden + i], o);
    a.out[static_cast<std::size_t>(token) * a.hidden + i] = o;
  }
}

// ---------------------------------------------------------------------------
// Down for every selected slot, then the combine over slots in the last
// workgroup to arrive at each 128-row output tile of a token.
//
// grid.x = n_tokens * (hidden / 128) * used * 2. Within a token, the 2*used
// workgroups of one output tile are adjacent.
// ---------------------------------------------------------------------------
template <int NB, bool kSplitInput>
__global__ __launch_bounds__(kThreads) void DownKernel(DownArgs a) {
  constexpr int kIn = NB * 128;
  __shared__ __attribute__((aligned(16))) _Float16 xh[kIn];
  __shared__ __attribute__((aligned(16))) _Float16 xl[kSplitInput ? kIn : 8];

  const int tiles = a.hidden / 128;
  const int per_tile = a.used * 2;
  const int per_token = tiles * per_tile;
  const int token = blockIdx.x / per_token;
  const int r = blockIdx.x % per_token;
  const int tile = r / per_tile;
  const int k = (r % per_tile) >> 1;
  const int half = r & 1;
  const int slot = token * a.used + k;
  const int tid = threadIdx.x;
  const int lane_row = tid >> 2;
  const int q = tid & 3;
  const int row = tile * 128 + half * 64 + lane_row;

  const int expert = a.ids[slot];
  const std::size_t row_codes = static_cast<std::size_t>(NB) * 64;
  const std::size_t exp_rows = static_cast<std::size_t>(a.hidden);
  const std::uint8_t* codes_row =
      a.down + (static_cast<std::size_t>(expert) * exp_rows + row) * row_codes;
  const __half* scales_row =
      reinterpret_cast<const __half*>(a.down + a.n_experts * exp_rows * row_codes) +
      (static_cast<std::size_t>(expert) * exp_rows + row) * NB;

  u32x4 c[NB];
#pragma unroll
  for (int b = 0; b < NB; ++b)
    c[b] = __builtin_nontemporal_load(
        reinterpret_cast<const u32x4*>(codes_row + b * 64 + q * 16));

  // Stage this slot's down input (already rotated by gate+up's finisher).
  const float* din = a.down_in + static_cast<std::size_t>(slot) * kIn;
  for (int i = tid; i < kIn; i += kThreads) StageValue<kSplitInput>(xh, xl, i, din[i]);
  __syncthreads();

  float s[NB];
#pragma unroll
  for (int b = 0; b < NB; ++b) s[b] = __half2float(scales_row[b]);
  float acc = 0.0f;
#pragma unroll
  for (int b = 0; b < NB; ++b)
    acc = fmaf(DotQuarter<kSplitInput>(c[b], xh + b * 128, xl + b * 128, q), s[b], acc);
  acc += __shfl_xor(acc, 1, kWave);
  acc += __shfl_xor(acc, 2, kWave);

  float* part = a.down_part + static_cast<std::size_t>(slot) * a.hidden;
  if (q == 0) part[row] = acc;

  if (!LastArrival(a.counters + token * tiles + tile, static_cast<unsigned>(per_tile)))
    return;
  if (tid >= kWave) return;

  const int lane = tid;
  float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int e = 0; e < a.used; ++e) {
    const int sl = token * a.used + e;
    const float w = a.weights[sl];
    float* p = a.down_part + static_cast<std::size_t>(sl) * a.hidden + tile * 128;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[j] = LoadPartial(p + lane + 32 * j);
      sum[j] = fmaf(w, v[j], sum[j]);
    }
    if (a.per_slot_out) {
      // Test seam: the unweighted, fully rotated output of this slot, in
      // the [slots x hidden] layout of the unfused path.
      Fwht128Wave(v);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int i = tile * 128 + lane + 32 * j;
        p[lane + 32 * j] = v[j] * a.sv_down[i];
      }
    }
  }
  Fwht128Wave(sum);
  const float gate = a.shared_out != nullptr
                         ? Sigmoid(a.shared_gate[static_cast<std::size_t>(token) *
                                                 a.shared_gate_stride])
                         : 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = tile * 128 + lane + 32 * j;
    float o = sum[j] * a.sv_down[i];
    if (a.shared_out != nullptr)
      o = fmaf(gate, a.shared_out[static_cast<std::size_t>(token) * a.hidden + i], o);
    a.out[static_cast<std::size_t>(token) * a.hidden + i] = o;
  }
}

// ---------------------------------------------------------------------------
// Down for a verify batch with experts shared across tokens; same layout and
// combine as DownKernel.
// Down for every selected slot, then the combine over slots in the last
// workgroup to arrive at each 128-row output tile of a token.
//
// grid.x = n_tokens * (hidden / 128) * used * 2. Within a token, the 2*used
// workgroups of one output tile are adjacent.
// ---------------------------------------------------------------------------
// Down of one expert's 64-row half tile for each member slot, then one
// arrival per member token's output tile; the last to arrive at a token's
// tile combines it. Shared by the verify (share) and prefill (grouped)
// kernels.
template <int NB, bool kSplitInput>
__device__ __forceinline__ void DownMembers(const DownArgs& a, int expert, int tile, int half,
                                            const int* members, int n_mem, _Float16* xh,
                                            _Float16* xl) {
  constexpr int kIn = NB * 128;
  const int tiles = a.hidden / 128;
  const int per_tile = a.used * 2;
  const int tid = threadIdx.x;
  const int lane_row = tid >> 2;
  const int q = tid & 3;
  const int row = tile * 128 + half * 64 + lane_row;
  const std::size_t row_codes = static_cast<std::size_t>(NB) * 64;
  const std::size_t exp_rows = static_cast<std::size_t>(a.hidden);
  const std::uint8_t* codes_row =
      a.down + (static_cast<std::size_t>(expert) * exp_rows + row) * row_codes;
  const __half* scales_row =
      reinterpret_cast<const __half*>(a.down + a.n_experts * exp_rows * row_codes) +
      (static_cast<std::size_t>(expert) * exp_rows + row) * NB;

  u32x4 c[NB];
#pragma unroll
  for (int b = 0; b < NB; ++b)
    c[b] = __builtin_nontemporal_load(
        reinterpret_cast<const u32x4*>(codes_row + b * 64 + q * 16));
#pragma unroll 1
  for (int m = 0; m < n_mem; ++m) {
    const int ms = members[m];
    if (m > 0) __syncthreads();  // the previous slot's dot is done with LDS
    // Keep the code unpacking inside the loop: unpacked once for all tokens
    // it would need far more registers than the packed codes.
#pragma unroll
    for (int b = 0; b < NB; ++b) asm volatile("" : "+v"(c[b]));
    // Stage this slot's down input (already rotated by gate+up's finisher).
    const float* din = a.down_in + static_cast<std::size_t>(ms) * kIn;
    for (int i = tid; i < kIn; i += kThreads) StageValue<kSplitInput>(xh, xl, i, din[i]);
    __syncthreads();

    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < NB; ++b)
      acc = fmaf(DotQuarter<kSplitInput>(c[b], xh + b * 128, xl + b * 128, q),
                 __half2float(scales_row[b]), acc);
    acc += __shfl_xor(acc, 1, kWave);
    acc += __shfl_xor(acc, 2, kWave);

    float* part = a.down_part + static_cast<std::size_t>(ms) * a.hidden;
    if (q == 0) part[row] = acc;
  }

  // One arrival per member token's tile (each slot still gets exactly two,
  // one per half); the last to arrive at a token's tile combines it.
#pragma unroll 1
  for (int m = 0; m < n_mem; ++m) {
    const int mt = members[m] / a.used;
    const bool last =
        LastArrival(a.counters + mt * tiles + tile, static_cast<unsigned>(per_tile));
    if (last && tid < kWave) Combine(a, mt, tile);
  }
}

template <int NB, bool kSplitInput>
__global__ __launch_bounds__(kThreads) void DownShareKernel(DownArgs a) {
  constexpr int kIn = NB * 128;
  __shared__ __attribute__((aligned(16))) _Float16 xh[kIn];
  __shared__ __attribute__((aligned(16))) _Float16 xl[kSplitInput ? kIn : 8];

  const int tiles = a.hidden / 128;
  const int per_tile = a.used * 2;
  const int per_token = tiles * per_tile;
  const int token = blockIdx.x / per_token;
  const int r = blockIdx.x % per_token;
  const int tile = r / per_tile;
  const int k = (r % per_tile) >> 1;
  const int half = r & 1;
  const int slot = token * a.used + k;
  const int n_tokens = static_cast<int>(gridDim.x) / per_token;
  __shared__ int members[kMaxMembers];
  const int n_mem = ExpertMembers(a.ids, slot, n_tokens * a.used, n_tokens, members);
  if (n_mem == 0) return;
  DownMembers<NB, kSplitInput>(a, a.ids[slot], tile, half, members, n_mem, xh, xl);
}

// Down for a prefill chunk, slots grouped by expert: grid.x = n_experts *
// (hidden / 128) * 2.
template <int NB, bool kSplitInput>
__global__ __launch_bounds__(kThreads) void DownGroupKernel(DownArgs a) {
  constexpr int kIn = NB * 128;
  __shared__ __attribute__((aligned(16))) _Float16 xh[kIn];
  __shared__ __attribute__((aligned(16))) _Float16 xl[kSplitInput ? kIn : 8];
  const int per_expert = (a.hidden / 128) * 2;
  const int e = blockIdx.x / per_expert;
  const int r = blockIdx.x % per_expert;
  const int off = a.group_offsets[e];
  const int n_mem = a.group_offsets[e + 1] - off;
  if (n_mem == 0) return;
  DownMembers<NB, kSplitInput>(a, e, r >> 1, r & 1, a.group_slots + off, n_mem, xh, xl);
}


// ---------------------------------------------------------------------------
// Slot grouping for prefill: offsets[e + 1] counts the slots that selected
// expert e, an exclusive scan turns counts into offsets, and a scatter writes
// each slot at its expert's cursor. The order within an expert depends on
// the atomics, but each slot's result does not depend on that order.
// ---------------------------------------------------------------------------
__global__ void GroupCountKernel(const std::int32_t* __restrict__ ids, int n_slots,
                                 std::int32_t* __restrict__ offsets) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_slots) atomicAdd(&offsets[ids[i] + 1], 1);
}

__global__ void GroupScanKernel(std::int32_t* __restrict__ offsets,
                                std::int32_t* __restrict__ cursor, int n_experts) {
  if (threadIdx.x != 0) return;
  std::int32_t run = 0;
  for (int e = 0; e < n_experts; ++e) {
    cursor[e] = run;
    run += offsets[e + 1];
    offsets[e + 1] = run;
  }
  offsets[0] = 0;
}

__global__ void GroupScatterKernel(const std::int32_t* __restrict__ ids, int n_slots,
                                   std::int32_t* __restrict__ cursor,
                                   std::int32_t* __restrict__ slots) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_slots) slots[atomicAdd(&cursor[ids[i]], 1)] = i;
}

// ---------------------------------------------------------------------------
// Load-time repack: interleaved 66-byte blocks -> codes region + scales
// region, for a whole [n_experts x rows] tensor. One thread per block.
// ---------------------------------------------------------------------------
__global__ void RepackKernel(const std::uint8_t* __restrict__ src,
                             std::uint8_t* __restrict__ dst,
                             std::size_t n_blocks) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_blocks) return;
  const std::uint8_t* b = src + i * 66;
  std::uint8_t* codes = dst + i * 64;
#pragma unroll
  for (int k = 0; k < 64; ++k) codes[k] = b[k];
  std::uint8_t* scales = dst + n_blocks * 64 + i * 2;
  scales[0] = b[64];
  scales[1] = b[65];
}

__global__ void RepackQ8Kernel(const std::uint8_t* __restrict__ src,
                               std::uint8_t* __restrict__ dst,
                               std::size_t n_blocks) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_blocks) return;
  const std::uint8_t* b = src + i * 34;
  std::uint8_t* codes = dst + i * 32;
#pragma unroll
  for (int k = 0; k < 32; ++k) codes[k] = b[2 + k];
  std::uint8_t* scales = dst + n_blocks * 32 + i * 2;
  scales[0] = b[0];
  scales[1] = b[1];
}

}  // namespace fused
}  // namespace quant
}  // namespace gufo

namespace {
using gufo::quant::fused::kThreads;

}  // namespace

extern "C" {

bool i4r_fused_gate_up(const I4rGateUpArgs* args, int n_slots, hipStream_t stream) {
  const I4rGateUpArgs& a = *args;
  if (a.ff % 128 != 0 || a.hidden % 128 != 0 || n_slots <= 0 || a.used <= 0 ||
      n_slots != a.n_tokens * a.used)
    return false;
  unsigned wgs = static_cast<unsigned>(n_slots) * (2u * a.ff / 64u);
  if (a.shared_down != nullptr) {
    if (a.shared_ff % 128 != 0 || a.shared_ff <= 0)
      return false;
    wgs += static_cast<unsigned>(a.n_tokens) * (a.hidden / 128u) * (a.shared_ff / 128u);
  }
  const bool group = a.group_offsets != nullptr;
  if (group) wgs = static_cast<unsigned>(a.n_experts) * (2u * a.ff / 64u) +
                   (wgs - static_cast<unsigned>(n_slots) * (2u * a.ff / 64u));
  const dim3 grid(wgs);
  const bool share = !group && gufo::quant::fused::ShareExperts(a.n_tokens, n_slots);
  switch (a.hidden / 128) {
#define GUFO_I4R_CASE(nb)                                                    \
  case nb:                                                                   \
    if (group && a.plain_f16)                                                \
      gufo::quant::fused::GateUpGroupKernel<nb, false><<<grid, kThreads, 0, stream>>>(a); \
    else if (group)                                                          \
      gufo::quant::fused::GateUpGroupKernel<nb, true><<<grid, kThreads, 0, stream>>>(a); \
    else if (share && a.plain_f16)                                           \
      gufo::quant::fused::GateUpShareKernel<nb, false><<<grid, kThreads, 0, stream>>>(a); \
    else if (share)                                                          \
      gufo::quant::fused::GateUpShareKernel<nb, true><<<grid, kThreads, 0, stream>>>(a); \
    else if (a.plain_f16)                                                    \
      gufo::quant::fused::GateUpKernel<nb, false><<<grid, kThreads, 0, stream>>>(a); \
    else                                                                     \
      gufo::quant::fused::GateUpKernel<nb, true><<<grid, kThreads, 0, stream>>>(a); \
    return true;
    GUFO_I4R_CASE(1)
    GUFO_I4R_CASE(2)
    GUFO_I4R_CASE(4)
    GUFO_I4R_CASE(5)
    GUFO_I4R_CASE(8)
    GUFO_I4R_CASE(16)
    GUFO_I4R_CASE(20)
    GUFO_I4R_CASE(32)
#undef GUFO_I4R_CASE
    default:
      return false;
  }
}

bool i4r_fused_down(const I4rDownArgs* args, int n_tokens, hipStream_t stream) {
  const I4rDownArgs& a = *args;
  if (a.ff % 128 != 0 || a.hidden % 128 != 0 || n_tokens <= 0) return false;
  const bool group = a.group_offsets != nullptr;
  const dim3 grid(group ? static_cast<unsigned>(a.n_experts) * (a.hidden / 128u) * 2u
                        : static_cast<unsigned>(n_tokens) * (a.hidden / 128u) * a.used * 2u);
  const bool share = !group && gufo::quant::fused::ShareExperts(n_tokens, n_tokens * a.used);
  switch (a.ff / 128) {
#define GUFO_I4R_CASE(nb)                                                    \
  case nb:                                                                   \
    if (group && a.plain_f16)                                                \
      gufo::quant::fused::DownGroupKernel<nb, false><<<grid, kThreads, 0, stream>>>(a); \
    else if (group)                                                          \
      gufo::quant::fused::DownGroupKernel<nb, true><<<grid, kThreads, 0, stream>>>(a); \
    else if (share && a.plain_f16)                                           \
      gufo::quant::fused::DownShareKernel<nb, false><<<grid, kThreads, 0, stream>>>(a); \
    else if (share)                                                          \
      gufo::quant::fused::DownShareKernel<nb, true><<<grid, kThreads, 0, stream>>>(a); \
    else if (a.plain_f16)                                                    \
      gufo::quant::fused::DownKernel<nb, false><<<grid, kThreads, 0, stream>>>(a); \
    else                                                                     \
      gufo::quant::fused::DownKernel<nb, true><<<grid, kThreads, 0, stream>>>(a); \
    return true;
    GUFO_I4R_CASE(1)
    GUFO_I4R_CASE(2)
    GUFO_I4R_CASE(4)
    GUFO_I4R_CASE(5)
    GUFO_I4R_CASE(8)
    GUFO_I4R_CASE(10)
    GUFO_I4R_CASE(16)
#undef GUFO_I4R_CASE
    default:
      return false;
  }
}

bool i4r_group_slots(const std::int32_t* ids, int n_slots, int n_experts,
                     std::int32_t* offsets, std::int32_t* cursor, std::int32_t* slots,
                     hipStream_t stream) {
  if (n_slots <= 0 || n_experts <= 0) return false;
  if (hipMemsetAsync(offsets, 0, (static_cast<std::size_t>(n_experts) + 1) * sizeof(std::int32_t),
                     stream) != hipSuccess)
    return false;
  const unsigned blocks = static_cast<unsigned>((n_slots + 255) / 256);
  gufo::quant::fused::GroupCountKernel<<<blocks, 256, 0, stream>>>(ids, n_slots, offsets);
  gufo::quant::fused::GroupScanKernel<<<1, 32, 0, stream>>>(offsets, cursor, n_experts);
  gufo::quant::fused::GroupScatterKernel<<<blocks, 256, 0, stream>>>(ids, n_slots, cursor, slots);
  return hipGetLastError() == hipSuccess;
}

bool i4r_repack_q8_split(const std::uint8_t* src, std::uint8_t* dst,
                         std::size_t n_blocks, hipStream_t stream) {
  if (n_blocks == 0) return true;
  const unsigned threads = 256;
  const unsigned blocks = static_cast<unsigned>((n_blocks + threads - 1) / threads);
  gufo::quant::fused::RepackQ8Kernel<<<blocks, threads, 0, stream>>>(src, dst, n_blocks);
  return hipGetLastError() == hipSuccess;
}

bool i4r_repack_split(const std::uint8_t* src, std::uint8_t* dst,
                      std::size_t n_blocks, hipStream_t stream) {
  if (n_blocks == 0) return true;
  const unsigned threads = 256;
  const unsigned blocks = static_cast<unsigned>((n_blocks + threads - 1) / threads);
  gufo::quant::fused::RepackKernel<<<blocks, threads, 0, stream>>>(src, dst, n_blocks);
  return hipGetLastError() == hipSuccess;
}

}  // extern "C"
