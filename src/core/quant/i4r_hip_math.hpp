#ifndef GUFO_CORE_QUANT_I4R_HIP_MATH_HPP_
#define GUFO_CORE_QUANT_I4R_HIP_MATH_HPP_

// Pure math for the I4R128 GPU path, shared verbatim by the HIP kernels and the
// host-emulation test. Everything here is __device__ __host__ so the exact
// functions the device runs can be exercised on the CPU against the
// RotatedGemv oracle without a GPU. The kernels add only memory movement
// (shared memory, block reduce) around these.
//
// Layout/contract authority: docs/internal/design/i4r128/RFC.md.
//
// A block is 128 values along the input axis: 64 bytes of offset-binary codes
// in GGUF nibble order (low nibble = value b, high nibble = value 64+b) plus
// an f16 scale. The scale is passed here already widened to float; the kernel
// converts the f16 once per block.

#include <cstdint>

// When this header is used outside a HIP compilation unit (the host-emulation
// test, built with the plain C++ compiler), __device__/__host__ are not
// defined. Neutralise them so the pure math compiles as ordinary host code.
// Under hipcc these are already defined by the HIP headers and are untouched.
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif

namespace gufo {
namespace quant {

// Elements per rotation block / quantization group.
#define I4R_BLOCK 128
// 1 / sqrt(128), the orthonormal scaling of Hn.
#define I4R_INV_SQRT_BLOCK 0.08838834764831845f

// Decode one stored value v (0..127) of a block from its 64 code bytes and
// float scale. Offset binary: nibble - 8 in [-8, 7].
__device__ __host__ inline float i4r_decode_value(const uint8_t* codes,
                                                  float scale, int v) {
  // GGUF nibble order: value v lives in byte (v & 63); low nibble for v < 64,
  // high nibble for v >= 64. Cast to int before subtracting 8: (byte & 0xF)
  // promotes to unsigned and would wrap.
  const int nib = (codes[v & 63] >> (v < 64 ? 0 : 4)) & 0xF;
  return static_cast<float>(nib - 8) * scale;
}

// Unpack a whole 66-byte block's codes into 128 floats (the stored number M).
__device__ __host__ inline void i4r_unpack_block(const uint8_t* codes,
                                                 float scale, float* out) {
  for (int v = 0; v < I4R_BLOCK; ++v)
    out[v] = i4r_decode_value(codes, scale, v);
}

// In-place unnormalised Walsh-Hadamard transform of 128 floats in Sylvester
// (natural) ordering: multiplies by H128 with h(i,j) = (-1)^popcount(i&j).
// Same butterfly schedule as the CPU Fwht128, so no bit-reversal pass. The
// device kernel runs this on shared memory with the ds4-style per-thread
// ownership (see i4r_kernels.hip.cpp); the recurrence is identical.
__device__ __host__ inline void i4r_fwht128(float* v) {
  for (unsigned step = 1u; step < I4R_BLOCK; step <<= 1u) {
    const unsigned span = step << 1u;
    for (unsigned base = 0u; base < I4R_BLOCK; base += span) {
      for (unsigned i = 0u; i < step; ++i) {
        const float a = v[base + i];
        const float b = v[base + i + step];
        v[base + i] = a + b;
        v[base + i + step] = a - b;
      }
    }
  }
}

// Orthonormal Hn applied to one 128-element block, in place.
__device__ __host__ inline void i4r_rotate_block128(float* v) {
  i4r_fwht128(v);
  for (int i = 0; i < I4R_BLOCK; ++i)
    v[i] *= I4R_INV_SQRT_BLOCK;
}

// Dot of one decoded block against a 128-element slice of the rotated input.
// Accumulates in float (what the kernel does); the host-emulation oracle uses
// double and the two are compared with a tolerance.
__device__ __host__ inline float i4r_dot_block(const uint8_t* codes,
                                               float scale, const float* xr) {
  float acc = 0.0f;
  for (int v = 0; v < I4R_BLOCK; ++v) {
    acc += i4r_decode_value(codes, scale, v) * xr[v];
  }
  return acc;
}

}  // namespace quant
}  // namespace gufo

#endif  // GUFO_CORE_QUANT_I4R_HIP_MATH_HPP_
