// Matrix-vector products for q6g64 weights (GgmlType::kQ6G64), read in place.
//
// Format (public hgn-spec v1.1.0, STORAGE.md "q6g64"): each row is
// pad(13 K / 16, 16) bytes: K/2 bytes of low nibbles (byte j: value 2j low,
// 2j+1 high), K/4 bytes of high bit pairs (value k at bits 2(k mod 4) of byte
// K/2 + k/4), then K/64 records (f16 scale, f16 bias). value = code * scale +
// bias, code = low | high << 4 in 0..63.
//
// y[t][r] = sum_g scale_g * (sum_k code_k x_k) + bias_g * (sum_k x_k), in f32.
//
// One kernel for every token count (1..8); each token's arithmetic is the
// same as for a single token, so verify batches give bit for bit the
// single-token results. The partial-sum scratch is unused (kept in the
// interface; q6g64_dense_scratch reports 0).

#ifndef GUFO_CORE_QUANT_Q6G64_DENSE_HPP_
#define GUFO_CORE_QUANT_Q6G64_DENSE_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

extern "C" {

struct Q6g64DenseArgs {
  const std::uint8_t* w;  ///< kQ6G64 payload, rows x cols
  const float* x;         ///< [n_tokens][x_stride]
  float* out;             ///< [n_tokens][out_stride]
  int rows;
  int cols;      ///< multiple of 64
  int n_tokens;  ///< 1..8
  int x_stride;
  int out_stride;
  float* partials;               ///< scratch when the input is split
  unsigned* counters;            ///< zeroed; reset by each use
  std::size_t partial_capacity;  ///< floats at partials
  std::size_t counter_capacity;  ///< counters available
};

/// Scratch one launch of this shape needs (0 floats when not split).
void q6g64_dense_scratch(int rows, int cols, std::size_t* partials,
                         std::size_t* counters);

/// Launches y = W x for args->n_tokens (1..8) tokens. False on a bad shape
/// or too little scratch.
bool q6g64_dense_gemv(const Q6g64DenseArgs* args, hipStream_t stream);

}  // extern "C"

#endif  // GUFO_CORE_QUANT_Q6G64_DENSE_HPP_
