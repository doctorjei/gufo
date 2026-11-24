// Wide (prefill) support for weights read in place during decode: the 4-bit
// ht trellis store (GgmlType::kHT4) and q6g64 (GgmlType::kQ6G64).
//
// Prefill multiplies a whole chunk of tokens by each matrix, so instead of the
// decode kernels (at most 8 tokens a launch) a matrix is decoded once to f16
// and handed to a dense GEMM with f16 activations and f32 accumulation.
//
// ht: W = diag(sv) . Hn . M . Hn . diag(su) / 128 (public hgn-spec v1.1.0,
// STORAGE.md "Tiles", "The 128-point rotation"), where M holds the codebook
// values (exact in f16), the column-side Hn acts on each 128-value block of a
// row and the row-side Hn across the 128 rows of a band. So
//   y = sv * Hn_rows( M . x' ),   x' = Hn_cols(su * x) / sqrt(128),
// with the row-side transform also scaled by 1/sqrt(128). The decode below
// writes M; the two rotations act on the activations.
//
// q6g64: value = code * scale + bias per 64-value group, written as f16.

#ifndef GUFO_CORE_QUANT_HGN_PREFILL_HPP_
#define GUFO_CORE_QUANT_HGN_PREFILL_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

extern "C" {

/// M of a kHT4 matrix (rows x cols, both multiples of 128 and 16) as
/// row-major f16, out[r * cols + c].
bool ht_decode_f16(const std::uint8_t* codes, int rows, int cols, void* out,
                   hipStream_t stream);

/// x' = Hn(su * x) / sqrt(128) per 128-value block, rounded to f16:
/// x is [n_tokens][cols] f32, out [n_tokens][cols] f16.
bool ht_rotate_input_f16(const float* x, const float* su, void* out,
                         int n_tokens, int cols, hipStream_t stream);

/// y = sv * Hn(y) / sqrt(128) per 128-row band, in place: y is
/// [n_tokens][rows] f32.
bool ht_rotate_output(float* y, const float* sv, int n_tokens, int rows,
                      hipStream_t stream);

/// A kQ6G64 matrix (rows x cols, cols a multiple of 64) as row-major f16.
bool q6g64_decode_f16(const std::uint8_t* w, int rows, int cols, void* out,
                      hipStream_t stream);

}  // extern "C"

#endif  // GUFO_CORE_QUANT_HGN_PREFILL_HPP_
