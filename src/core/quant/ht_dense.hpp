// Matrix-vector products for 4-bit ht weights (GgmlType::kHT4), read in place.
//
// The technique is Peonist's halogen-flash engine's, reimplemented clean-room:
// built only from a functional spec of halogen's ht decode kernels (the "ht
// dense decode spec") and the public hgn-spec v1.1.0 format. See
// docs/internal/design/hgn/PROVENANCE.md; design notes in HT-DENSE.md there.
//
// y[t] = W x[t] for up to 8 tokens per launch, W = diag(sv) Hn M Hn diag(su),
// with M the ht stored numbers. One launch per call: input rotation fused
// into the shared-memory staging, the trellis decoded in registers, the
// cross-workgroup sum finished by the last workgroup of each 128-row band,
// which also applies the output rotation and sv.

#ifndef GUFO_CORE_QUANT_HT_DENSE_HPP_
#define GUFO_CORE_QUANT_HT_DENSE_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

extern "C" {

struct HtDenseArgs {
  const std::uint8_t* codes;  ///< kHT4 payload, rows x cols
  const float* su;            ///< [cols] input-side vector
  const float* sv;            ///< [rows] output-side vector
  const float* x;             ///< [n_tokens][x_stride] input rows
  float* out;                 ///< [n_tokens][out_stride] output rows
  float* partials;            ///< scratch, HtDensePartialFloats() floats
  unsigned* counters;         ///< [rows/128] zeroed; each use resets them
  int rows;
  int cols;
  int n_tokens;  ///< 1..8
  int x_stride;
  int out_stride;
};

/// Matrices that read the same input can share one launch (up to this many).
#define HT_DENSE_MAX_GROUP 4

struct HtMatrix {
  const std::uint8_t* codes;  ///< kHT4 payload, rows x cols
  const float* su;            ///< [cols]
  const float* sv;            ///< [rows]
  float* out;                 ///< [n_tokens][out_stride]
  int rows;
  int out_stride;
};

struct HtGroupArgs {
  HtMatrix m[HT_DENSE_MAX_GROUP];
  int count;  ///< 1..HT_DENSE_MAX_GROUP
  const float* x;
  int cols;      ///< shared input width
  int n_tokens;  ///< 1..8
  int x_stride;
  float* partials;               ///< scratch shared by the group
  unsigned* counters;            ///< zeroed; reset by each use
  std::size_t partial_capacity;  ///< floats available at partials
  std::size_t counter_capacity;  ///< counters available
};

/// One launch for every matrix of the group (y_i = W_i x). False on a bad
/// shape or too little scratch.
bool ht_dense_group(const HtGroupArgs* args, hipStream_t stream);

/// Scratch a group of matrices with these row counts needs.
void ht_dense_group_scratch(const int* rows, int count, int cols,
                            std::size_t* partials, std::size_t* counters);

/// Input split (workgroups per 128-row band) the launcher uses.
/// The split plan is one per matrix shape, the same at every token count
/// (verify batches give bit for bit the single-token results). Ask for at
/// least `blocks` 128-column blocks per workgroup (0 = about 640 workgroups;
/// the default is 7).
void ht_dense_set_min_blocks(int blocks);

int ht_dense_splits(int rows, int cols);

/// Scratch floats one launch needs (0 when the input is not split).
std::size_t ht_dense_partial_floats(int rows, int cols);

/// Launches y = W x for args->n_tokens (1..8) tokens. Returns false on an
/// unsupported shape.
bool ht_dense_gemv(const HtDenseArgs* args, hipStream_t stream);

}  // extern "C"

#endif  // GUFO_CORE_QUANT_HT_DENSE_HPP_
