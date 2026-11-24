#ifndef GUFO_CORE_QUANT_I4R_LAUNCH_HPP_
#define GUFO_CORE_QUANT_I4R_LAUNCH_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

// Host-callable launchers for the I4R128 rotated-expert kernels. Defined in
// i4r_kernels.hip.cpp; declared here so the HIP executor can call them from
// a plain C++ translation unit. Launch geometry lives with the definitions.

extern "C" {

// out[0..n_blocks*128) = Hn((pre ? in*pre : in)) [*(post ? post : 1)].
void i4r_launch_rotate(const float* in, float* out, const float* pre_scale,
                       const float* post_scale, int n_blocks,
                       hipStream_t stream);

// y[0..n_out) = M * xr for one expert (single token).
void i4r_launch_mmvq(const uint8_t* M, const float* xr, float* y, int n_out,
                     int n_blocks_in, hipStream_t stream);

// Y[t][0..n_out) = M * X[t] for one expert, `tokens` token rows.
void i4r_launch_mmq(const uint8_t* M, const float* X, float* Y, int n_out,
                    int n_in, int tokens, hipStream_t stream);

}  // extern "C"

#endif  // GUFO_CORE_QUANT_I4R_LAUNCH_HPP_
