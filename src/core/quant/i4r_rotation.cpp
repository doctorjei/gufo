#include "src/core/quant/i4r_rotation.hpp"

#include <cstddef>
#include <vector>

namespace gufo::quant {

void Fwht128(float* v) noexcept {
  // Sylvester butterfly. Stage `step` pairs indices differing in bit
  // log2(step), which is exactly the recursive H_{2n} = [[H_n, H_n], [H_n,
  // -H_n]] layout, so the result is ordered as h(i, j) = (-1)^popcount(i & j)
  // with no bit-reversal pass.
  for (std::size_t step = 1; step < kI4rBlock; step <<= 1) {
    const std::size_t span = step << 1;
    for (std::size_t base = 0; base < kI4rBlock; base += span) {
      for (std::size_t i = 0; i < step; ++i) {
        const float a = v[base + i];
        const float b = v[base + i + step];
        v[base + i] = a + b;
        v[base + i + step] = a - b;
      }
    }
  }
}

void RotateBlock128(float* v) noexcept {
  Fwht128(v);
  for (std::size_t i = 0; i < kI4rBlock; ++i)
    v[i] *= kI4rInvSqrtBlock;
}

void RotateBlocks(float* data, std::size_t count) noexcept {
  for (std::size_t base = 0; base + kI4rBlock <= count; base += kI4rBlock) {
    RotateBlock128(data + base);
  }
}

void RotatedGemv(const float* m, std::size_t n_out, std::size_t n_in,
                 const float* x, const float* su, const float* sv, float* y) {
  // Stages 1 and 2: scale by su, then rotate each input block with Hn.
  std::vector<float> xr(n_in);
  for (std::size_t i = 0; i < n_in; ++i)
    xr[i] = x[i] * su[i];
  RotateBlocks(xr.data(), n_in);

  // Stage 3: plain matmul against the decoded stored numbers M.
  for (std::size_t o = 0; o < n_out; ++o) {
    const float* row = m + o * n_in;
    double acc = 0.0;
    for (std::size_t i = 0; i < n_in; ++i)
      acc += static_cast<double>(row[i]) * xr[i];
    y[o] = static_cast<float>(acc);
  }

  // Stages 4 and 5: rotate each output block with Hn, then scale by sv.
  RotateBlocks(y, n_out);
  for (std::size_t o = 0; o < n_out; ++o)
    y[o] *= sv[o];
}

}  // namespace gufo::quant
