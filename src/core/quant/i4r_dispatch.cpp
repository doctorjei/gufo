#include "src/core/quant/i4r_dispatch.hpp"

#include <cstddef>
#include <cstdint>

#include "src/core/quant/i4r_rotation.hpp"

namespace gufo::quant {

void RotateLayerInput(RotatedOps& ops, const float* x, const RotatedScales& s,
                      std::size_t hidden, std::size_t n_tokens, float* xr) {
  // Each token row is block-aligned (hidden % kI4rBlock == 0), so the input
  // scale su_experts[hidden] lines up with every row independently: rotating
  // row by row is the same math as one fused pass over a tiled scale, just
  // without the tiling. The rotated row is written once and reused by every
  // expert of that token.
  for (std::size_t t = 0; t < n_tokens; ++t) {
    ops.Rotate(x + t * hidden, xr + t * hidden, s.su_experts, nullptr, hidden);
  }
}

void RotatedGateUp(RotatedOps& ops, const std::uint8_t* gate,
                   const std::uint8_t* up, const RotatedScales& s,
                   const float* xr, const std::int32_t* ids, std::size_t hidden,
                   std::size_t ff, std::size_t n_slots, std::uint32_t used,
                   bool prefill, float* gate_out, float* up_out,
                   float* scratch_gate, float* scratch_up) {
  const std::size_t gate_stride = I4rExpertStride(ff, hidden);
  const std::size_t up_stride = I4rExpertStride(ff, hidden);
  const std::size_t blocks_in = hidden / kI4rBlock;
  for (std::size_t sl = 0; sl < n_slots; ++sl) {
    const std::size_t t = sl / used;
    const std::int32_t e = ids[sl];
    const float* xrow = xr + t * hidden;
    const std::uint8_t* mg = gate + static_cast<std::size_t>(e) * gate_stride;
    const std::uint8_t* mu = up + static_cast<std::size_t>(e) * up_stride;
    float* g = gate_out + sl * ff;
    float* u = up_out + sl * ff;
    // The matmul lands in a temp and stage 2 rotates temp -> output. Doing
    // it in place would break the kernel's __restrict__ contract (see the
    // header), and the CPU stand-in's temporary is not evidence otherwise.
    if (prefill) {
      ops.Gemm(mg, xrow, scratch_gate, ff, hidden, 1);
      ops.Gemm(mu, xrow, scratch_up, ff, hidden, 1);
    } else {
      ops.Matvec(mg, xrow, scratch_gate, ff, blocks_in);
      ops.Matvec(mu, xrow, scratch_up, ff, blocks_in);
    }
    // Stage 2: rotate each output with its own sv. The gate/up outputs are
    // separate buffers, so the "fused 1280-wide" rotation is two launches
    // here; the blocks never cross the boundary either way.
    ops.Rotate(scratch_gate, g, nullptr, s.sv_gate, ff);
    ops.Rotate(scratch_up, u, nullptr, s.sv_up, ff);
  }
}

void RotatedDown(RotatedOps& ops, const std::uint8_t* down,
                 const RotatedScales& s, const float* h,
                 const std::int32_t* ids, std::size_t hidden, std::size_t ff,
                 std::size_t n_slots, std::uint32_t used, bool prefill,
                 float* down_out, float* scratch_in, float* scratch_out) {
  const std::size_t down_stride = I4rExpertStride(hidden, ff);
  const std::size_t blocks_in = ff / kI4rBlock;
  (void)used;  // The down side keys only on the expert id per slot.
  for (std::size_t sl = 0; sl < n_slots; ++sl) {
    const std::int32_t e = ids[sl];
    const std::uint8_t* md = down + static_cast<std::size_t>(e) * down_stride;
    const float* hin = h + sl * ff;
    // Stage 3: rotate the down input with su_down into scratch_in.
    ops.Rotate(hin, scratch_in, s.su_down, nullptr, ff);
    // The matmul lands in scratch_out; stage 4 rotates it into place.
    if (prefill) {
      ops.Gemm(md, scratch_in, scratch_out, hidden, ff, 1);
    } else {
      ops.Matvec(md, scratch_in, scratch_out, hidden, blocks_in);
    }
    // Stage 4: rotate the down output with sv_down.
    ops.Rotate(scratch_out, down_out + sl * hidden, nullptr, s.sv_down, hidden);
  }
}

}  // namespace gufo::quant
