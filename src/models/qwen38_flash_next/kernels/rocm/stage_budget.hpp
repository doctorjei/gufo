#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_STAGE_BUDGET_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_STAGE_BUDGET_HPP_

#include <cstddef>

#include "src/core/quant/i4r_dispatch.hpp"

// The linear, attention and routed-expert stages run sequentially and share
// one scratch block sized by the maximum over them. That sharing makes the
// reservation a single number with no local connection to the slices carved
// out of it, which is exactly how the rotated path first went wrong: it
// added an input buffer and rotation temps without growing the expert term,
// so a shape where the expert stage dominates hands those slices memory
// nobody reserved.
//
// The layout therefore lives here, as pure arithmetic with no HIP types, and
// the executor takes exactly the sizes this returns. A CPU test can then
// assert the expert-dominant case is covered without a device.

namespace gufo::models::qwen38_flash_next::rocm {

/// The routed-expert stage's slices, in the order the executor takes them.
/// The rotated members stay zero for an unrotated model.
struct ExpertStageSlices {
  std::size_t gate_e{0};
  std::size_t up_e{0};
  std::size_t down_e{0};
  std::size_t x_rot{0};
  std::size_t rot_gate_pre{0};
  std::size_t rot_up_pre{0};
  std::size_t rot_down_in{0};
  std::size_t rot_down_pre{0};

  /// What the expert stage must reserve: the sum of the slices taken from it.
  [[nodiscard]] constexpr std::size_t Total() const noexcept {
    return gate_e + up_e + down_e + x_rot + rot_gate_pre + rot_up_pre +
           rot_down_in + rot_down_pre;
  }
};

/// Lays out the expert stage. `slots` is tokens * experts_used;
/// `max_tokens` is the session's maximum batch, which bounds the hoisted
/// rotated input. The rotated temps are per-slot constants reused across
/// the slot loop, so they do not scale with `slots`.
[[nodiscard]] constexpr ExpertStageSlices ExpertStageLayout(
    std::size_t slots, std::size_t expert_ff, std::size_t hidden, bool rotated,
    std::size_t max_tokens) noexcept {
  ExpertStageSlices s;
  s.gate_e = slots * expert_ff;
  s.up_e = slots * expert_ff;
  s.down_e = slots * hidden;
  if (rotated) {
    s.x_rot = max_tokens * hidden;
    s.rot_gate_pre = expert_ff;
    s.rot_up_pre = expert_ff;
    s.rot_down_in = expert_ff;
    s.rot_down_pre = hidden;
  }
  return s;
}

/// Cross-check against the dispatch's own count of the rotated buffers: the
/// stage total must equal the pre-rotation plain expert formula plus the
/// dispatch's rotated scratch, for the same geometry. Two independent
/// expressions of one contract; if they drift this fires at compile time.
static_assert(ExpertStageLayout(6, 128, 256, true, 1).Total() ==
                  6 * (2 * 128 + 256) +
                      gufo::quant::RotatedScratchFloats(256, 128, 1),
              "expert stage total disagrees with plain + RotatedScratchFloats");
static_assert(ExpertStageLayout(6, 128, 256, false, 1).Total() ==
                  6 * (2 * 128 + 256),
              "unrotated expert stage must not reserve rotated scratch");

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_STAGE_BUDGET_HPP_
