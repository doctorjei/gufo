// Point 85, step 2 review: the shared stage scratch must budget the rotated
// path's own slices.
//
// The linear, attention and routed-expert stages share one block sized by
// the maximum over them. The rotated (I4R128) dispatch adds an input buffer
// plus out-of-place rotation temps. If those are not inside the expert term,
// a shape where the expert stage dominates hands the rotated slices memory
// that was never reserved -- and the delivered two-expert tiny fixture can
// hide that behind spare linear-stage capacity, so it needs an explicitly
// expert-dominant case.
//
// This is a CPU test of the arithmetic the executor actually uses
// (ExpertStageLayout), not a re-derivation of it: the executor takes these
// exact sizes, so the reservation and the takes cannot drift apart.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "src/models/qwen38_flash_next/kernels/rocm/stage_budget.hpp"

namespace es = gufo::models::qwen38_flash_next::rocm;
namespace qr = gufo::quant;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

// Mirrors the executor's max over the sequential stages. Only the expert
// term varies here; the other terms stand in for shapes that are smaller or
// larger than it.
std::size_t StageMax(std::size_t linear, std::size_t attention,
                     std::size_t expert) {
  return std::max({linear, attention, expert});
}

// The pre-fix expert reservation, kept explicit so the shortfall is a
// number in this test rather than a claim in prose.
std::size_t OldExpertFloats(std::size_t slots, std::size_t ff,
                            std::size_t hidden) {
  return slots * (2 * ff + hidden);
}

// The expert-dominant case from the review: many experts per token, so the
// expert term is the stage maximum and nothing else can absorb the overrun.
void CheckExpertDominant() {
  const std::size_t hidden = 256;
  const std::size_t ff = 128;
  const std::size_t used = 8;  // experts per token
  const std::size_t T = 32;    // session max batch
  const std::size_t slots = T * used;

  const std::size_t old_expert = OldExpertFloats(slots, ff, hidden);
  const auto laid = es::ExpertStageLayout(slots, ff, hidden, true, T);
  const std::size_t new_expert = laid.Total();

  Require(new_expert > old_expert,
          "rotated expert stage did not grow over the old formula");
  const std::size_t shortfall = new_expert - old_expert;
  Require(shortfall == qr::RotatedScratchFloats(hidden, ff, T),
          "rotated growth is not exactly the dispatch's scratch: got " +
              std::to_string(shortfall) + " want " +
              std::to_string(qr::RotatedScratchFloats(hidden, ff, T)));

  // The slices the dispatch actually touches, each individually nonzero.
  Require(laid.x_rot == T * hidden, "x_rot mis-sized");
  Require(laid.rot_gate_pre == ff && laid.rot_up_pre == ff,
          "gate/up pre-rotation temps mis-sized");
  Require(laid.rot_down_in == ff, "down-input rotation temp mis-sized");
  Require(laid.rot_down_pre == hidden, "down pre-rotation buffer mis-sized");

  // With the expert stage as the maximum, the block must cover the new
  // total. Under the old formula it would not: show the gap explicitly.
  const std::size_t small_linear = old_expert / 4;
  const std::size_t small_attention = old_expert / 2;
  const std::size_t block_new =
      StageMax(small_linear, small_attention, new_expert);
  const std::size_t block_old =
      StageMax(small_linear, small_attention, old_expert);
  Require(block_new >= new_expert,
          "expert-dominant block does not cover the rotated stage");
  Require(block_old < new_expert,
          "old formula unexpectedly covered the rotated stage -- the test "
          "shape is no longer expert-dominant");
  std::printf(
      "expert-dominant (slots=%zu ff=%zu hidden=%zu): old %zu new %zu "
      "(+%zu) block_old %zu < needed %zu <= block_new %zu PASS\n",
      slots, ff, hidden, old_expert, new_expert, shortfall, block_old,
      new_expert, block_new);
}

// When another stage is the maximum, the rotated extra must be absorbed by
// that stage rather than inflate the reservation on its own account.
void CheckNonDominantAbsorbed() {
  const std::size_t hidden = 256, ff = 128, T = 4, used = 2;
  const std::size_t slots = T * used;
  const auto laid = es::ExpertStageLayout(slots, ff, hidden, true, T);
  const std::size_t big_attention = laid.Total() * 8;
  const std::size_t block = StageMax(0, big_attention, laid.Total());
  Require(block == big_attention,
          "attention-dominant shape should not be inflated by the expert term");
  std::printf("attention-dominant: block %zu == attention %zu PASS\n", block,
              big_attention);
}

// An unrotated model must reserve exactly what it always did -- no rotated
// scratch charged to a plain model.
void CheckUnrotatedUnchanged() {
  const std::size_t hidden = 256, ff = 128, T = 32, used = 8;
  const std::size_t slots = T * used;
  const auto plain = es::ExpertStageLayout(slots, ff, hidden, false, T);
  Require(plain.Total() == OldExpertFloats(slots, ff, hidden),
          "unrotated stage changed size");
  Require(plain.x_rot == 0 && plain.rot_gate_pre == 0 &&
              plain.rot_up_pre == 0 && plain.rot_down_in == 0 &&
              plain.rot_down_pre == 0,
          "unrotated stage reserved rotated scratch");
  std::printf("unrotated: total %zu == plain formula %zu PASS\n", plain.Total(),
              OldExpertFloats(slots, ff, hidden));
}

}  // namespace

int main() {
  try {
    CheckExpertDominant();
    CheckNonDominantAbsorbed();
    CheckUnrotatedUnchanged();
  } catch (const std::exception& e) {
    std::printf("FAIL: %s\n", e.what());
    return 1;
  }
  std::printf("i4r_stage_budget ALL PASS\n");
  return 0;
}
