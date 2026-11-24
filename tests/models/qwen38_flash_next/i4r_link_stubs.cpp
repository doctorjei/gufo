// Link stub for the CPU-only i4r MoE parity harness.
//
// The reference model's Moe path never touches the vision rope layout; the
// symbol is pulled in only because reference.cpp shares a link closure with
// vision code that the tiny fixture does not exercise. Aborting here (rather
// than returning a plausible value) means a future change that DOES reach it
// fails loudly instead of silently reading an unimplemented layout.
//
// Not part of the shipped library; test-harness scaffolding only.

#include <cstdio>
#include <cstdlib>

#include "src/families/qwen/vision/prompt.hpp"

namespace gufo::models::qwen::vision {

std::array<std::int32_t, 3> RopeLayout::Position(std::uint32_t) const {
  std::fprintf(
      stderr,
      "i4r parity link stub: vision::RopeLayout::Position reached on a "
      "CPU MoE-only path -- this was not expected\n");
  std::abort();
}

}  // namespace gufo::models::qwen::vision
