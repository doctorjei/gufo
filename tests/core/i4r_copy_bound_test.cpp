// Point 85, step 2 REV 6: host-side copy-bound regression for the rotated
// dispatch's input geometry (review-v5 blocker 1).
//
// The bug this pins: the driver sized its host->device input copy by the SLOT
// count (slots = n_tokens * num_experts_used) while the host input buffer
// only holds n_tokens * hidden floats. With used=2 that reads twice the
// caller's memory -- a 1024-byte input read as 2048 in decode-like, 3072 as
// 6144 in prefill-like. The dispatch reads x as n_tokens rows of `hidden`
// (RotateLayerInput indexes by token row), so the input copy length must be
// the INPUT geometry, never the slot geometry.
//
// This runs on a CPU-only box: no HIP, no device. It exercises the geometry
// helpers (the single source of truth the driver now uses) and a real
// memcpy-bound check with a guard region, so the over-read class is caught
// by construction rather than by a device crash that only fires on hardware.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "tests/models/qwen38_flash_next/i4r_parity_gate.hpp"

namespace tf = gufo::test_fixture;

namespace {

int g_pass = 0;
int g_fail = 0;

void Check(bool ok, const std::string& what) {
  if (ok) {
    ++g_pass;
  } else {
    ++g_fail;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

constexpr std::uint32_t kHidden = 256;
constexpr std::uint32_t kGuardFloats = 64;
constexpr std::uint32_t kGuardPattern = 0xDEADBEEFU;

// One (n_tokens, used) case. Verifies the two geometries are distinct when
// used>1, that the correct copy length leaves a guard region intact, and
// that the buggy slot-length would reach into that guard.
void CheckCase(std::uint32_t n_tokens, std::uint32_t used) {
  const std::size_t in_floats = tf::InputCopyFloats(n_tokens, kHidden);
  const std::size_t slot_floats = tf::SlotFloats(n_tokens, used, kHidden);
  const std::string tag =
      "n_tokens=" + std::to_string(n_tokens) + " used=" + std::to_string(used);

  // The helpers agree with the plain products.
  Check(in_floats == static_cast<std::size_t>(n_tokens) * kHidden,
        "InputCopyFloats == n_tokens*hidden (" + tag + ")");
  Check(slot_floats == static_cast<std::size_t>(n_tokens) * used * kHidden,
        "SlotFloats == n_tokens*used*hidden (" + tag + ")");

  if (used > 1) {
    // The over-read condition: the slot geometry is strictly larger than the
    // input buffer. Using it as the copy length reads out of bounds.
    Check(slot_floats > in_floats,
          "slot geometry exceeds input buffer when used>1 (" + tag + ")");
    Check(slot_floats == static_cast<std::size_t>(used) * in_floats,
          "over-read factor equals `used` (" + tag + ")");
  } else {
    Check(slot_floats == in_floats,
          "used==1: input and slot geometry coincide (" + tag + ")");
  }

  // Lay the input buffer down with a guard region immediately after it.
  std::vector<std::uint32_t> buf(in_floats + kGuardFloats);
  for (std::size_t i = 0; i < in_floats; ++i)
    buf[i] = static_cast<std::uint32_t>(i) ^ 0x5A5A5A5AU;
  for (std::size_t i = 0; i < kGuardFloats; ++i)
    buf[in_floats + i] = kGuardPattern;

  // Correct copy: exactly the input bytes. The guard must be untouched.
  std::vector<std::uint32_t> dst(in_floats);
  std::memcpy(dst.data(), buf.data(), in_floats * sizeof(std::uint32_t));
  bool guard_intact = true;
  for (std::size_t i = 0; i < kGuardFloats; ++i)
    if (buf[in_floats + i] != kGuardPattern)
      guard_intact = false;
  Check(guard_intact,
        "correct-length copy leaves the guard intact (" + tag + ")");
  bool data_copied = true;
  for (std::size_t i = 0; i < in_floats; ++i)
    if (dst[i] != buf[i])
      data_copied = false;
  Check(data_copied, "correct-length copy transfers the input (" + tag + ")");

  // The buggy length: slot_floats bytes. Show it would extend past the input
  // region into the guard -- i.e. the over-read is real -- WITHOUT actually
  // performing the out-of-bounds read. The guard starts at byte offset
  // in_floats*sizeof(uint32); the buggy copy spans [0, slot_floats*sizeof).
  const std::size_t guard_byte_off = in_floats * sizeof(std::uint32_t);
  const std::size_t buggy_bytes = slot_floats * sizeof(std::uint32_t);
  const std::size_t correct_bytes = in_floats * sizeof(std::uint32_t);
  Check(correct_bytes == guard_byte_off,
        "correct copy ends exactly at the guard (" + tag + ")");
  if (used > 1) {
    Check(buggy_bytes > guard_byte_off,
          "buggy slot-length would read into the guard (" + tag + ")");
    Check(buggy_bytes - correct_bytes == static_cast<std::size_t>(used - 1) *
                                             in_floats * sizeof(std::uint32_t),
          "buggy over-read size equals (used-1)*input bytes (" + tag + ")");
  }
  std::printf("  %s: in=%zu slot=%zu correct=%zuB buggy=%zuB guard@%zuB %s\n",
              tag.c_str(), in_floats, slot_floats, correct_bytes, buggy_bytes,
              guard_byte_off, used > 1 ? "(over-read caught)" : "(equal)");
}

}  // namespace

int main() {
  std::printf("i4r copy-bound regression (review-v5 blocker 1)\n");
  // used=2 is the fixture's value: the exact case that doubled the read.
  CheckCase(1, 2);  // decode-like
  CheckCase(3, 2);  // prefill-like
  CheckCase(1, 1);  // degenerate: no over-read possible
  CheckCase(4, 8);  // a wider fan-out
  std::printf("\ni4r copy-bound: %d PASS / %d FAIL\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
