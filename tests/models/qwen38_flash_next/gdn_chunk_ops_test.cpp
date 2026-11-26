// Chunked gated delta rule (GatedDeltaNet(..., chunked = true), halogen's
// prefill form) against the token-by-token kernel: output and state error,
// bit-identical results when a pass is split on a 16-token boundary or
// checkpointed there, and timings. Needs a GPU.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

constexpr std::uint32_t kKHeads = 16;
constexpr std::uint32_t kVHeads = 48;
constexpr std::uint32_t kDim = 128;
constexpr std::uint32_t kKernel = 4;
constexpr std::uint32_t kChannels = 2 * kKHeads * kDim + kVHeads * kDim;
constexpr std::uint32_t kZ = kVHeads * kDim;
constexpr float kEps = 1e-6F;
constexpr std::size_t kStateCount = static_cast<std::size_t>(kVHeads) * kDim * kDim;
constexpr std::size_t kConvState = static_cast<std::size_t>(kKernel - 1) * kChannels;

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

struct Buf {
  float* p = nullptr;
  std::size_t n = 0;
  explicit Buf(std::size_t count) : n(count) { Check(hipMalloc(&p, count * sizeof(float)), "hipMalloc"); }
  ~Buf() { (void)hipFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  void up(const std::vector<float>& v) { Check(hipMemcpy(p, v.data(), n * 4, hipMemcpyHostToDevice), "up"); }
  std::vector<float> down() const {
    std::vector<float> v(n);
    Check(hipMemcpy(v.data(), p, n * 4, hipMemcpyDeviceToHost), "down");
    return v;
  }
};

std::vector<float> Values(std::size_t count, std::uint32_t seed, float scale, float offset = 0.0F) {
  std::vector<float> v(count);
  for (float& x : v) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    x = offset + scale * static_cast<float>(static_cast<int>(seed & 0xFFFFU) - 32768) / 32768.0F;
  }
  return v;
}

double WorstOverRms(const std::vector<float>& ref, const std::vector<float>& got) {
  double ss = 0.0, worst = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (!std::isfinite(got[i])) return std::numeric_limits<double>::infinity();
    ss += static_cast<double>(ref[i]) * ref[i];
    worst = std::max(worst, std::abs(static_cast<double>(ref[i]) - got[i]));
  }
  return worst / std::sqrt(ss / static_cast<double>(ref.size()) + 1e-30);
}

struct Inputs {
  std::uint32_t n;
  std::vector<float> qkv, z, ab, conv_w, a, dt, norm_w, conv_state, state;
  explicit Inputs(std::uint32_t tokens) : n(tokens) {
    qkv = Values(static_cast<std::size_t>(n) * kChannels, 0x1234ABCDU, 1.0F);
    z = Values(static_cast<std::size_t>(n) * kZ, 0x0BADF00DU, 2.0F);
    ab = Values(static_cast<std::size_t>(n) * 2 * kVHeads, 0xBADC0FFEU, 2.0F);
    conv_w = Values(static_cast<std::size_t>(kChannels) * kKernel, 0xDEADBEEFU, 0.5F);
    a = Values(kVHeads, 0xC0FFEE11U, 1.0F, -1.5F);
    dt = Values(kVHeads, 0xFEEDFACEU, 1.0F);
    norm_w = Values(kDim, 0x600DCAFEU, 0.5F, 1.0F);
    conv_state = Values(kConvState, 0x13579BDFU, 1.0F);
    state = Values(kStateCount, 0x2468ACE0U, 0.1F);
  }
};

struct Result {
  std::vector<float> out, state;
  float ms = 0.0F;
};

// Runs tokens [first, first + count) of `in` as one pass starting from the
// given conv and recurrent states (updated in place).
void Pass(const Inputs& in, std::uint32_t first, std::uint32_t count, bool chunked, Buf& d_conv_state,
          Buf& d_state, std::vector<float>* out, q::GdnCheckpoint cp = {}, int reps = 1,
          float* ms = nullptr, const std::vector<float>* conv_rows = nullptr, bool half_out = false) {
  Buf d_qkv(static_cast<std::size_t>(count) * kChannels), d_z(static_cast<std::size_t>(count) * kZ),
      d_ab(static_cast<std::size_t>(count) * 2 * kVHeads), d_cw(in.conv_w.size()), d_a(kVHeads),
      d_dt(kVHeads), d_nw(kDim), d_scratch(static_cast<std::size_t>(count + kKernel) * kChannels),
      d_qn(static_cast<std::size_t>(count) * kKHeads * kDim), d_kn(static_cast<std::size_t>(count) * kKHeads * kDim),
      d_raw(static_cast<std::size_t>(count) * kZ), d_out(static_cast<std::size_t>(count) * kZ);
  d_qkv.up(std::vector<float>(in.qkv.begin() + static_cast<std::size_t>(first) * kChannels,
                              in.qkv.begin() + static_cast<std::size_t>(first + count) * kChannels));
  d_z.up(std::vector<float>(in.z.begin() + static_cast<std::size_t>(first) * kZ,
                            in.z.begin() + static_cast<std::size_t>(first + count) * kZ));
  d_ab.up(std::vector<float>(in.ab.begin() + static_cast<std::size_t>(first) * 2 * kVHeads,
                             in.ab.begin() + static_cast<std::size_t>(first + count) * 2 * kVHeads));
  d_cw.up(in.conv_w);
  d_a.up(in.a);
  d_dt.up(in.dt);
  d_nw.up(in.norm_w);
  const auto conv0 = d_conv_state.down();
  const auto state0 = d_state.down();
  hipEvent_t e0, e1;
  Check(hipEventCreate(&e0), "event");
  Check(hipEventCreate(&e1), "event");
  for (int r = 0; r < reps; ++r) {
    if (r > 0) {
      d_conv_state.up(conv0);
      d_state.up(state0);
    }
    if (conv_rows)
      Check(hipMemcpy(d_scratch.p, conv_rows->data(), conv_rows->size() * 4, hipMemcpyHostToDevice), "rows");
    if (r == reps - 1) Check(hipEventRecord(e0, nullptr), "record");
    if (!q::GatedDeltaNet(d_qkv.p, kChannels, d_z.p, kZ, d_ab.p, d_cw.p, d_a.p, d_dt.p, d_nw.p,
                          d_conv_state.p, d_scratch.p, d_qn.p, d_kn.p, d_raw.p, d_state.p, d_out.p,
                          nullptr, {}, {}, count, kKHeads, kVHeads, kDim, kKernel, true, conv_rows != nullptr, kEps,
                          nullptr, half_out ? reinterpret_cast<__half*>(d_out.p) : nullptr, cp,
                          half_out ? kZ + 32 : 0, nullptr, nullptr, chunked))
      throw std::runtime_error("GatedDeltaNet refused");
    if (r == reps - 1) Check(hipEventRecord(e1, nullptr), "record");
  }
  Check(hipEventSynchronize(e1), "sync");
  float t = 0.0F;
  Check(hipEventElapsedTime(&t, e0, e1), "elapsed");
  if (ms) *ms = t;
  if (out) *out = d_out.down();
}

Result Run(const Inputs& in, bool chunked, int reps = 1, const std::vector<float>* conv_rows = nullptr) {
  Buf cs(kConvState), st(kStateCount);
  cs.up(in.conv_state);
  st.up(in.state);
  Result r;
  Pass(in, 0, in.n, chunked, cs, st, &r.out, {}, reps, &r.ms, conv_rows);
  r.state = st.down();
  return r;
}

float HostBf16(float x) {
  std::uint32_t u;
  std::memcpy(&u, &x, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  u &= 0xFFFF0000u;
  std::memcpy(&x, &u, 4);
  return x;
}

// Convolved rows rounded to bf16 on the host (halogen's convolution form,
// spec 5.8.4), fed to both routes so only the recurrence arithmetic differs.
std::vector<float> Bf16ConvRows(const Inputs& in) {
  std::vector<float> rows(static_cast<std::size_t>(in.n) * kChannels);
  for (std::uint32_t t = 0; t < in.n; ++t)
    for (std::uint32_t c = 0; c < kChannels; ++c) {
      float acc = 0.0F;
      for (int k = 0; k < 4; ++k) {
        const int src = static_cast<int>(t) - 3 + k;
        const float x = src >= 0 ? in.qkv[static_cast<std::size_t>(src) * kChannels + c]
                                 : in.conv_state[static_cast<std::size_t>(3 + src) * kChannels + c];
        const float w = in.conv_w[static_cast<std::size_t>(c) * 4 + k];
        acc = k == 0 ? w * HostBf16(x) : std::fma(w, HostBf16(x), acc);
      }
      rows[static_cast<std::size_t>(t) * kChannels + c] = HostBf16(acc / (1.0F + std::exp2(acc * -1.44269504088896341F)));
    }
  return rows;
}

}  // namespace

int main() {
  int devs = 0;
  if (hipGetDeviceCount(&devs) != hipSuccess || devs == 0) {
    std::printf("SKIP: no GPU\n");
    return 77;
  }
  int fails = 0;
  for (std::uint32_t n : {16u, 37u, 100u, 1024u, 2304u}) {
    const Inputs in(n);
    const int reps = n >= 1024 ? 3 : 1;
    const Result tok = Run(in, false, reps);
    const Result chk = Run(in, true, reps);
    const double eo = WorstOverRms(tok.out, chk.out);
    const double es = WorstOverRms(tok.state, chk.state);
    std::printf("n=%5u  output err/rms %.3e  state err/rms %.3e  token %.3f ms  chunked %.3f ms\n", n, eo, es,
                tok.ms, chk.ms);
    (void)eo;  // f32 token inputs vs bf16 chunk inputs: informational
  }
  // Both routes on the same bf16 convolved rows: the recurrence arithmetic
  // alone (half-precision operands, chunked order) separates them.
  for (std::uint32_t n : {16u, 37u, 100u, 1024u, 2304u}) {
    const Inputs in(n);
    const auto rows = Bf16ConvRows(in);
    const Result tok = Run(in, false, 1, &rows);
    const Result chk = Run(in, true, 1, &rows);
    const double eo = WorstOverRms(tok.out, chk.out), es = WorstOverRms(tok.state, chk.state);
    std::printf("same bf16 rows n=%5u: output err/rms %.3e  state err/rms %.3e\n", n, eo, es);
    if (!(eo < 5e-2 && es < 5e-2)) {
      std::printf("FAIL: chunked recurrence disagrees with the token kernel on identical inputs at n=%u\n", n);
      ++fails;
    }
  }
  // The f16-output variant with the padded row stride the prompt pass uses.
  {
    const Inputs in(2304);
    Buf cs(kConvState), st(kStateCount);
    for (const bool chunked : {false, true}) {
      cs.up(in.conv_state);
      st.up(in.state);
      float ms = 0.0F;
      Pass(in, 0, in.n, chunked, cs, st, nullptr, {}, 3, &ms, nullptr, true);
      std::printf("f16 rows, stride %u: %s %.3f ms\n", kZ + 32, chunked ? "chunked" : "token", ms);
    }
  }
  // Split invariance on a 16-token boundary, and a checkpoint there.
  {
    const std::uint32_t n = 100, cut = 48;
    const Inputs in(n);
    const Result whole = Run(in, true);
    Buf cs(kConvState), st(kStateCount);
    cs.up(in.conv_state);
    st.up(in.state);
    std::vector<float> o1, o2;
    Pass(in, 0, cut, true, cs, st, &o1);
    Pass(in, cut, n - cut, true, cs, st, &o2);
    o1.insert(o1.end(), o2.begin(), o2.end());
    const bool same_out = std::memcmp(o1.data(), whole.out.data(), o1.size() * 4) == 0;
    const bool same_state = std::memcmp(st.down().data(), whole.state.data(), kStateCount * 4) == 0;
    // Checkpoint at the cut inside the whole pass.
    Buf cs2(kConvState), st2(kStateCount), cap(kStateCount), hist(kConvState);
    cs2.up(in.conv_state);
    st2.up(in.state);
    q::GdnCheckpoint cp{};
    cp.count = 1;
    cp.tokens[0] = cut;
    cp.state[0] = cap.p;
    cp.history[0] = hist.p;
    Pass(in, 0, n, true, cs2, st2, nullptr, cp);
    Buf cs3(kConvState), st3(kStateCount);
    cs3.up(in.conv_state);
    st3.up(in.state);
    Pass(in, 0, cut, true, cs3, st3, nullptr);
    const bool same_cp = std::memcmp(cap.down().data(), st3.down().data(), kStateCount * 4) == 0;
    std::printf("split at %u: outputs %s, state %s; checkpoint %s\n", cut, same_out ? "identical" : "DIFFER",
                same_state ? "identical" : "DIFFER", same_cp ? "identical" : "DIFFER");
    if (!same_out || !same_state || !same_cp) {
      std::printf("FAIL: chunked route is not split-invariant on a 16-token boundary\n");
      ++fails;
    }
  }
  return fails == 0 ? 0 : 1;
}
