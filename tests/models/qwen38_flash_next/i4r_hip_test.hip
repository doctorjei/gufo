// GPU test for the I4R128 HIP kernels. Launches the real kernels
// (i4r_kernels.hip.cpp) and checks them against the CPU oracle RotatedGemv /
// RotateBlocks / DequantizeI4R128. Run on gfx1151 once the GPU is free.
//
// The host-emulation test (i4r_hip_math_host_test.cpp) already proves the
// math matches the oracle without a GPU; this closes the loop on actual device
// execution: memory movement, shared-memory butterfly, block reduce, launch
// geometry, and the f16 scale load on device.
//
// Tolerance: the kernels accumulate in float, the oracle in double. Over a
// 2560-element dot the float-vs-double gap is ~1e-5 relative; 5e-3 is a
// generous ceiling that still catches any real defect (a wrong stage, a bad
// nibble order, a misaligned block produces O(1) error).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/i4r_rotation.hpp"

extern "C" {
void i4r_launch_rotate(const float* in, float* out, const float* pre_scale,
                       const float* post_scale, int n_blocks, hipStream_t stream);
void i4r_launch_mmvq(const uint8_t* M, const float* xr, float* y, int n_out,
                     int n_blocks_in, hipStream_t stream);
void i4r_launch_mmq(const uint8_t* M, const float* X, float* Y, int n_out,
                    int n_in, int tokens, hipStream_t stream);
}

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool ok, const char* what) {
  if (ok) {
    ++g_pass;
  } else {
    ++g_fail;
    std::printf("FAIL: %s\n", what);
  }
}

void need(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::printf("FAIL: HIP %s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

double rel_err(double a, double b) {
  const double m = std::fmax(1.0, std::fabs(b));
  return std::fabs(a - b) / m;
}

// Geometry contract: the I4R128 rotate kernel processes whole 128-value
// blocks and the stage-4 output rotation is launched as n_out/128 blocks.
// A short n_out silently launches ZERO blocks (no kernel-side guard by
// design), leaving the output unrotated/unscaled while the CPU oracle
// rotates fully -- a silent mismatch, not a crash. Every end-to-end fixture
// must satisfy this predicate before launching. (Found in review: the old
// 64x128 fixture violated it.)
bool whole_blocks_ok(int n_out, int n_in) {
  return n_out > 0 && n_in > 0 && n_out % 128 == 0 && n_in % 128 == 0;
}

// Finite-value guard run BEFORE any max-relative-error accumulation.
// std::fmax returns the non-NaN operand, so fmax(max_rel, NaN) silently
// keeps max_rel small -- a NaN/Inf in the GPU output would pass the
// tolerance check. Check finiteness explicitly and fail with the index.
bool all_finite(const float* v, size_t n, const char* label) {
  for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(v[i])) {
      std::printf("FAIL: %s: non-finite value at index %zu: %g\n", label, i,
                 static_cast<double>(v[i]));
      return false;
    }
  return true;
}

uint16_t f32_to_f16(float value) {
  uint32_t x;
  std::memcpy(&x, &value, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    const int shift = 14 - exp;
    return static_cast<uint16_t>(sign | ((mant >> shift) + ((mant >> (shift - 1)) & 1u)));
  }
  uint32_t r = (mant + 0x0FFFu + ((mant >> 13) & 1u)) >> 13;
  if (r & 0x400u) {
    exp += 1;
    r = 0;
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (r & 0x3FFu));
}

std::vector<uint8_t> make_matrix(std::mt19937& rng, int n_out, int n_in) {
  const int nb = n_in / 128;
  std::vector<uint8_t> M(static_cast<size_t>(n_out) * nb * 66);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_real_distribution<float> scale_dist(-2.0f, 2.0f);
  for (int o = 0; o < n_out; ++o)
    for (int b = 0; b < nb; ++b) {
      uint8_t* blk = M.data() + (static_cast<size_t>(o) * nb + b) * 66;
      for (int i = 0; i < 64; ++i)
        blk[i] = static_cast<uint8_t>(byte_dist(rng));
      const uint16_t bits = f32_to_f16(scale_dist(rng));
      blk[64] = static_cast<uint8_t>(bits & 0xFF);
      blk[65] = static_cast<uint8_t>(bits >> 8);
    }
  return M;
}

std::vector<float> decode_matrix(const std::vector<uint8_t>& M, int n_out,
                                int n_in) {
  const int nb = n_in / 128;
  std::vector<float> m(static_cast<size_t>(n_out) * n_in);
  for (int o = 0; o < n_out; ++o)
    gufo::quant::DequantizeI4R128(M.data() + static_cast<size_t>(o) * nb * 66,
                                   m.data() + static_cast<size_t>(o) * n_in, n_in);
  return m;
}

// End-to-end: stage1 rotate -> mmvq -> stage4 rotate, vs RotatedGemv.
// REQUIRES whole_blocks_ok(n_out, n_in); asserted before any launch.
void test_mmvq_end_to_end(int n_out, int n_in, unsigned seed, const char* label) {
  check(whole_blocks_ok(n_out, n_in), label);  // geometry precondition
  if (!whole_blocks_ok(n_out, n_in)) {
    std::printf("  %s: REJECTED shape n_out=%d n_in=%d (need multiples of 128)\n",
                label, n_out, n_in);
    return;
  }
  std::mt19937 rng(seed);
  const int nb = n_in / 128;
  auto M = make_matrix(rng, n_out, n_in);
  const auto m_dec = decode_matrix(M, n_out, n_in);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x(n_in), su(n_in), sv(n_out), y_oracle(n_out);
  for (auto& v : x) v = nd(rng);
  for (auto& v : su) v = nd(rng);
  for (auto& v : sv) v = nd(rng);
  gufo::quant::RotatedGemv(m_dec.data(), n_out, n_in, x.data(), su.data(),
                           sv.data(), y_oracle.data());

  float *dx, *dxr, *dy, *dy_rot, *dsu, *dsv;
  uint8_t* dM;
  need(hipMalloc(&dx, n_in * 4), "malloc x");
  need(hipMalloc(&dxr, n_in * 4), "malloc xr");
  need(hipMalloc(&dy, n_out * 4), "malloc y");
  need(hipMalloc(&dy_rot, n_out * 4), "malloc y_rot");
  need(hipMalloc(&dsu, n_in * 4), "malloc su");
  need(hipMalloc(&dsv, n_out * 4), "malloc sv");
  need(hipMalloc(&dM, M.size()), "malloc M");
  need(hipMemcpy(dx, x.data(), n_in * 4, hipMemcpyHostToDevice), "cpy x");
  need(hipMemcpy(dsu, su.data(), n_in * 4, hipMemcpyHostToDevice), "cpy su");
  need(hipMemcpy(dsv, sv.data(), n_out * 4, hipMemcpyHostToDevice), "cpy sv");
  need(hipMemcpy(dM, M.data(), M.size(), hipMemcpyHostToDevice), "cpy M");

  i4r_launch_rotate(dx, dxr, dsu, nullptr, nb, 0);  // stage 1
  i4r_launch_mmvq(dM, dxr, dy, n_out, nb, 0);       // stage 3
  // Stage 4 + sv writes to a SEPARATE buffer: the rotate kernel is
  // __restrict__-aliased, so rotating dy in place is undefined (the rev3
  // finding). Out-of-place matches the production dispatch.
  i4r_launch_rotate(dy, dy_rot, nullptr, dsv, n_out / 128, 0);  // stage 4 + sv
  need(hipDeviceSynchronize(), "sync");

  std::vector<float> y_gpu(n_out);
  need(hipMemcpy(y_gpu.data(), dy_rot, n_out * 4, hipMemcpyDeviceToHost),
       "cpy y back");

  // Two-sided finite guard: GPU output AND CPU oracle must both be finite
  // before fmax, or a NaN on either side silently scores zero error.
  check(all_finite(y_gpu.data(), y_gpu.size(), label), label);
  check(all_finite(y_oracle.data(), y_oracle.size(), label), label);
  double max_rel = 0.0;
  for (int i = 0; i < n_out; ++i)
    max_rel = std::fmax(max_rel, rel_err(y_gpu[i], y_oracle[i]));
  std::printf("  %s: mmvq end-to-end max rel err vs oracle: %.3e\n", label, max_rel);
  check(max_rel < 5e-3, label);

  hipFree(dx); hipFree(dxr); hipFree(dy); hipFree(dy_rot);
  hipFree(dsu); hipFree(dsv); hipFree(dM);
}

// Rotate kernel alone vs RotateBlocks (with pre and post scale).
void test_rotate_kernel() {
  std::mt19937 rng(99);
  const int n = 2560, nb = n / 128;
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> in(n), pre(n), post(n), expect(n), out(n);
  for (auto& v : in) v = nd(rng);
  for (auto& v : pre) v = nd(rng);
  for (auto& v : post) v = nd(rng);

  // expect = rotate(in * pre) then * post  (stage1 + stage4 combined)
  for (int i = 0; i < n; ++i) expect[i] = in[i] * pre[i];
  gufo::quant::RotateBlocks(expect.data(), n);
  for (int i = 0; i < n; ++i) expect[i] *= post[i];

  float *din, *dpre, *dpost, *dout;
  need(hipMalloc(&din, n * 4), "rot malloc in");
  need(hipMalloc(&dpre, n * 4), "rot malloc pre");
  need(hipMalloc(&dpost, n * 4), "rot malloc post");
  need(hipMalloc(&dout, n * 4), "rot malloc out");
  need(hipMemcpy(din, in.data(), n * 4, hipMemcpyHostToDevice), "rot cpy in");
  need(hipMemcpy(dpre, pre.data(), n * 4, hipMemcpyHostToDevice), "rot cpy pre");
  need(hipMemcpy(dpost, post.data(), n * 4, hipMemcpyHostToDevice), "rot cpy post");
  i4r_launch_rotate(din, dout, dpre, dpost, nb, 0);
  need(hipDeviceSynchronize(), "rot sync");
  need(hipMemcpy(out.data(), dout, n * 4, hipMemcpyDeviceToHost), "rot cpy out");
  check(all_finite(out.data(), out.size(), "rotate kernel out"),
        "rotate kernel output finite");

  double max_rel = 0.0;
  for (int i = 0; i < n; ++i)
    max_rel = std::fmax(max_rel, rel_err(out[i], expect[i]));
  std::printf("  rotate kernel (pre+post scale) max rel err: %.3e\n", max_rel);
  check(max_rel < 1e-5, "rotate kernel matches RotateBlocks with pre/post scale");

  hipFree(din); hipFree(dpre); hipFree(dpost); hipFree(dout);
}

// MMQ prefill: multi-token GEMM vs per-token oracle.
void test_mmq_prefill() {
  std::mt19937 rng(555);
  const int n_out = 256, n_in = 640, tokens = 7;
  const int nb = n_in / 128;
  auto M = make_matrix(rng, n_out, n_in);
  const auto m_dec = decode_matrix(M, n_out, n_in);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> su(n_in), sv(n_out);
  for (auto& v : su) v = nd(rng);
  for (auto& v : sv) v = nd(rng);
  std::vector<float> X(static_cast<size_t>(tokens) * n_in);
  for (auto& v : X) v = nd(rng);

  // Rotate each token's input (stage 1) on host for the oracle path.
  std::vector<float> Xrot(X);
  for (int t = 0; t < tokens; ++t) {
    for (int i = 0; i < n_in; ++i) Xrot[t * n_in + i] = X[t * n_in + i] * su[i];
    gufo::quant::RotateBlocks(Xrot.data() + t * n_in, n_in);
  }
  std::vector<float> y_oracle(static_cast<size_t>(tokens) * n_out);
  for (int t = 0; t < tokens; ++t)
    gufo::quant::RotatedGemv(m_dec.data(), n_out, n_in, Xrot.data() + t * n_in,
                             su.data(), sv.data(),
                             y_oracle.data() + t * n_out);
  // NOTE: oracle applies su again inside RotatedGemv; to match the kernel
  // (which consumes already-rotated X and applies sv only), compare the
  // kernel's raw M*Xrot against the oracle's pre-sv stage. Simpler: run the
  // kernel as M*Xrot (no su), then rotate+sv on host, vs full oracle.
  // We instead test the kernel's GEMM piece directly:
  uint8_t* dM;
  float *dXrot, *dY;
  need(hipMalloc(&dM, M.size()), "mmq malloc M");
  need(hipMalloc(&dXrot, Xrot.size() * 4), "mmq malloc Xrot");
  need(hipMalloc(&dY, static_cast<size_t>(tokens) * n_out * 4), "mmq malloc Y");
  need(hipMemcpy(dM, M.data(), M.size(), hipMemcpyHostToDevice), "mmq cpy M");
  need(hipMemcpy(dXrot, Xrot.data(), Xrot.size() * 4, hipMemcpyHostToDevice), "mmq cpy Xrot");
  i4r_launch_mmq(dM, dXrot, dY, n_out, n_in, tokens, 0);
  need(hipDeviceSynchronize(), "mmq sync");
  std::vector<float> y_gpu(static_cast<size_t>(tokens) * n_out);
  need(hipMemcpy(y_gpu.data(), dY, y_gpu.size() * 4, hipMemcpyDeviceToHost), "mmq cpy Y");
  check(all_finite(y_gpu.data(), y_gpu.size(), "mmq prefill Y"),
        "mmq prefill output finite");

  // Kernel computes Y[t][o] = dot(M_row_o, Xrot[t]). Compare to a host
  // double-precision dot of the decoded matrix against Xrot.
  double max_rel = 0.0;
  for (int t = 0; t < tokens; ++t)
    for (int o = 0; o < n_out; ++o) {
      double acc = 0.0;
      for (int i = 0; i < n_in; ++i)
        acc += static_cast<double>(m_dec[o * n_in + i]) * Xrot[t * n_in + i];
      max_rel = std::fmax(max_rel, rel_err(y_gpu[t * n_out + o], acc));
    }
  std::printf("  mmq prefill (%d tokens) max rel err vs host dot: %.3e\n", tokens, max_rel);
  check(max_rel < 5e-3, "mmq prefill matches host double dot on rotated input");

  hipFree(dM); hipFree(dXrot); hipFree(dY);
}

// Geometry negative control: the harness predicate must REJECT the old
// short-output shape (64x128 -> zero-block stage-4 launch) and ACCEPT
// whole-block shapes. Source-level control; no launch involved.
void test_geometry_contract() {
  check(!whole_blocks_ok(64, 128), "negative control: 64x128 rejected (short n_out)");
  check(!whole_blocks_ok(128, 64), "negative control: 128x64 rejected (short n_in)");
  check(!whole_blocks_ok(0, 128), "negative control: 0x128 rejected");
  check(whole_blocks_ok(128, 128), "positive control: 128x128 accepted");
  check(whole_blocks_ok(256, 2560), "positive control: 256x2560 accepted");
  check(whole_blocks_ok(128, 640), "positive control: 128x640 accepted");
  std::printf("  geometry contract: negative controls for short shapes pass\n");
}

}  // namespace

int main() {
  int dev = 0;
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
    std::printf("SKIP: no GPU device available\n");
    return 77;
  }
  std::printf("GPU: %s\n", prop.name);

  test_geometry_contract();
  test_rotate_kernel();
  test_mmvq_end_to_end(256, 2560, 101, "mmvq gate/up shape (256x2560)");
  test_mmvq_end_to_end(128, 640, 202, "mmvq down shape (128x640)");
  test_mmvq_end_to_end(128, 128, 303, "mmvq minimal whole-block (128x128)");
  test_mmq_prefill();

  std::printf("\ni4r gpu test: %d PASS / %d FAIL\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
