// GPU check of ht_dense_gemv against the CPU decoder (hgn_decode) for random
// ht payloads: split and unsplit inputs, 1..8 tokens, plus a negative
// control. Exit 0 = pass.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "src/core/quant/hgn_decode.hpp"
#include "src/core/quant/ht_dense.hpp"
#include "src/core/quant/hgn_prefill.hpp"

namespace {

int failures = 0;

#define HIP_OK(x)                                                         \
  do {                                                                    \
    hipError_t e_ = (x);                                                  \
    if (e_ != hipSuccess) {                                               \
      std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_),       \
                  __FILE__, __LINE__);                                    \
      std::exit(2);                                                       \
    }                                                                     \
  } while (0)

template <typename T>
T* Up(const std::vector<T>& h) {
  T* d = nullptr;
  HIP_OK(hipMalloc(&d, h.size() * sizeof(T) + 4096));
  HIP_OK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
  return d;
}

double RunCase(int rows, int cols, int n, std::mt19937& rng, bool corrupt) {
  std::vector<std::uint8_t> codes(static_cast<std::size_t>(rows) * cols / 2);
  for (auto& b : codes) b = static_cast<std::uint8_t>(rng());
  std::vector<std::uint16_t> su(cols), sv(rows);
  std::vector<float> suf(cols), svf(rows);
  std::uniform_real_distribution<float> mag(0.005f, 0.03f);
  for (int c = 0; c < cols; ++c) {
    suf[c] = (rng() & 1) ? 1.0f : -1.0f;
    gufo::quant::FloatToHalfBits(&suf[c], &su[c], 1);
  }
  for (int r = 0; r < rows; ++r) {
    float v = mag(rng) * ((rng() & 1) ? 1.0f : -1.0f);
    gufo::quant::FloatToHalfBits(&v, &sv[r], 1);
    _Float16 h;
    std::memcpy(&h, &sv[r], 2);
    svf[r] = static_cast<float>(h);
  }
  std::vector<float> w(static_cast<std::size_t>(rows) * cols);
  std::string err;
  if (!gufo::quant::DecodeHgnToF32(gufo::core::GgmlType::kHT4, codes.data(),
                                   cols, rows, su.data(), sv.data(), w.data(),
                                   &err)) {
    std::printf("CPU decode failed: %s\n", err.c_str());
    std::exit(2);
  }
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x(static_cast<std::size_t>(n) * cols);
  for (auto& v : x) v = nd(rng);
  if (corrupt)
    for (std::size_t i = codes.size() / 3; i < codes.size() / 3 + 4096; ++i) codes[i] ^= 0x5A;
  auto* dcodes = Up(codes);
  auto* dsu = Up(suf);
  auto* dsv = Up(svf);
  auto* dx = Up(x);
  float* dout = nullptr;
  HIP_OK(hipMalloc(&dout, static_cast<std::size_t>(n) * rows * sizeof(float)));
  const std::size_t pf = ht_dense_partial_floats(rows, cols);
  float* dpart = nullptr;
  unsigned* dcnt = nullptr;
  HIP_OK(hipMalloc(&dpart, (pf + 1) * sizeof(float)));
  HIP_OK(hipMalloc(&dcnt, (rows / 128 + 1) * sizeof(unsigned)));
  HIP_OK(hipMemset(dcnt, 0, (rows / 128 + 1) * sizeof(unsigned)));
  HtDenseArgs a{dcodes, dsu, dsv, dx, dout, dpart, dcnt, rows, cols, n, cols, rows};
  // Twice: the second run checks that the counters reset themselves.
  for (int rep = 0; rep < 2; ++rep) {
    if (!ht_dense_gemv(&a, nullptr)) {
      std::printf("launch refused\n");
      std::exit(2);
    }
  }
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> y(static_cast<std::size_t>(n) * rows);
  HIP_OK(hipMemcpy(y.data(), dout, y.size() * sizeof(float), hipMemcpyDeviceToHost));
  double worst = 0.0;
  for (int t = 0; t < n; ++t) {
    double peak = 0.0, err_max = 0.0;
    for (int r = 0; r < rows; ++r) {
      double ref = 0.0;
      const float* wr = w.data() + static_cast<std::size_t>(r) * cols;
      const float* xt = x.data() + static_cast<std::size_t>(t) * cols;
      for (int c = 0; c < cols; ++c) ref += static_cast<double>(wr[c]) * xt[c];
      peak = std::max(peak, std::fabs(ref));
      err_max = std::max(err_max, std::fabs(ref - y[static_cast<std::size_t>(t) * rows + r]));
    }
    worst = std::max(worst, err_max / peak);
  }
  (void)hipFree(dcodes);
  (void)hipFree(dsu);
  (void)hipFree(dsv);
  (void)hipFree(dx);
  (void)hipFree(dout);
  (void)hipFree(dpart);
  (void)hipFree(dcnt);
  return worst;
}

struct Mat {
  std::vector<std::uint8_t> codes;
  std::vector<float> w, su, sv;
};

Mat MakeMat(int rows, int cols, std::mt19937& rng) {
  Mat m;
  m.codes.resize(static_cast<std::size_t>(rows) * cols / 2);
  for (auto& b : m.codes) b = static_cast<std::uint8_t>(rng());
  std::vector<std::uint16_t> su(cols), sv(rows);
  m.su.resize(cols);
  m.sv.resize(rows);
  std::uniform_real_distribution<float> mag(0.005f, 0.03f);
  for (int c = 0; c < cols; ++c) {
    m.su[c] = (rng() & 1) ? 1.0f : -1.0f;
    gufo::quant::FloatToHalfBits(&m.su[c], &su[c], 1);
  }
  for (int r = 0; r < rows; ++r) {
    float v = mag(rng) * ((rng() & 1) ? 1.0f : -1.0f);
    gufo::quant::FloatToHalfBits(&v, &sv[r], 1);
    _Float16 h;
    std::memcpy(&h, &sv[r], 2);
    m.sv[r] = static_cast<float>(h);
  }
  m.w.resize(static_cast<std::size_t>(rows) * cols);
  std::string err;
  if (!gufo::quant::DecodeHgnToF32(gufo::core::GgmlType::kHT4, m.codes.data(), cols, rows,
                                   su.data(), sv.data(), m.w.data(), &err)) {
    std::printf("CPU decode failed: %s\n", err.c_str());
    std::exit(2);
  }
  return m;
}

// One grouped launch over matrices sharing an input; worst error over all.
double RunGroup(const std::vector<int>& rows, int cols, int n, std::mt19937& rng) {
  std::vector<Mat> mats;
  for (int r : rows) mats.push_back(MakeMat(r, cols, rng));
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x(static_cast<std::size_t>(n) * cols);
  for (auto& v : x) v = nd(rng);
  auto* dx = Up(x);
  HtGroupArgs g{};
  std::vector<float*> outs;
  for (std::size_t i = 0; i < mats.size(); ++i) {
    float* o = nullptr;
    HIP_OK(hipMalloc(&o, static_cast<std::size_t>(n) * rows[i] * sizeof(float)));
    outs.push_back(o);
    g.m[g.count++] = HtMatrix{Up(mats[i].codes), Up(mats[i].su), Up(mats[i].sv), o, rows[i], rows[i]};
  }
  std::size_t pf = 0, nc = 0;
  ht_dense_group_scratch(rows.data(), static_cast<int>(rows.size()), cols, &pf, &nc);
  HIP_OK(hipMalloc(&g.partials, (pf + 1) * sizeof(float)));
  HIP_OK(hipMalloc(&g.counters, (nc + 1) * sizeof(unsigned)));
  HIP_OK(hipMemset(g.counters, 0, (nc + 1) * sizeof(unsigned)));
  g.partial_capacity = pf;
  g.counter_capacity = nc;
  g.x = dx;
  g.cols = cols;
  g.n_tokens = n;
  g.x_stride = cols;
  for (int rep = 0; rep < 2; ++rep)
    if (!ht_dense_group(&g, nullptr)) {
      std::printf("group launch refused\n");
      std::exit(2);
    }
  HIP_OK(hipDeviceSynchronize());
  double worst = 0.0;
  for (std::size_t i = 0; i < mats.size(); ++i) {
    std::vector<float> y(static_cast<std::size_t>(n) * rows[i]);
    HIP_OK(hipMemcpy(y.data(), outs[i], y.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (int t = 0; t < n; ++t) {
      double peak = 0.0, emax = 0.0;
      for (int r = 0; r < rows[i]; ++r) {
        double ref = 0.0;
        for (int c = 0; c < cols; ++c)
          ref += static_cast<double>(mats[i].w[static_cast<std::size_t>(r) * cols + c]) *
                 x[static_cast<std::size_t>(t) * cols + c];
        peak = std::max(peak, std::fabs(ref));
        emax = std::max(emax, std::fabs(ref - y[static_cast<std::size_t>(t) * rows[i] + r]));
      }
      worst = std::max(worst, emax / peak);
    }
  }
  return worst;
}

// Wide prefill path (hgn_prefill.hpp): M decoded to f16 on the GPU, input
// rotated to f16 on the GPU, y' = M x' on the host in double, output rotated
// on the GPU; compared with the CPU-decoded weights times x.
double RunWide(int rows, int cols, int n, std::mt19937& rng) {
  const Mat m = MakeMat(rows, cols, rng);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x(static_cast<std::size_t>(n) * cols);
  for (auto& v : x) v = nd(rng);
  auto* dcodes = Up(m.codes);
  auto* dsu = Up(m.su);
  auto* dsv = Up(m.sv);
  auto* dx = Up(x);
  _Float16* dm = nullptr;
  _Float16* dxr = nullptr;
  float* dy = nullptr;
  HIP_OK(hipMalloc(&dm, static_cast<std::size_t>(rows) * cols * 2));
  HIP_OK(hipMalloc(&dxr, static_cast<std::size_t>(n) * cols * 2));
  HIP_OK(hipMalloc(&dy, static_cast<std::size_t>(n) * rows * 4));
  if (!ht_decode_f16(dcodes, rows, cols, dm, nullptr) ||
      !ht_rotate_input_f16(dx, dsu, dxr, n, cols, nullptr)) {
    std::printf("wide launch refused\n");
    std::exit(2);
  }
  HIP_OK(hipDeviceSynchronize());
  std::vector<_Float16> hm(static_cast<std::size_t>(rows) * cols), hx(static_cast<std::size_t>(n) * cols);
  HIP_OK(hipMemcpy(hm.data(), dm, hm.size() * 2, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(hx.data(), dxr, hx.size() * 2, hipMemcpyDeviceToHost));
  std::vector<float> yp(static_cast<std::size_t>(n) * rows);
  for (int t = 0; t < n; ++t)
    for (int r = 0; r < rows; ++r) {
      double acc = 0.0;
      for (int c = 0; c < cols; ++c)
        acc += static_cast<double>(hm[static_cast<std::size_t>(r) * cols + c]) *
               static_cast<double>(hx[static_cast<std::size_t>(t) * cols + c]);
      yp[static_cast<std::size_t>(t) * rows + r] = static_cast<float>(acc);
    }
  HIP_OK(hipMemcpy(dy, yp.data(), yp.size() * 4, hipMemcpyHostToDevice));
  if (!ht_rotate_output(dy, dsv, n, rows, nullptr)) {
    std::printf("wide output rotation refused\n");
    std::exit(2);
  }
  std::vector<float> y(yp.size());
  HIP_OK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost));
  double worst = 0.0;
  for (int t = 0; t < n; ++t) {
    double peak = 0.0, emax = 0.0;
    for (int r = 0; r < rows; ++r) {
      double ref = 0.0;
      for (int c = 0; c < cols; ++c)
        ref += static_cast<double>(m.w[static_cast<std::size_t>(r) * cols + c]) *
               x[static_cast<std::size_t>(t) * cols + c];
      peak = std::max(peak, std::fabs(ref));
      emax = std::max(emax, std::fabs(ref - y[static_cast<std::size_t>(t) * rows + r]));
    }
    worst = std::max(worst, emax / peak);
  }
  for (void* p : {static_cast<void*>(dcodes), static_cast<void*>(dsu), static_cast<void*>(dsv),
                  static_cast<void*>(dx), static_cast<void*>(dm), static_cast<void*>(dxr),
                  static_cast<void*>(dy)})
    (void)hipFree(p);
  return worst;
}

// q6g64_decode_f16 against the CPU decoder: max error over all values,
// relative to the largest value.
double RunQ6Decode(int rows, int cols, std::mt19937& rng) {
  const std::size_t S = (static_cast<std::size_t>(13) * cols / 16 + 15) / 16 * 16;
  std::vector<std::uint8_t> w(static_cast<std::size_t>(rows) * S);
  for (auto& b : w) b = static_cast<std::uint8_t>(rng());
  for (int r = 0; r < rows; ++r)
    for (int g = 0; g < cols / 64; ++g) {
      // Sane f16 scale and bias records (random bits could be inf/nan).
      const float sc = 0.001f + 0.01f * static_cast<float>(rng() % 1000) / 1000.0f;
      const float bi = -0.03f + 0.06f * static_cast<float>(rng() % 1000) / 1000.0f;
      std::uint16_t h[2];
      gufo::quant::FloatToHalfBits(&sc, &h[0], 1);
      gufo::quant::FloatToHalfBits(&bi, &h[1], 1);
      std::memcpy(&w[static_cast<std::size_t>(r) * S + 3 * cols / 4 + 4 * g], h, 4);
    }
  std::vector<float> ref(static_cast<std::size_t>(rows) * cols);
  std::string err;
  if (!gufo::quant::DecodeHgnToF32(gufo::core::GgmlType::kQ6G64, w.data(), cols, rows, nullptr,
                                   nullptr, ref.data(), &err)) {
    std::printf("CPU q6g64 decode failed: %s\n", err.c_str());
    std::exit(2);
  }
  auto* dw = Up(w);
  _Float16* dout = nullptr;
  HIP_OK(hipMalloc(&dout, ref.size() * 2));
  if (!q6g64_decode_f16(dw, rows, cols, dout, nullptr)) {
    std::printf("q6g64 decode refused\n");
    std::exit(2);
  }
  std::vector<_Float16> got(ref.size());
  HIP_OK(hipMemcpy(got.data(), dout, got.size() * 2, hipMemcpyDeviceToHost));
  double peak = 0.0, emax = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    peak = std::max(peak, std::fabs(static_cast<double>(ref[i])));
    emax = std::max(emax, std::fabs(static_cast<double>(ref[i]) - static_cast<double>(got[i])));
  }
  (void)hipFree(dw);
  (void)hipFree(dout);
  return emax / peak;
}

}  // namespace

int main() {
  std::mt19937 rng(1234);
  // f16 rounding of the rotated input bounds the error at ~1e-3 of peak.
  constexpr double kGate = 4e-3;
  const struct {
    int rows, cols;
  } shapes[] = {{256, 384}, {640, 2560}, {2560, 6144}, {512, 2560}, {128 * 700, 256}};
  for (const auto& s : shapes) {
    for (int n : {1, 2, 3, 5, 6, 7, 8}) {
      if (s.rows == 128 * 700 && n > 2) continue;
      const double e = RunCase(s.rows, s.cols, n, rng, false);
      const bool ok = e < kGate;
      std::printf("%s rows=%d cols=%d n=%d splits=%d err=%.3e\n", ok ? "PASS" : "FAIL",
                  s.rows, s.cols, n, ht_dense_splits(s.rows, s.cols), e);
      failures += ok ? 0 : 1;
    }
  }
  // The verify-batch plan with fewer splits must give the same results.
  ht_dense_set_batch_target(0);
  for (const auto& s2 : std::vector<std::pair<int, int>>{{640, 2560}, {2560, 6144}, {6144, 2560}}) {
    for (int n : {2, 5, 8}) {
      const double e = RunCase(s2.first, s2.second, n, rng, false);
      const bool ok = e < kGate;
      std::printf("%s batch plan rows=%d cols=%d n=%d splits=%d err=%.3e\n", ok ? "PASS" : "FAIL",
                  s2.first, s2.second, n, ht_dense_splits_for(s2.first, s2.second, n), e);
      failures += ok ? 0 : 1;
    }
  }
  ht_dense_set_batch_target(0);
  const double neg = RunCase(640, 2560, 1, rng, true);
  const bool neg_ok = neg > 10 * kGate;
  std::printf("%s negative control err=%.3e\n", neg_ok ? "PASS" : "FAIL", neg);
  failures += neg_ok ? 0 : 1;
  for (const auto& grp : std::vector<std::vector<int>>{{1280, 768}, {1536, 128, 128}, {640, 640}}) {
    for (int n : {1, 3, 8}) {
      const double e = RunGroup(grp, 2560, n, rng);
      const bool ok = e < kGate;
      std::printf("%s group of %zu (rows %d..) cols=2560 n=%d err=%.3e\n", ok ? "PASS" : "FAIL",
                  grp.size(), grp[0], n, e);
      failures += ok ? 0 : 1;
    }
  }
  for (const auto& s : std::vector<std::pair<int, int>>{{640, 2560}, {2560, 6144}, {512, 2560}}) {
    const double e = RunWide(s.first, s.second, 24, rng);
    const bool ok = e < kGate;
    std::printf("%s wide prefill rows=%d cols=%d n=24 err=%.3e\n", ok ? "PASS" : "FAIL", s.first,
                s.second, e);
    failures += ok ? 0 : 1;
  }
  {
    // f16 rounding of the decoded values: below 1e-3 of the largest value.
    const double e = RunQ6Decode(300, 2560, rng);
    const bool ok = e < 1e-3;
    std::printf("%s q6g64 f16 decode err=%.3e\n", ok ? "PASS" : "FAIL", e);
    failures += ok ? 0 : 1;
  }
  std::printf("%s\n", failures == 0 ? "ALL PASS" : "FAILURES");

  // Timing at the Flash-Next shapes (random codes; speed does not depend on
  // the values). Bytes = rows * cols / 2.
  const struct {
    const char* name;
    int rows, cols;
  } real[] = {{"linear qkv", 10240, 2560}, {"linear gate", 6144, 2560},
              {"attn q+gate", 12288, 2560}, {"out proj", 2560, 6144},
              {"indexer", 640, 2560},       {"shexp gate", 640, 2560},
              {"lm_head", 248320, 2560}};
  std::vector<std::uint8_t> big(static_cast<std::size_t>(248320) * 2560 / 2);
  for (auto& b : big) b = static_cast<std::uint8_t>(rng());
  auto* dcodes = Up(big);
  std::vector<float> ones(248320, 1.0f), xin(8 * 6144, 0.5f);
  auto* dv = Up(ones);
  auto* dx = Up(xin);
  float* dout = nullptr;
  HIP_OK(hipMalloc(&dout, static_cast<std::size_t>(8) * 248320 * sizeof(float)));
  float* dpart = nullptr;
  unsigned* dcnt = nullptr;
  HIP_OK(hipMalloc(&dpart, 16u << 20));
  HIP_OK(hipMalloc(&dcnt, 4096 * sizeof(unsigned)));
  HIP_OK(hipMemset(dcnt, 0, 4096 * sizeof(unsigned)));
  hipEvent_t e0, e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  // Verify-batch plan sweep: time the batch sizes at several workgroup
  // targets (640 = the single-token plan), then keep the default.
  for (int target : {0, 640}) {  // 0 = the default batch plan
    ht_dense_set_batch_target(target);
    for (const auto& r : real) {
      if (r.rows > 100000) continue;
      for (int n : {2, 4, 5, 6, 8}) {
        HtDenseArgs a{dcodes, dv, dv, dx, dout, dpart, dcnt, r.rows, r.cols, n, r.cols, r.rows};
        for (int i = 0; i < 5; ++i) (void)ht_dense_gemv(&a, nullptr);
        HIP_OK(hipEventRecord(e0, nullptr));
        for (int i = 0; i < 200; ++i) (void)ht_dense_gemv(&a, nullptr);
        HIP_OK(hipEventRecord(e1, nullptr));
        HIP_OK(hipEventSynchronize(e1));
        float ms = 0.0f;
        HIP_OK(hipEventElapsedTime(&ms, e0, e1));
        std::printf("SWEEP target=%-3d %-12s n=%d splits=%-2d %8.1f us\n", target, r.name, n,
                    ht_dense_splits_for(r.rows, r.cols, n), 1000.0 * ms / 200);
      }
    }
  }
  ht_dense_set_batch_target(0);
  for (const auto& r : real) {
    for (int n : {1, 4, 8}) {
      HtDenseArgs a{dcodes, dv, dv, dx, dout, dpart, dcnt, r.rows, r.cols, n, r.cols, r.rows};
      const int iters = r.rows > 100000 ? 20 : 200;
      for (int i = 0; i < 5; ++i) (void)ht_dense_gemv(&a, nullptr);
      HIP_OK(hipEventRecord(e0, nullptr));
      for (int i = 0; i < iters; ++i) (void)ht_dense_gemv(&a, nullptr);
      HIP_OK(hipEventRecord(e1, nullptr));
      HIP_OK(hipEventSynchronize(e1));
      float ms = 0.0f;
      HIP_OK(hipEventElapsedTime(&ms, e0, e1));
      const double us = 1000.0 * ms / iters;
      const double gbs = (static_cast<double>(r.rows) * r.cols / 2) / (us * 1e3);
      std::printf("TIME %-12s %6dx%-5d n=%d splits=%-2d %9.1f us %6.1f GB/s\n", r.name,
                  r.rows, r.cols, n, ht_dense_splits(r.rows, r.cols), us, gbs);
    }
  }
  // Grouped against separate launches at the Flash-Next shared-input sets
  // (one token; the matrices share one code buffer, so this compares launch
  // structure, not DRAM bandwidth).
  const std::vector<std::pair<const char*, std::vector<int>>> sets = {
      {"linear qkv+gate", {10240, 6144}}, {"attn q+k+v", {12288, 512, 512}},
      {"shexp gate+up", {640, 640}}};
  for (const auto& [name, rows] : sets) {
    HtGroupArgs g{};
    for (int r : rows) g.m[g.count++] = HtMatrix{dcodes, dv, dv, dout, r, r};
    g.x = dx;
    g.cols = 2560;
    g.n_tokens = 1;
    g.x_stride = 2560;
    g.partials = dpart;
    g.counters = dcnt;
    g.partial_capacity = (16u << 20) / sizeof(float);
    g.counter_capacity = 4096;
    const auto time = [&](auto&& launch) {
      for (int i = 0; i < 5; ++i) launch();
      HIP_OK(hipEventRecord(e0, nullptr));
      for (int i = 0; i < 200; ++i) launch();
      HIP_OK(hipEventRecord(e1, nullptr));
      HIP_OK(hipEventSynchronize(e1));
      float ms = 0.0f;
      HIP_OK(hipEventElapsedTime(&ms, e0, e1));
      return 1000.0 * ms / 200;
    };
    const double grouped = time([&] { (void)ht_dense_group(&g, nullptr); });
    const double separate = time([&] {
      for (int i = 0; i < g.count; ++i) {
        HtDenseArgs a{dcodes, dv, dv, dx, dout, dpart, dcnt, rows[i], 2560, 1, 2560, rows[i]};
        (void)ht_dense_gemv(&a, nullptr);
      }
    });
    std::printf("TIME group %-16s grouped %7.1f us  separate %7.1f us\n", name, grouped, separate);
  }
  return failures == 0 ? 0 : 1;
}
