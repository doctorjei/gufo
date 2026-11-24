// GPU check of q6g64_dense_gemv against the CPU decoder (hgn_decode) for
// random q6g64 payloads, plus a per-token timing of the Flash-Next
// hyper-connection shapes (48 layers x 2 mixers x down/up, distinct buffers
// so the cache does not flatter the result). Exit 0 = pass.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "src/core/quant/hgn_decode.hpp"
#include "src/core/quant/q6g64_dense.hpp"

namespace {

#define HIP_OK(x)                                                         \
  do {                                                                    \
    hipError_t e_ = (x);                                                  \
    if (e_ != hipSuccess) {                                               \
      std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_),       \
                  __FILE__, __LINE__);                                    \
      std::exit(2);                                                       \
    }                                                                     \
  } while (0)

std::size_t RowBytes(int K) { return (static_cast<std::size_t>(13) * K / 16 + 15) / 16 * 16; }

std::vector<std::uint8_t> RandomPayload(int N, int K, std::mt19937& rng) {
  const std::size_t S = RowBytes(K);
  std::vector<std::uint8_t> p(static_cast<std::size_t>(N) * S, 0);
  std::uniform_real_distribution<float> sc(0.0005f, 0.004f), bi(-0.1f, 0.0f);
  for (int r = 0; r < N; ++r) {
    std::uint8_t* row = p.data() + static_cast<std::size_t>(r) * S;
    for (int i = 0; i < 3 * K / 4; ++i) row[i] = static_cast<std::uint8_t>(rng());
    for (int g = 0; g < K / 64; ++g) {
      float s = sc(rng), b = bi(rng);
      std::uint16_t hs, hb;
      gufo::quant::FloatToHalfBits(&s, &hs, 1);
      gufo::quant::FloatToHalfBits(&b, &hb, 1);
      std::memcpy(row + 3 * K / 4 + 4 * g, &hs, 2);
      std::memcpy(row + 3 * K / 4 + 4 * g + 2, &hb, 2);
    }
  }
  return p;
}

template <typename T>
T* Up(const std::vector<T>& h) {
  T* d = nullptr;
  HIP_OK(hipMalloc(&d, h.size() * sizeof(T) + 4096));
  HIP_OK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
  return d;
}

double RunCase(int N, int K, int n, std::mt19937& rng) {
  const auto pay = RandomPayload(N, K, rng);
  std::vector<float> w(static_cast<std::size_t>(N) * K);
  std::string err;
  if (!gufo::quant::DecodeHgnToF32(gufo::core::GgmlType::kQ6G64, pay.data(), K, N,
                                   nullptr, nullptr, w.data(), &err)) {
    std::printf("CPU decode failed: %s\n", err.c_str());
    std::exit(2);
  }
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x(static_cast<std::size_t>(n) * K);
  for (auto& v : x) v = nd(rng);
  auto* dw = Up(pay);
  auto* dx = Up(x);
  float* dout = nullptr;
  HIP_OK(hipMalloc(&dout, static_cast<std::size_t>(n) * N * sizeof(float)));
  std::size_t pf = 0, nc = 0;
  q6g64_dense_scratch(N, K, &pf, &nc);
  float* dpart = nullptr;
  unsigned* dcnt = nullptr;
  HIP_OK(hipMalloc(&dpart, (pf + 1) * sizeof(float)));
  HIP_OK(hipMalloc(&dcnt, (nc + 1) * sizeof(unsigned)));
  HIP_OK(hipMemset(dcnt, 0, (nc + 1) * sizeof(unsigned)));
  Q6g64DenseArgs a{dw, dx, dout, N, K, n, K, N, dpart, dcnt, pf, nc};
  for (int rep = 0; rep < 2; ++rep)  // the second run checks counter reset
    if (!q6g64_dense_gemv(&a, nullptr)) {
      std::printf("launch refused\n");
      std::exit(2);
    }
  HIP_OK(hipDeviceSynchronize());
  std::vector<float> y(static_cast<std::size_t>(n) * N);
  HIP_OK(hipMemcpy(y.data(), dout, y.size() * sizeof(float), hipMemcpyDeviceToHost));
  double worst = 0.0;
  for (int t = 0; t < n; ++t) {
    double peak = 0.0, emax = 0.0;
    for (int r = 0; r < N; ++r) {
      double ref = 0.0;
      for (int k = 0; k < K; ++k)
        ref += static_cast<double>(w[static_cast<std::size_t>(r) * K + k]) * x[static_cast<std::size_t>(t) * K + k];
      peak = std::max(peak, std::fabs(ref));
      emax = std::max(emax, std::fabs(ref - y[static_cast<std::size_t>(t) * N + r]));
    }
    worst = std::max(worst, emax / peak);
  }
  (void)hipFree(dw);
  (void)hipFree(dx);
  (void)hipFree(dout);
  (void)hipFree(dpart);
  (void)hipFree(dcnt);
  return worst;
}

}  // namespace

int main() {
  std::mt19937 rng(77);
  int failures = 0;
  constexpr double kGate = 1e-5;  // f32 accumulation of exact values
  const struct {
    int N, K;
  } shapes[] = {{320, 10240}, {10240, 320}, {4, 10240}, {100, 640}, {33, 2048}};
  for (const auto& s : shapes) {
    for (int n : {1, 2, 3, 5, 8}) {
      const double e = RunCase(s.N, s.K, n, rng);
      const bool ok = e < kGate;
      std::printf("%s rows=%d cols=%d n=%d err=%.3e\n", ok ? "PASS" : "FAIL", s.N, s.K, n, e);
      failures += ok ? 0 : 1;
    }
  }
  std::printf("%s\n", failures == 0 ? "ALL PASS" : "FAILURES");

  // Per-token timing: 96 mixers x (down 320x10240 + up 10240x320) + the
  // final mixer, each its own buffer.
  const int kMixers = 97;
  std::vector<std::uint8_t*> downs, ups;
  for (int i = 0; i < kMixers; ++i) {
    downs.push_back(Up(RandomPayload(320, 10240, rng)));
    ups.push_back(Up(RandomPayload(10240, 320, rng)));
  }
  std::vector<float> xs(8 * 10240, 0.25f);
  auto* dx = Up(xs);
  float *dlo = nullptr, *dhi = nullptr;
  HIP_OK(hipMalloc(&dlo, 8 * 320 * sizeof(float)));
  HIP_OK(hipMalloc(&dhi, 8 * 10240 * sizeof(float)));
  float* tpart = nullptr;
  unsigned* tcnt = nullptr;
  HIP_OK(hipMalloc(&tpart, 4u << 20));
  HIP_OK(hipMalloc(&tcnt, 4096 * sizeof(unsigned)));
  HIP_OK(hipMemset(tcnt, 0, 4096 * sizeof(unsigned)));
  hipEvent_t e0, e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  const double bytes = kMixers * (320.0 * RowBytes(10240) + 10240.0 * RowBytes(320));
  for (int n : {1, 4, 8}) {
    for (int rep = 0; rep < 2; ++rep) {
      HIP_OK(hipEventRecord(e0, nullptr));
      for (int it = 0; it < 10; ++it) {
        for (int i = 0; i < kMixers; ++i) {
          Q6g64DenseArgs d{downs[i], dx, dlo, 320, 10240, n, 10240, 320, tpart, tcnt, (4u << 20) / 4, 4096};
          Q6g64DenseArgs u{ups[i], dlo, dhi, 10240, 320, n, 320, 10240, tpart, tcnt, (4u << 20) / 4, 4096};
          (void)q6g64_dense_gemv(&d, nullptr);
          (void)q6g64_dense_gemv(&u, nullptr);
        }
      }
      HIP_OK(hipEventRecord(e1, nullptr));
      HIP_OK(hipEventSynchronize(e1));
      float ms = 0.0f;
      HIP_OK(hipEventElapsedTime(&ms, e0, e1));
      if (rep == 1)
        std::printf("TIME hc mixers per token n=%d: %.3f ms for %.1f MB (%.1f GB/s)\n", n,
                    ms / 10, bytes / 1e6, bytes / (ms / 10 * 1e6));
    }
  }
  return failures == 0 ? 0 : 1;
}
