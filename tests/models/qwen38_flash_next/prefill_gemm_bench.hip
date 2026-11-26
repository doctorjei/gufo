// Prefill dense GEMM microbenchmark (manual; needs a GPU): out[n][m] =
// x[n][k] . W[m][k]^T with f16 W and x and f32 out, at the v2 prefill shapes
// (ht and q6g64 matrices decoded to f16) for one 2,304-token chunk. Times the
// engine's own kernel (UnquantizedF16Gemm, one K order at any token position)
// against hipBLASLt's heuristic candidates with no workspace and with a
// 64 MiB workspace (the best of up to 8 each), and checks each library result
// against the engine kernel's.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {

#define HIP_OK(x)                                                                    \
  do {                                                                               \
    hipError_t e_ = (x);                                                             \
    if (e_ != hipSuccess) {                                                          \
      std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(2);                                                                  \
    }                                                                                \
  } while (0)

struct Shape {
  const char* name;
  int m, k;
  int per_token;  // launches of this shape per token (for a per-token total)
};

}  // namespace

int main() {
  int devs = 0;
  if (hipGetDeviceCount(&devs) != hipSuccess || devs == 0) {
    std::printf("SKIP: no GPU\n");
    return 77;
  }
  const int n = 2304;
  // Per-layer v2 dense shapes (36 linear-attention, 12 full-attention layers;
  // hyper-connection down/up twice per layer).
  const Shape shapes[] = {{"linear qkv", 10240, 2560, 36}, {"linear gate", 6144, 2560, 36},
                          {"attn q+gate", 12288, 2560, 12}, {"attn k", 512, 2560, 12},
                          {"attn v", 512, 2560, 12},        {"out proj", 2560, 6144, 48},
                          {"shexp gate", 640, 2560, 48},    {"shexp up", 640, 2560, 48},
                          {"shexp down", 2560, 640, 48},    {"hc down", 320, 10240, 97},
                          {"hc up", 10240, 320, 97}};
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  hipblasLtHandle_t lt = nullptr;
  if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) {
    std::printf("hipBLASLt init failed\n");
    return 2;
  }
  hipEvent_t e0, e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  const bool cold = std::getenv("PGB_COLD") != nullptr && std::getenv("PGB_COLD")[0] == '1';
  void* flush = nullptr;
  const std::size_t flush_bytes = 256u << 20;
  HIP_OK(hipMalloc(&flush, flush_bytes));
  std::printf("mode: %s\n", cold ? "cold (256 MiB written before each call)" : "warm (back to back)");
  void* ws = nullptr;
  const std::size_t ws_bytes = 64u << 20;
  HIP_OK(hipMalloc(&ws, ws_bytes));
  double tot_own = 0, tot_lib0 = 0, tot_libw = 0;
  for (const Shape& s : shapes) {
    std::vector<__half> w(static_cast<std::size_t>(s.m) * s.k), x(static_cast<std::size_t>(n) * s.k);
    for (auto& v : w) v = __float2half(0.02f * nd(rng));
    for (auto& v : x) v = __float2half(nd(rng));
    __half *dw = nullptr, *dx = nullptr;
    float *d_own = nullptr, *d_lib = nullptr;
    HIP_OK(hipMalloc(&dw, w.size() * 2));
    HIP_OK(hipMalloc(&dx, x.size() * 2));
    HIP_OK(hipMalloc(&d_own, static_cast<std::size_t>(n) * s.m * 4));
    HIP_OK(hipMalloc(&d_lib, static_cast<std::size_t>(n) * s.m * 4));
    HIP_OK(hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    const auto time = [&](auto&& fn) {
      for (int i = 0; i < 3; ++i) fn();
      if (cold) {
        // PGB_COLD=1: each call timed alone after writing a buffer larger than
        // the last-level cache, so weights and inputs start from memory.
        float sum = 0;
        for (int i = 0; i < 10; ++i) {
          HIP_OK(hipMemsetAsync(flush, i & 0xFF, flush_bytes, nullptr));
          HIP_OK(hipEventRecord(e0, nullptr));
          fn();
          HIP_OK(hipEventRecord(e1, nullptr));
          HIP_OK(hipEventSynchronize(e1));
          float ms = 0;
          HIP_OK(hipEventElapsedTime(&ms, e0, e1));
          sum += ms;
        }
        return sum / 10;
      }
      HIP_OK(hipEventRecord(e0, nullptr));
      for (int i = 0; i < 10; ++i) fn();
      HIP_OK(hipEventRecord(e1, nullptr));
      HIP_OK(hipEventSynchronize(e1));
      float ms = 0;
      HIP_OK(hipEventElapsedTime(&ms, e0, e1));
      return ms / 10;
    };
    using gufo::models::qwen38_flash_next::rocm::UnquantizedF16Gemm;
    const double t_own =
        time([&] { (void)UnquantizedF16Gemm(dw, dx, d_own, n, s.m, s.k, nullptr); });
    std::vector<float> ref(static_cast<std::size_t>(n) * s.m), got(ref.size());
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipMemcpy(ref.data(), d_own, ref.size() * 4, hipMemcpyDeviceToHost));
    double peak = 0;
    for (float v : ref) peak = std::max(peak, std::fabs(static_cast<double>(v)));
    // hipBLASLt: out (m x n, col-major) = W^T (m x k) . x (k x n).
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t la, lb, lc;
    const hipblasOperation_t tA = HIPBLAS_OP_T, tB = HIPBLAS_OP_N;
    hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
    hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
    hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, s.k, s.m, s.k);
    hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, s.k, n, s.k);
    hipblasLtMatrixLayoutCreate(&lc, HIP_R_32F, s.m, n, s.m);
    const float one = 1.0f, zero = 0.0f;
    double best[2] = {1e30, 1e30}, err_at_best[2] = {0, 0};
    for (int wsi = 0; wsi < 2; ++wsi) {
      const std::size_t limit = wsi == 0 ? 0 : ws_bytes;
      hipblasLtMatmulPreference_t pref = nullptr;
      hipblasLtMatmulPreferenceCreate(&pref);
      hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit,
                                            sizeof(limit));
      hipblasLtMatmulHeuristicResult_t cand[8];
      int count = 0;
      hipblasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 8, cand, &count);
      hipblasLtMatmulPreferenceDestroy(pref);
      for (int c = 0; c < count; ++c) {
        if (cand[c].state != HIPBLAS_STATUS_SUCCESS || cand[c].workspaceSize > limit) continue;
        const double t = time([&] {
          (void)hipblasLtMatmul(lt, op, &one, dw, la, dx, lb, &zero, d_lib, lc, d_lib, lc,
                                &cand[c].algo, ws, cand[c].workspaceSize, nullptr);
        });
        if (t < best[wsi]) {
          HIP_OK(hipDeviceSynchronize());
          HIP_OK(hipMemcpy(got.data(), d_lib, got.size() * 4, hipMemcpyDeviceToHost));
          double e = 0;
          for (std::size_t i = 0; i < got.size(); ++i)
            e = std::max(e, std::fabs(static_cast<double>(got[i]) - ref[i]));
          best[wsi] = t;
          err_at_best[wsi] = e / peak;
        }
      }
    }
    const double flop = 2.0 * n * s.m * s.k;
    std::printf("GEMM %-12s %5dx%-5d n=%d  own %7.3f ms (%5.1f TF/s)  lib ws0 %7.3f ms (%5.1f TF/s, err %.1e)"
                "  lib ws64M %7.3f ms (%5.1f TF/s, err %.1e)\n",
                s.name, s.m, s.k, n, t_own, flop / t_own / 1e9, best[0], flop / best[0] / 1e9,
                err_at_best[0], best[1], flop / best[1] / 1e9, err_at_best[1]);
    tot_own += t_own * s.per_token;
    tot_lib0 += std::min(t_own, best[0]) * s.per_token;
    tot_libw += std::min(t_own, best[1]) * s.per_token;
    hipblasLtMatrixLayoutDestroy(la);
    hipblasLtMatrixLayoutDestroy(lb);
    hipblasLtMatrixLayoutDestroy(lc);
    hipblasLtMatmulDescDestroy(op);
    (void)hipFree(dw);
    (void)hipFree(dx);
    (void)hipFree(d_own);
    (void)hipFree(d_lib);
  }
  std::printf("TOTAL per prompt token (all layers, n=%d): own %.4f ms; best of own/lib-ws0 %.4f ms; "
              "best of own/lib-ws64M %.4f ms\n",
              n, tot_own / n, tot_lib0 / n, tot_libw / n);
  hipblasLtDestroy(lt);
  return 0;
}
