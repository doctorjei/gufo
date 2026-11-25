// Microbenchmark and real-geometry cross-check for the batched I4R128 MoE
// decode kernels (i4r_fused.hpp) against the unbatched host-routed dispatch
// they replace (i4r_launch.hpp, the sequence Executor::MoeExperts ran).
//
// Geometry defaults to the Flash-Next layer: hidden 2560, expert ff 640,
// 10 experts per token. Synthetic random weights (uniform codes, random f16
// scales, +-1 rotation sign vectors) over several independent layers so the
// timed loop does not run out of the GPU's caches; expert ids are drawn at
// random per step and live on the device.
//
// Usage: i4r_fused_bench [experts_per_layer=128] [layers=8] [iters=480]
// Prints one line per measurement and a final JSON line (prefix "JSON ").

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "src/core/quant/i4r_fused.hpp"
#include "src/core/quant/i4r_launch.hpp"

namespace {

constexpr int kHidden = 2560;
constexpr int kFf = 640;
constexpr int kUsed = 10;

#define HIP_OK(expr)                                                       \
  do {                                                                     \
    const hipError_t e_ = (expr);                                          \
    if (e_ != hipSuccess) {                                                \
      std::fprintf(stderr, "HIP error %s at %s:%d: %s\n",                  \
                   hipGetErrorString(e_), __FILE__, __LINE__, #expr);      \
      std::exit(2);                                                        \
    }                                                                      \
  } while (0)

__device__ std::uint32_t Hash(std::uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return static_cast<std::uint32_t>(x);
}

// Fills n_blocks interleaved 66-byte I4R128 blocks.
__global__ void FillBlocks(std::uint8_t* dst, std::size_t n_blocks, std::uint64_t seed) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_blocks) return;
  std::uint8_t* b = dst + i * 66;
  for (int k = 0; k < 64; k += 4) {
    const std::uint32_t h = Hash(seed ^ (i * 64 + k));
    b[k] = h & 0xFF;
    b[k + 1] = (h >> 8) & 0xFF;
    b[k + 2] = (h >> 16) & 0xFF;
    b[k + 3] = (h >> 24) & 0xFF;
  }
  const std::uint32_t h = Hash(seed * 31 + i);
  const float sc = (0.004f + 0.004f * static_cast<float>(h & 0xFFFF) / 65535.0f) *
                   ((h >> 16) & 1 ? -1.0f : 1.0f);
  const __half hs = __float2half(sc);
  const std::uint16_t bits = __half_as_ushort(hs);
  b[64] = bits & 0xFF;
  b[65] = bits >> 8;
}

// Fills n_blocks Q8_0 blocks (f16 scale + 32 int8).
__global__ void FillQ8(std::uint8_t* dst, std::size_t n_blocks, std::uint64_t seed) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_blocks) return;
  std::uint8_t* b = dst + i * 34;
  const std::uint32_t h = Hash(seed * 17 + i);
  const __half hs = __float2half(0.0005f + 0.0005f * static_cast<float>(h & 0xFFFF) / 65535.0f);
  const std::uint16_t bits = __half_as_ushort(hs);
  b[0] = bits & 0xFF;
  b[1] = bits >> 8;
  for (int k = 0; k < 32; ++k) b[2 + k] = Hash(seed ^ (i * 32 + k + 0x9e37)) & 0xFF;
}

__global__ void SwigluKernel(float* g, const float* u, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) g[i] = g[i] / (1.0f + __expf(-g[i])) * u[i];
}

__global__ void EpilogueKernel(const float* down, const float* w, const float* shared,
                               const float* gate, float* out, int k, int dim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  float acc = 0.0f;
  for (int s = 0; s < k; ++s) acc += w[s] * down[s * dim + i];
  out[i] = acc + shared[i] / (1.0f + __expf(-gate[0]));
}

struct Layer {
  std::uint8_t* gate_old;
  std::uint8_t* up_old;
  std::uint8_t* down_old;
  std::uint8_t* gate_new;
  std::uint8_t* up_new;
  std::uint8_t* down_new;
  float* su_experts;
  float* sv_gate;
  float* sv_up;
  float* su_down;
  float* sv_down;
  std::uint8_t* shared_old;  // Q8_0 [hidden x ff], interleaved
  std::uint8_t* shared_new;  // repacked
};

float* UploadSigns(int n, std::mt19937& rng) {
  std::vector<float> v(n);
  for (auto& x : v) x = (rng() & 1) ? 1.0f : -1.0f;
  float* d = nullptr;
  HIP_OK(hipMalloc(&d, n * sizeof(float)));
  HIP_OK(hipMemcpy(d, v.data(), n * sizeof(float), hipMemcpyHostToDevice));
  return d;
}

std::uint8_t* MakeTensor(std::size_t n_blocks, std::uint64_t seed) {
  std::uint8_t* d = nullptr;
  HIP_OK(hipMalloc(&d, n_blocks * 66));
  FillBlocks<<<static_cast<unsigned>((n_blocks + 255) / 256), 256>>>(d, n_blocks, seed);
  HIP_OK(hipGetLastError());
  return d;
}

std::uint8_t* Repacked(const std::uint8_t* src, std::size_t n_blocks) {
  std::uint8_t* d = nullptr;
  HIP_OK(hipMalloc(&d, n_blocks * 66));
  if (!i4r_repack_split(src, d, n_blocks, nullptr)) std::exit(2);
  return d;
}

void FreeMemLine(const char* when) {
  std::size_t free_b = 0, total_b = 0;
  (void)hipMemGetInfo(&free_b, &total_b);
  std::printf("heartbeat %s: device free %.1f GiB of %.1f GiB\n", when,
              free_b / 1073741824.0, total_b / 1073741824.0);
  std::fflush(stdout);
}

struct Scratch {
  float* x;          // [hidden]
  float* x_rot;      // [hidden]
  float* gate_e;     // [used x ff]
  float* up_e;       // [used x ff]
  float* down_e;     // [used x hidden]
  float* rot_a;      // [ff]
  float* rot_b;      // [ff]
  float* rot_c;      // [hidden]
  float* weights;    // [used]
  float* shared;     // [hidden]
  float* gate_logit; // [1]
  float* out;        // [hidden]
  unsigned* cnt_up;
  unsigned* cnt_down;
  unsigned* cnt_shared;
  float* shared_in;    // [ff] shared SwiGLU output (input of shared down)
  float* shared_part;  // [ff/128 x hidden]
  std::int32_t* ids_host;  // pinned
};

// The unbatched host-routed sequence, as Executor::MoeExperts ran it for one
// decode token. Returns the number of kernel launches issued.
int OldLayer(const Layer& L, const Scratch& s, const std::int32_t* ids_dev,
             hipStream_t st) {
  const int nb_h = kHidden / 128, nb_f = kFf / 128;
  const std::size_t gate_stride = std::size_t{kFf} * nb_h * 66;
  const std::size_t down_stride = std::size_t{kHidden} * nb_f * 66;
  int launches = 0;
  i4r_launch_rotate(s.x, s.x_rot, L.su_experts, nullptr, nb_h, st), ++launches;
  HIP_OK(hipMemcpyAsync(s.ids_host, ids_dev, kUsed * sizeof(std::int32_t),
                        hipMemcpyDeviceToHost, st));
  HIP_OK(hipStreamSynchronize(st));
  for (int sl = 0; sl < kUsed; ++sl) {
    const std::int32_t e = s.ids_host[sl];
    i4r_launch_mmvq(L.gate_old + e * gate_stride, s.x_rot, s.rot_a, kFf, nb_h, st);
    i4r_launch_mmvq(L.up_old + e * gate_stride, s.x_rot, s.rot_b, kFf, nb_h, st);
    i4r_launch_rotate(s.rot_a, s.gate_e + sl * kFf, nullptr, L.sv_gate, nb_f, st);
    i4r_launch_rotate(s.rot_b, s.up_e + sl * kFf, nullptr, L.sv_up, nb_f, st);
    launches += 4;
  }
  SwigluKernel<<<(kUsed * kFf + 255) / 256, 256, 0, st>>>(s.gate_e, s.up_e, kUsed * kFf);
  ++launches;
  for (int sl = 0; sl < kUsed; ++sl) {
    const std::int32_t e = s.ids_host[sl];
    i4r_launch_rotate(s.gate_e + sl * kFf, s.rot_a, L.su_down, nullptr, nb_f, st);
    i4r_launch_mmvq(L.down_old + e * down_stride, s.rot_a, s.rot_c, kHidden, nb_f, st);
    i4r_launch_rotate(s.rot_c, s.down_e + sl * kHidden, nullptr, L.sv_down, nb_h, st);
    launches += 3;
  }
  EpilogueKernel<<<(kHidden + 255) / 256, 256, 0, st>>>(s.down_e, s.weights, s.shared,
                                                       s.gate_logit, s.out, kUsed, kHidden);
  ++launches;
  return launches;
}

int NewLayer(const Layer& L, const Scratch& s, const std::int32_t* ids_dev,
             std::size_t n_experts, hipStream_t st, bool per_slot, bool plain,
             bool do_gate_up = true, bool do_down = true, bool shared = false) {
  int launches = 0;
  if (do_gate_up) {
    I4rGateUpArgs gu{};
    gu.gate = L.gate_new;
    gu.up = L.up_new;
    gu.n_experts = n_experts;
    gu.x = s.x;
    gu.ids = ids_dev;
    gu.su_experts = L.su_experts;
    gu.sv_gate = L.sv_gate;
    gu.sv_up = L.sv_up;
    gu.su_down = L.su_down;
    gu.gate_part = s.gate_e;
    gu.up_part = s.up_e;
    gu.counters = s.cnt_up;
    gu.hidden = kHidden;
    gu.ff = kFf;
    gu.used = kUsed;
    gu.plain_f16 = plain ? 1 : 0;
    gu.n_tokens = 1;
    if (shared) {
      gu.shared_down = L.shared_new;
      gu.shared_in = s.shared_in;
      gu.shared_part = s.shared_part;
      gu.shared_out = s.shared;
      gu.shared_counters = s.cnt_shared;
      gu.shared_ff = kFf;
    }
    if (!i4r_fused_gate_up(&gu, kUsed, st)) std::exit(3);
    ++launches;
  }
  if (do_down) {
    I4rDownArgs dn{};
    dn.down = L.down_new;
    dn.n_experts = n_experts;
    dn.down_in = s.gate_e;
    dn.ids = ids_dev;
    dn.weights = s.weights;
    dn.sv_down = L.sv_down;
    dn.shared_out = s.shared;
    dn.shared_gate = s.gate_logit;
    dn.shared_gate_stride = 1;
    dn.down_part = s.down_e;
    dn.out = s.out;
    dn.counters = s.cnt_down;
    dn.hidden = kHidden;
    dn.ff = kFf;
    dn.used = kUsed;
    dn.per_slot_out = per_slot ? 1 : 0;
    dn.plain_f16 = plain ? 1 : 0;
    if (!i4r_fused_down(&dn, 1, st)) std::exit(3);
    ++launches;
  }
  return launches;
}

// A verify-style batch: n_tokens tokens, each with kUsed experts, through the
// two batched launches (no shared expert). Buffers sized for n_tokens.
struct Batch {
  float* x;       // [T x hidden]
  float* gate_e;  // [T*used x ff]
  float* up_e;
  float* down_e;  // [T*used x hidden]
  float* weights; // [T*used]
  float* out;     // [T x hidden]
  unsigned* cnt_up;
  unsigned* cnt_down;
};

struct Group {
  std::int32_t* offsets = nullptr;
  std::int32_t* cursor = nullptr;
  std::int32_t* slots = nullptr;
  unsigned* counters = nullptr;
  std::uint16_t* xr_hi = nullptr;
  std::uint16_t* xr_lo = nullptr;
  void* items = nullptr;  // prefill v2 tile list
  std::int32_t* n_items = nullptr;
  const float* su = nullptr;
};

void RunBatch(const Layer& L, const Batch& b, const std::int32_t* ids, const float* x,
              const float* weights, float* out, int n_tokens, std::size_t n_experts,
              hipStream_t st, const Group* grp = nullptr) {
  if (grp != nullptr &&
      (!i4r_group_slots(ids, n_tokens * kUsed, static_cast<int>(n_experts), grp->offsets,
                        grp->cursor, grp->slots, st) ||
       !i4r_rotate_expert_input(x, L.su_experts, grp->xr_hi, grp->xr_lo, n_tokens, kHidden, st) ||
       (grp->items != nullptr &&
        !i4r_prefill_tiles(grp->offsets, static_cast<int>(n_experts), grp->items, grp->n_items, st))))
    std::exit(3);
  I4rGateUpArgs gu{};
  gu.gate = L.gate_new;
  gu.up = L.up_new;
  gu.n_experts = n_experts;
  gu.x = x;
  gu.ids = ids;
  gu.su_experts = L.su_experts;
  gu.sv_gate = L.sv_gate;
  gu.sv_up = L.sv_up;
  gu.su_down = L.su_down;
  gu.gate_part = b.gate_e;
  gu.up_part = b.up_e;
  gu.counters = b.cnt_up;
  gu.hidden = kHidden;
  gu.ff = kFf;
  gu.used = kUsed;
  gu.n_tokens = n_tokens;
  if (grp != nullptr) {
    gu.group_offsets = grp->offsets;
    gu.group_slots = grp->slots;
    gu.group_counters = grp->counters;
    gu.group_xr_hi = grp->xr_hi;
    gu.group_xr_lo = grp->xr_lo;
    gu.group_items = grp->items;
    gu.group_n_items = grp->n_items;
    gu.group_max_items = i4r_prefill_max_tiles(n_tokens * kUsed, static_cast<int>(n_experts));
  }
  if (!i4r_fused_gate_up(&gu, n_tokens * kUsed, st)) std::exit(3);
  I4rDownArgs dn{};
  dn.down = L.down_new;
  dn.n_experts = n_experts;
  dn.down_in = b.gate_e;
  dn.ids = ids;
  dn.weights = weights;
  dn.sv_down = L.sv_down;
  dn.down_part = b.down_e;
  dn.out = out;
  dn.counters = b.cnt_down;
  dn.hidden = kHidden;
  dn.ff = kFf;
  dn.used = kUsed;
  if (grp != nullptr) {
    dn.group_offsets = grp->offsets;
    dn.group_slots = grp->slots;
    dn.group_items = grp->items;
    dn.group_n_items = grp->n_items;
    dn.group_max_items = i4r_prefill_max_tiles(n_tokens * kUsed, static_cast<int>(n_experts));
    dn.hgn_bf16 = std::getenv("I4R_HGN") != nullptr ? 1 : 0;  // I4R_HGN=1: bf16 slot outputs (halogen prefill store)
  }
  if (!i4r_fused_down(&dn, n_tokens, st)) std::exit(3);
}

std::vector<float> Download(const float* d, std::size_t n) {
  std::vector<float> h(n);
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipMemcpy(h.data(), d, n * sizeof(float), hipMemcpyDeviceToHost));
  return h;
}

double PeakRel(const std::vector<float>& a, const std::vector<float>& b, std::size_t off,
               std::size_t n) {
  double peak = 0.0, err = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    peak = std::max(peak, std::abs(static_cast<double>(b[off + i])));
    err = std::max(err, std::abs(static_cast<double>(a[off + i]) - b[off + i]));
  }
  return peak > 0.0 ? err / peak : (err > 0.0 ? 1e30 : 0.0);
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t n_exp = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 128;
  const int n_layers = argc > 2 ? std::atoi(argv[2]) : 8;
  const int iters = argc > 3 ? std::atoi(argv[3]) : 480;
  int dev_count = 0;
  if (hipGetDeviceCount(&dev_count) != hipSuccess || dev_count == 0) {
    std::printf("SKIP: no GPU\n");
    return 77;
  }
  hipDeviceProp_t prop{};
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  std::printf("device %s (%s), %d CUs; experts/layer %zu, layers %d, iters %d\n",
              prop.name, prop.gcnArchName, prop.multiProcessorCount, n_exp, n_layers,
              iters);
  FreeMemLine("start");

  std::mt19937 rng(1234);
  const std::size_t gu_blocks = n_exp * kFf * (kHidden / 128);
  const std::size_t dn_blocks = n_exp * kHidden * (kFf / 128);
  std::vector<Layer> layers(n_layers);
  for (int l = 0; l < n_layers; ++l) {
    Layer& L = layers[l];
    L.gate_old = MakeTensor(gu_blocks, 0x1000 + l * 3);
    L.up_old = MakeTensor(gu_blocks, 0x1001 + l * 3);
    L.down_old = MakeTensor(dn_blocks, 0x1002 + l * 3);
    L.gate_new = Repacked(L.gate_old, gu_blocks);
    L.up_new = Repacked(L.up_old, gu_blocks);
    L.down_new = Repacked(L.down_old, dn_blocks);
    L.su_experts = UploadSigns(kHidden, rng);
    L.sv_gate = UploadSigns(kFf, rng);
    L.sv_up = UploadSigns(kFf, rng);
    L.su_down = UploadSigns(kFf, rng);
    L.sv_down = UploadSigns(kHidden, rng);
    const std::size_t q8_blocks = std::size_t{kHidden} * (kFf / 32);
    HIP_OK(hipMalloc(&L.shared_old, q8_blocks * 34));
    FillQ8<<<static_cast<unsigned>((q8_blocks + 255) / 256), 256>>>(L.shared_old, q8_blocks,
                                                                     0x7000 + l);
    HIP_OK(hipMalloc(&L.shared_new, q8_blocks * 34));
    if (!i4r_repack_q8_split(L.shared_old, L.shared_new, q8_blocks, nullptr)) std::exit(2);
  }
  HIP_OK(hipDeviceSynchronize());
  FreeMemLine("weights ready");

  Scratch s{};
  auto falloc = [](std::size_t n) {
    float* p = nullptr;
    HIP_OK(hipMalloc(&p, n * sizeof(float)));
    HIP_OK(hipMemset(p, 0, n * sizeof(float)));
    return p;
  };
  s.x = falloc(kHidden);
  s.x_rot = falloc(kHidden);
  s.gate_e = falloc(kUsed * kFf);
  s.up_e = falloc(kUsed * kFf);
  s.down_e = falloc(kUsed * kHidden);
  s.rot_a = falloc(kFf);
  s.rot_b = falloc(kFf);
  s.rot_c = falloc(kHidden);
  s.weights = falloc(kUsed);
  s.shared = falloc(kHidden);
  s.gate_logit = falloc(1);
  s.out = falloc(kHidden);
  HIP_OK(hipMalloc(&s.cnt_up, kUsed * (kFf / 128) * sizeof(unsigned)));
  HIP_OK(hipMemset(s.cnt_up, 0, kUsed * (kFf / 128) * sizeof(unsigned)));
  HIP_OK(hipMalloc(&s.cnt_down, (kHidden / 128) * sizeof(unsigned)));
  HIP_OK(hipMemset(s.cnt_down, 0, (kHidden / 128) * sizeof(unsigned)));
  HIP_OK(hipMalloc(&s.cnt_shared, (kHidden / 128) * sizeof(unsigned)));
  HIP_OK(hipMemset(s.cnt_shared, 0, (kHidden / 128) * sizeof(unsigned)));
  s.shared_in = falloc(kFf);
  s.shared_part = falloc((kFf / 128) * kHidden);
  HIP_OK(hipHostMalloc(&s.ids_host, kUsed * sizeof(std::int32_t)));

  {
    std::vector<float> x(kHidden), sh(kHidden), w(kUsed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = nd(rng);
    for (auto& v : sh) v = 0.1f * nd(rng);
    float wsum = 0.0f;
    for (auto& v : w) wsum += (v = 0.2f + std::abs(nd(rng)));
    for (auto& v : w) v /= wsum;
    const float gl = 0.37f;
    HIP_OK(hipMemcpy(s.x, x.data(), kHidden * sizeof(float), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(s.shared, sh.data(), kHidden * sizeof(float), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(s.weights, w.data(), kUsed * sizeof(float), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(s.gate_logit, &gl, sizeof(float), hipMemcpyHostToDevice));
  }

  // Random distinct expert ids, many sets, on the device.
  constexpr int kSets = 1024;
  std::int32_t* ids_all = nullptr;
  {
    std::vector<std::int32_t> ids(kSets * kUsed);
    std::vector<std::int32_t> perm(n_exp);
    for (std::size_t i = 0; i < n_exp; ++i) perm[i] = static_cast<std::int32_t>(i);
    for (int set = 0; set < kSets; ++set) {
      std::shuffle(perm.begin(), perm.end(), rng);
      for (int k = 0; k < kUsed; ++k) ids[set * kUsed + k] = perm[k];
    }
    HIP_OK(hipMalloc(&ids_all, ids.size() * sizeof(std::int32_t)));
    HIP_OK(hipMemcpy(ids_all, ids.data(), ids.size() * sizeof(std::int32_t),
                     hipMemcpyHostToDevice));
  }

  hipStream_t st;
  HIP_OK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
  hipEvent_t e0, e1;
  HIP_OK(hipEventCreate(&e0));
  HIP_OK(hipEventCreate(&e1));
  int fails = 0;

  // ---- Real-geometry cross-check against the unbatched path ----
  double worst_slot[2] = {0, 0}, worst_comb[2] = {0, 0};
  for (int l = 0; l < std::min(n_layers, 3); ++l) {
    for (int set = 0; set < 2; ++set) {
      const std::int32_t* ids = ids_all + set * kUsed;
      OldLayer(layers[l], s, ids, st);
      const auto ref_slots = Download(s.down_e, kUsed * kHidden);
      const auto ref_out = Download(s.out, kHidden);
      for (int plain = 0; plain < 2; ++plain) {
        NewLayer(layers[l], s, ids, n_exp, st, true, plain != 0);
        const auto got_slots = Download(s.down_e, kUsed * kHidden);
        const auto got_out = Download(s.out, kHidden);
        for (int sl = 0; sl < kUsed; ++sl)
          worst_slot[plain] = std::max(worst_slot[plain],
                                       PeakRel(got_slots, ref_slots, sl * kHidden, kHidden));
        worst_comb[plain] = std::max(worst_comb[plain], PeakRel(got_out, ref_out, 0, kHidden));
      }
    }
  }
  // Negative control: a different routing must not match.
  OldLayer(layers[0], s, ids_all, st);
  const auto ref0 = Download(s.out, kHidden);
  NewLayer(layers[0], s, ids_all + kUsed, n_exp, st, false, false);
  const double neg = PeakRel(Download(s.out, kHidden), ref0, 0, kHidden);
  std::printf("cross-check (peak-normalised max error vs unbatched path):\n");
  std::printf("  split-f16 input: per-slot %.3e, combined %.3e\n", worst_slot[0], worst_comb[0]);
  std::printf("  plain-f16 input: per-slot %.3e, combined %.3e\n", worst_slot[1], worst_comb[1]);
  std::printf("  negative control (other routing): %.3e\n", neg);
  if (!(worst_slot[0] < 1e-4 && worst_comb[0] < 1e-4)) {
    std::printf("FAIL: split-input batched path exceeds 1e-4 against the unbatched path\n");
    ++fails;
  }
  if (!(neg > 1e-1)) {
    std::printf("FAIL: negative control did not separate\n");
    ++fails;
  }
  // Shared-expert range (addendum 1) against a host double reference.
  double shared_err[2] = {0, 0};
  {
    std::vector<float> in(kFf);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : in) v = nd(rng);
    HIP_OK(hipMemcpy(s.shared_in, in.data(), kFf * sizeof(float), hipMemcpyHostToDevice));
    const std::size_t q8_blocks = std::size_t{kHidden} * (kFf / 32);
    std::vector<std::uint8_t> w(q8_blocks * 34);
    HIP_OK(hipMemcpy(w.data(), layers[0].shared_old, w.size(), hipMemcpyDeviceToHost));
    std::vector<float> ref(kHidden);
    for (int r = 0; r < kHidden; ++r) {
      double acc = 0.0;
      for (int b = 0; b < kFf / 32; ++b) {
        const std::uint8_t* blk = &w[(static_cast<std::size_t>(r) * (kFf / 32) + b) * 34];
        const std::uint16_t bits = blk[0] | (blk[1] << 8);
        const double d = static_cast<double>(__half2float(__ushort_as_half(bits)));
        for (int k = 0; k < 32; ++k)
          acc += d * static_cast<std::int8_t>(blk[2 + k]) * in[b * 32 + k];
      }
      ref[r] = static_cast<float>(acc);
    }
    for (int plain = 0; plain < 2; ++plain) {
      HIP_OK(hipMemset(s.shared, 0, kHidden * sizeof(float)));
      NewLayer(layers[0], s, ids_all, n_exp, st, false, plain != 0, true, false, true);
      shared_err[plain] = PeakRel(Download(s.shared, kHidden), ref, 0, kHidden);
    }
    std::printf("shared-expert down range vs host double: split %.3e, plain %.3e\n",
                shared_err[0], shared_err[1]);
    if (!(shared_err[0] < 1e-4)) {
      std::printf("FAIL: shared-expert range exceeds 1e-4\n");
      ++fails;
    }
  }
  FreeMemLine("cross-check done");

  // ---- Verify batches: experts shared across tokens ----
  // 8 tokens, each drawing kUsed distinct experts from a pool. With a small
  // pool most experts are shared, and each expert's workgroups serve every
  // token that selected it. The batched output must equal, bit for bit, each
  // token run on its own.
  double t_batch_shared = 0, t_batch_distinct = 0, t_batch_one = 0;
  {
    constexpr int kT = 8;
    Batch b{};
    b.x = falloc(kT * kHidden);
    b.gate_e = falloc(kT * kUsed * kFf);
    b.up_e = falloc(kT * kUsed * kFf);
    b.down_e = falloc(kT * kUsed * kHidden);
    b.weights = falloc(kT * kUsed);
    b.out = falloc(kT * kHidden);
    HIP_OK(hipMalloc(&b.cnt_up, kT * kUsed * (kFf / 128) * sizeof(unsigned)));
    HIP_OK(hipMemset(b.cnt_up, 0, kT * kUsed * (kFf / 128) * sizeof(unsigned)));
    HIP_OK(hipMalloc(&b.cnt_down, kT * (kHidden / 128) * sizeof(unsigned)));
    HIP_OK(hipMemset(b.cnt_down, 0, kT * (kHidden / 128) * sizeof(unsigned)));
    float* ref = falloc(kT * kHidden);
    {
      std::vector<float> x(kT * kHidden), w(kT * kUsed);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (auto& v : x) v = nd(rng);
      for (auto& v : w) v = 0.05f + 0.1f * std::abs(nd(rng));
      HIP_OK(hipMemcpy(b.x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
      HIP_OK(hipMemcpy(b.weights, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice));
    }
    auto make_ids = [&](std::size_t pool) {
      std::vector<std::int32_t> ids(kT * kUsed), perm(pool);
      for (std::size_t i = 0; i < pool; ++i) perm[i] = static_cast<std::int32_t>(i);
      for (int t = 0; t < kT; ++t) {
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int k = 0; k < kUsed; ++k) ids[t * kUsed + k] = perm[k];
      }
      std::int32_t* d = nullptr;
      HIP_OK(hipMalloc(&d, ids.size() * sizeof(std::int32_t)));
      HIP_OK(hipMemcpy(d, ids.data(), ids.size() * sizeof(std::int32_t), hipMemcpyHostToDevice));
      std::vector<bool> seen(pool, false);
      int n = 0;
      for (auto e : ids) if (!seen[e]) { seen[e] = true; ++n; }
      return std::make_pair(d, n);
    };
    const std::size_t pool = std::min<std::size_t>(16, n_exp);
    auto [ids_shared, n_shared] = make_ids(pool);
    auto [ids_distinct, n_distinct] = make_ids(n_exp);
    int mism = 0;
    for (int which = 0; which < 2; ++which) {
      const std::int32_t* ids = which == 0 ? ids_shared : ids_distinct;
      for (int t = 0; t < kT; ++t)
        RunBatch(layers[0], b, ids + t * kUsed, b.x + t * kHidden, b.weights + t * kUsed,
                 ref + t * kHidden, 1, n_exp, st);
      const auto r = Download(ref, kT * kHidden);
      RunBatch(layers[0], b, ids, b.x, b.weights, b.out, kT, n_exp, st);
      const auto g = Download(b.out, kT * kHidden);
      int m = 0;
      for (std::size_t i = 0; i < r.size(); ++i) m += std::memcmp(&r[i], &g[i], sizeof(float)) != 0;
      std::printf("batch of %d, %s experts (%d distinct of %d slots): %d of %zu outputs differ from per-token runs\n",
                  kT, which == 0 ? "shared" : "mostly distinct", which == 0 ? n_shared : n_distinct,
                  kT * kUsed, m, r.size());
      mism += m;
    }
    if (mism != 0) {
      std::printf("FAIL: shared-expert batch is not bit-identical to per-token runs\n");
      ++fails;
    }
    auto time_batch = [&](const std::int32_t* ids, int n_tokens) {
      for (int i = 0; i < 16; ++i)
        RunBatch(layers[i % n_layers], b, ids, b.x, b.weights, b.out, n_tokens, n_exp, st);
      HIP_OK(hipEventRecord(e0, st));
      for (int i = 0; i < iters; ++i)
        RunBatch(layers[i % n_layers], b, ids, b.x, b.weights, b.out, n_tokens, n_exp, st);
      HIP_OK(hipEventRecord(e1, st));
      HIP_OK(hipEventSynchronize(e1));
      float ms = 0;
      HIP_OK(hipEventElapsedTime(&ms, e0, e1));
      return 1000.0 * ms / iters;
    };
    t_batch_one = time_batch(ids_distinct, 1);
    t_batch_distinct = time_batch(ids_distinct, kT);
    t_batch_shared = time_batch(ids_shared, kT);
    std::printf("TIME batch, per MoE layer: 1 token %.1f us; 8 tokens mostly distinct (%d experts) %.1f us; "
                "8 tokens shared (%d experts) %.1f us\n",
                t_batch_one, n_distinct, t_batch_distinct, n_shared, t_batch_shared);
  }

  // ---- Prefill chunks: slots grouped by expert ----
  // 256 and 1,024 tokens with random experts (about 5 and 20 slots per
  // expert at 512 experts: the dot-product and the matrix-core grouped
  // kernels), checked against per-token runs and timed against the per-slot
  // launches.
  for (const int kP : {256, 1024, 2304}) {
    Batch b{};
    b.x = falloc(kP * kHidden);
    b.gate_e = falloc(kP * kUsed * kFf);
    b.up_e = falloc(kP * kUsed * kFf);
    b.down_e = falloc(kP * kUsed * kHidden);
    b.weights = falloc(kP * kUsed);
    b.out = falloc(kP * kHidden);
    HIP_OK(hipMalloc(&b.cnt_up, kP * kUsed * (kFf / 128) * sizeof(unsigned)));
    HIP_OK(hipMemset(b.cnt_up, 0, kP * kUsed * (kFf / 128) * sizeof(unsigned)));
    HIP_OK(hipMalloc(&b.cnt_down, kP * (kHidden / 128) * sizeof(unsigned)));
    HIP_OK(hipMemset(b.cnt_down, 0, kP * (kHidden / 128) * sizeof(unsigned)));
    Group g;
    HIP_OK(hipMalloc(&g.offsets, (n_exp + 1) * sizeof(std::int32_t)));
    HIP_OK(hipMalloc(&g.cursor, n_exp * sizeof(std::int32_t)));
    HIP_OK(hipMalloc(&g.slots, kP * kUsed * sizeof(std::int32_t)));
    HIP_OK(hipMalloc(&g.counters, n_exp * (kFf / 128) * sizeof(unsigned)));
    HIP_OK(hipMemset(g.counters, 0, n_exp * (kFf / 128) * sizeof(unsigned)));
    HIP_OK(hipMalloc(&g.xr_hi, kP * kHidden * sizeof(std::uint16_t)));
    HIP_OK(hipMalloc(&g.xr_lo, kP * kHidden * sizeof(std::uint16_t)));
    HIP_OK(hipMalloc(&g.items, i4r_prefill_max_tiles(kP * kUsed, static_cast<int>(n_exp)) * 8));
    HIP_OK(hipMalloc(&g.n_items, sizeof(std::int32_t)));
    float* ref = falloc(kP * kHidden);
    std::vector<std::int32_t> ids(kP * kUsed), perm(n_exp);
    for (std::size_t i = 0; i < n_exp; ++i) perm[i] = static_cast<std::int32_t>(i);
    for (int t = 0; t < kP; ++t) {
      std::shuffle(perm.begin(), perm.end(), rng);
      for (int k = 0; k < kUsed; ++k) ids[t * kUsed + k] = perm[k];
    }
    std::int32_t* dids = nullptr;
    HIP_OK(hipMalloc(&dids, ids.size() * sizeof(std::int32_t)));
    HIP_OK(hipMemcpy(dids, ids.data(), ids.size() * sizeof(std::int32_t), hipMemcpyHostToDevice));
    {
      std::vector<float> x(kP * kHidden), w(kP * kUsed);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      for (auto& v : x) v = nd(rng);
      for (auto& v : w) v = 0.05f + 0.1f * std::abs(nd(rng));
      HIP_OK(hipMemcpy(b.x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
      HIP_OK(hipMemcpy(b.weights, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice));
    }
    for (int t = 0; t < kP; ++t)
      RunBatch(layers[0], b, dids + t * kUsed, b.x + t * kHidden, b.weights + t * kUsed,
               ref + t * kHidden, 1, n_exp, st);
    const auto r = Download(ref, kP * kHidden);
    // At 256 tokens the dot-product grouped kernels run (bit-identical); at
    // 1,024 the prefill v2 matrix-core kernels (single-pass f16 weights and
    // inputs, about 5e-4 of peak by design), so the gate is 2e-3.
    double worst = 0.0;
    for (int rep = 0; rep < 2; ++rep) {  // twice: counters must reset themselves
      RunBatch(layers[0], b, dids, b.x, b.weights, b.out, kP, n_exp, st, &g);
      const auto got = Download(b.out, kP * kHidden);
      for (int t = 0; t < kP; ++t)
        worst = std::max(worst, PeakRel(got, r, static_cast<std::size_t>(t) * kHidden, kHidden));
    }
    std::printf("prefill chunk of %d tokens, grouped by expert: worst peak-normalised error vs per-token runs %.3e\n",
                kP, worst);
    if (!(worst < 2e-3)) {
      std::printf("FAIL: grouped prefill exceeds 2e-3 against per-token runs\n");
      ++fails;
    }
    auto time_p = [&](const Group* grp) {
      for (int i = 0; i < 4; ++i)
        RunBatch(layers[i % n_layers], b, dids, b.x, b.weights, b.out, kP, n_exp, st, grp);
      const int it = std::max(8, iters / 20);
      HIP_OK(hipEventRecord(e0, st));
      for (int i = 0; i < it; ++i)
        RunBatch(layers[i % n_layers], b, dids, b.x, b.weights, b.out, kP, n_exp, st, grp);
      HIP_OK(hipEventRecord(e1, st));
      HIP_OK(hipEventSynchronize(e1));
      float ms = 0;
      HIP_OK(hipEventElapsedTime(&ms, e0, e1));
      return 1000.0 * ms / it;
    };
    const double t_slot = time_p(nullptr);
    const double t_group = time_p(&g);
    std::printf("TIME prefill chunk %d tokens, per MoE layer: per-slot %.1f us (%.1f us/token); grouped %.1f us (%.1f us/token)\n",
                kP, t_slot, t_slot / kP, t_group, t_group / kP);
  }

  // ---- Timing ----
  const double gu_bytes = kUsed * 2.0 * kFf * kHidden * 4.125 / 8.0;
  const double dn_bytes = kUsed * 1.0 * kHidden * kFf * 4.125 / 8.0;
  auto time_new = [&](bool plain, bool gu, bool dn, int* launches, bool sh = false) {
    for (int i = 0; i < 16; ++i)
      NewLayer(layers[i % n_layers], s, ids_all + (i % kSets) * kUsed, n_exp, st, false,
               plain, gu, dn, sh);
    HIP_OK(hipEventRecord(e0, st));
    int n = 0;
    for (int i = 0; i < iters; ++i)
      n = NewLayer(layers[i % n_layers], s, ids_all + ((i * 7) % kSets) * kUsed, n_exp, st,
                   false, plain, gu, dn, sh);
    HIP_OK(hipEventRecord(e1, st));
    HIP_OK(hipEventSynchronize(e1));
    float ms = 0;
    HIP_OK(hipEventElapsedTime(&ms, e0, e1));
    if (launches) *launches = n;
    return 1000.0 * ms / iters;  // us per layer
  };
  int new_launches = 0;
  const double t_new = time_new(false, true, true, &new_launches);
  const double t_new_gu = time_new(false, true, false, nullptr);
  const double t_new_dn = time_new(false, false, true, nullptr);
  const double t_plain = time_new(true, true, true, nullptr);
  const double t_plain_gu = time_new(true, true, false, nullptr);
  const double t_plain_dn = time_new(true, false, true, nullptr);
  const double t_sh = time_new(false, true, true, nullptr, true);
  const double t_sh_gu = time_new(false, true, false, nullptr, true);
  const double sh_bytes = std::size_t{kHidden} * kFf * 34.0 / 32.0;

  // Unbatched path, including its per-layer id download and host wait.
  const int old_iters = std::max(48, iters / 5);
  int old_launches = 0;
  for (int i = 0; i < 4; ++i) OldLayer(layers[i % n_layers], s, ids_all, st);
  HIP_OK(hipStreamSynchronize(st));
  const auto c0 = std::chrono::steady_clock::now();
  for (int i = 0; i < old_iters; ++i)
    old_launches = OldLayer(layers[i % n_layers], s, ids_all + ((i * 7) % kSets) * kUsed, st);
  HIP_OK(hipStreamSynchronize(st));
  const double t_old =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - c0).count() /
      old_iters;
  FreeMemLine("timing done");

  const double gbs = [](double bytes, double us) { return bytes / (us * 1e3); }(gu_bytes + dn_bytes, t_new);
  std::printf("timing, per MoE layer (one decode token, %d experts):\n", kUsed);
  std::printf("  unbatched (old): %8.1f us, %d kernel launches + id download + host wait, %.1f GB/s\n",
              t_old, old_launches, (gu_bytes + dn_bytes) / (t_old * 1e3));
  std::printf("  batched split  : %8.1f us, %d launches, %.1f GB/s (gate+up %.1f us %.1f GB/s; down %.1f us %.1f GB/s)\n",
              t_new, new_launches, gbs, t_new_gu, gu_bytes / (t_new_gu * 1e3), t_new_dn,
              dn_bytes / (t_new_dn * 1e3));
  std::printf("  batched plain  : %8.1f us, %.1f GB/s (gate+up %.1f us %.1f GB/s; down %.1f us %.1f GB/s)\n",
              t_plain, (gu_bytes + dn_bytes) / (t_plain * 1e3), t_plain_gu,
              gu_bytes / (t_plain_gu * 1e3), t_plain_dn, dn_bytes / (t_plain_dn * 1e3));
  std::printf("  batched split + shared down range: %8.1f us (gate+up launch %.1f us, %.1f GB/s incl. shared %.2f MB)\n",
              t_sh, t_sh_gu, (gu_bytes + sh_bytes) / (t_sh_gu * 1e3), sh_bytes / 1e6);
  std::printf("  x48 layers: old %.2f ms/token, batched split %.2f ms/token, plain %.2f ms/token\n",
              t_old * 48 / 1000, t_new * 48 / 1000, t_plain * 48 / 1000);
  std::printf(
      "JSON {\"experts\":%zu,\"layers\":%d,\"iters\":%d,\"old_us\":%.2f,\"old_launches\":%d,"
      "\"new_us\":%.2f,\"new_launches\":%d,\"new_gu_us\":%.2f,\"new_dn_us\":%.2f,"
      "\"plain_us\":%.2f,\"plain_gu_us\":%.2f,\"plain_dn_us\":%.2f,\"bytes_gu\":%.0f,"
      "\"bytes_dn\":%.0f,\"err_slot_split\":%.3e,\"err_comb_split\":%.3e,"
      "\"err_slot_plain\":%.3e,\"err_comb_plain\":%.3e,\"neg\":%.3e,\"sh_us\":%.2f,"
      "\"sh_gu_us\":%.2f,\"sh_err\":%.3e,\"fails\":%d}\n",
      n_exp, n_layers, iters, t_old, old_launches, t_new, new_launches, t_new_gu, t_new_dn,
      t_plain, t_plain_gu, t_plain_dn, gu_bytes, dn_bytes, worst_slot[0], worst_comb[0],
      worst_slot[1], worst_comb[1], neg, t_sh, t_sh_gu, shared_err[0], fails);
  return fails == 0 ? 0 : 1;
}
