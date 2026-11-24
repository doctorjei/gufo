#ifndef GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_I4R_TINY_FIXTURE_HPP_
#define GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_I4R_TINY_FIXTURE_HPP_

// Shared fixture for the point-85 CPU end-to-end tests: a complete, tiny
// `qwen4exp` GGUF carrying I4R128 rotated experts, the five F16 rotation
// vectors per layer, and the rotation marker.
//
// The architecture gate in Config::FromGguf pins the production shape (48
// layers, 512 experts), so these fixtures bind through
// ModelWeights::BindWithConfig with a hand-built Config of the same layout at
// reduced dimensions. The rotation geometry that matters is preserved: every
// rotated axis is a whole number of 128-element blocks.
//
// The fixture also materialises the true weights W = diag(sv)*Hn*M*Hn*diag(su)
// for each expert, so a test can compare the rotated storage path against a
// plain dense matmul of the same mathematical weights.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/quant/i4r_rotation.hpp"
#include "src/models/qwen38_flash_next/config.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::test_fixture {

namespace q = gufo::core;
namespace qr = gufo::quant;
namespace fn = gufo::models::qwen38_flash_next;

// Tiny-but-block-aligned geometry. hidden/ff are multiples of 128 so the
// rotation contract (whole blocks on every axis) holds exactly.
inline constexpr std::uint64_t kTinyHidden = 256;  // 2 blocks
inline constexpr std::uint64_t kTinyVocab = 32;
inline constexpr std::uint64_t kTinyExpertFf = 128;  // 1 block
inline constexpr std::uint64_t kTinySharedFf = 128;
inline constexpr std::uint32_t kTinyExperts = 2;
inline constexpr std::uint32_t kTinyExpertsUsed = 2;
inline constexpr std::uint32_t kTinyLayers = 4;  // 3 linear + 1 attention
inline constexpr std::uint64_t kAlign = 32;

inline fn::Config TinyConfig() {
  fn::Config c;
  c.num_layers = kTinyLayers;
  c.num_layers_all = kTinyLayers;
  c.hidden_size = static_cast<std::uint32_t>(kTinyHidden);
  c.vocab_size = static_cast<std::uint32_t>(kTinyVocab);
  c.context_length = 64;
  c.rms_eps = 1e-6F;
  c.hc_count = 2;
  c.hc_low_rank = 32;
  c.full_attention_interval = 4;
  // FF-GEOM256 (Point 85 approved full-forward fixture): the full-attention
  // layer carries the geometry the current ROCm backend actually executes --
  // the WMMA contract 24 query / 2 KV heads at head_dim 256 with
  // compress_ratio 4 (also fallback-valid: 256-wide heads, group 12 % 4 == 0).
  // The prior tiny geometry (64-wide, 2q/1kv, ratio 2) is supported by
  // NEITHER device path and is rejected by ValidateBackendGeometry(kFullForward).
  c.num_heads = 24;
  c.num_kv_heads = 2;
  c.head_dim = 256;
  c.rotary_dim = 32;
  c.rope_sections = {6, 5, 5, 0};
  c.rope_theta = 1e7F;
  // FF-GEOM256-V2 (review finding, 2026-11-11): the
  // indexer must carry the dimensions the device selector is compiled for.
  // kernels.hip.cpp fixes kSelectHeads = 4 / kSelectDim = 128 (L2736-2737)
  // and SelectBlocks returns WITHOUT writing scores or mask when the
  // caller's heads/dim differ (L8935-8937); Executor::Attention still
  // enters sparse selection on decode and hands s_.mask to attention
  // (L2996-3041), so a 2x32 indexer is not a harmless unused path.
  // 4x128 is also the production profile (config.cpp L254). The prior
  // 2x32 tiny indexer passed ValidateBackendGeometry only because that
  // validator does not look at the indexer at all -- see the v2 README.
  c.indexer_heads = 4;
  c.indexer_head_dim = 128;
  c.indexer_top_k = 8;
  // FF-GEOM256: ratio 4 matches the WMMA contract (kWmmaRatio) and keeps the
  // indexer block budget (top_k / ratio = 2) below the 12-token sequence's
  // complete-block count, so the sparse-selection path stays exercised.
  c.compress_ratio = 4;
  c.ssm_conv_kernel = 4;
  // The ROCm GDN kernels are compiled for a 128-wide head (kGdnDim) and
  // never receive the runtime head dimension, so the fixture must use the
  // supported width. A 64-wide fixture falls through to GdnKernel, which
  // addresses state with d=128 against buffers sized for d=64 -- a 4x
  // over-read/over-write. See gdn_geometry.hpp and the point-85 fault
  // review. Production admits only 128 (Config::FromGguf).
  c.ssm_head_dim = 128;
  c.ssm_num_k_heads = 2;
  c.ssm_num_v_heads = 4;
  c.ssm_inner_size = c.SsmValueDim();
  c.num_experts = kTinyExperts;
  c.num_experts_used = kTinyExpertsUsed;
  c.expert_ff = static_cast<std::uint32_t>(kTinyExpertFf);
  c.shared_expert_ff = static_cast<std::uint32_t>(kTinySharedFf);
  c.ple_layer = 0;
  c.ple_ngram_size = 3;
  c.ple_heads_per_ngram = 2;
  c.ple_heads = 4;
  c.ple_head_dim = 32;
  c.ple_conv_kernel = 4;
  c.ple_eos_token = 0;
  c.ple_multipliers = {2654435761ULL, 40503ULL, 12345ULL};
  for (std::uint32_t h = 0; h < c.ple_heads; ++h) {
    c.ple_head_offsets[h] = 0;
    c.ple_head_vocab[h] = 16;
  }
  c.ple_rows = 16;
  c.nextn_layers = 0;
  return c;
}

/// Deterministic pseudo-random generator so every run builds the same bytes.
struct Lcg {
  std::uint64_t s{0x243F6A8885A308D3ULL};
  std::uint32_t Next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(s >> 32);
  }
  float Uniform() {
    return (static_cast<float>(Next()) / 4294967295.0F) * 2.0F - 1.0F;
  }
};

inline std::uint16_t FloatToHalf(float v) {
  // Copy the encoding, not the numeric value: static_cast<uint16_t> of a
  // half truncates 0.1 to 0 instead of yielding 0x2E66.
  const _Float16 h = static_cast<_Float16>(v);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

/// Round-trips a float vector through the F16 encoding it will be stored
/// with. The loader widens the stored halves back to F32, so the oracle has
/// to see the ROUND-TRIPPED values. Generating F32 scales, materialising
/// the dense weights from those, and only rounding to F16 on the way into
/// the tensor makes the dense and loaded paths multiply different
/// mathematical weights -- the difference then shows up as a residual that
/// looks like Hadamard-chain error but is really scale precision.
inline std::vector<float> RoundTripF16(const std::vector<float>& v) {
  std::vector<float> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    out[i] = static_cast<float>(static_cast<_Float16>(v[i]));
  }
  return out;
}

/// One tensor's declaration plus the payload bytes that back it.
struct TensorSpec {
  std::string name;
  std::vector<std::uint64_t> dims;
  q::GgmlType type;
  std::vector<std::uint8_t> payload;
  std::uint64_t offset{0};  ///< Byte offset from the GGUF data start.
};

inline void AppendBytes(std::vector<std::uint8_t>& buf, const void* data,
                        std::size_t len) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  buf.insert(buf.end(), p, p + len);
}

template<class T>
inline void AppendPod(std::vector<std::uint8_t>& buf, T v) {
  AppendBytes(buf, &v, sizeof(T));
}

inline void AppendStr(std::vector<std::uint8_t>& buf, const std::string& s) {
  AppendPod(buf, static_cast<std::uint64_t>(s.size()));
  AppendBytes(buf, s.data(), s.size());
}

/// Minimal GGUF writer with real payloads (the shared GgufBuilder fills a
/// dummy blob, which is not enough to run the model).
class PayloadGguf {
public:
  void AddKvString(const std::string& key, const std::string& value) {
    kv_.emplace_back(key, value);
  }
  void AddTensor(TensorSpec spec) { tensors_.push_back(std::move(spec)); }

  std::vector<std::uint8_t> Finish() const {
    std::vector<std::uint8_t> buf;
    AppendBytes(buf, "GGUF", 4);  // raw magic, not a length-prefixed string
    AppendPod(buf, static_cast<std::uint32_t>(3));
    AppendPod(buf, static_cast<std::uint64_t>(tensors_.size()));
    AppendPod(buf, static_cast<std::uint64_t>(kv_.size()));
    for (const auto& [key, value] : kv_) {
      AppendStr(buf, key);
      AppendPod(buf, static_cast<std::uint32_t>(q::GgufValueType::kString));
      AppendStr(buf, value);
    }
    for (const auto& t : tensors_) {
      AppendStr(buf, t.name);
      AppendPod(buf, static_cast<std::uint32_t>(t.dims.size()));
      for (auto d : t.dims) {
        AppendPod(buf, d);
      }
      AppendPod(buf, static_cast<std::uint32_t>(t.type));
      AppendPod(buf, t.offset);
    }
    while (buf.size() % kAlign != 0) {
      buf.push_back(0);
    }
    const std::size_t data_start = buf.size();
    for (const auto& t : tensors_) {
      buf.resize(data_start + t.offset + t.payload.size(), 0);
      std::memcpy(buf.data() + data_start + t.offset, t.payload.data(),
                  t.payload.size());
    }
    return buf;
  }

  /// Lays out tensor offsets sequentially, aligned to kAlign.
  void Layout() {
    std::uint64_t off = 0;
    for (auto& t : tensors_) {
      std::size_t elements = 1;
      for (auto d : t.dims) {
        elements *= d;
      }
      const std::size_t bytes = qr::EncodedSizeBytes(t.type, elements);
      t.offset = off;
      off += bytes;
      off = (off + kAlign - 1) / kAlign * kAlign;
    }
  }

private:
  std::vector<std::pair<std::string, std::string>> kv_;
  std::vector<TensorSpec> tensors_;
};

/// Encodes one I4R128 matrix (row-major [rows x cols]) from its stored
/// rotated-basis values M, and returns the payload bytes.
inline std::vector<std::uint8_t> PackI4R128(const std::vector<float>& m,
                                            std::uint64_t rows,
                                            std::uint64_t cols) {
  std::vector<std::uint8_t> out;
  const std::size_t blocks = cols / qr::kI4rBlock;
  out.resize(rows * blocks * 66);
  for (std::uint64_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0; b < blocks; ++b) {
      // Scale chosen so the codes span the block's peak exactly.
      float peak = 0.0F;
      for (std::size_t i = 0; i < qr::kI4rBlock; ++i) {
        peak = std::max(peak, std::abs(m[r * cols + b * qr::kI4rBlock + i]));
      }
      const float d = peak / 8.0F;
      const float safe_d = d == 0.0F ? 1.0F : d;
      std::uint8_t* blk = out.data() + (r * blocks + b) * 66;
      for (std::size_t i = 0; i < 64; ++i) {
        const int lo = static_cast<int>(std::lround(
                           m[r * cols + b * qr::kI4rBlock + i] / safe_d)) +
                       8;
        const int hi = static_cast<int>(std::lround(
                           m[r * cols + b * qr::kI4rBlock + 64 + i] / safe_d)) +
                       8;
        blk[i] =
            static_cast<std::uint8_t>((std::clamp(lo, 0, 15) & 0xF) |
                                      ((std::clamp(hi, 0, 15) & 0xF) << 4));
      }
      const std::uint16_t dh = FloatToHalf(d);
      std::memcpy(blk + 64, &dh, 2);
    }
  }
  return out;
}

/// Applies the orthonormal Hn to every 128-block along the row axis of a
/// row-major [rows x cols] matrix (the left multiply Hn * M).
inline void HadamardAlongRows(std::vector<float>& m, std::uint64_t rows,
                              std::uint64_t cols) {
  std::vector<float> col(rows);
  for (std::uint64_t c = 0; c < cols; ++c) {
    for (std::uint64_t r = 0; r < rows; ++r) {
      col[r] = m[r * cols + c];
    }
    qr::RotateBlocks(col.data(), col.size());
    for (std::uint64_t r = 0; r < rows; ++r) {
      m[r * cols + c] = col[r];
    }
  }
}

/// W = diag(sv) * Hn * M * Hn * diag(su), materialised for the oracle.
inline std::vector<float> MaterializeW(const std::vector<float>& m,
                                       std::uint64_t rows, std::uint64_t cols,
                                       const std::vector<float>& su,
                                       const std::vector<float>& sv) {
  std::vector<float> w = m;
  for (std::uint64_t r = 0; r < rows; ++r) {
    qr::RotateBlocks(w.data() + r * cols, cols);  // M * Hn along cols
  }
  HadamardAlongRows(w, rows, cols);  // Hn * (M * Hn) along rows
  for (std::uint64_t r = 0; r < rows; ++r) {
    for (std::uint64_t c = 0; c < cols; ++c) {
      w[r * cols + c] *= sv[r] * su[c];
    }
  }
  return w;
}

/// A built fixture: the GGUF bytes, the config to bind with, and the
/// materialised true weights for the parity oracle.
struct TinyHybrid {
  std::vector<std::uint8_t> bytes;
  fn::Config config;
  std::vector<std::vector<float>> w_gate;  // [expert][hidden*ff] row-major
  std::vector<std::vector<float>> w_up;
  std::vector<std::vector<float>> w_down;  // [expert][ff*hidden]
  std::vector<float> router;               // [experts] per-token logits weights
};

/// Knobs for the loader's negative cases; the default builds the positive
/// tiny hybrid.
struct HybridVariant {
  bool rotated = true;
  bool marker = true;
  std::string marker_value = "hadamard128";
  int drop_rot = -1;  // 0=su_experts 1=sv_gate 2=sv_up 3=su_down 4=sv_down
  bool rot_as_f32 = false;
  bool mixed = false;  // gate rotated, up/down plain
};

/// Builds the tiny hybrid GGUF. A rotated variant carries I4R128 experts
/// with the five rotation vectors and the marker; a plain variant carries
/// Q4_K experts and no rotation metadata.
inline TinyHybrid BuildTinyHybrid(const HybridVariant& v = {}) {
  const fn::Config c = TinyConfig();
  Lcg rng;
  // Dedicated stream for the GDN population (full-forward parity
  // candidate). The adopted 128-wide fixture draws NO random values for
  // the SSM tensors (they were add_zeros), so drawing the populated GDN
  // weights from this separate stream leaves the main stream — and every
  // non-GDN tensor in the fixture — byte-identical to the adopted
  // 8ed1955c428ec4607f44334d8bb9d82e7d244b1f8dc6a02b72e60002fc286bed
  // artifact. Seed is a documented constant; the delta is auditable as
  // "only the GDN tensor payloads changed".
  Lcg gdn_rng{0xC0FFEE8585C0FFEEULL};
  // FF-GEOM256: dedicated stream for the full-attention population. The
  // prior fixture left the attention tensors zero; populated, nonzero
  // weights from this separate stream make the attention path numerically
  // live (a device silent-skip is then distinguishable from a correct
  // zero). Drawing from attn_rng leaves the main stream and gdn_rng
  // untouched, so every non-attention tensor -- including the populated
  // 128-wide GDN payloads -- stays byte-identical to the prior populated
  // fixture 8732168e...b1efb8.
  Lcg attn_rng{0xA77E2560FF2560A7ULL};
  TinyHybrid f;
  f.config = c;

  PayloadGguf g;
  g.AddKvString("general.architecture", "qwen4exp");
  if (v.marker) {
    g.AddKvString(std::string(fn::kRotationMarkerKey), v.marker_value);
  }

  auto add_f32 = [&](const std::string& name, std::vector<float> values,
                     std::vector<std::uint64_t> dims) {
    // Validate the payload against the declared shape at the writer. A short
    // vector against a larger declared tensor used to be silently padded by
    // later layout expansion, masking the remainder with zeros (the router
    // 2-float / 512-float bug). Fail loudly instead.
    std::size_t expected = 1;
    for (auto d : dims) {
      expected *= d;
    }
    if (values.size() != expected) {
      std::fprintf(stderr,
                   "i4r fixture: payload size %zu != declared %zu for %s\n",
                   values.size(), expected, name.c_str());
      std::abort();
    }
    TensorSpec t;
    t.name = name;
    t.dims = std::move(dims);
    t.type = q::GgmlType::kF32;
    t.payload.resize(values.size() * sizeof(float));
    if (!values.empty()) {
      std::memcpy(t.payload.data(), values.data(), t.payload.size());
    }
    g.AddTensor(std::move(t));
  };
  // Populate a fully-random F32 tensor of the declared shape. Used where the
  // math depends on nonzero values (HC norm gammas, mixer projections) so a
  // real Forward pass produces non-vacuous activations.
  auto add_random_f32_src = [&](Lcg& src, const std::string& name,
                                std::vector<std::uint64_t> dims, float scale,
                                float offset) {
    std::size_t elements = 1;
    for (auto d : dims) {
      elements *= d;
    }
    std::vector<float> values(elements);
    for (auto& v : values) {
      v = offset + scale * src.Uniform();
    }
    add_f32(name, std::move(values), std::move(dims));
  };
  auto add_random_f32 = [&](const std::string& name,
                            std::vector<std::uint64_t> dims, float scale,
                            float offset) {
    add_random_f32_src(rng, name, std::move(dims), scale, offset);
  };
  auto add_zeros = [&](const std::string& name, std::vector<std::uint64_t> dims,
                       q::GgmlType type) {
    TensorSpec t;
    t.name = name;
    t.dims = std::move(dims);
    t.type = type;
    std::size_t elements = 1;
    for (auto d : t.dims) {
      elements *= d;
    }
    t.payload =
        std::vector<std::uint8_t>(qr::EncodedSizeBytes(type, elements), 0);
    g.AddTensor(std::move(t));
  };
  auto add_f16 = [&](const std::string& name, std::vector<float> values) {
    TensorSpec t;
    t.name = name;
    t.dims = {values.size()};
    t.type = q::GgmlType::kF16;
    t.payload.resize(values.size() * 2);
    for (std::size_t i = 0; i < values.size(); ++i) {
      const std::uint16_t h = FloatToHalf(values[i]);
      std::memcpy(t.payload.data() + 2 * i, &h, 2);
    }
    g.AddTensor(std::move(t));
  };

  // Model-level tensors.
  std::vector<float> embd(kTinyHidden * kTinyVocab);
  for (auto& v : embd) {
    v = rng.Uniform();
  }
  add_f32("token_embd.weight", embd, {kTinyHidden, kTinyVocab});
  add_f32("output.weight", embd, {kTinyHidden, kTinyVocab});
  const std::uint64_t hc_dim = 2 * kTinyHidden;
  // HC norm gammas and mixer projections are populated, not zero. HcMix
  // RMS-normalises the residual with the gamma vector and the epilogue
  // multiplies those normalised rows by the gate from the down/up
  // projections; a zero gamma (or zero projections) collapses the mixed
  // input to zero and makes every downstream expert input vacuous. Values
  // are small and positive around 1.0 for the gammas, small around 0.1 for
  // the projections, so the mix is nontrivial but bounded.
  add_random_f32("output_hc_norm.weight", {hc_dim}, 0.2F, 0.9F);
  add_random_f32("output_hc_down.weight", {hc_dim, 32}, 0.2F, -0.1F);
  add_random_f32("output_hc_up.weight", {32, hc_dim}, 0.2F, -0.1F);
  add_zeros("per_layer_token_embd.weight", {32, 16}, q::GgmlType::kQ8_0);

  const auto dense = q::GgmlType::kF32;
  // Q8_0 rather than Q4_K: the tiny expert_ff (128) is not a multiple of the
  // 256-element Q4_K block, so only the 32-block formats can carry these
  // experts at fixture scale.
  const auto expert_plain = q::GgmlType::kQ8_0;

  for (std::uint32_t il = 0; il < kTinyLayers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    for (const std::string mix : {"hc_attn", "hc_ffn"}) {
      // Nonzero gammas + projections so the per-layer mixed input to the
      // FFN/experts is non-vacuous (see the model-level HC note above).
      // The inject projection stays zero: it is an additive residual term
      // and does not gate the mixed row.
      add_random_f32(p + mix + "_norm.weight", {hc_dim}, 0.2F, 0.9F);
      add_random_f32(p + mix + "_down.weight", {hc_dim, 32}, 0.2F, -0.1F);
      add_random_f32(p + mix + "_up.weight", {32, hc_dim}, 0.2F, -0.1F);
      add_zeros(p + mix + "_inject.weight", {hc_dim, 2}, q::GgmlType::kF32);
    }
    const bool linear = ((il + 1) % 4) != 0;
    if (linear) {
      // Every SSM shape is derived from the config rather than written as a
      // literal, so it tracks ssm_head_dim. The binder checks each tensor
      // against the same helpers (weights.cpp Layer), so a fixture that
      // drifted from the config would fail to bind.
      //
      // FULL-FORWARD PARITY CANDIDATE: the GDN tensors are POPULATED, from
      // the dedicated gdn_rng stream (see the note at BuildTinyHybrid).
      // The adopted fixture's all-zero SSM weights make a full-model
      // forward GDN-vacuous (device GDN output is zero regardless of
      // kernel correctness); nonzero weights are what let the forward
      // comparison exercise the corrected 128-wide GDN path numerically.
      // Ranges mirror the v4 B4 HC precedent: small and bounded. ssm_a is
      // drawn negative so the per-step decay exp(a*softplus(alpha+dt))
      // stays below 1 — the recurrent state decays instead of compounding.
      add_random_f32_src(gdn_rng, p + "attn_qkv.weight",
                         {kTinyHidden, c.SsmConvChannels()}, 0.2F, 0.0F);
      add_random_f32_src(gdn_rng, p + "attn_gate.weight",
                         {kTinyHidden, c.SsmValueDim()}, 0.2F, 0.0F);
      add_random_f32_src(gdn_rng, p + "ssm_conv1d.weight",
                         {c.ssm_conv_kernel, c.SsmConvChannels()}, 0.2F, 0.0F);
      add_random_f32_src(gdn_rng, p + "ssm_alpha.weight",
                         {kTinyHidden, c.ssm_num_v_heads}, 0.1F, 0.0F);
      add_random_f32_src(gdn_rng, p + "ssm_beta.weight",
                         {kTinyHidden, c.ssm_num_v_heads}, 0.5F, 0.0F);
      add_random_f32_src(gdn_rng, p + "ssm_dt.bias", {c.ssm_num_v_heads}, 0.1F,
                         0.0F);
      add_random_f32_src(gdn_rng, p + "ssm_a", {c.ssm_num_v_heads}, 0.25F,
                         -0.75F);
      add_random_f32_src(gdn_rng, p + "ssm_norm.weight", {c.ssm_head_dim}, 0.2F,
                         0.9F);
      add_random_f32_src(gdn_rng, p + "ssm_out.weight",
                         {c.SsmValueDim(), kTinyHidden}, 0.2F, 0.0F);
    } else {
      // FF-GEOM256: attention layer populated from attn_rng with
      // config-derived shapes (attn_q carries the interleaved q|gate
      // projection, hence 2x Q-dim; norms are per-head-dim). Ranges mirror
      // the GDN population: small, bounded, nonzero.
      add_random_f32_src(attn_rng, p + "attn_q.weight",
                         {kTinyHidden, 2ULL * c.AttentionQDim()}, 0.2F, 0.0F);
      add_random_f32_src(attn_rng, p + "attn_k.weight",
                         {kTinyHidden, c.AttentionKvDim()}, 0.2F, 0.0F);
      add_random_f32_src(attn_rng, p + "attn_v.weight",
                         {kTinyHidden, c.AttentionKvDim()}, 0.2F, 0.0F);
      add_random_f32_src(attn_rng, p + "attn_output.weight",
                         {c.AttentionQDim(), kTinyHidden}, 0.2F, 0.0F);
      add_random_f32_src(attn_rng, p + "attn_q_norm.weight", {c.head_dim}, 0.2F,
                         0.9F);
      add_random_f32_src(attn_rng, p + "attn_k_norm.weight", {c.head_dim}, 0.2F,
                         0.9F);
      add_random_f32_src(attn_rng, p + "indexer.q_proj.weight",
                         {kTinyHidden,
                          static_cast<std::uint64_t>(c.indexer_heads) *
                              c.indexer_head_dim},
                         0.2F, 0.0F);
      add_random_f32_src(attn_rng, p + "indexer.k_proj.weight",
                         {kTinyHidden, c.indexer_head_dim}, 0.2F, 0.0F);
      add_random_f32_src(attn_rng, p + "indexer.q_norm.weight",
                         {c.indexer_head_dim}, 0.2F, 0.9F);
      add_random_f32_src(attn_rng, p + "indexer.k_norm.weight",
                         {c.indexer_head_dim}, 0.2F, 0.9F);
    }
    if (il == 0) {
      add_zeros(p + "ple_key.weight", {128, hc_dim}, dense);
      add_zeros(p + "ple_value.weight", {128, kTinyHidden}, dense);
      add_zeros(p + "ple_norm_key.weight", {hc_dim}, q::GgmlType::kF32);
      add_zeros(p + "ple_norm_query.weight", {hc_dim}, q::GgmlType::kF32);
      add_zeros(p + "ple_norm_conv.weight", {hc_dim}, q::GgmlType::kF32);
      add_zeros(p + "ple_conv1d.weight", {4, hc_dim}, q::GgmlType::kF32);
    }

    // Router is the [hidden -> experts] projection matrix, not a per-token
    // vector: the declared {hidden, experts} tensor needs hidden*experts
    // floats. The old fixture supplied only `experts` floats and relied on
    // later zero-padding, which masked the remainder. Fill the whole
    // payload deterministically; add_f32 now validates the length.
    add_random_f32(p + "ffn_gate_inp.weight", {kTinyHidden, kTinyExperts}, 2.0F,
                   0.0F);

    // The rotation vectors are per layer; the same vectors serve every
    // expert, exactly as the real artifact does.
    std::vector<float> su_experts(kTinyHidden), sv_gate(kTinyExpertFf),
        sv_up(kTinyExpertFf), su_down(kTinyExpertFf), sv_down(kTinyHidden);
    for (auto& v : su_experts) {
      v = 0.5F + 0.5F * rng.Uniform();
    }
    for (auto& v : sv_gate) {
      v = 0.5F + 0.5F * rng.Uniform();
    }
    for (auto& v : sv_up) {
      v = 0.5F + 0.5F * rng.Uniform();
    }
    for (auto& v : su_down) {
      v = 0.5F + 0.5F * rng.Uniform();
    }
    for (auto& v : sv_down) {
      v = 0.5F + 0.5F * rng.Uniform();
    }
    // Round-trip BEFORE anything reads them: MaterializeW uses these vectors
    // for the dense oracle and add_f16 writes them to the tensor, so both
    // must carry the same F16-representable values. Doing it here (rather
    // than only at the write) is what keeps the two paths multiplying
    // identical weights.
    su_experts = RoundTripF16(su_experts);
    sv_gate = RoundTripF16(sv_gate);
    sv_up = RoundTripF16(sv_up);
    su_down = RoundTripF16(su_down);
    sv_down = RoundTripF16(sv_down);

    for (int role = 0; role < 3; ++role) {
      const bool role_rot = v.rotated && !(v.mixed && role != 0);
      const std::uint64_t cols = role < 2 ? kTinyHidden : kTinyExpertFf;
      const std::uint64_t rows = role < 2 ? kTinyExpertFf : kTinyHidden;
      const std::string base =
          role == 0 ? "ffn_gate_exps"
                    : (role == 1 ? "ffn_up_exps" : "ffn_down_exps");
      std::vector<float> whole;  // concatenated per-expert stored values
      for (std::uint32_t e = 0; e < kTinyExperts; ++e) {
        std::vector<float> m(rows * cols);
        for (auto& value : m) {
          value = rng.Uniform();
        }
        whole.insert(whole.end(), m.begin(), m.end());
      }
      if (role_rot) {
        std::vector<std::uint8_t> payload =
            PackI4R128(whole, rows * kTinyExperts, cols);
        TensorSpec t;
        t.name = p + base + ".weight";
        t.dims = {cols, rows, kTinyExperts};
        t.type = q::GgmlType::kI4R128;
        t.payload = payload;
        g.AddTensor(std::move(t));
        // Materialise the true weights from the DECODED codes, not the
        // pre-quantisation values: the parity oracle must see exactly what
        // the loader sees.
        std::vector<float> decoded(whole.size());
        qr::DequantizeI4R128(payload.data(), decoded.data(), decoded.size());
        const std::vector<float>& su = role < 2 ? su_experts : su_down;
        const std::vector<float>& sv =
            role == 0 ? sv_gate : (role == 1 ? sv_up : sv_down);
        auto& out_w = role == 0 ? f.w_gate : (role == 1 ? f.w_up : f.w_down);
        for (std::uint32_t e = 0; e < kTinyExperts; ++e) {
          std::vector<float> m_dec(decoded.begin() + e * rows * cols,
                                   decoded.begin() + (e + 1) * rows * cols);
          out_w.push_back(MaterializeW(m_dec, rows, cols, su, sv));
        }
      } else {
        // Plain experts: the loader path only, payload content is not read.
        add_zeros(p + base + ".weight", {cols, rows, kTinyExperts},
                  expert_plain);
      }
    }
    if (v.rotated) {
      const std::vector<float>* rots[5] = {&su_experts, &sv_gate, &sv_up,
                                           &su_down, &sv_down};
      const char* names[5] = {"ffn_experts_rot.su", "ffn_gate_exps_rot.sv",
                              "ffn_up_exps_rot.sv", "ffn_down_exps_rot.su",
                              "ffn_down_exps_rot.sv"};
      for (int i = 0; i < 5; ++i) {
        if (v.drop_rot == i) {
          continue;
        }
        if (v.rot_as_f32) {
          add_f32(p + names[i], *rots[i], {rots[i]->size()});
        } else {
          add_f16(p + names[i], *rots[i]);
        }
      }
    }

    add_zeros(p + "ffn_gate_inp_shexp.weight", {kTinyHidden, 1},
              q::GgmlType::kF32);
    add_zeros(p + "ffn_gate_shexp.weight", {kTinyHidden, kTinySharedFf}, dense);
    add_zeros(p + "ffn_up_shexp.weight", {kTinyHidden, kTinySharedFf}, dense);
    add_zeros(p + "ffn_down_shexp.weight", {kTinySharedFf, kTinyHidden}, dense);
  }

  g.Layout();
  f.bytes = g.Finish();
  return f;
}

}  // namespace gufo::test_fixture

#endif  // GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_I4R_TINY_FIXTURE_HPP_
