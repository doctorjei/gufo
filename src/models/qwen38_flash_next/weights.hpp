#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen38_flash_next/config.hpp"

namespace gufo::models::qwen38_flash_next {

/// Optional host destinations for independent predictor qualification.
/// Every populated span contains the final complete row of its stage.
struct MtpTrace {
  std::span<float> normalized_hidden;
  std::span<float> fused;
  std::span<float> attention;
  std::span<float> hidden;
  std::span<float> head;
  std::span<float> ffn_input;
  std::span<float> ffn_output;
  [[nodiscard]] bool Valid(std::size_t H, std::size_t D) const noexcept {
    for (const auto row : {normalized_hidden, fused, attention, hidden})
      if (!row.empty() && row.size() != D)
        return false;
    for (const auto row : {head, ffn_input, ffn_output})
      if (!row.empty() && row.size() != H)
        return false;
    return true;
  }
};

/// Non-owning view of one GGUF tensor. `rows` x `cols` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole matrices.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::uint64_t file_offset{0};  ///< Byte offset inside the owning shard.
  std::uint32_t shard{0};        ///< Mapped region index in the reader.
  std::string_view name;

  /// `shard` of a tensor decoded on the host at bind time (an HGN store, see
  /// src/core/quant/hgn_decode.hpp): `data` points at the decoded F16/F32
  /// values, owned by the bound weights, and uploads copy from it instead of
  /// reading the file.
  static constexpr std::uint32_t kHostShard = 0xFFFFFFFFu;
  /// kHT4 (read in place by the ht kernels): the F16 rotation vectors
  /// `<name>_rot.su` [cols] and `<name>_rot.sv` [rows], mapped from the file.
  const void* rot_su{nullptr};
  const void* rot_sv{nullptr};
  [[nodiscard]] bool HostDecoded() const noexcept {
    return shard == kHostShard;
  }

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    return cols * rows * experts;
  }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
  /// Base address of one stacked expert matrix.
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
};

/// Two hyper-connection mixers per layer plus the output head mixer share
/// this shape; `inject` is absent on the head mixer.
struct HcMixer {
  TensorRef norm;    ///< [hc_dim] grouped RMSNorm gamma, (1 + w) folded.
  TensorRef down;    ///< [hc_dim -> low_rank]
  TensorRef up;      ///< [low_rank -> hc_dim]
  TensorRef inject;  ///< [hc_dim -> hc_count], empty on the head mixer.
};

/// The five F16 rotation vectors a rotated (I4R128) expert layer carries,
/// resolved through the pairing table of I4R-GGUF-SPEC.md section 5:
///
///   gate_exps <- ffn_experts_rot.su     [hidden]  + ffn_gate_exps_rot.sv [ff]
///   up_exps   <- ffn_experts_rot.su     [hidden]  + ffn_up_exps_rot.sv   [ff]
///   down_exps <- ffn_down_exps_rot.su   [ff]      + ffn_down_exps_rot.sv
///   [hidden]
///
/// Gate and up share one input vector, which is why the layer carries five
/// vectors rather than six. `rotated` is set only when all three expert
/// tensors are I4R128 and every vector bound cleanly; a layer that is only
/// partly rotated is a load error, never a silent mix.
struct LayerRotation {
  bool rotated{false};
  TensorRef su_experts;  ///< [hidden] shared gate/up input scale
  TensorRef sv_gate;     ///< [expert_ff] gate output scale
  TensorRef sv_up;       ///< [expert_ff] up output scale
  TensorRef su_down;     ///< [expert_ff] down input scale
  TensorRef sv_down;     ///< [hidden] down output scale
};

/// Global GGUF marker naming the rotation applied to I4R128 codes
/// (I4R-GGUF-SPEC.md section 6). A file that carries rotated codes must
/// carry this marker with this exact value; a loader must never fall back to
/// reading rotated codes as plain.
inline constexpr std::string_view kRotationMarkerKey =
    "qwen4exp.experts.rotation";
inline constexpr std::string_view kRotationMarkerValue = "hadamard128";

struct LayerWeights {
  bool linear{false};
  HcMixer hc_attn;
  HcMixer hc_ffn;

  // Gated DeltaNet (linear layers).
  TensorRef ssm_qkv;     ///< [hidden -> 2*key_dim + value_dim]
  TensorRef ssm_gate;    ///< [hidden -> value_dim], the z output gate.
  TensorRef ssm_conv1d;  ///< [conv_kernel, channels]
  TensorRef ssm_alpha;   ///< [hidden -> v_heads]
  TensorRef ssm_beta;    ///< [hidden -> v_heads]
  TensorRef ssm_dt;      ///< [v_heads] softplus bias
  TensorRef ssm_a;       ///< [v_heads] = -exp(A_log)
  TensorRef ssm_norm;    ///< [ssm_head_dim]
  TensorRef ssm_out;     ///< [value_dim -> hidden]

  // Gated GQA + QSA indexer (attention layers).
  TensorRef attn_q;       ///< [hidden -> heads * 2 * head_dim], q|gate per head
  TensorRef attn_k;       ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_v;       ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_out;     ///< [heads * head_dim -> hidden]
  TensorRef attn_q_norm;  ///< [head_dim]
  TensorRef attn_k_norm;  ///< [head_dim]
  TensorRef indexer_q;    ///< [hidden -> indexer_heads * indexer_dim]
  TensorRef indexer_k;    ///< [hidden -> indexer_dim]
  TensorRef indexer_q_norm;
  TensorRef indexer_k_norm;

  // PLE n-gram injection (one linear layer).
  TensorRef ple_key;         ///< [ple_dim -> hc_dim]
  TensorRef ple_value;       ///< [ple_dim -> hidden]
  TensorRef ple_norm_key;    ///< [hc_dim]
  TensorRef ple_norm_query;  ///< [hc_dim]
  TensorRef ple_norm_conv;   ///< [hc_dim]
  TensorRef ple_conv1d;      ///< [conv_kernel, hc_dim]

  // Mixture of experts.
  TensorRef router;         ///< [hidden -> num_experts] F32
  TensorRef ffn_gate_exps;  ///< [hidden -> expert_ff] x experts
  TensorRef ffn_up_exps;    ///< [hidden -> expert_ff] x experts
  TensorRef ffn_down_exps;  ///< [expert_ff -> hidden] x experts
  /// Rotation vectors when the three tensors above are I4R128.
  LayerRotation rotation;
  TensorRef shexp_gate_inp;  ///< [hidden] -> scalar sigmoid gate
  TensorRef shexp_gate;      ///< [hidden -> shared_ff]
  TensorRef shexp_up;        ///< [hidden -> shared_ff]
  TensorRef shexp_down;      ///< [shared_ff -> hidden]

  // Speculative `nextn` block only.
  TensorRef nextn_enorm;  ///< [hidden]
  TensorRef nextn_hnorm;  ///< [hc_dim], one RMS denominator across all streams
  TensorRef nextn_eh_proj;  ///< GGUF rows: [fc_embedding | fc_hidden]
  HcMixer nextn_head;       ///< Output mixer of the draft block.
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;  ///< [hidden -> vocab]
  TensorRef output;      ///< [hidden -> vocab]
  HcMixer hc_head;       ///< Final mixer, doubles as the output norm.
  TensorRef ple_table;   ///< [ple_head_dim, ple_rows], read from disk only.
  /// Set when the trunk keeps its n-gram table in a separate GGUF next to it
  /// (metadata qwen4exp.ple.table_file); ple_table is then bound by the
  /// caller from that file with BindPleTable.
  std::string ple_table_file;
  std::vector<LayerWeights> layers;
  /// Host buffers of tensors decoded at bind time (TensorRef::kHostShard).
  std::vector<std::shared_ptr<const std::vector<std::uint8_t>>> decoded;

  /// Binds and validates the trunk artifact. Tensor payloads stay mapped and
  /// untouched; only headers are read.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Binds against a caller-supplied configuration, skipping the production
  /// geometry gate that `Bind` applies through `Config::FromGguf`. This exists
  /// so the CPU-oracle tests can bind fixtures that use the real tensor
  /// layout at reduced dimensions; production loads always go through
  /// `Bind`, which keeps the gate.
  [[nodiscard]] static std::optional<ModelWeights> BindWithConfig(
      const core::GgufReader& reader, const Config& config,
      std::string* error_msg = nullptr);
};

/// The MTP draft block from the `shared-*` sidecar: one attention layer with
/// its own hyper-connection mixers and experts; token embedding and LM head
/// are borrowed from the trunk.
struct MtpWeights {
  Config config;
  LayerWeights block;
  std::vector<std::shared_ptr<const std::vector<std::uint8_t>>> decoded;

  [[nodiscard]] static std::optional<MtpWeights> Bind(
      const core::GgufReader& reader, const Config& trunk,
      std::string* error_msg = nullptr);
};

/// Binds the n-gram (PLE) table of a trunk with configuration `c` from
/// `reader`: the trunk itself, or the separate table GGUF it names.
[[nodiscard]] std::optional<TensorRef> BindPleTable(
    const core::GgufReader& reader, const Config& c,
    std::string* error_msg = nullptr);

/// Global scale of an FP8G n-gram table (the f32 after its codes).
[[nodiscard]] float Fp8gTableScale(const TensorRef& table) noexcept;

/// Hyper-connection inject and GDN alpha/beta tensors of `layer` stored as
/// Q8_0. The official trunk GGUF keeps them in F32; the device expands Q8_0
/// copies to F32 at load, because their W8A8 route breaks copied identifiers.
[[nodiscard]] std::size_t ExpandedQ8_0Tensors(
    const LayerWeights& layer) noexcept;

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHTS_HPP_
