#include "src/models/qwen38_flash_next/weights.hpp"

#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/core/quant/hgn_decode.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

using core::GgmlType;

struct Format {
  std::uint32_t block;
  std::uint32_t bytes;
};

/// Storage geometry of the formats this runtime decodes. Unknown types get a
/// zero block, which every validation below rejects.
[[nodiscard]] Format FormatOf(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
      return {1, 4};
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return {1, 2};
    case GgmlType::kQ8_0:
      return {32, 34};
    case GgmlType::kQ5_1:
      return {32, 24};
    case GgmlType::kIQ4_NL:
      return {32, 18};
    case GgmlType::kQ4_K:
      return {256, 144};
    case GgmlType::kQ5_K:
      return {256, 176};
    case GgmlType::kQ6_K:
      return {256, 210};
    case GgmlType::kI4R128:
      // 128 offset-binary 4-bit codes + one f16 scale = 66 bytes per block.
      return {128, 66};
    case GgmlType::kFP8G:
      // n-gram table rows: 160 fp8 codes (the global scale follows them).
      return {160, 160};
    case GgmlType::kQ6G64:
      // 64 values in 52 bytes; rows are padded to 16 bytes (RowBytes).
      return {64, 52};
    case GgmlType::kHT4:
      // 16 x 16 trellis tiles, 4 bits a value; the size is cols/2 bytes a
      // row, but rows are only addressable as whole 128-row bands.
      return {128, 64};
    default:
      return {0, 0};
  }
}

struct Binder {
  const core::GgufReader& reader;
  std::string* error;
  bool ok{true};
  bool marker_checked{false};
  bool marker_ok{false};
  /// MTP sidecar: HGN stores are refused. Loading the v2 head through the
  /// decode-at-load path faulted the GPU in the draft forward, so the head is
  /// converted to the stock formats instead (hgn2gguf --mtp-stock-types).
  bool refuse_hgn{false};
  /// Owner of host buffers for HGN stores decoded at bind time.
  std::vector<std::shared_ptr<const std::vector<std::uint8_t>>>* decoded{
      nullptr};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  /// Checks the global rotation marker once, the first time a layer turns out
  /// to carry rotated experts. Absence or any other value is a hard error:
  /// rotated codes must never be read as plain.
  bool RotationMarkerOk() {
    if (marker_checked) {
      return marker_ok;
    }
    marker_checked = true;
    const auto value = reader.GetMetadataString(kRotationMarkerKey);
    if (!value.has_value()) {
      Fail("rotated I4R128 experts present but the rotation marker '" +
           std::string(kRotationMarkerKey) +
           "' is missing; refusing to read rotated codes as plain");
      marker_ok = false;
    } else if (*value != kRotationMarkerValue) {
      Fail("rotation marker '" + std::string(kRotationMarkerKey) + "' names '" +
           std::string(*value) +
           "', which this runtime cannot decode (expected '" +
           std::string(kRotationMarkerValue) + "')");
      marker_ok = false;
    } else {
      marker_ok = true;
    }
    return marker_ok;
  }

  /// Resolves the pairing table (I4R-GGUF-SPEC.md section 5) for one layer.
  /// A layer is either fully rotated or not rotated at all; a mix is rejected
  /// because half-rotated experts decode to plausible garbage.
  void BindRotation(const Config& c, std::uint32_t il, LayerWeights& l) {
    const bool gate_rot = l.ffn_gate_exps.type == GgmlType::kI4R128;
    const bool up_rot = l.ffn_up_exps.type == GgmlType::kI4R128;
    const bool down_rot = l.ffn_down_exps.type == GgmlType::kI4R128;
    if (!gate_rot && !up_rot && !down_rot) {
      return;
    }
    if (!gate_rot || !up_rot || !down_rot) {
      Fail("layer " + std::to_string(il) +
           " mixes rotated (I4R128) and plain expert tensors; gate, up and "
           "down must share one format");
      return;
    }
    if (!RotationMarkerOk()) {
      return;
    }
    const std::string p = "blk." + std::to_string(il) + ".";
    const auto f16 = {GgmlType::kF16};
    l.rotation.su_experts =
        Get(p + "ffn_experts_rot.su", c.hidden_size, 1, 1, f16);
    l.rotation.sv_gate =
        Get(p + "ffn_gate_exps_rot.sv", c.expert_ff, 1, 1, f16);
    l.rotation.sv_up = Get(p + "ffn_up_exps_rot.sv", c.expert_ff, 1, 1, f16);
    l.rotation.su_down =
        Get(p + "ffn_down_exps_rot.su", c.expert_ff, 1, 1, f16);
    l.rotation.sv_down =
        Get(p + "ffn_down_exps_rot.sv", c.hidden_size, 1, 1, f16);
    if (!ok) {
      return;
    }
    l.rotation.rotated = true;
  }

  /// Binds `name` with the exact shape and one of the accepted formats.
  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::uint64_t experts, std::initializer_list<GgmlType> types,
                bool required = true) {
    TensorRef t;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      if (required) {
        Fail("missing tensor " + name);
      }
      return t;
    }
    const auto& d = info->dimensions;
    const std::uint64_t got_cols = d.size() > 0 ? d[0] : 1;
    const std::uint64_t got_rows = d.size() > 1 ? d[1] : 1;
    const std::uint64_t got_experts = d.size() > 2 ? d[2] : 1;
    if (got_cols != cols || got_rows != rows || got_experts != experts ||
        d.size() > 3) {
      Fail("tensor " + name + " has shape [" + std::to_string(got_cols) + ", " +
           std::to_string(got_rows) + ", " + std::to_string(got_experts) +
           "], expected [" + std::to_string(cols) + ", " +
           std::to_string(rows) + ", " + std::to_string(experts) + "]");
      return t;
    }
    // An HGN store stays in place when the role lists it (its kernel reads
    // it); otherwise, or with the decode-at-load reference switch, it is
    // decoded on the host below.
    bool keep_ht = false;
    if ((info->type == GgmlType::kHT4 || info->type == GgmlType::kQ6G64) &&
        !HtDecodeAtLoad()) {
      for (auto type : types) {
        keep_ht = keep_ht || type == info->type;
      }
    }
    if (refuse_hgn && quant::IsHgnStore(info->type)) {
      Fail("MTP sidecar tensor " + name + " is stored as " +
           std::string(core::ToString(info->type)) +
           "; convert the head with tools/hgn2gguf --mtp --mtp-stock-types");
      return TensorRef{};
    }
    if (quant::IsHgnStore(info->type) && info->type != GgmlType::kFP8G &&
        !keep_ht) {
      return DecodeHgn(name, info->type, info->data, info->name, cols, rows,
                       experts, types);
    }
    bool type_ok = false;
    for (auto type : types) {
      type_ok = type_ok || info->type == type;
    }
    const Format format = FormatOf(info->type);
    if (!type_ok || format.block == 0 || cols % format.block != 0) {
      Fail("tensor " + name + " has unsupported format " +
           std::string(core::ToString(info->type)));
      return t;
    }
    t.data = info->data;
    t.type = info->type;
    t.cols = cols;
    t.rows = rows;
    t.experts = experts;
    t.name = info->name;
    // Locate the shard so disk readers can address the payload directly, and
    // so a payload running past its shard is caught here rather than as a
    // fault later.
    const auto regions = reader.GetMappedRegions();
    const auto address = reinterpret_cast<std::uintptr_t>(info->data);
    bool inside = false;
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
      if (address >= base && address < base + regions[i].size) {
        t.shard = i;
        t.file_offset = address - base;
        inside = t.SizeBytes() <= regions[i].size - t.file_offset;
        break;
      }
    }
    if (!inside) {
      Fail("tensor " + name + " is truncated");
      return TensorRef{};
    }
    if (t.type == GgmlType::kHT4 && !BindHtVectors(name, t)) {
      return TensorRef{};
    }
    return t;
  }

  /// GUFO_HT_DECODE_AT_LOAD=1 (or GUFO_HGN_DECODE_AT_LOAD=1) decodes the ht
  /// and q6g64 matrices at load instead of running their kernels (the
  /// correctness reference path).
  static bool HtDecodeAtLoad() {
    for (const char* key : {"GUFO_HT_DECODE_AT_LOAD", "GUFO_HGN_DECODE_AT_LOAD"}) {
      const char* v = std::getenv(key);
      if (v != nullptr && v[0] == '1') {
        return true;
      }
    }
    return false;
  }

  bool BindHtVectors(const std::string& name, TensorRef& t) {
    if (!DenseRotationOk()) {
      return false;
    }
    const std::string base = name.substr(0, name.size() - 7);  // ".weight"
    const auto* su = reader.FindTensor(base + "_rot.su");
    const auto* sv = reader.FindTensor(base + "_rot.sv");
    const auto count = [](const auto* info) -> std::uint64_t {
      std::uint64_t n = 1;
      for (auto d : info->dimensions) {
        n *= d;
      }
      return n;
    };
    if (su == nullptr || sv == nullptr || su->type != GgmlType::kF16 ||
        sv->type != GgmlType::kF16 || count(su) != t.cols ||
        count(sv) != t.rows || t.rows % 128 != 0 || t.cols % 128 != 0 ||
        t.experts != 1) {
      Fail("ht tensor " + name + " needs 128-aligned shape and F16 vectors " +
           base + "_rot.su/.sv");
      return false;
    }
    t.rot_su = su->data;
    t.rot_sv = sv->data;
    return true;
  }

  /// Checks the dense rotation marker once (kHT4/kHT3 tensors).
  bool dense_marker_checked{false};
  bool dense_marker_ok{false};
  bool DenseRotationOk() {
    if (!dense_marker_checked) {
      dense_marker_checked = true;
      const auto value = reader.GetMetadataString("qwen4exp.dense.rotation");
      dense_marker_ok = value.has_value() && *value == "hadamard128";
      if (!dense_marker_ok) {
        Fail("rotated ht tensors present but qwen4exp.dense.rotation is not "
             "'hadamard128'; refusing to read rotated codes as plain");
      }
    }
    return dense_marker_ok;
  }

  /// Decodes an HGN store (src/core/quant/hgn_decode.hpp) on the host into
  /// the role's F16 (preferred) or F32 format. This is the load-time path:
  /// correct, but it reads full-precision weights at run time. Stores in
  /// stacked expert tensors need dedicated kernels and are refused here.
  TensorRef DecodeHgn(const std::string& name, GgmlType store,
                      const void* payload, std::string_view tensor_name,
                      std::uint64_t cols, std::uint64_t rows,
                      std::uint64_t experts,
                      std::initializer_list<GgmlType> types) {
    bool f16 = false;
    bool f32 = false;
    for (auto type : types) {
      f16 = f16 || type == GgmlType::kF16;
      f32 = f32 || type == GgmlType::kF32;
    }
    if (experts != 1 || (!f16 && !f32) || decoded == nullptr) {
      Fail("tensor " + name + " is stored as " +
           std::string(core::ToString(store)) +
           ", which this role cannot decode at load (needs an F16 or F32 "
           "route)");
      return {};
    }
    const std::uint16_t* su = nullptr;
    const std::uint16_t* sv = nullptr;
    if (quant::IsHgnRotated(store)) {
      if (!DenseRotationOk()) {
        return {};
      }
      const std::string base = name.substr(0, name.size() - 7);  // ".weight"
      const auto* a = reader.FindTensor(base + "_rot.su");
      const auto* b = reader.FindTensor(base + "_rot.sv");
      const auto count = [](const auto* info) -> std::uint64_t {
        std::uint64_t n = 1;
        for (auto d : info->dimensions) {
          n *= d;
        }
        return n;
      };
      if (a == nullptr || b == nullptr || a->type != GgmlType::kF16 ||
          b->type != GgmlType::kF16 || count(a) != cols || count(b) != rows) {
        Fail("rotated tensor " + name + " lacks F16 vectors " + base +
             "_rot.su [" + std::to_string(cols) + "] and " + base +
             "_rot.sv [" + std::to_string(rows) + "]");
        return {};
      }
      su = static_cast<const std::uint16_t*>(a->data);
      sv = static_cast<const std::uint16_t*>(b->data);
    }
    const std::size_t count = static_cast<std::size_t>(cols * rows);
    std::vector<float> values;
    auto buffer = std::make_shared<std::vector<std::uint8_t>>();
    try {
      values.resize(count);
      buffer->resize(count * (f16 ? 2 : 4));
    } catch (...) {
      Fail("out of host memory decoding " + name);
      return {};
    }
    std::string why;
    if (!quant::DecodeHgnToF32(store, payload, cols, rows, su, sv,
                               values.data(), &why)) {
      Fail("cannot decode " + name + ": " + why);
      return {};
    }
    if (f16) {
      quant::FloatToHalfBits(values.data(),
                             reinterpret_cast<std::uint16_t*>(buffer->data()),
                             count);
    } else {
      std::memcpy(buffer->data(), values.data(), count * sizeof(float));
    }
    TensorRef t;
    t.data = buffer->data();
    t.type = f16 ? GgmlType::kF16 : GgmlType::kF32;
    t.cols = cols;
    t.rows = rows;
    t.experts = 1;
    t.name = tensor_name;
    t.shard = TensorRef::kHostShard;
    decoded->push_back(std::move(buffer));
    return t;
  }

  HcMixer Mixer(const std::string& prefix, const Config& c, bool with_inject) {
    HcMixer m;
    const std::uint64_t hc_dim = c.HcDim();
    m.norm = Get(prefix + "_norm.weight", hc_dim, 1, 1, {GgmlType::kF32});
    m.down = Get(prefix + "_down.weight", hc_dim, c.hc_low_rank, 1,
                 {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16,
                  GgmlType::kF32, GgmlType::kQ6G64});
    m.up = Get(prefix + "_up.weight", c.hc_low_rank, hc_dim, 1,
               {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16,
                GgmlType::kF32, GgmlType::kQ6G64});
    if (with_inject) {
      m.inject = Get(prefix + "_inject.weight", hc_dim, c.hc_count, 1,
                     {GgmlType::kF32, GgmlType::kQ8_0, GgmlType::kBF16});
    }
    return m;
  }

  LayerWeights Layer(const Config& c, std::uint32_t il, bool linear, bool ple,
                     bool nextn) {
    LayerWeights l;
    l.linear = linear;
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::uint64_t hidden = c.hidden_size;
    const std::uint64_t hc_dim = c.HcDim();
    const auto dense = {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16,
                        GgmlType::kF32, GgmlType::kHT4};
    const auto experts = {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
                          GgmlType::kQ5_1, GgmlType::kQ8_0, GgmlType::kI4R128};

    l.hc_attn = Mixer(p + "hc_attn", c, true);
    l.hc_ffn = Mixer(p + "hc_ffn", c, true);

    if (linear) {
      l.ssm_qkv =
          Get(p + "attn_qkv.weight", hidden, c.SsmConvChannels(), 1, dense);
      l.ssm_gate =
          Get(p + "attn_gate.weight", hidden, c.SsmValueDim(), 1, dense);
      l.ssm_conv1d = Get(p + "ssm_conv1d.weight", c.ssm_conv_kernel,
                         c.SsmConvChannels(), 1, {GgmlType::kF32});
      l.ssm_alpha = Get(p + "ssm_alpha.weight", hidden, c.ssm_num_v_heads, 1,
                        {GgmlType::kF32, GgmlType::kBF16, GgmlType::kQ8_0});
      l.ssm_beta = Get(p + "ssm_beta.weight", hidden, c.ssm_num_v_heads, 1,
                       {GgmlType::kF32, GgmlType::kBF16, GgmlType::kQ8_0});
      l.ssm_dt =
          Get(p + "ssm_dt.bias", c.ssm_num_v_heads, 1, 1, {GgmlType::kF32});
      l.ssm_a = Get(p + "ssm_a", c.ssm_num_v_heads, 1, 1, {GgmlType::kF32});
      l.ssm_norm =
          Get(p + "ssm_norm.weight", c.ssm_head_dim, 1, 1, {GgmlType::kF32});
      l.ssm_out = Get(p + "ssm_out.weight", c.SsmValueDim(), hidden, 1, dense);
    } else {
      l.attn_q =
          Get(p + "attn_q.weight", hidden, 2 * c.AttentionQDim(), 1, dense);
      l.attn_k = Get(p + "attn_k.weight", hidden, c.AttentionKvDim(), 1, dense);
      l.attn_v = Get(p + "attn_v.weight", hidden, c.AttentionKvDim(), 1, dense);
      l.attn_out =
          Get(p + "attn_output.weight", c.AttentionQDim(), hidden, 1, dense);
      l.attn_q_norm =
          Get(p + "attn_q_norm.weight", c.head_dim, 1, 1, {GgmlType::kF32});
      l.attn_k_norm =
          Get(p + "attn_k_norm.weight", c.head_dim, 1, 1, {GgmlType::kF32});
      l.indexer_q = Get(p + "indexer.q_proj.weight", hidden,
                        c.indexer_heads * c.indexer_head_dim, 1, dense);
      l.indexer_k = Get(p + "indexer.k_proj.weight", hidden, c.indexer_head_dim,
                        1, dense);
      l.indexer_q_norm = Get(p + "indexer.q_norm.weight", c.indexer_head_dim, 1,
                             1, {GgmlType::kF32});
      l.indexer_k_norm = Get(p + "indexer.k_norm.weight", c.indexer_head_dim, 1,
                             1, {GgmlType::kF32});
    }

    if (ple) {
      const std::uint64_t ple_dim = c.PleEmbeddingDim();
      l.ple_key = Get(p + "ple_key.weight", ple_dim, hc_dim, 1, dense);
      l.ple_value = Get(p + "ple_value.weight", ple_dim, hidden, 1, dense);
      l.ple_norm_key =
          Get(p + "ple_norm_key.weight", hc_dim, 1, 1, {GgmlType::kF32});
      l.ple_norm_query =
          Get(p + "ple_norm_query.weight", hc_dim, 1, 1, {GgmlType::kF32});
      l.ple_norm_conv =
          Get(p + "ple_norm_conv.weight", hc_dim, 1, 1, {GgmlType::kF32});
      l.ple_conv1d = Get(p + "ple_conv1d.weight", c.ple_conv_kernel, hc_dim, 1,
                         {GgmlType::kF32});
    }

    l.router = Get(p + "ffn_gate_inp.weight", hidden, c.num_experts, 1,
                   {GgmlType::kF32});
    l.ffn_gate_exps = Get(p + "ffn_gate_exps.weight", hidden, c.expert_ff,
                          c.num_experts, experts);
    l.ffn_up_exps = Get(p + "ffn_up_exps.weight", hidden, c.expert_ff,
                        c.num_experts, experts);
    l.ffn_down_exps = Get(p + "ffn_down_exps.weight", c.expert_ff, hidden,
                          c.num_experts, experts);
    BindRotation(c, il, l);
    l.shexp_gate_inp =
        Get(p + "ffn_gate_inp_shexp.weight", hidden, 1, 1, {GgmlType::kF32});
    l.shexp_gate =
        Get(p + "ffn_gate_shexp.weight", hidden, c.shared_expert_ff, 1, dense);
    l.shexp_up =
        Get(p + "ffn_up_shexp.weight", hidden, c.shared_expert_ff, 1, dense);
    l.shexp_down =
        Get(p + "ffn_down_shexp.weight", c.shared_expert_ff, hidden, 1, dense);

    if (nextn) {
      l.nextn_enorm =
          Get(p + "nextn.enorm.weight", hidden, 1, 1, {GgmlType::kF32});
      l.nextn_hnorm =
          Get(p + "nextn.hnorm.weight", hc_dim, 1, 1, {GgmlType::kF32});
      l.nextn_eh_proj =
          Get(p + "nextn.eh_proj.weight", 2 * hidden, hidden, 1, dense);
      l.nextn_head = Mixer(p + "nextn.hc_head", c, false);
    }
    return l;
  }
};

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  if (type == GgmlType::kQ6G64) {
    return cols % 64 == 0 ? static_cast<std::size_t>((13 * cols / 16 + 15) / 16 * 16)
                          : 0;
  }
  const Format f = FormatOf(type);
  if (f.block == 0 || cols % f.block != 0) {
    return 0;
  }
  return static_cast<std::size_t>(cols / f.block) * f.bytes;
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               std::string* error_msg) {
  auto config = Config::FromGguf(reader, true, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  return BindWithConfig(reader, *config, error_msg);
}

std::optional<ModelWeights> ModelWeights::BindWithConfig(
    const core::GgufReader& reader, const Config& config,
    std::string* error_msg) {
  Binder b{reader, error_msg};
  ModelWeights w;
  b.decoded = &w.decoded;
  w.config = config;
  const Config& c = w.config;
  const auto dense = {GgmlType::kQ8_0, GgmlType::kQ6_K, GgmlType::kBF16,
                      GgmlType::kF16,  GgmlType::kF32, GgmlType::kHT4};

  {
    // The vocabulary is the embedding row count; no metadata key carries it.
    const auto* info = reader.FindTensor("token_embd.weight");
    if (info == nullptr || info->dimensions.size() != 2) {
      b.Fail("missing token_embd.weight");
      return std::nullopt;
    }
    w.config.vocab_size = static_cast<std::uint32_t>(info->dimensions[1]);
  }
  w.token_embd =
      b.Get("token_embd.weight", c.hidden_size, c.vocab_size, 1, dense);
  w.output =
      b.Get("output.weight", c.hidden_size, c.vocab_size, 1, dense, false);
  if (w.output.empty()) {
    w.output = w.token_embd;
  }
  w.hc_head = b.Mixer("output_hc", c, false);
  if (c.ple_layer >= 0) {
    const auto table_file = reader.GetMetadataString("qwen4exp.ple.table_file");
    if (reader.FindTensor("per_layer_token_embd.weight") == nullptr &&
        table_file.has_value() && !table_file->empty()) {
      // A separate table GGUF next to the trunk; the caller binds it.
      w.ple_table_file = std::string(*table_file);
    } else if (auto table = BindPleTable(reader, c, error_msg)) {
      w.ple_table = *table;
    } else {
      b.ok = false;
    }
  }
  w.layers.reserve(c.num_layers);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    w.layers.push_back(
        b.Layer(c, il, c.IsLinearLayer(il), c.IsPleLayer(il), false));
  }
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

std::optional<MtpWeights> MtpWeights::Bind(const core::GgufReader& reader,
                                           const Config& trunk,
                                           std::string* error_msg) {
  auto config = Config::FromGguf(reader, false, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  if (!config->MtpMatches(trunk)) {
    b.Fail(
        "MTP sidecar architecture or inference constants differ from the "
        "trunk");
    return std::nullopt;
  }
  MtpWeights m;
  b.decoded = &m.decoded;
  b.refuse_hgn = true;
  m.config = *config;
  m.config.vocab_size = trunk.vocab_size;
  // The draft block is a full-attention layer at index num_layers.
  m.block = b.Layer(m.config, m.config.num_layers, false, false, true);
  if (!b.ok) {
    return std::nullopt;
  }
  return m;
}

std::optional<TensorRef> BindPleTable(const core::GgufReader& reader,
                                      const Config& c,
                                      std::string* error_msg) {
  Binder b{reader, error_msg};
  // The converter may pad the table beyond the hashed range, so take the
  // row count from the file and only check that it covers every head.
  const auto* info = reader.FindTensor("per_layer_token_embd.weight");
  const std::uint64_t rows = info != nullptr && info->dimensions.size() == 2
                                 ? info->dimensions[1]
                                 : 0;
  if (rows < c.ple_rows) {
    b.Fail("per_layer_token_embd.weight is missing or too short");
    return std::nullopt;
  }
  auto t = b.Get("per_layer_token_embd.weight", c.ple_head_dim, rows, 1,
                 {GgmlType::kIQ4_NL, GgmlType::kQ8_0, GgmlType::kBF16,
                  GgmlType::kFP8G});
  if (!b.ok) {
    return std::nullopt;
  }
  return t;
}

float Fp8gTableScale(const TensorRef& table) noexcept {
  if (table.type != GgmlType::kFP8G || table.empty()) {
    return 0.0f;
  }
  const std::size_t codes = static_cast<std::size_t>(table.rows) * 160;
  const std::size_t at = (codes + 63) / 64 * 64;
  float scale = 0.0f;
  std::memcpy(&scale, static_cast<const std::uint8_t*>(table.data) + at,
              sizeof(scale));
  return scale;
}

std::size_t ExpandedQ8_0Tensors(const LayerWeights& layer) noexcept {
  std::size_t count = 0;
  for (const TensorRef* t : {&layer.hc_attn.inject, &layer.hc_ffn.inject,
                             &layer.ssm_alpha, &layer.ssm_beta}) {
    count += !t->empty() && t->type == GgmlType::kQ8_0;
  }
  return count;
}

}  // namespace gufo::models::qwen38_flash_next
