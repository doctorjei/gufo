// CPU decoders for the stores of halogen's published HGN checkpoints, as kept
// in GGUF by tools/hgn2gguf (types kHT4..kQ8G64, see TYPES.md there). The
// formats are the public hgn-spec v1.1.0 (MIT OR CC-BY-4.0), STORAGE.md;
// tests/core/hgn_decode_test.cpp checks them against its TEST-VECTORS.md.
//
// These are load-time decoders: they turn a payload into the true weights so
// the existing F16/F32 kernels can run them. Fast kernels that read the
// payloads in place are separate work.

#ifndef GUFO_CORE_QUANT_HGN_DECODE_HPP_
#define GUFO_CORE_QUANT_HGN_DECODE_HPP_

#include <cstddef>
#include <cstdint>
#include <string>

#include "src/core/gguf_reader.hpp"

namespace gufo::quant {

/// True for the HGN store types (kHT4, kHT3, kQ6G64, kFP8G, kQ4C, kQ8G64).
[[nodiscard]] constexpr bool IsHgnStore(core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kHT4:
    case core::GgmlType::kHT3:
    case core::GgmlType::kQ6G64:
    case core::GgmlType::kFP8G:
    case core::GgmlType::kQ4C:
    case core::GgmlType::kQ8G64:
      return true;
    default:
      return false;
  }
}

/// True for the stores in the rotated (128-point Hadamard) basis.
[[nodiscard]] constexpr bool IsHgnRotated(core::GgmlType type) noexcept {
  return type == core::GgmlType::kHT4 || type == core::GgmlType::kHT3;
}

/// Payload bytes of a tensor with `elements` values and row length `cols`
/// (GGUF ne0), or 0 when the shape breaks the store's rules.
[[nodiscard]] std::size_t HgnEncodedSizeBytes(core::GgmlType type,
                                              std::size_t elements,
                                              std::uint64_t cols) noexcept;

/// Decodes a whole tensor of `rows` rows of `cols` values into row-major F32
/// true weights. Rotated stores need `su` (cols f16 bit patterns) and `sv`
/// (rows f16 bit patterns); the others ignore them. kFP8G is not decoded here
/// (the n-gram table is read row by row; see Fp8Value).
/// Uses up to `threads` threads (0 = hardware concurrency). Returns false
/// with `error` set on an unsupported type or shape.
bool DecodeHgnToF32(core::GgmlType type, const void* payload,
                    std::uint64_t cols, std::uint64_t rows,
                    const std::uint16_t* su, const std::uint16_t* sv,
                    float* out, std::string* error,
                    unsigned threads = 0) noexcept;

/// Decodes a whole tensor straight into GGUF Q8_0 blocks (34 bytes per 32
/// values, row-major), a band or a row at a time, so no full F32 copy is
/// staged. Q8_0 re-quantizes the true weights (about 0.4% relative error): use
/// it only where the values steer and do not decide, such as a draft head
/// whose proposals the full model verifies. Same types and arguments as
/// DecodeHgnToF32; `out` must hold rows * cols / 32 * 34 bytes.
bool DecodeHgnToQ8_0(core::GgmlType type, const void* payload,
                     std::uint64_t cols, std::uint64_t rows,
                     const std::uint16_t* su, const std::uint16_t* sv,
                     void* out, std::string* error,
                     unsigned threads = 0) noexcept;

/// The value of one fp8 code (STORAGE.md "The fp8 encoding": bias 7, no
/// infinity or NaN, 0x7F = 480).
[[nodiscard]] float Fp8Value(std::uint8_t code) noexcept;

/// The ht codebook 1 value of a 16-bit state, as float (exactly the f16).
[[nodiscard]] float HtCodebook(std::uint16_t state) noexcept;

/// The 256 trellis states of one ht tile (`b` = 3 or 4), in value order.
void HtTileStates(const std::uint8_t* tile, unsigned b,
                  std::uint16_t* states) noexcept;

/// Rounds F32 to F16 bit patterns, to nearest with ties to even.
void FloatToHalfBits(const float* src, std::uint16_t* dst,
                     std::size_t count) noexcept;

}  // namespace gufo::quant

#endif  // GUFO_CORE_QUANT_HGN_DECODE_HPP_
