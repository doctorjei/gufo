#!/usr/bin/env python3
"""Check hgnfmt's decoders against hgn-spec v1.1.0 TEST-VECTORS.md.
Run: python3 -I test_vectors.py   (exit 0 = all pass)."""
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hgnfmt as H  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


# Checksum
check("checksum", H.xor_fold(bytes(range(1, 11))) == 0x0C040E0D)

# fp8 codes
t = H.fp8_table()
for c, v in [(0x00, 0.0), (0x01, 0.001953125), (0x07, 0.013671875), (0x08, 0.015625),
             (0x38, 1.0), (0x3F, 1.875), (0x40, 2.0), (0x77, 240.0), (0x7E, 448.0),
             (0x7F, 480.0), (0xB8, -1.0), (0xFF, -480.0)]:
    check(f"fp8 {c:#04x}", t[c] == v, f"{t[c]} vs {v}")
check("fp8 0x80 is -0", t[0x80] == 0 and np.signbit(t[0x80]))

# ht codebook
cb = H.ht_codebook()
for s, bits in [(0x0000, 0xC2E8), (0x0001, 0x3921), (0x0002, 0xB72D), (0x1234, 0xB89E),
                (0x8000, 0xBAF2), (0xABCD, 0xBD42), (0xFFFF, 0xB936)]:
    got = int(cb[s:s + 1].view(np.uint16)[0])
    check(f"ht codebook {s:#06x}", got == bits, f"{got:#06x}")
check("ht codebook distinct", len(np.unique(cb)) == 913)
check("ht codebook range", float(cb.min()) == -3.453125 and float(cb.max()) == 3.34765625)


def golden_words(n):
    w = (np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x9E3779B9)) & np.uint64(0xFFFFFFFF)
    return w.astype("<u4").tobytes()


# ht tile, b = 4
tile4 = golden_words(32)
check("ht tile4 bytes", tile4[:32].hex() == "b979379e72f36e3c2b6da6dae4e6dd789d60151756da4cb50f548453c8cdbbf1")
st = H.ht_tile_states(np.frombuffer(tile4, np.uint8).reshape(1, 128), 4)[0]
for i, s in [(0, 0x7209), (1, 0x209E), (2, 0x09E3), (7, 0x79B9), (8, 0x9B93),
             (127, 0x9B90), (128, 0xB908), (255, 0x3720)]:
    check(f"ht4 state {i}", int(st[i]) == s, f"{int(st[i]):#06x}")
vals = cb[st]
check("ht4 sha", hashlib.sha256(vals.astype("<f2").tobytes()).hexdigest()
      == "32ae1812fa6a2dc63150621a31ec916757607f03eb9058d51923ab7b92a78d9b")
dec = H.ht_decode_tiles(np.frombuffer(tile4, np.uint8).reshape(1, 128), 4)[0]
for i, (r, c), v in [(0, (0, 0), 1.48046875), (8, (1, 0), 0.1064453125),
                     (128, (0, 8), -1.341796875), (255, (15, 15), 1.568359375)]:
    check(f"ht4 pos {i}", float(dec[r, c]) == v, f"{float(dec[r, c])}")

# ht tile, b = 3
tile3 = tile4[:96]
st3 = H.ht_tile_states(np.frombuffer(tile3, np.uint8).reshape(1, 96), 3)[0]
for i, s in [(0, 0x4AC4), (1, 0x5627), (2, 0xB13C), (7, 0x3779), (8, 0xBBCD),
             (127, 0xB4AC), (128, 0xA560), (255, 0x6958)]:
    check(f"ht3 state {i}", int(st3[i]) == s, f"{int(st3[i]):#06x}")
check("ht3 sha", hashlib.sha256(cb[st3].astype("<f2").tobytes()).hexdigest()
      == "415420d5d59d5f2b288f563473b597b207e94852290120291887286c5f00555b")

# 128-point rotation of a [128,128] ht tensor
pay = golden_words(2048)
check("rot payload sha", hashlib.sha256(pay).hexdigest()
      == "cfab929290971c665ca08cb3222f626e41bfb1b013837ced3b7178136b8f2ca1")
M = H.ht_decode_bands(pay, 128, 4, 0, 1)
c = np.arange(128)
su = np.where((5 * c) % 7 < 3, -1.0, 1.0)
sv = (8 + (c % 8)) / 512.0 * np.where(c % 3 == 0, -1.0, 1.0)
W = H.rotate(M, su, sv)
for (r, cc), v in [((0, 0), -0.0005001630634069443), ((0, 1), -0.0008721128106117249),
                   ((1, 0), -0.021587662398815155), ((1, 1), 0.006169011816382408),
                   ((64, 64), -0.002170378342270851), ((127, 127), 0.003455530386418104)]:
    check(f"rot ({r},{cc})", np.float32(W[r, cc]) == np.float32(v), f"{W[r, cc]!r}")
check("rot sha", hashlib.sha256(W.astype("<f4").tobytes()).hexdigest()
      == "1a1f0f4bee42296506b55f20ae44b69a6b558ee16bf5c70ec9519fa52cbfc1b2")

# q6g64
row = bytes.fromhex("0ba54fe9832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9832dc761"
                    "dc21b7d82176c82d768b1d728bdc6287002800bc000000000000000000000000")
v = H.q6g64_decode(row, 1, 64)[0]
for p, val in [(0, -0.65625), (1, 0.5), (2, -0.34375), (3, 0.8125), (4, -0.03125),
               (5, -0.875), (6, 0.28125), (7, -0.5625)]:
    check(f"q6g64 pos {p}", v[p] == val, f"{v[p]}")
codes = (37 * np.arange(64) + 11) % 64
check("q6g64 all", np.array_equal(v, (codes * 0.03125 - 1.0).astype(np.float32)))

# i4r row (HGN sequential pairs) and its GGUF I4R128 re-layout
cbytes = bytes.fromhex("832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9" * 2)
codes = np.frombuffer(cbytes, np.uint8).reshape(1, 64)
scale = np.array([[0.015625]], np.float16)
m = H.i4r_M(codes, scale)[0]
nib = (5 * np.arange(128) + 3) % 16
check("i4r M", np.array_equal(m, ((nib - 8) * 0.015625).astype(np.float32)))
g = H.i4r128_from_hgn(codes, scale)
check("i4r128 relayout", np.array_equal(H.i4r128_M(g.reshape(-1), 1, 128)[0], m))

# head permutation
table = {0: 0, 1: 16, 2: 32, 3: 1, 12: 4, 13: 20, 24: 8, 35: 43, 36: 12, 46: 31, 47: 47}
check("head t(h)", all(H.head_t(h) == t for h, t in table.items()))
perm = H.gguf_from_hgn_heads()
check("head inverse", all(perm[H.head_t(h)] == h for h in range(48)))

print(f"\n{'ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
