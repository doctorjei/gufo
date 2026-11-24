"""hgnfmt.py - HGN container reader, CPU decoders and a small GGUF reader.

Everything here follows the public hgn-spec v1.1.0 (MIT OR CC-BY-4.0,
https://github.com/jtsylve/hgn-spec): CONTAINER.md for the file, STORAGE.md
for the stores, TEST-VECTORS.md for the checks in test_vectors.py.

The decoders are reference code for verification, not fast paths: they turn
stored payloads into numbers so the converter's output can be compared with
the source and with an independent GGUF of the same model.
"""
import struct

import numpy as np

# ------------------------------------------------------------------ stores
ST_BF16, ST_F32, ST_F16, ST_I64 = 0, 1, 2, 4
ST_Q4C, ST_Q8G64, ST_FP8G, ST_HT, ST_I4R, ST_Q6G64 = 5, 7, 10, 16, 23, 24
STORE_NAME = {0: "bf16", 1: "f32", 2: "f16", 4: "i64", 5: "q4c", 7: "q8g64",
              10: "fp8g", 16: "ht", 23: "i4r", 24: "q6g64"}


def pad(n, a):
    return (n + a - 1) // a * a


# --------------------------------------------------------------- container
class Ent:
    __slots__ = ("name", "code", "shape", "off", "size", "ck", "var")

    def __init__(self, name, code, shape, off, size, ck, var):
        self.name, self.code, self.shape = name, code, tuple(shape)
        self.off, self.size, self.ck, self.var = off, size, ck, var

    @property
    def K(self):
        return self.shape[-1]

    @property
    def N(self):
        n = 1
        for d in self.shape[:-1]:
            n *= d
        return n

    def __repr__(self):
        return (f"Ent({self.name} {STORE_NAME.get(self.code, self.code)} "
                f"{list(self.shape)} var={self.var:#x} size={self.size})")


class Hgn:
    """Read-only HGN file: header and table checks per CONTAINER.md."""

    def __init__(self, path):
        self.path = path
        self.fh = open(path, "rb")
        h = self.fh.read(0x68)
        if len(h) < 0x68 or h[:4] != b"HGN1":
            raise ValueError(f"{path}: not an HGN file")
        self.version, = struct.unpack_from("<I", h, 4)
        if self.version not in (1, 2):
            raise ValueError(f"{path}: HGN version {self.version}")
        n, toff, _doff, fsize = struct.unpack_from("<QQQQ", h, 8)
        self.fh.seek(0, 2)
        if self.fh.tell() != fsize:
            raise ValueError(f"{path}: size field {fsize} != file size")
        self.identity = h[0x28:0x68].split(b"\0")[0].decode("ascii")
        if toff + 0xA0 * n > fsize:
            raise ValueError(f"{path}: table does not fit")
        self.fh.seek(toff)
        raw = self.fh.read(0xA0 * n)
        self.ents, spans = {}, []
        for i in range(n):
            e = raw[i * 0xA0:(i + 1) * 0xA0]
            name = e[:96].split(b"\0")[0].decode("ascii")
            code, rank = struct.unpack_from("<II", e, 0x60)
            shape = struct.unpack_from("<4q", e, 0x68)[:rank]
            off, size, ck, var = struct.unpack_from("<QQII", e, 0x88)
            if off + size > fsize:
                raise ValueError(f"{path}: {name} runs past the end")
            if name in self.ents:
                raise ValueError(f"{path}: duplicate name {name}")
            self.ents[name] = Ent(name, code, shape, off, size, ck, var)
            spans.append((off, off + size, name))
        spans.sort()
        for (a0, a1, an), (b0, _b1, bn) in zip(spans, spans[1:]):
            if b0 < a1:
                raise ValueError(f"{path}: payloads {an} and {bn} overlap")

    def __getitem__(self, name):
        return self.ents[name]

    def __contains__(self, name):
        return name in self.ents

    def read(self, ent, verify=True):
        """Whole payload as bytes; verifies the XOR-fold checksum."""
        self.fh.seek(ent.off)
        b = self.fh.read(ent.size)
        if len(b) != ent.size:
            raise IOError(f"{ent.name}: short read")
        if verify and xor_fold(b) != ent.ck:
            raise ValueError(f"{ent.name}: checksum mismatch")
        return b

    def stream(self, ent, chunk=64 << 20):
        """Yield the payload in chunks (multiples of 4) and check the
        checksum at the end. Raises after the last chunk on a mismatch."""
        assert chunk % 4 == 0
        self.fh.seek(ent.off)
        rem, acc = ent.size, 0
        while rem:
            b = self.fh.read(min(chunk, rem))
            if not b:
                raise IOError(f"{ent.name}: short read")
            acc ^= xor_fold(b)
            rem -= len(b)
            yield b
        if acc != ent.ck:
            raise ValueError(f"{ent.name}: checksum mismatch")

    def read_at(self, ent, rel, n):
        self.fh.seek(ent.off + rel)
        b = self.fh.read(n)
        if len(b) != n:
            raise IOError(f"{ent.name}: short read at +{rel}")
        return b


def xor_fold(b):
    """CONTAINER.md checksum: XOR of little-endian u32 words, last word
    zero-filled."""
    if len(b) % 4:
        b = bytes(b) + b"\0" * (4 - len(b) % 4)
    if not b:
        return 0
    return int(np.bitwise_xor.reduce(np.frombuffer(b, "<u4")))


# ------------------------------------------------------------------- fp8
def fp8_table():
    """STORAGE.md 'The fp8 encoding': bias 7, no inf/NaN, 0x7F = 480."""
    t = np.empty(256, np.float64)
    for c in range(256):
        e, m = (c >> 3) & 15, c & 7
        v = m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        t[c] = -v if c & 0x80 else v
    return t


# -------------------------------------------------------------------- ht
_HT_TABLE = None


def ht_codebook():
    """Codebook 1 for all 65,536 states, as float16."""
    global _HT_TABLE
    if _HT_TABLE is None:
        s = np.arange(65536, dtype=np.uint64)
        x = (s * np.uint64(0x83DCD12D)) & np.uint64(0xFFFFFFFF)
        a = ((x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF)
             + (x >> 24)).astype(np.int64)
        _HT_TABLE = ((1774 * a - 905216) / 2.0 ** 18).astype(np.float16)
    return _HT_TABLE


def _ht_index(b):
    L = 256 * b
    i = np.arange(256)
    start = (b * (i + 1) - 16) % L
    return (start[:, None] + np.arange(16)[None, :]) % L     # [256, 16]


def ht_tile_states(tiles_u8, b):
    """tiles_u8: uint8 [n, 32*b] (each tile 8b little-endian u32 words).
    Returns uint16 states [n, 256] in value order i."""
    n = tiles_u8.shape[0]
    words = np.ascontiguousarray(tiles_u8).view("<u4")       # [n, 8b]
    be = words.astype(">u4").view(np.uint8).reshape(n, 32 * b)
    bits = np.unpackbits(be, axis=1)                         # MSB-first per word
    idx = _ht_index(b)
    w = (1 << np.arange(15, -1, -1)).astype(np.uint32)
    return (bits[:, idx].astype(np.uint32) * w).sum(-1).astype(np.uint16)


# value i of a tile sits at tile row (i/8) mod 16, tile column 8(i/128) + i mod 8
_I = np.arange(256)
HT_POS = ((_I // 8) % 16) * 16 + 8 * (_I // 128) + (_I % 8)  # row-major slot


def ht_decode_tiles(tiles_u8, b):
    """float16 [n, 16, 16] tile values (row-major within each tile)."""
    vals = ht_codebook()[ht_tile_states(tiles_u8, b)]
    out = np.empty_like(vals)
    out[:, HT_POS] = vals
    return out.reshape(-1, 16, 16)


def ht_bits(var):
    if var not in (0x1206, 0x1208):
        raise ValueError(f"ht variant {var:#x} not established")
    return (var & 0xFF) // 2


def ht_band_bytes(K, b):
    return 128 * K * b // 8


def ht_decode_bands(payload, K, b, band_lo, band_hi):
    """Stored numbers M (float32 [rows, K]) for 128-row bands
    [band_lo, band_hi) of a 2-D ht payload (STORAGE.md 'Tiles')."""
    tb = 32 * b
    bb = ht_band_bytes(K, b)
    raw = np.frombuffer(payload, np.uint8)[band_lo * bb:band_hi * bb]
    nb = band_hi - band_lo
    t = ht_decode_tiles(raw.reshape(-1, tb), b)              # [nb*KT*8,16,16]
    KT = K // 16
    t = t.reshape(nb, KT, 8, 16, 16).transpose(0, 2, 3, 1, 4)
    return t.reshape(nb * 128, K).astype(np.float32)


# -------------------------------------------------------------- rotation
_H = None


def h128():
    """Sylvester Hadamard, entries +-1 (exact)."""
    global _H
    if _H is None:
        H = np.array([[1.0]])
        while H.shape[0] < 128:
            H = np.block([[H, H], [H, -H]])
        _H = H
    return _H


def rotate(M, su, sv):
    """W = diag(sv) Hn M Hn diag(su), Hn = H/sqrt(128), block-diagonal.
    M float [N, K] with N, K multiples of 128. Uses the +-1 matrix and one
    division by 128, in float64: exact for f16 inputs (STORAGE.md)."""
    N, K = M.shape
    H = h128()
    A = np.einsum("rgb,ba->rga", M.reshape(N, K // 128, 128).astype(np.float64), H)
    B = np.einsum("ab,pbc->pac", H, A.reshape(N // 128, 128, K)) / 128.0
    return sv[:, None].astype(np.float64) * B.reshape(N, K) * su[None, :].astype(np.float64)


# ----------------------------------------------------------------- q6g64
def q6g64_row_bytes(K):
    return pad(13 * K // 16, 16)


def q6g64_decode(payload, N, K, rows=None):
    S = q6g64_row_bytes(K)
    raw = np.frombuffer(payload, np.uint8)[:N * S].reshape(N, S)
    if rows is not None:
        raw = raw[rows]
    lo = np.empty((raw.shape[0], K), np.uint8)
    lo[:, 0::2] = raw[:, :K // 2] & 15
    lo[:, 1::2] = raw[:, :K // 2] >> 4
    hb = raw[:, K // 2:3 * K // 4]
    hi = np.empty((raw.shape[0], K), np.uint8)
    for j in range(4):
        hi[:, j::4] = (hb >> (2 * j)) & 3
    c = (lo | (hi << 4)).astype(np.float32)
    rec = raw[:, 3 * K // 4:13 * K // 16].copy().view(np.float16).reshape(-1, K // 64, 2)
    sc = rec[:, :, 0].astype(np.float32).repeat(64, axis=1)
    bi = rec[:, :, 1].astype(np.float32).repeat(64, axis=1)
    return c * sc + bi


# ----------------------------------------------------------------- q8g64
def q8g64_decode(payload, N, K, rows=None):
    S = K + K // 16
    raw = np.frombuffer(payload, np.uint8)[:N * S].reshape(N, S)
    if rows is not None:
        raw = raw[rows]
    c = raw[:, :K].astype(np.float32)
    rec = raw[:, K:].copy().view(np.float16).reshape(-1, K // 64, 2)
    sc = rec[:, :, 0].astype(np.float32).repeat(64, axis=1)
    bi = rec[:, :, 1].astype(np.float32).repeat(64, axis=1)
    return c * sc + bi


# ------------------------------------------------------------ q4c var 2
def q4c_layout(N, K):
    S = 64 + pad(N * K // 2, 64)
    R = pad(K // 16, 16)
    return S, R, S + N * R


def q4c_decode(payload, N, K, rows=None):
    S, R, _ = q4c_layout(N, K)
    p = np.frombuffer(payload, np.uint8)
    cb = p[:64].copy().view("<f4")
    codes = p[64:64 + N * K // 2].reshape(N, K // 2)
    sc = p[S:S + N * R].reshape(N, R)[:, :K // 16].copy().view(np.float16)
    if rows is not None:
        codes, sc = codes[rows], sc[rows]
    c = np.empty((codes.shape[0], K), np.uint8)
    c[:, 0::2] = codes & 15
    c[:, 1::2] = codes >> 4
    return cb[c] * sc.astype(np.float32).repeat(32, axis=1)


def q4c_permute_rows(payload, N, K, perm):
    """New payload whose row j is source row perm[j] (codes and scales)."""
    S, R, total = q4c_layout(N, K)
    p = np.frombuffer(payload, np.uint8)
    out = bytearray(payload)
    o = np.frombuffer(out, np.uint8)
    codes = p[64:64 + N * K // 2].reshape(N, K // 2)
    o[64:64 + N * K // 2] = codes[perm].reshape(-1)
    sc = p[S:S + N * R].reshape(N, R)
    o[S:S + N * R] = sc[perm].reshape(-1)
    return bytes(out)


def q4c_select_rows(payload, N, K, rows):
    """New q4c variant-2 payload holding only `rows` (same codebook)."""
    S, R, _ = q4c_layout(N, K)
    p = np.frombuffer(payload, np.uint8)
    n2 = len(rows)
    S2, R2, total2 = q4c_layout(n2, K)
    out = np.zeros(total2, np.uint8)
    out[:64] = p[:64]
    out[64:64 + n2 * K // 2] = p[64:64 + N * K // 2].reshape(N, K // 2)[rows].reshape(-1)
    out[S2:S2 + n2 * R2] = p[S:S + N * R].reshape(N, R)[rows].reshape(-1)
    return out.tobytes()


def q8g64_concat_k(a, b, N, Ka, Kb):
    """Row-wise concatenation [A | B] of two q8g64 matrices with N rows:
    codes of A then B, then the records of A then B (exact)."""
    ra = np.frombuffer(a, np.uint8).reshape(N, Ka + Ka // 16)
    rb = np.frombuffer(b, np.uint8).reshape(N, Kb + Kb // 16)
    return np.concatenate([ra[:, :Ka], rb[:, :Kb], ra[:, Ka:], rb[:, Kb:]], axis=1).tobytes()


# ------------------------------------------------------------------- i4r
def i4r_M(codes_u8, scales_f16):
    """HGN i4r rows -> stored numbers M. codes: [n, K/2] sequential pairs;
    offset binary c = nibble - 8."""
    n, hb = codes_u8.shape
    v = np.empty((n, hb * 2), np.int8)
    v[:, 0::2] = (codes_u8 & 15).astype(np.int8) - 8
    v[:, 1::2] = (codes_u8 >> 4).astype(np.int8) - 8
    return v.astype(np.float32) * scales_f16.astype(np.float32).repeat(128, axis=1)


def i4r128_from_hgn(codes_u8, scales_f16):
    """HGN i4r rows -> GGUF I4R128 bytes [n, K/128, 66] (GGUF nibble order:
    byte b of a block holds value b low, value 64+b high)."""
    n, hb = codes_u8.shape
    K = hb * 2
    G = K // 128
    vals = np.empty((n, K), np.uint8)
    vals[:, 0::2] = codes_u8 & 15
    vals[:, 1::2] = codes_u8 >> 4
    blk = vals.reshape(n, G, 128)
    out = np.empty((n, G, 66), np.uint8)
    out[:, :, :64] = blk[:, :, :64] | (blk[:, :, 64:] << 4)
    out[:, :, 64:] = scales_f16.reshape(n, G).copy().view(np.uint8).reshape(n, G, 2)
    return out


def i4r128_M(payload_u8, n, K):
    G = K // 128
    p = payload_u8.reshape(n, G, 66)
    vals = np.empty((n, G, 128), np.int16)
    vals[:, :, :64] = p[:, :, :64] & 15
    vals[:, :, 64:] = p[:, :, :64] >> 4
    sc = p[:, :, 64:66].copy().view(np.float16).reshape(n, G, 1)
    return ((vals - 8).astype(np.float32) * sc.astype(np.float32)).reshape(n, K)


def quant_q8_0(rows):
    """float [n, K] -> GGUF Q8_0 bytes (per 32 values: f16 d = amax/127, then
    int8 round(x/d)). Re-quantizes: for draft heads only."""
    n, K = rows.shape
    b = rows.astype(np.float32).reshape(n, K // 32, 32)
    d = np.abs(b).max(axis=2) / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0.0)
    q = np.clip(np.rint(b * inv[:, :, None]), -127, 127).astype(np.int8)
    out = np.empty((n, K // 32, 34), np.uint8)
    out[:, :, :2] = d.astype(np.float16).view(np.uint8).reshape(n, K // 32, 2)
    out[:, :, 2:] = q.view(np.uint8)
    return out.tobytes()


# -------------------------------------------------------------- bf16 etc
def bf16_to_f32(b):
    u = np.frombuffer(b, "<u2").astype(np.uint32) << 16
    return u.view(np.float32)


# ------------------------------------------------------- head permutation
def head_t(h):
    """GGUF-IMPORT.md 'Head reorder': HGN value head h is GGUF head t(h)."""
    return h // 3 + 16 * (h % 3)


def gguf_from_hgn_heads(n=48):
    """perm with gguf[g] = hgn[perm[g]]."""
    perm = np.empty(n, np.int64)
    for h in range(n):
        perm[head_t(h)] = h
    return perm


# ------------------------------------------------------ minimal GGUF read
_GT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
       7: "<B", 10: "<Q", 11: "<q", 12: "<d"}


class Gguf:
    """Header, KV and tensor-info reader that does not need to know the
    tensor types (so it reads the custom ones)."""

    def __init__(self, path):
        self.path = path
        f = self.fh = open(path, "rb")
        assert f.read(4) == b"GGUF"
        ver, nt, nkv = struct.unpack("<IQQ", f.read(20))
        assert ver == 3
        self.kv = {}
        for _ in range(nkv):
            k = self._str()
            t, = struct.unpack("<I", f.read(4))
            self.kv[k] = self._val(t)
        self.alignment = int(self.kv.get("general.alignment", 32))
        self.tensors = {}
        order = []
        for _ in range(nt):
            name = self._str()
            nd, = struct.unpack("<I", f.read(4))
            dims = list(struct.unpack(f"<{nd}Q", f.read(8 * nd)))
            typ, off = struct.unpack("<IQ", f.read(12))
            self.tensors[name] = [dims, typ, off, None]
            order.append(name)
        self.data = pad(f.tell(), self.alignment)
        self.order = order

    def _str(self):
        n, = struct.unpack("<Q", self.fh.read(8))
        return self.fh.read(n).decode("utf-8")

    def _val(self, t):
        if t == 8:
            return self._str()
        if t == 9:
            st, n = struct.unpack("<IQ", self.fh.read(12))
            return [self._val(st) for _ in range(n)]
        fmt = _GT[t]
        v, = struct.unpack(fmt, self.fh.read(struct.calcsize(fmt)))
        return bool(v) if t == 7 else v

    def read(self, name, nbytes, rel=0):
        dims, typ, off, _ = self.tensors[name]
        self.fh.seek(self.data + off + rel)
        b = self.fh.read(nbytes)
        if len(b) != nbytes:
            raise IOError(f"{self.path}: short read on {name}")
        return b
