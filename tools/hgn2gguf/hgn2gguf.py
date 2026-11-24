#!/usr/bin/env python3
"""hgn2gguf.py - halogen Flash-Next v2 (HGN) -> GGUF for Gufo, losslessly.

Takes every weight from the published v2 checkpoint (and the separate n-gram
table file) and writes a GGUF with the tensor names, shapes and metadata of
the stock Flash-Next GGUF (unsloth UD-Q4_K_XL, used as the skeleton for
names, order, shard split, tokenizer and hyper-parameters; none of its
weights are copied).

The quantized payloads keep their HGN bytes under new GGUF type ids (see
TYPES.md). Changes are only the ones a GGUF engine needs:
  * GGUF names and dimension order (GGUF-IMPORT.md "Name map", "Shapes");
  * DeltaNet value heads back from grouped to tiled order, by whole rows,
    128-row bands or 128-column blocks (GGUF-IMPORT.md "Head reorder",
    applied in reverse), with the rotation vectors permuted to match;
  * zero-centred norm gains -> GGUF gains (+1, f32); A_log -> ssm_a;
  * the fused indexer q/k matrix split back into q_proj and k_proj (by
    128-row bands, so ht tiles stay whole);
  * i4r experts re-laid out to the existing I4R128 GGUF type (lossless).
Every HGN payload read is checked against its XOR-fold checksum.

Usage:
  hgn2gguf.py --v2 V2.hgn --ngram NGRAM.hgn --skeleton 'DIR/*.gguf' \
      --out-prefix OUT [--only REGEX] [--dry-run]
"""
import argparse
import glob
import json
import os
import re
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hgnfmt as H  # noqa: E402

VERSION = "hgn2gguf 0.1 (hgn-spec 1.1.0)"

# GGUF type ids. 0/1/30 are upstream; 64 is Gufo's I4R128; 65.. are new
# (TYPES.md). They sit far above upstream GGML_TYPE_COUNT.
F32, F16, BF16 = 0, 1, 30
Q8_0 = 8
I4R128, HT4, HT3, Q6G64, FP8G, Q4C, Q8G64 = 64, 65, 66, 67, 68, 69, 70

ALIGN = 64
GV = dict(UINT8=0, INT8=1, UINT16=2, INT16=3, UINT32=4, INT32=5, FLOAT32=6,
          BOOL=7, STRING=8, ARRAY=9, UINT64=10, INT64=11, FLOAT64=12)
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
          10: "<Q", 11: "<q", 12: "<d"}

# Norm gains that GGUF stores with +1 (GGUF-IMPORT.md "Norm offsets").
OFFSET_SUFFIXES = ("hc_norm.weight", "self_attn.q_norm.weight",
                   "self_attn.k_norm.weight", "q_layernorm.weight",
                   "k_layernorm.weight", "ple.norm_key.weight",
                   "ple.norm_query.weight", "ple.norm_conv.weight",
                   "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight")

HEADS = H.gguf_from_hgn_heads(48)          # gguf head g <- hgn head HEADS[g]


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ------------------------------------------------------------------ plan
class Item:
    def __init__(self, name, dims, typ, nbytes, write, src=""):
        self.name, self.dims, self.typ = name, list(dims), typ
        self.nbytes, self.write, self.src = nbytes, write, src


class Converter:
    def __init__(self, v2, ngram):
        self.v2, self.ng = v2, ngram

    # ---- producers --------------------------------------------------
    def verbatim(self, h, ent):
        def w(f):
            for b in h.stream(ent):
                f.write(b)
        return w

    def f16_vec(self, ent, block_perm=None, sl=None):
        n = ent.shape[0]
        if ent.code != H.ST_F16 or len(ent.shape) != 1 or ent.size != 2 * n:
            raise ValueError(f"{ent.name}: expected 1-D f16 vector")

        def w(f):
            a = np.frombuffer(self.v2.read(ent), "<f2")
            if block_perm is not None:
                a = a.reshape(-1, 128)[block_perm].reshape(-1)
            if sl is not None:
                a = a[sl]
            f.write(a.tobytes())
        cnt = n if sl is None else len(range(n)[sl])
        return w, cnt

    def ht(self, ent, band_perm=None, col_perm=None, bands=None):
        if ent.code != H.ST_HT or len(ent.shape) != 2:
            raise ValueError(f"{ent.name}: expected 2-D ht")
        b = H.ht_bits(ent.var)
        N, K = ent.shape
        if N % 128 or K % 128 or ent.size != N * K * b // 8:
            raise ValueError(f"{ent.name}: ht shape/size rule")
        nb = N // 128 if bands is None else bands[1] - bands[0]

        def w(f):
            p = np.frombuffer(self.v2.read(ent), np.uint8).reshape(N // 128, -1)
            if bands is not None:
                p = p[bands[0]:bands[1]]
            if band_perm is not None:
                p = p[band_perm]
            if col_perm is not None:
                p = p.reshape(p.shape[0], K // 128, -1)[:, col_perm]
            f.write(np.ascontiguousarray(p).tobytes())
        return w, (HT4 if b == 4 else HT3), nb * 128, nb * 128 * K * b // 8

    def floats(self, ent, offset=False, perm=None, perm_unit=1, out=F32, neg_exp=False):
        if ent.code != H.ST_BF16:
            raise ValueError(f"{ent.name}: expected bf16, got {ent.code}")
        n = ent.N * ent.K

        def w(f):
            raw = self.v2.read(ent)
            if out == BF16 and not (offset or neg_exp or perm is not None):
                f.write(raw)
                return
            a = H.bf16_to_f32(raw).copy()
            if offset:
                a = (a + np.float32(1.0)).astype(np.float32)
            if neg_exp:
                a = (-np.exp(a.astype(np.float64))).astype(np.float32)
            if perm is not None:
                a = permute_units(a, perm, perm_unit)
            if out == F32:
                f.write(a.astype("<f4").tobytes())
            else:
                raise ValueError("bf16 output only for unchanged tensors")
        return w, n * (4 if out == F32 else 2)

    def q4c_rows(self, ent, perm=None):
        if ent.code != H.ST_Q4C or ent.var != 2:
            raise ValueError(f"{ent.name}: expected q4c variant 2")
        N, K = ent.N, ent.K
        if ent.size != H.q4c_layout(N, K)[2]:
            raise ValueError(f"{ent.name}: q4c size rule")

        def w(f):
            p = self.v2.read(ent)
            if perm is not None:
                p = H.q4c_permute_rows(p, N, K, perm)
            f.write(p)
        return w, ent.size

    def i4r_expert(self, ent, which, acc):
        """Stream gate|up|down of one layer as I4R128, expert by expert.
        The XOR checksum is accumulated on the gate and down passes, which
        between them read both payloads completely."""
        E, rows, K = ent.shape
        G = K // 128
        code_bytes = E * rows * K // 2
        if ent.code != H.ST_I4R or ent.size != code_bytes + E * rows * G * 2:
            raise ValueError(f"{ent.name}: i4r size rule")
        half = rows // 2
        out_rows = half if which in ("gate", "up") else rows
        nbytes = E * out_rows * G * 66
        check = which in ("gate", "down")

        def w(f):
            x = 0
            for e in range(E):
                c = np.frombuffer(self.v2.read_at(ent, e * rows * K // 2, rows * K // 2),
                                  np.uint8).reshape(rows, K // 2)
                sb = self.v2.read_at(ent, code_bytes + e * rows * G * 2, rows * G * 2)
                if check:
                    x ^= H.xor_fold(c.tobytes()) ^ H.xor_fold(sb)
                s = np.frombuffer(sb, "<f2").reshape(rows, G)
                if which == "gate":
                    c, s = c[:half], s[:half]
                elif which == "up":
                    c, s = c[half:], s[half:]
                f.write(H.i4r128_from_hgn(c, s).tobytes())
            if check and x != ent.ck:
                raise ValueError(f"{ent.name}: checksum mismatch")
        return w, nbytes


def permute_units(a, perm, unit):
    """Reorder the LAST len(perm) units of `unit` elements of a 1-D array
    (unit j of the result is source unit perm[j]); earlier elements stay.
    The DeltaNet value heads are always the tail (conv1d channels
    4096..10239) or the whole array (48 values)."""
    span = len(perm) * unit
    head, tail = a[:a.size - span], a[a.size - span:]
    tail = tail.reshape(len(perm), -1)[perm].reshape(-1)
    return np.concatenate([head, tail])


# --------------------------------------------------------- name mapping
def plan_tensor(cv, name, sdims, kvs):
    """Items replacing stock tensor `name` (stock dims `sdims`)."""
    v2 = cv.v2
    E = v2.ents
    items = []

    def add(gname, dims, typ, nbytes, w, src):
        items.append(Item(gname, dims, typ, nbytes, w, src))

    def rot(gbase, hbase, row_band_perm=None, col_block_perm=None, sv_sl=None):
        su_w, su_n = cv.f16_vec(E[hbase + ".suh"], block_perm=col_block_perm)
        sv_w, sv_n = cv.f16_vec(E[hbase + ".svh"], block_perm=row_band_perm, sl=sv_sl)
        add(gbase + "_rot.su", [su_n], F16, 2 * su_n, su_w, hbase + ".suh")
        add(gbase + "_rot.sv", [sv_n], F16, 2 * sv_n, sv_w, hbase + ".svh")

    def ht2(gname, hname, band_perm=None, col_perm=None, bands=None, sv_sl=None):
        w, typ, n_rows, nb = cv.ht(E[hname], band_perm, col_perm, bands)
        add(gname, [E[hname].K, n_rows], typ, nb, w, hname)
        hbase = hname[:-len(".weight")]
        rot(gname[:-len(".weight")], hbase,
            row_band_perm=None if band_perm is None else band_perm,
            col_block_perm=col_perm, sv_sl=sv_sl)

    def fl(gname, hname, dims, **kw):
        w, nb = cv.floats(E[hname], **kw)
        add(gname, dims, kw.get("out", F32), nb, w, hname)

    whole = {
        "token_embd.weight": "embed_tokens.weight",
        "output.weight": "lm_head.weight",
        "output_hc_down.weight": "hyper_connection_mixer.input_mix_weight_down.weight",
        "output_hc_up.weight": "hyper_connection_mixer.input_mix_weight_up.weight",
        "output_hc_norm.weight": "hyper_connection_mixer.hc_norm.weight",
    }
    if name == "per_layer_token_embd.weight":
        ent = cv.ng["layers.1.ple.ngram_embedding.weight"]
        rows = ent.shape[0] * ent.shape[1]
        if ent.code != H.ST_FP8G or ent.shape[2] != 160 or ent.size != H.pad(160 * rows, 64) + 4:
            raise ValueError("n-gram table: fp8g rule")
        add(name, [160, rows], FP8G, ent.size, cv.verbatim(cv.ng, ent), "ngram:" + ent.name)
        return items
    if name in whole:
        hn = whole[name]
        ent = E[hn]
        if ent.code == H.ST_Q4C:
            w, nb = cv.q4c_rows(ent)
            add(name, [ent.K, ent.N], Q4C, nb, w, hn)
        elif ent.code == H.ST_HT:
            ht2(name, hn)
        elif ent.code == H.ST_Q6G64:
            add(name, [ent.K, ent.N], Q6G64, ent.size, cv.verbatim(v2, ent), hn)
        elif ent.code == H.ST_BF16:
            fl(name, hn, sdims, offset=hn.endswith(OFFSET_SUFFIXES))
        else:
            raise ValueError(f"{name}: unhandled store {ent.code}")
        return items

    m = re.match(r"blk\.(\d+)\.(.+)$", name)
    if not m:
        raise KeyError(f"unmapped tensor {name}")
    L, suf = int(m.group(1)), m.group(2)
    p = f"layers.{L}."
    qkv_perm = np.concatenate([np.arange(32), 32 + HEADS])   # 80 bands

    simple_q6 = {"hc_attn_down.weight": "attn_hyper_connection.input_mix_weight_down.weight",
                 "hc_attn_up.weight": "attn_hyper_connection.input_mix_weight_up.weight",
                 "hc_attn_inject.weight": "attn_hyper_connection.block_inject_weight.weight",
                 "hc_ffn_down.weight": "mlp_hyper_connection.input_mix_weight_down.weight",
                 "hc_ffn_up.weight": "mlp_hyper_connection.input_mix_weight_up.weight",
                 "hc_ffn_inject.weight": "mlp_hyper_connection.block_inject_weight.weight"}
    if suf in simple_q6:
        ent = E[p + simple_q6[suf]]
        if ent.code != H.ST_Q6G64 or ent.var != 64 or ent.size != ent.N * H.q6g64_row_bytes(ent.K):
            raise ValueError(f"{ent.name}: q6g64 rule")
        add(name, [ent.K, ent.N], Q6G64, ent.size, cv.verbatim(v2, ent), ent.name)
        return items

    norms = {"hc_attn_norm.weight": "attn_hyper_connection.hc_norm.weight",
             "hc_ffn_norm.weight": "mlp_hyper_connection.hc_norm.weight",
             "attn_q_norm.weight": "self_attn.q_norm.weight",
             "attn_k_norm.weight": "self_attn.k_norm.weight",
             "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
             "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
             "ple_norm_key.weight": "ple.norm_key.weight",
             "ple_norm_query.weight": "ple.norm_query.weight",
             "ple_norm_conv.weight": "ple.norm_conv.weight",
             "ssm_norm.weight": "linear_attn.norm.weight"}
    if suf in norms:
        hn = p + norms[suf]
        fl(name, hn, sdims, offset=hn.endswith(OFFSET_SUFFIXES))
        return items

    ht_plain = {"attn_q.weight": "self_attn.q_proj.weight",
                "attn_k.weight": "self_attn.k_proj.weight",
                "attn_v.weight": "self_attn.v_proj.weight",
                "attn_output.weight": "self_attn.o_proj.weight",
                "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
                "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
                "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight"}
    if suf in ht_plain:
        ht2(name, p + ht_plain[suf])
        return items
    if suf == "attn_qkv.weight":
        ht2(name, p + "linear_attn.in_proj_qkv.weight", band_perm=qkv_perm)
        return items
    if suf == "attn_gate.weight":
        ht2(name, p + "linear_attn.in_proj_z.weight", band_perm=HEADS)
        return items
    if suf == "ssm_out.weight":
        ht2(name, p + "linear_attn.out_proj.weight", col_perm=HEADS)
        return items
    if suf in ("indexer.q_proj.weight", "indexer.k_proj.weight"):
        hn = p + "self_attn.indexer.index_qk_proj.weight"
        nq = sdims[1] if suf.startswith("indexer.q") else None
        q_rows = kvs["qwen4exp.attention.indexer.head_count"] * kvs["qwen4exp.attention.indexer.key_length"]
        total = E[hn].shape[0]
        if q_rows % 128 or (total - q_rows) % 128:
            raise ValueError(f"{hn}: q/k split not on 128-row bands")
        if suf.startswith("indexer.q"):
            ht2(name, hn, bands=(0, q_rows // 128), sv_sl=slice(0, q_rows))
        else:
            ht2(name, hn, bands=(q_rows // 128, total // 128), sv_sl=slice(q_rows, total))
        del nq
        return items
    if suf in ("ssm_alpha.weight", "ssm_beta.weight"):
        ent = E[p + ("linear_attn.in_proj_a.weight" if suf == "ssm_alpha.weight"
                     else "linear_attn.in_proj_b.weight")]
        w, nb = cv.q4c_rows(ent, perm=HEADS)
        add(name, [ent.K, ent.N], Q4C, nb, w, ent.name)
        return items
    if suf == "ssm_a":
        fl(name, p + "linear_attn.A_log", sdims, neg_exp=True, perm=HEADS, perm_unit=1)
        return items
    if suf == "ssm_dt.bias":
        fl(name, p + "linear_attn.dt_bias", sdims, perm=HEADS, perm_unit=1)
        return items
    if suf == "ssm_conv1d.weight":
        # HGN [C, 1, W] == GGUF [W, C] in memory; reorder value channels
        # 4096..10239 by 128-channel bands.
        fl(name, p + "linear_attn.conv1d.weight", sdims, perm=HEADS, perm_unit=128 * 4)
        return items
    if suf == "ple_conv1d.weight":
        fl(name, p + "ple.conv1d.weight", sdims)
        return items
    if suf in ("ple_key.weight", "ple_value.weight"):
        hn = p + ("ple.key_proj.weight" if suf == "ple_key.weight" else "ple.value_proj.weight")
        fl(name, hn, sdims, out=BF16)
        return items
    if suf == "ffn_gate_inp.weight":
        fl(name, p + "mlp.gate.weight", sdims)
        return items
    if suf == "ffn_gate_inp_shexp.weight":
        w, nb = cv.q4c_rows(E[p + "mlp.shared_expert_gate.weight"])
        add(name, sdims, Q4C, nb, w, p + "mlp.shared_expert_gate.weight")
        return items
    me = re.match(r"ffn_(gate|up|down)_exps\.weight$", suf)
    if me:
        which = me.group(1)
        hn = p + ("mlp.experts.down_proj.weight" if which == "down"
                  else "mlp.experts.gate_up_proj.weight")
        ent = E[hn]
        Ex, rows, K = ent.shape
        w, nb = cv.i4r_expert(ent, which, None)
        add(name, [K, rows // 2 if which != "down" else rows, Ex], I4R128, nb, w, hn)
        if which == "down":
            # Rotation vectors per docs/internal/design/i4r128/RFC.md.
            gu = E[p + "mlp.experts.gate_up_proj.weight"]
            half = gu.shape[1] // 2
            for gname, hname, sl in (
                    (f"blk.{L}.ffn_experts_rot.su", p + "mlp.experts.su", None),
                    (f"blk.{L}.ffn_gate_exps_rot.sv", p + "mlp.experts.gate_up_proj.svh", slice(0, half)),
                    (f"blk.{L}.ffn_up_exps_rot.sv", p + "mlp.experts.gate_up_proj.svh", slice(half, 2 * half)),
                    (f"blk.{L}.ffn_down_exps_rot.su", p + "mlp.experts.down_proj.suh", None),
                    (f"blk.{L}.ffn_down_exps_rot.sv", p + "mlp.experts.down_proj.svh", None)):
                vw, cnt = cv.f16_vec(E[hname], sl=sl)
                add(gname, [cnt], F16, 2 * cnt, vw, hname)
        return items
    raise KeyError(f"unmapped tensor {name}")


# ------------------------------------------------------------- MTP head
def plan_mtp(cv, name, sdims, stock_types=False):
    """blk.48.* of Gufo's MTP sidecar from v2's mtp.* tensors
    (PORTING.md "MTP head"; the head is the published head file's bytes).
    stock_types: write the stock sidecar's formats instead (q8g64 and q4c
    re-quantized to Q8_0, the shared-expert gate as F32), so the head runs on
    Gufo's stock MTP path. A draft head only steers speculation: the trunk
    verifies every token, so this changes acceptance, never output."""
    E = cv.v2.ents
    items = []

    def add(dims, typ, nbytes, w, src):
        items.append(Item(name, dims, typ, nbytes, w, src))

    def q8(ent, rows=None):
        if ent.code != H.ST_Q8G64 or ent.size != ent.N * (ent.K + ent.K // 16):
            raise ValueError(f"{ent.name}: q8g64 rule")
        rb = ent.K + ent.K // 16
        if stock_types:
            r0, r1 = (0, ent.N) if rows is None else (rows.start, rows.stop)

            def w(f):
                vals = H.q8g64_decode(cv.v2.read(ent), ent.N, ent.K, np.arange(r0, r1))
                f.write(H.quant_q8_0(vals))
            add([ent.K, r1 - r0], Q8_0, (r1 - r0) * ent.K // 32 * 34, w, ent.name)
            return
        if rows is None:
            add([ent.K, ent.N], Q8G64, ent.size, cv.verbatim(cv.v2, ent), ent.name)
        else:
            def w(f):
                f.write(cv.v2.read(ent)[rows.start * rb:rows.stop * rb])
            add([ent.K, rows.stop - rows.start], Q8G64, (rows.stop - rows.start) * rb, w, ent.name)

    m = re.match(r"blk\.48\.(.+)$", name)
    if not m:
        return items                      # output/token_embd: Gufo uses the trunk's
    suf = m.group(1)
    p = "mtp.layers.0."
    if suf == "nextn.eh_proj.weight":
        ea, eb = E["mtp.fc_embedding.weight"], E["mtp.fc_hidden.weight"]
        if stock_types:
            def w(f):
                va = H.q8g64_decode(cv.v2.read(ea), ea.N, ea.K)
                vb = H.q8g64_decode(cv.v2.read(eb), eb.N, eb.K)
                f.write(H.quant_q8_0(np.concatenate([va, vb], axis=1)))
            add([ea.K + eb.K, ea.N], Q8_0, ea.N * (ea.K + eb.K) // 32 * 34, w,
                ea.name + "+" + eb.name)
            return items

        def w(f):
            f.write(H.q8g64_concat_k(cv.v2.read(ea), cv.v2.read(eb), ea.N, ea.K, eb.K))
        add([ea.K + eb.K, ea.N], Q8G64, ea.size + eb.size, w, ea.name + "+" + eb.name)
        return items
    floats = {"nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
              "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
              "nextn.hc_head_norm.weight": "mtp.hyper_connection_mixer.hc_norm.weight",
              "attn_q_norm.weight": p + "self_attn.q_norm.weight",
              "attn_k_norm.weight": p + "self_attn.k_norm.weight",
              "indexer.q_norm.weight": p + "self_attn.indexer.q_layernorm.weight",
              "indexer.k_norm.weight": p + "self_attn.indexer.k_layernorm.weight",
              "hc_attn_norm.weight": p + "attn_hyper_connection.hc_norm.weight",
              "hc_ffn_norm.weight": p + "mlp_hyper_connection.hc_norm.weight",
              "ffn_gate_inp.weight": p + "mlp.gate.weight"}
    if suf in floats:
        hn = floats[suf]
        w, nb = cv.floats(E[hn], offset=hn.endswith(OFFSET_SUFFIXES))
        add(sdims, F32, nb, w, hn)
        return items
    dense = {"nextn.hc_head_down.weight": "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
             "nextn.hc_head_up.weight": "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
             "attn_q.weight": p + "self_attn.q_proj.weight",
             "attn_k.weight": p + "self_attn.k_proj.weight",
             "attn_v.weight": p + "self_attn.v_proj.weight",
             "attn_output.weight": p + "self_attn.o_proj.weight",
             "ffn_gate_shexp.weight": p + "mlp.shared_expert.gate_proj.weight",
             "ffn_up_shexp.weight": p + "mlp.shared_expert.up_proj.weight",
             "ffn_down_shexp.weight": p + "mlp.shared_expert.down_proj.weight",
             "hc_attn_down.weight": p + "attn_hyper_connection.input_mix_weight_down.weight",
             "hc_attn_up.weight": p + "attn_hyper_connection.input_mix_weight_up.weight",
             "hc_attn_inject.weight": p + "attn_hyper_connection.block_inject_weight.weight",
             "hc_ffn_down.weight": p + "mlp_hyper_connection.input_mix_weight_down.weight",
             "hc_ffn_up.weight": p + "mlp_hyper_connection.input_mix_weight_up.weight",
             "hc_ffn_inject.weight": p + "mlp_hyper_connection.block_inject_weight.weight"}
    if suf in dense:
        q8(E[dense[suf]])
        return items
    if suf in ("indexer.q_proj.weight", "indexer.k_proj.weight"):
        ent = E[p + "self_attn.indexer.index_qk_proj.weight"]
        nq = sdims[1] if suf.startswith("indexer.q") else ent.N - sdims[1]
        q8(ent, slice(0, nq) if suf.startswith("indexer.q") else slice(nq, ent.N))
        return items
    if suf == "ffn_gate_inp_shexp.weight":
        ent = E[p + "mlp.shared_expert_gate.weight"]
        if stock_types:
            def w(f):
                f.write(H.q4c_decode(cv.v2.read(ent), ent.N, ent.K).astype("<f4").tobytes())
            add(sdims, F32, ent.N * ent.K * 4, w, ent.name)
            return items
        w, nb = cv.q4c_rows(ent)
        add(sdims, Q4C, nb, w, ent.name)
        return items
    me = re.match(r"ffn_(gate|up|down)_exps\.weight$", suf)
    if me and stock_types:
        which = me.group(1)
        hn = p + ("mlp.experts.down_proj.weight" if which == "down" else "mlp.experts.gate_up_proj.weight")
        ent = E[hn]
        Ex, rows, K = ent.shape
        out_rows = rows if which == "down" else rows // 2
        base = 0 if which != "up" else rows // 2

        def w(f):
            p_ = cv.v2.read(ent)
            for e in range(Ex):
                vals = H.q4c_decode(p_, Ex * rows, K, np.arange(e * rows + base, e * rows + base + out_rows))
                f.write(H.quant_q8_0(vals))
        add([K, out_rows, Ex], Q8_0, Ex * out_rows * K // 32 * 34, w, hn)
        return items
    if me:
        which = me.group(1)
        if which == "down":
            ent = E[p + "mlp.experts.down_proj.weight"]
            Ex, rows, K = ent.shape
            w, nb = cv.q4c_rows(ent)
            add([K, rows, Ex], Q4C, nb, w, ent.name)
            return items
        ent = E[p + "mlp.experts.gate_up_proj.weight"]
        Ex, rows, K = ent.shape
        half = rows // 2
        base = 0 if which == "gate" else half
        sel = (np.arange(Ex)[:, None] * rows + base + np.arange(half)[None, :]).reshape(-1)
        nb = H.q4c_layout(Ex * half, K)[2]

        def w(f):
            p_ = cv.v2.read(ent)
            f.write(H.q4c_select_rows(p_, Ex * rows, K, sel))
        add([K, half, Ex], Q4C, nb, w, ent.name)
        return items
    raise KeyError(f"unmapped MTP tensor {name}")


# ------------------------------------------------------------ GGUF write
def w_str(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)) + b)


def w_val(f, t, v, sub=None):
    if t == GV["STRING"]:
        w_str(f, v)
    elif t == GV["ARRAY"]:
        f.write(struct.pack("<IQ", sub, len(v)))
        for x in v:
            w_val(f, sub, x)
    elif t == GV["BOOL"]:
        f.write(struct.pack("<B", 1 if v else 0))
    else:
        f.write(struct.pack(SCALAR[t], v))


def skeleton(pattern):
    if os.environ.get("GGUF_PY"):  # a llama.cpp gguf-py checkout; else an installed gguf
    sys.path.insert(0, os.environ["GGUF_PY"])
    from gguf import GGUFReader
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(pattern)
    r0 = GGUFReader(shards[0])
    drop = ("split.", "GGUF.", "quantize.", "general.quantized_by", "general.repo_url", "general.name",
            "general.tags", "general.file_type", "general.quantization_version",
            "general.alignment")
    kvs = []
    for k, fld in r0.fields.items():
        if k.startswith(drop):
            continue
        t = int(fld.types[0])
        sub = int(fld.types[1]) if t == GV["ARRAY"] else None
        kvs.append((k, t, fld.contents(), sub))
    tensors = []
    for si, s in enumerate(shards):
        for t in GGUFReader(s).tensors:
            tensors.append((si, t.name, [int(d) for d in t.shape]))
    return shards, kvs, tensors


TABLE_KEYS = ("general.architecture", "qwen4exp.embedding_length_per_layer_input",
              "qwen4exp.ple.layers", "qwen4exp.ple.ngram_size", "qwen4exp.ple.heads_per_ngram",
              "qwen4exp.ple.layer_multipliers", "qwen4exp.ple.head_offsets",
              "qwen4exp.ple.head_vocab_sizes")


def write_gguf(out, allkv, items, si):
    """Write one GGUF file (resumable: skipped when its .done marker exists)."""
    done = out + ".done"
    if os.path.exists(done):
        log(f"skip {out} (done marker present)")
        return json.load(open(done))
    manifest = []
    tmp = out + ".part"
    with open(tmp, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, len(items), len(allkv)))
        for k, t, v, sub in allkv:
            w_str(f, k)
            f.write(struct.pack("<I", t))
            w_val(f, t, v, sub)
        off = 0
        offs = []
        for it in items:
            w_str(f, it.name)
            f.write(struct.pack("<I", len(it.dims)) + struct.pack(f"<{len(it.dims)}Q", *it.dims))
            f.write(struct.pack("<IQ", it.typ, off))
            offs.append(off)
            off = H.pad(off + it.nbytes, ALIGN)
        base = H.pad(f.tell(), ALIGN)
        f.write(b"\0" * (base - f.tell()))
        t0 = time.time()
        done_b = 0
        for it, o in zip(items, offs):
            assert f.tell() == base + o, it.name
            it.write(f)
            got = f.tell() - (base + o)
            if got != it.nbytes:
                raise ValueError(f"{it.name}: wrote {got} B, planned {it.nbytes}")
            f.write(b"\0" * (H.pad(it.nbytes, ALIGN) - it.nbytes))
            manifest.append(dict(name=it.name, type=it.typ, dims=it.dims,
                                 nbytes=it.nbytes, src=it.src, shard=si, file=os.path.basename(out)))
            done_b += it.nbytes
            if it.nbytes > 200 << 20:
                el = time.time() - t0
                log(f"  {it.name} {it.nbytes / 1e9:.2f} GB; file {done_b / 1e9:.1f} GB "
                    f"in {el:.0f}s ({done_b / max(el, 1e-9) / 1e6:.0f} MB/s)")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, out)
    with open(done, "w") as d:
        json.dump(manifest, d)
    log(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB, {len(items)} tensors)")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--ngram", required=True)
    ap.add_argument("--skeleton", required=True, help="glob of the stock GGUF shards")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--only", help="regex: write only matching tensors, one file (test)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ple-table", help="write the n-gram (PLE) table to this separate GGUF "
                    "and record its file name in the trunk (qwen4exp.ple.table_file)")
    ap.add_argument("--mtp-stock-types", action="store_true",
                    help="with --mtp: write the head in the stock sidecar formats (Q8_0/F32)")
    ap.add_argument("--mtp", action="store_true",
                    help="write Gufo's MTP sidecar (blk.48.*) from v2's mtp.* "
                         "tensors; --skeleton is then the stock MTP GGUF")
    a = ap.parse_args()

    v2, ng = H.Hgn(a.v2), H.Hgn(a.ngram)
    log(f"v2 {v2.identity} {len(v2.ents)} tensors; ngram {ng.identity}")
    shards, kvs, tensors = skeleton(a.skeleton)
    kvd = {k: v for k, _t, v, _s in kvs}
    cv = Converter(v2, ng)

    # PLE index vectors must equal the skeleton's metadata arrays.
    for key, hn in () if a.mtp else (("qwen4exp.ple.layer_multipliers", "layers.1.ple.ple_embedding.layer_multipliers"),
                    ("qwen4exp.ple.head_offsets", "layers.1.ple.ple_embedding.ngram_heads_offsets"),
                    ("qwen4exp.ple.head_vocab_sizes", "layers.1.ple.ple_embedding.ngram_heads_vocab_sizes")):
        hv = np.frombuffer(v2.read(v2[hn]), "<i8").tolist()
        if hv != list(kvd[key]):
            raise ValueError(f"{key} differs from {hn}")
    log("PLE index vectors match the skeleton metadata")

    plan = []          # (shard, Item)
    used = set()
    for si, name, sdims in tensors:
        if a.only and not re.search(a.only, name):
            continue
        for it in (plan_mtp(cv, name, sdims, a.mtp_stock_types) if a.mtp
                   else plan_tensor(cv, name, sdims, kvd)):
            if it.name == name and it.dims != sdims:
                raise ValueError(f"{name}: dims {it.dims} != skeleton {sdims}")
            used.add(it.src.split(":")[-1])
            plan.append((0 if a.only else si, it))

    # Every v2 tensor except the MTP head and the PLE index vectors must be used.
    used = {u for x in used for u in x.split("+")}
    if a.mtp:
        leftover = sorted(n for n in v2.ents if n.startswith("mtp.") and n not in used)
        if leftover:
            raise ValueError(f"{len(leftover)} mtp tensors unused, e.g. {leftover[:5]}")
    elif not a.only:
        leftover = sorted(n for n in v2.ents if n not in used and not n.startswith("mtp.")
                          and not n.startswith("layers.1.ple.ple_embedding."))
        if leftover:
            raise ValueError(f"{len(leftover)} v2 tensors unused, e.g. {leftover[:5]}")
    total = sum(it.nbytes for _s, it in plan)
    log(f"plan: {len(plan)} tensors, {total / 1e9:.2f} GB")
    if a.dry_run:
        by = {}
        for _s, it in plan:
            by.setdefault(it.typ, [0, 0])
            by[it.typ][0] += 1
            by[it.typ][1] += it.nbytes
        for t, (c, b) in sorted(by.items()):
            print(f"  type {t:3d}: {c:5d} tensors {b / 1e9:9.3f} GB")
        return

    table, table_path = [], None
    if a.ple_table and not a.mtp:
        table = [it for _s, it in plan if it.name == "per_layer_token_embd.weight"]
        plan = [(s_, it) for s_, it in plan if it.name != "per_layer_token_embd.weight"]
        table_path = a.ple_table
        if os.path.dirname(os.path.abspath(table_path)) != os.path.dirname(os.path.abspath(a.out_prefix)):
            raise ValueError("--ple-table must sit next to the trunk (it is found by file name)")
    n_out = 1 if (a.only or a.mtp) else len(shards)
    extra = [("general.alignment", GV["UINT32"], ALIGN, None),
             ("general.name", GV["STRING"], "Qwen3.8 Flash Next (halogen v2 weights)", None),
             ("general.source.converter", GV["STRING"], VERSION, None),
             ("general.source.checkpoint", GV["STRING"],
              f"{os.path.basename(a.v2)} ({v2.identity}, {os.path.getsize(a.v2)} bytes) + "
              f"{os.path.basename(a.ngram)} ({ng.identity})", None),
             ("general.source.url", GV["STRING"],
              "https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next", None),
             ("general.license", GV["STRING"], "other", None),
             ("general.license.name", GV["STRING"], "qwen-community-1.0", None),
             ("qwen4exp.experts.rotation", GV["STRING"], "hadamard128", None),
             ("qwen4exp.experts.rotation_spec", GV["STRING"], "hgn-spec 1.1.0 STORAGE.md#i4r", None),
             ("qwen4exp.experts.rotation_scale", GV["STRING"], "1/sqrt(128) per side, Sylvester", None),
             ("qwen4exp.dense.rotation", GV["STRING"], "hadamard128", None),
             ("qwen4exp.dense.rotation_spec", GV["STRING"], "hgn-spec 1.1.0 STORAGE.md#ht", None),
             ("qwen4exp.ple.table_store", GV["STRING"], "fp8g", None),
             *([("qwen4exp.ple.table_file", GV["STRING"], os.path.basename(table_path), None)]
               if table_path else []),
             ("split.count", GV["UINT16"], n_out, None),
             ("split.tensors.count", GV["INT32"], len(plan), None)]
    if a.mtp:
        extra = [e for e in extra if not e[0].startswith(("qwen4exp.experts.rotation",
                                                          "qwen4exp.dense.", "qwen4exp.ple."))]
        extra[1] = ("general.name", GV["STRING"], "Qwen3.8 Flash Next MTP head (halogen v2 weights)", None)
    ekeys = {e[0] for e in extra}
    kvs = [kv for kv in kvs if kv[0] not in ekeys]
    manifest = []
    for si in range(n_out):
        out = f"{a.out_prefix}-{si + 1:05d}-of-{n_out:05d}.gguf"
        items = [it for s_, it in plan if s_ == si]
        allkv = kvs + extra + [("split.no", GV["UINT16"], si, None)]
        manifest += write_gguf(out, allkv, items, si)
    if table:
        tkv = [kv for kv in kvs if kv[0] in TABLE_KEYS] + [
            e for e in extra if e[0].startswith(("general.alignment", "general.source.",
                                                 "general.license", "qwen4exp.ple."))
            and e[0] != "qwen4exp.ple.table_file"]
        tkv.append(("general.name", GV["STRING"], "Qwen3.8 Flash Next n-gram table (halogen fp8g)", None))
        tkv.append(("general.type", GV["STRING"], "ple_table", None))
        manifest += write_gguf(table_path, tkv, table, -1)
    with open(a.out_prefix + ".manifest.json", "w") as d:
        json.dump(manifest, d, indent=0)
    log("done")


if __name__ == "__main__":
    main()
