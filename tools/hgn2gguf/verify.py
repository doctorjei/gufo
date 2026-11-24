#!/usr/bin/env python3
"""verify.py - compare a converted GGUF with the stock GGUF of the same model.

Both are quantizations of the same BF16 weights, so decoded rows must agree
closely (4-bit vs Q8_0: cosine about 0.99+). A wrong layout, orientation,
head order or rotation gives cosine near 0. For every matching tensor this
decodes a few sampled rows from each file and reports cosine and relative
RMS difference, plus a negative control (our row vs a different stock row).

Usage: verify.py --ours 'OUT-*.gguf' --skeleton 'STOCK/*.gguf' [--names REGEX]
                 [--json REPORT.json]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hgnfmt as H  # noqa: E402

if os.environ.get("GGUF_PY"):  # a llama.cpp gguf-py checkout; else an installed gguf
    sys.path.insert(0, os.environ["GGUF_PY"])
from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

F32, F16, BF16 = 0, 1, 30
I4R128, HT4, HT3, Q6G64, FP8G, Q4C, Q8G64 = 64, 65, 66, 67, 68, 69, 70
RNG = np.random.default_rng(1234)


class Ours:
    def __init__(self, pattern):
        self.files = [H.Gguf(p) for p in sorted(glob.glob(pattern))]
        self.where = {n: g for g in self.files for n in g.tensors}

    def info(self, name):
        g = self.where[name]
        dims, typ, _off, _ = g.tensors[name]
        return g, dims, typ

    def raw(self, name, nbytes, rel=0):
        return self.where[name].read(name, nbytes, rel)

    def vec(self, name):
        g, dims, typ = self.info(name)
        n = dims[0]
        assert typ == F16
        return np.frombuffer(self.raw(name, 2 * n), "<f2").astype(np.float64)

    def rows(self, name, rows):
        """Decoded float rows (true weights) for 2-D tensors; for 3-D experts
        `rows` index the flattened [expert*ne1 + r]."""
        g, dims, typ = self.info(name)
        K = dims[0]
        N = int(np.prod(dims[1:])) if len(dims) > 1 else 1
        rows = np.asarray(rows)
        if typ == F32:
            a = np.frombuffer(self.raw(name, 4 * N * K), "<f4").reshape(N, K)
            return a[rows].astype(np.float64)
        if typ == BF16:
            a = H.bf16_to_f32(self.raw(name, 2 * N * K)).reshape(N, K)
            return a[rows].astype(np.float64)
        if typ == Q6G64:
            p = self.raw(name, N * H.q6g64_row_bytes(K))
            return H.q6g64_decode(p, N, K, rows).astype(np.float64)
        if typ == 8:  # Q8_0
            raw = np.frombuffer(self.raw(name, N * K // 32 * 34), np.uint8).reshape(N, K // 32, 34)[rows]
            d = raw[:, :, :2].copy().view(np.float16).astype(np.float64)
            q = raw[:, :, 2:].copy().view(np.int8).astype(np.float64)
            return (q * d).reshape(len(rows), K)
        if typ == Q8G64:
            p = self.raw(name, N * (K + K // 16))
            return H.q8g64_decode(p, N, K, rows).astype(np.float64)
        if typ == Q4C:
            p = self.raw(name, H.q4c_layout(N, K)[2])
            return H.q4c_decode(p, N, K, rows).astype(np.float64)
        if typ == FP8G:
            t = H.fp8_table()
            scale = np.frombuffer(self.raw(name, 4, H.pad(160 * N, 64)), "<f4")[0]
            out = []
            for r in rows:
                c = np.frombuffer(self.raw(name, 160, 160 * int(r)), np.uint8)
                out.append(t[c] * scale)
            return np.array(out)
        if typ in (HT4, HT3):
            b = 4 if typ == HT4 else 3
            base = name[:-len(".weight")]
            su, sv = self.vec(base + "_rot.su"), self.vec(base + "_rot.sv")
            bb = H.ht_band_bytes(K, b)
            out = []
            for r in rows:
                band = int(r) // 128
                p = self.raw(name, bb, band * bb)
                M = H.ht_decode_bands(p, K, b, 0, 1)
                W = H.rotate(M, su, sv[band * 128:(band + 1) * 128])
                out.append(W[int(r) % 128])
            return np.array(out)
        if typ == I4R128:
            E = dims[2]
            ne1 = dims[1]
            G = K // 128
            L = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps", name)
            l, which = L.group(1), L.group(2)
            su = self.vec(f"blk.{l}.ffn_experts_rot.su" if which != "down" else f"blk.{l}.ffn_down_exps_rot.su")
            sv = self.vec(f"blk.{l}.ffn_{which}_exps_rot.sv")
            out = []
            for r in rows:
                r = int(r)
                band0 = (r // 128) * 128
                p = np.frombuffer(self.raw(name, 128 * G * 66, band0 * G * 66), np.uint8)
                M = H.i4r128_M(p, 128, K)
                rr = band0 % ne1
                W = H.rotate(M, su, sv[rr:rr + 128])
                out.append(W[r - band0])
            return np.array(out)
        raise ValueError(f"{name}: type {typ}")


class Stock:
    def __init__(self, pattern):
        self.t = {}
        for p in sorted(glob.glob(pattern)):
            for t in GGUFReader(p).tensors:
                self.t[t.name] = t

    def rows(self, name, rows):
        t = self.t[name]
        dims = [int(d) for d in t.shape]
        K = dims[0]
        N = int(np.prod(dims[1:])) if len(dims) > 1 else 1
        qt = t.tensor_type
        bs, ts = GGML_QUANT_SIZES[qt]
        rb = K // bs * ts
        flat = np.asarray(t.data).reshape(-1).view(np.uint8)
        out = []
        for r in rows:
            seg = flat[int(r) * rb:(int(r) + 1) * rb]
            out.append(np.asarray(dequantize(seg, qt), np.float64).reshape(-1)[:K])
        return np.array(out), N


def compare(a, b):
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-300)
    rel = np.linalg.norm(a - b, axis=1) / (np.linalg.norm(b, axis=1) + 1e-300)
    return cos, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--names", default=".")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--json")
    a = ap.parse_args()
    ours, stock = Ours(a.ours), Stock(a.skeleton)
    report = []
    worst = 1.0
    for name in sorted(ours.where):
        if "_rot." in name or not re.search(a.names, name) or name not in stock.t:
            continue
        g, dims, typ = ours.info(name)
        N = int(np.prod(dims[1:])) if len(dims) > 1 else 1
        pick = RNG.choice(N, min(a.rows, N), replace=False)
        rows = pick if name.startswith("per_layer_token_embd") else np.unique(np.concatenate([[0, N - 1], pick]))
        mine = ours.rows(name, rows)
        ref, _ = stock.rows(name, rows)
        cos, rel = compare(mine, ref)
        ctrl, _ = stock.rows(name, (rows + max(1, N // 3)) % N)
        ccos, _ = compare(mine, ctrl)
        ent = dict(name=name, type=typ, dims=dims, stock_type=stock.t[name].tensor_type.name,
                   rows=[int(r) for r in rows], cos_min=float(cos.min()), rel_max=float(rel.max()),
                   control_cos_max=float(np.abs(ccos).max()))
        report.append(ent)
        worst = min(worst, ent["cos_min"])
        print(f"{name:42s} t{typ:<3d} vs {ent['stock_type']:7s} cos_min {ent['cos_min']:.5f} "
              f"rel_max {ent['rel_max']:.4f}  control |cos| {ent['control_cos_max']:.3f}", flush=True)
    print(f"\n{len(report)} tensors checked; worst cosine {worst:.5f}")
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
