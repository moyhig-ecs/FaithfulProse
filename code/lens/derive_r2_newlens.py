#!/usr/bin/env python3
"""C (2026-09-22): re-derive the C-lens-pos R-2 table (relative coordinates, P/O buckets) from THIS campaign's c_lens_pos.json,
using run_r3.py's own verbatim aggregator (p_bucket / o_bucket / aggregate — the same code path as run_r3's 統制 A), so that
run_r3.py's 統制 A ("aggregator reproduces c_lens_pos_r2.json") is self-consistent under the new lens. Forward zero.
Run through lens_swap.py so the read of c_lens_pos.json and the write of c_lens_pos_r2.json both land in results/lens_<tag>/:
  python3 lens_swap.py --runner derive_r2_newlens.py --tag c6ee7d22 --fresh-read c_lens_pos.json,c_lens_pos_r2.json
Frozen files are not touched (run_r3.py / run_c_lens_pos.py imported, not edited)."""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "probes/c-lens-pos"))
import run_r3 as R3                                      # noqa: E402  ★ verbatim aggregator (frozen)

SRC = REPO / "probes/c-lens-pos/results/c_lens_pos.json"      # read  (redirected by --fresh-read)
DST = REPO / "probes/c-lens-pos/results/c_lens_pos_r2.json"   # write (redirected)

doc = json.loads(SRC.read_text())
stored = doc["measurements"]
rows = []
for m in stored:                                          # == run_r3.control_a's construction, verbatim
    if m["group"] != "G":
        continue
    for axis, fn in (("P", R3.p_bucket), ("O", R3.o_bucket)):
        b = fn(m["position"], 501)
        if b:
            rows.append({**m, "bkt": b, "axis": axis, "prompt_md5": "cc05d498c10ca9c4acf3b030a5884236"})
cells = R3.aggregate([r for r in rows if r["axis"] == "P"], "P") + R3.aggregate([r for r in rows if r["axis"] == "O"], "O")
prov = doc.get("provenance", {})
out = {"check": "R-2 相対座標での再集計（帰属診断・⛔ 判定は動かさない）— 新 lens 再走からの再導出（C・2026-09-22）",
       "prereg": "PREREG_relative_coordinates.en.md",
       "n_prompt": 501,
       "source": f"results/lens_*/c_lens_pos.json（新走行ゼロ・lens_md5 {doc.get('provenance', {}).get('lens_md5', '?')}・measurements {len(stored)}・sdpa texts {sorted({m['text'] for m in stored if m['attn'] == 'sdpa'})}）",
       "aggregator": "probes/c-lens-pos/run_r3.py p_bucket/o_bucket/aggregate（verbatim import・統制 A と同一経路）",
       "P_buckets": [[e[0], e[1], e[2] if e[2] is not None else 501] for e in R3.P_EDGES], "O_buckets": [[e[0], e[1], e[2]] for e in R3.O_EDGES],
       "cells": cells}
DST.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
print(f"-> cells {len(cells)} ({sorted({(c['axis'], c['attn']) for c in cells})}) from {len(stored)} measurements; wrote {DST}")
