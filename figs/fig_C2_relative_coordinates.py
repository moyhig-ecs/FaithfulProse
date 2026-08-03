#!/usr/bin/env python3
"""C2 --- fidelity vs RELATIVE position.  FROZEN EMPTY FRAME, drawn before any
aggregation.

Frozen in ../frozen/PREREG_relative_coordinates.en.md.
The instruction under which this was written:
  "Freeze the bucket definitions for the re-aggregation and the empty frame of
   this figure **before computing anything**. Do not cut buckets after seeing
   the data --- apply to the next figure exactly the discipline the first one
   just demonstrated."

This script draws an **empty frame** and exits if the aggregation is absent.
The axes, buckets and decision lines are frozen in the pre-registration and are
not moved after seeing the results. The values in this figure are **not used
for any decision** (it is an attribution diagnostic) --- the decision lines are
drawn for reference only. The prose corpus is not plotted here: it has no
prompt/generation distinction and therefore a different coordinate system.

Deterministic: fixed rcParams, no timestamps, no randomness.
Run as:  python3 fig_C2_relative_coordinates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = next(a for a in (HERE, *HERE.parents) if (a / "probes").is_dir())
sys.path.insert(0, str(REPO / "probes" / "olmo3-f5-think-prefix" / "figs"))
import figstyle                                              # noqa: E402  ★ 家風

DATA = REPO / "probes" / "c-lens-pos" / "results" / "c_lens_pos_r2.json"
STEM = HERE / "fig_C2_relative_coordinates"

# ── ★ PREREG §1 で凍結した相対座標（⛔ 計算前に固定・見てから切らない） ──────
P_BUCKETS = [("P1\n1–\n128", 1, 128), ("P2\n129–\n320", 129, 320),
             ("P3\n321–\n501", 321, 501)]
O_BUCKETS = [("O1\n1–\n128", 1, 128), ("O2\n129–\n512", 129, 512),
             ("O3\n513–\n2048", 513, 2048), ("O4\n2049–\n4096", 2049, 4096),
             ("O5\n4097–\n8191", 4097, 8191), ("O6\n8192–\n16384", 8192, 16384)]
N_PROMPT = 501               # G 群（F5b-1 系）の prompt 長
THRESH_HOLD, THRESH_OUT = 0.10, 0.30      # ★ C1 と同一位置（⛔ 参照のためだけ）
ATTN = [("eager", "-"), ("sdpa", (0, (3, 2)))]
PANELS = [("rankcorr degradation", "rankcorr_deg"),
          ("seatrank increase\n(Δ log10 rank)", "seatrank_deg")]
AXES = [("P", "P  prompt depth", P_BUCKETS),
        ("O", "O  offset from the first generated token", O_BUCKETS)]

figstyle.use_house_style()
fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharey="row",
                         gridspec_kw=dict(width_ratios=[0.62, 1.0], wspace=0.10,
                                          hspace=0.40, left=0.105, right=0.985,
                                          bottom=0.235, top=0.845))

cells = None
if DATA.exists():
    cells = json.loads(DATA.read_text()).get("cells")

TOP, BOT = 0.595, -0.015
for row, (title, key) in enumerate(PANELS):
    for col, (axis_id, axis_name, BK) in enumerate(AXES):
        ax = axes[row][col]
        x = list(range(len(BK)))
        ax.axhspan(THRESH_OUT, 1.0, color="#fdecea", zorder=0)
        ax.axhspan(THRESH_HOLD, THRESH_OUT, color="#fff8e1", zorder=0)
        ax.axhline(THRESH_HOLD, color="#f9a825", linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
        ax.axhline(THRESH_OUT, color="#c62828", linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
        ax.axvline(0.5, color="#2e5c2e", linewidth=0.8, zorder=1)     # fit 域幅 128 の右端
        if col == 1:
            ax.axvline(-0.45, color="#8d6e63", linewidth=1.2, zorder=1)   # 生成境界 o = 1
        if cells:
            for ai, (attn, ls) in enumerate(ATTN):
                # ★ ラベル衝突の回避のみ（2026-08-03）: eager は点の左、sdpa は点の右へ。
                #   ⛔ 軸・bucket・判定線・データ位置は一切動かさない（注記の置き場だけ）。
                dx, ha = (-0.17, "right") if ai == 0 else (0.17, "left")
                sel = [c for c in cells if c["axis"] == axis_id and c["attn"] == attn]
                if not sel:
                    continue
                ys = [next((c[key] for c in sel if c["bucket"] == b[0][:2]), None) for b in BK]
                ns = [next((c.get("n") for c in sel if c["bucket"] == b[0][:2]), 0) for b in BK]
                xs2 = [xi for xi, yi in zip(x, ys) if yi is not None]
                ys2 = [min(max(yi, BOT), TOP) for yi in ys if yi is not None]
                ax.plot(xs2, ys2, marker="o", color="#1a1a1a", markersize=4.2,
                        linewidth=1.0, linestyle=ls, zorder=5, label=f"G · {attn}")
                for xi, (yi, ni) in enumerate(zip(ys, ns)):
                    if yi is None:
                        continue
                    if yi > TOP or yi < BOT:                  # ⛔ 軸は広げない
                        edge = TOP if yi > TOP else BOT
                        ax.plot(xi, edge, marker="^" if yi > TOP else "v",
                                color="#1a1a1a", markersize=7, zorder=7)
                        ax.text(xi + dx, edge + (0.008 if yi > TOP else -0.008),
                                f"{yi:+.2f}", fontsize=5.4, ha=ha,
                                va="bottom" if yi > TOP else "top",
                                color="#1a1a1a", zorder=7)
                    else:
                        ax.text(xi + dx, yi + 0.018, f"n={ni}", fontsize=5.0, ha=ha,
                                color="#555555", zorder=6)
        else:
            ax.text(len(BK) / 2 - 0.5, 0.30, "FROZEN\nEMPTY FRAME", fontsize=7.4,
                    ha="center", va="center", color="#9e9e9e", style="italic")
        ax.set_xticks(x)
        ax.set_xticklabels([b[0] for b in BK], fontsize=6.0)
        ax.set_ylim(-0.03, 0.62)
        ax.set_xlim(-0.5, len(BK) - 0.5)
        ax.tick_params(length=2.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if row == 0:
            ax.set_title(axis_name, fontsize=7.4, pad=4)
        if col == 0:
            ax.set_ylabel(f"{title}\n(vs P1 / O1)", fontsize=7.0)

if cells:
    h, lb = axes[0][0].get_legend_handles_labels()
    # ★ 2026-08-03: tick label と重なっていたので下げ、title を 2 行に折った（配置のみ）。
    fig.legend(h, lb, frameon=False, fontsize=6.0, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.108),
               title="G group only  ·  solid = eager  ·  dashed = sdpa\n"
                     "triangles = beyond the frozen axis (axis NOT widened)  ·  "
                     "labels: eager left of the marker, sdpa right",
               title_fontsize=6.0)

# ⚠ 2026-08-03 訂正: 旧題は「the same 700 measurements」だったが、本図は G 群のみ
#    （W 群 348 は prompt/生成の区別を持たず絶対座標に据え置き）⇒ 数えれば偽。
#    REDLINE_faithfulprose_results_v1_20260803 住所 B の是正を図側にも適用。
fig.text(0.5, 0.966,
         "C-lens-pos R-2  —  the G-group measurements (352 of 700), in relative coordinates",
         fontsize=9.3, ha="center", va="center")
fig.text(0.5, 0.928,
         f"G group only (n_prompt = {N_PROMPT});  W group has no prompt/generation split "
         "and stays in absolute coordinates",
         fontsize=6.3, ha="center", va="center", color="#555555")
fig.text(0.5, 0.900,
         "attribution diagnostic — the decision table is NOT re-applied here; "
         "the 10% / 30% lines are drawn for reference only",
         fontsize=6.3, ha="center", va="center", color="#c62828")

figstyle.footer(
    fig,
    sources=["c_lens_pos.json@f0a7787 (re-aggregated, no new run)"],
    date="2026-08-02", commit=figstyle.head_commit(REPO),
    lens_md5="c73a32d1f72968bd73c104c06445a482",
    extra=("Buckets and this frame were frozen in PREREG_relative_coordinates.en "
           "before any aggregation was computed."),
    provisional=True, y=0.012, dy=0.019, fontsize=5.3)

fig.savefig(f"{STEM}.pdf")
fig.savefig(f"{STEM}.png")
print(f"-> {STEM}.pdf / .png   ({'DATA' if cells else '★ FROZEN EMPTY FRAME'})")
