#!/usr/bin/env python3
"""C1 --- fidelity vs absolute position.  FROZEN EMPTY FRAME, drawn before any
data exists.

The instruction under which this was written:
  "Freeze the figure template --- bucket axis, fit-domain band, empty frame ---
   together with the design review, as the figure counterpart of the decision
   table. It looks like play; it is discipline: it prevents drawing after
   seeing."

This script **draws an empty frame and exits if no data is present**. The axes,
bucket boundaries, decision lines and footer are fixed before the results
arrive, so neither an axis nor a threshold can be moved after seeing them.

The decision table is frozen in ../frozen/DECISION_TABLE.en.md:
  decision = worst-of {rankcorr, seatrank}; decision group = the reasoning
  traces; ratio to the B1 median; 10% / 30%.

Deterministic: fixed rcParams, no timestamps, no randomness.
Run as:  python3 fig_C1_fidelity_vs_position.py
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

DATA = REPO / "probes" / "c-lens-pos" / "results" / "c_lens_pos.json"
STEM = HERE / "fig_C1_fidelity_vs_position"

# ── ★ 凍結された軸（⛔ 結果を見てから動かさない） ────────────────────────────
BUCKETS = [                       # (label, lo, hi)  — DESIGN v1 §3 / redline §2-1 承認済
    ("B1\n1–\n128", 1, 128),
    ("B2\n129–\n512", 129, 512),
    ("B3\n513–\n2048", 513, 2048),
    ("B4\n2049–\n4096", 2049, 4096),
    ("B5\n4097–\n8191", 4097, 8191),
    ("B6\n8192–\n16384", 8192, 16384),
]
FIT_MAX_SEQ = 128            # lens fit setting (all 6 lenses)
SLIDING_WINDOW = 4096        # model config
NATIVE_CTX = 8192            # YaRN original_max_position_embeddings
THRESH_HOLD, THRESH_OUT = 0.10, 0.30      # ★ 凍結閾値（DESIGN v2 §4）
# ★ ADJUDICATION_c_lens_pos_attn_ruling: marker 凡例の追加のみ許可。
#   ⛔ 軸・bucket・判定線は触れない（凍結 frame は不変）。
#   eager = B1–B5（較正器と被較正器の呼び出し形が一致）/ sdpa = B1–B6（B6 判定は sdpa 内で完結）
GROUPS = [("G  geometry traces (decision group)", "#1a1a1a", "o"),
          ("W  wikitext (attribution diagnostic)", "#9e9e9e", "s")]
ATTN_STYLE = {"eager": ("-", 1.0), "sdpa": ((0, (3, 2)), 1.0)}
PANELS = [("rankcorr degradation  (1 − bucket/B1)", "rankcorr_deg"),
          ("seatrank increase  (Δ log10 rank vs B1)", "seatrank_deg")]

figstyle.use_house_style()
fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.3), sharex=True,
                         gridspec_kw=dict(wspace=0.24, left=0.085, right=0.985,
                                          bottom=0.415, top=0.800))

# ★ 凍結 frame の軸は「B1 比の劣化」である（軸ラベルどおり）。cells は生値なので、
#   凍結定義 (b)(b′) で劣化に変換する: rankcorr = 1 − bucket/B1（比）/ seatrank = bucket − B1（絶対 Δ桁）。
#   ⛔ 軸・bucket・判定線・閾値は一切動かしていない（データの写像を定義どおりに書いただけ）。
rows = None
if DATA.exists():
    cells = json.loads(DATA.read_text()).get("cells")
    ref = {(c["attn"], c["group"]): c for c in cells if c["bucket"] == "B1"}
    rows = []
    for c in cells:
        b1 = ref.get((c["attn"], c["group"]))
        if b1 is None:
            continue
        rows.append({**c,
                     "rankcorr_deg": 1 - c["rankcorr"] / b1["rankcorr"],
                     "seatrank_deg": c["seatrank"] - b1["seatrank"]})

x = list(range(len(BUCKETS)))
for ax, (title, key) in zip(axes, PANELS):
    # ★ 判定線（凍結）
    ax.axhspan(THRESH_OUT, 1.0, color="#fdecea", zorder=0)
    ax.axhspan(THRESH_HOLD, THRESH_OUT, color="#fff8e1", zorder=0)
    ax.axhline(THRESH_HOLD, color="#f9a825", linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
    ax.axhline(THRESH_OUT, color="#c62828", linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.text(len(BUCKETS) - 0.55, THRESH_HOLD + 0.012, "10%  hold", fontsize=6.0,
            color="#f9a825", ha="right")
    ax.text(len(BUCKETS) - 0.55, THRESH_OUT + 0.012, "30%  out of domain", fontsize=6.0,
            color="#c62828", ha="right")
    # ★ 構造境界（凍結）
    ax.axvline(0.5, color="#2e5c2e", linewidth=0.8, zorder=1)          # fit 域の右端 (B1|B2)
    ax.axvline(3.5, color="#1565c0", linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)  # sliding
    ax.axvline(4.5, color="#6a1b9a", linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)  # YaRN
    if rows:
        for label, colour, marker in GROUPS:
            g = label.split()[0]
            for ai, (attn, (ls, lw)) in enumerate(ATTN_STYLE.items()):
                # ★ 2026-08-03: 注記の重なり回避のみ。eager は点の左 / sdpa は右へ。
                #   ⛔ 軸・bucket・判定線・データ位置は一切動かさない（置き場だけ）。
                dx, ha_ = (-0.16, "right") if ai == 0 else (0.16, "left")
                sel = [c for c in rows if c["group"] == g and c.get("attn") == attn]
                if not sel:
                    continue
                ys = [next((c[key] for c in sel if c["bucket"] == b[0][:2]), None)
                      for b in BUCKETS]
                ns = [next((c.get("n") for c in sel if c["bucket"] == b[0][:2]), 0)
                      for b in BUCKETS]
                # ★ 凍結 ylim (-0.03..0.62) の内側の掲示線。⛔ 軸は広げない。
                TOP, BOT = 0.595, -0.015
                xs2 = [xi for xi, yi in zip(x, ys) if yi is not None]
                ys2 = [min(max(yi, BOT), TOP) for yi in ys if yi is not None]
                ax.plot(xs2, ys2, marker=marker, color=colour, markersize=4.5,
                        linewidth=lw, linestyle=ls, zorder=5,
                        label=f"{label}  ·  {attn}")
                # ⚠ 上下いずれのはみ出しも ▲▼ と実値で明示する
                for xi, yi in zip(x, ys):
                    if yi is None:
                        continue
                    if yi > TOP:
                        ax.plot(xi, TOP, marker="^", color=colour, markersize=7, zorder=7)
                        ax.text(xi + dx, TOP + 0.008, f"{yi:+.2f}", fontsize=5.4,
                                ha=ha_, va="bottom", color=colour, zorder=7)
                    elif yi < BOT:
                        ax.plot(xi, BOT, marker="v", color=colour, markersize=7, zorder=7)
                        ax.text(xi + dx, BOT - 0.008, f"{yi:+.2f}", fontsize=5.4,
                                ha=ha_, va="top", color=colour, zorder=7)
                for xi, (yi, ni) in enumerate(zip(ys, ns)):   # ★ 実 n を隠さない
                    if yi is not None and BOT <= yi <= TOP:
                        ax.text(xi + dx, yi + 0.018, f"n={ni}", fontsize=5.0, ha=ha_,
                                color=colour, zorder=6)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in BUCKETS], fontsize=6.2)
    ax.set_ylim(-0.03, 0.62)
    ax.set_xlim(-0.5, len(BUCKETS) - 0.5)
    ax.set_title(title, fontsize=7.4, pad=4)
    ax.tick_params(length=2.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

axes[0].set_ylabel("degradation vs B1  (worst-of decides)")
if rows:
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, fontsize=6.0, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.205), columnspacing=2.4,
               title="solid = eager (B1–B5)   ·   dashed = sdpa (B1–B6)   ·   "
                     "triangles = beyond the frozen axis (value printed; axis NOT widened)",
               title_fontsize=5.9)
else:
    for ax in axes:
        ax.text(2.5, 0.31, "FROZEN EMPTY FRAME\naxes, buckets and decision lines\n"
                           "fixed before any data existed",
                fontsize=7.0, ha="center", va="center", color="#9e9e9e", style="italic")

fig.text(0.5, 0.955, "C-lens-pos  —  lens fidelity against absolute position",
         fontsize=9.5, ha="center", va="center")
fig.text(0.5, 0.905,
         f"green = fit domain edge ({FIT_MAX_SEQ})   ·   blue = sliding_window ({SLIDING_WINDOW})   "
         f"·   purple = YaRN native context ({NATIVE_CTX})",
         fontsize=6.3, ha="center", va="center", color="#555555")
fig.text(0.5, 0.872,
         "decision = worst-of {rankcorr, seatrank}, group G, ratio to the B1 median",
         fontsize=6.3, ha="center", va="center", color="#555555")

figstyle.footer(
    fig,
    sources=["c_lens_pos.json@f0a7787" if rows else "c_lens_pos.json (pending)"],
    date="2026-08-02", commit=figstyle.head_commit(REPO),
    lens_md5="c73a32d1f72968bd73c104c06445a482",
    extra=("Frame frozen before measurement; thresholds frozen in DESIGN_c_lens_pos_v2 §4 and "
           "applied mechanically. Triangles = points beyond the frozen y-limit, value printed; "
           "the axis was NOT widened."),
    provisional=True, y=0.016, dy=0.022, fontsize=5.4)

fig.savefig(f"{STEM}.pdf")
fig.savefig(f"{STEM}.png")
print(f"-> {STEM}.pdf / .png   ({'DATA' if rows else '★ FROZEN EMPTY FRAME'})")
