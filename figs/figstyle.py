#!/usr/bin/env python3
"""House style for figures --- shared rcParams and the standard footer.

The standard footer carries: data file@commit / lens md5 (where a lens is
used) / the date / the provisional clause. Every subsequent figure then wears
the same discipline automatically.

Determinism: nothing equivalent to `Date.now()` is used --- the date is passed
in **by the caller**. Burning a timestamp into a figure would mean the same
data no longer produces the same bytes.

This module neither decides nor interprets anything.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

# 家風の rcParams（CommitStage paper/figs と同値）
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
}

# 差し戻し行 —— L1/L2 分離の図版版（全図共通・⛔ 文言を弱めない）
REMAND = "L1 only — measured values; the reading belongs to the judgement layer."

# provisional 注 —— word-group readout / J-space を使う図でのみ付す
# ★ 2026-08-02 更新（ADJUDICATION_c_lens_pos_verdict 帳簿 2）: probe は pending ではなく判定済み。
#   B6 は域外確定・FLAG は恒久条項へ昇格。⛔ ただし交絡 3 件を必ず併記する。
PROVISIONAL = ("FLAG-lens-pos-domain is now a standing clause: C-lens-pos found the deep-position "
               "seat readout has no verifiable anchor (B6 out of domain; B4/B5 beyond 0.30 digits). "
               "Three confounds stand with the verdict: W group does not degrade with position, "
               "the G buckets straddle the prompt/generation boundary, and the bridge term varied "
               "attention implementation and sequence length together.")


def use_house_style() -> None:
    plt.rcParams.update(RC)


def head_commit(repo: Path) -> str:
    """Read HEAD in the same form as the run provenance; 'unknown' if unavailable."""
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def footer(fig, *, sources: list[str], date: str, commit: str,
           lens_md5: str | None = None, provisional: bool = False,
           extra: str | None = None, y: float = 0.052, dy: float = 0.022,
           fontsize: float = 5.7, color: str = "#777777",
           wrap: int = 150) -> None:
    """Stack the standard footer at the bottom of the figure.

    sources   : 'file.json@commit' entries --- the scope tag: which real outputs the figure was drawn from
    date      : passed in by the caller --- the run time is never burned into the figure, for determinism
    commit    : repo HEAD at which the figure was generated
    lens_md5  : only for figures that use a lens
    provisional: if True, append the standing position-domain clause
    extra     : one figure-specific line (smoothing window, exclusion rule, and so on)
    """
    lines = [REMAND]
    if extra:
        lines.append(extra)
    if provisional:
        lines.append(PROVISIONAL)
    tag = "  ·  ".join(sources)
    lines.append(f"data: {tag}   ·   figure at {commit}   ·   {date}"
                 + (f"   ·   lens {lens_md5[:8]}…" if lens_md5 else ""))
    # ★ 2026-08-03: 長い行が figure 幅から溢れて左端で切れていた（PROVISIONAL 等）。
    #   ⇒ 描画前に折り返す。⛔ 文言は一字も変えない・行の順序も不変（体裁のみ）。
    import textwrap
    wrapped: list[str] = []
    for ln in lines:
        wrapped.extend(textwrap.wrap(ln, width=wrap) if len(ln) > wrap else [ln])
    lines = wrapped
    n = len(lines)
    for i, ln in enumerate(lines):          # lines[0] is the topmost, y is the bottom line
        fig.text(0.5, y + (n - 1 - i) * dy, ln,
                 fontsize=fontsize, ha="center", va="center", color=color)
