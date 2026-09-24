#!/usr/bin/env python3
"""C-lens-pos (FaithfulProse position calibration): per (group, attn, bucket) medians of kl / top1 / rankcorr for
OLD (frozen 08-02, lens c73a32d1) vs CTL (today, same lens) vs NEW (today, refit lens). Buckets B1..B6 = position depth."""
import json, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "c-lens-pos" / "results"
P = {"OLD": R / "c_lens_pos.json", "CTL": R / "lens_c73a32d1-rerun" / "c_lens_pos.json", "NEW": R / "lens_c6ee7d22" / "c_lens_pos.json"}
D = {}
for k, p in P.items():
    if p.exists():
        d = json.load(open(p)); D[k] = d.get("measurements", []); print(f"{k}: {len(D[k])} measurements ({p.relative_to(R.parent.parent)})")
    else: print(f"{k}: (missing)")
def med(ms, group, attn, bucket, key):
    v = [m[key] for m in ms if m.get("group") == group and m.get("attn") == attn and m.get("bucket") == bucket and m.get(key) is not None]
    return (st.median(v), len(v)) if v else (None, 0)
groups = sorted({m.get("group") for ms in D.values() for m in ms}); attns = sorted({m.get("attn") for ms in D.values() for m in ms}); buckets = sorted({m.get("bucket") for ms in D.values() for m in ms})
for key in ("kl", "top1", "rankcorr"):
    print(f"\n### {key} (median per cell) — columns: " + " | ".join(D.keys()))
    print("| group | attn | bucket | " + " | ".join(D.keys()) + " |"); print("|---|---|---|" + "---|" * len(D))
    for g in groups:
        for a in attns:
            for b in buckets:
                cells = [med(ms, g, a, b, key) for ms in D.values()]
                if all(c[1] == 0 for c in cells): continue
                print(f"| {g} | {a} | {b} | " + " | ".join(f"{c[0]:.4f} (n={c[1]})" if c[0] is not None else "—" for c in cells) + " |")
