#!/usr/bin/env python3
"""B: old lens (results/tsd_r2 = c73a32d1, fitted on transformers 5.11.0) vs new lens (results/tsd_r2_lens_c6ee7d22 = fitted
on 5.17.0), same runner / model / forward (5.14.1) / rows. Prints what moved. Usage: compare_lensdelta.py [stage ...]"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
OLD, NEW = HERE / "results/tsd_r2", HERE / "results/tsd_r2_lens_c6ee7d22"
stages = sys.argv[1:] or ["arm_a_v3", "arm_c_v3", "cii_l1_v3", "cii_l2_v3", "cii_nulls_v3"]

def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{p}.{k}" if p else k)
    elif isinstance(o, list) and o and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in o):
        yield p, ("series", o)
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flat(v, f"{p}[{i}]")
    else:
        yield p, ("scalar", o)

for st in stages:
    fo, fn = OLD / f"{st}.json", NEW / f"{st}.json"
    if not fn.exists(): print(f"\n## {st}: new output not present yet"); continue
    a, b = json.load(open(fo)), json.load(open(fn))
    fa, fb = dict(flat(a)), dict(flat(b))
    keys = [k for k in fa if k in fb and not k.startswith("meta")]
    same = moved = 0; lines = []
    for k in keys:
        (ta, va), (tb, vb) = fa[k], fb[k]
        if ta == "scalar":
            if va == vb: same += 1
            else: moved += 1; lines.append(f"| `{k}` | {va} | {vb} |")
        else:
            if va == vb: same += 1
            else:
                moved += 1
                n = min(len(va), len(vb)); diffs = [abs(x - y) for x, y in zip(va, vb)]
                lines.append(f"| `{k}` (series n={len(va)}) | min {min(va)} / med {sorted(va)[n//2]} | min {min(vb)} / med {sorted(vb)[n//2]} · changed {sum(1 for d in diffs if d)}/{n} · max|Δ| {max(diffs):.4g} |")
    print(f"\n## {st}: leaves compared {same + moved} · unchanged {same} · moved {moved}")
    # headline fields
    for hk in ("m_v3", "m_legacy", "controls.vn_v3_min", "controls.determinism_pass", "calibration.tau98_v3", "calibration.spread_v3", "calibration.coherent_v3", "summary.all_gt_1000", "summary.min_cell", "summary.margin_dex"):
        if hk in fa and hk in fb: print(f"  {hk}: {fa[hk][1]} -> {fb[hk][1]}")
    if lines:
        print("| leaf | old (c73a32d1) | new (c6ee7d22) |\n|---|---|---|")
        print("\n".join(lines[:80]))
        if len(lines) > 80: print(f"… {len(lines) - 80} more")
