#!/usr/bin/env python3
"""Three-way comparison at verdict level: OLD = frozen run (08-26, lens c73a32d1), CTL = same lens re-run today, NEW = refitted lens today.
noise = OLD vs CTL (same lens, different day); lens = CTL vs NEW (same day, different lens)."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
D = {"OLD": HERE / "results/tsd_r2", "CTL": HERE / "results/tsd_r2_lens_c73a32d1_rerun", "NEW": HERE / "results/tsd_r2_lens_c6ee7d22"}
def load(tag, st):
    p = D[tag] / f"{st}.json"
    return json.load(open(p)) if p.exists() else None
def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{p}.{k}" if p else k)
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flat(v, f"{p}[{i}]")
    else: yield p, o
def n_moved(a, b, pred):
    fa, fb = dict(flat(a)), dict(flat(b))
    ks = [k for k in fa if k in fb and not k.startswith("meta") and pred(k)]
    return sum(1 for k in ks if fa[k] != fb[k]), len(ks)
rows = []
def add(name, f):
    vals = {}
    for tag in D:
        try: vals[tag] = f(tag)
        except Exception as e: vals[tag] = f"n/a"
    rows.append((name, vals))
# Arm A
add("Arm A m_v3 / m_legacy", lambda t: f"{load(t,'arm_a_v3')['m_v3']}/5 · {load(t,'arm_a_v3')['m_legacy']}/5")
add("Arm A windows (v3)", lambda t: [r.get("window_start_v3") for r in load(t,'arm_a_v3')["rows"]])
add("Arm A V-N min rank", lambda t: load(t,'arm_a_v3')["controls"]["vn_v3_min"])
# Arm C
add("Arm C τ98 v3", lambda t: load(t,'arm_c_v3')["calibration"]["tau98_v3"])
add("Arm C spread / coherent", lambda t: f"{load(t,'arm_c_v3')['calibration']['spread_v3']} / {load(t,'arm_c_v3')['calibration']['coherent_v3']}")
add("Arm C rel onset (f0,fmid,fbig) per row", lambda t: {r["id"]: [r["conds"][c]["v3"].get("onset_rel_landmark") for c in ("f0","fmid","fbig")] for r in load(t,'arm_c_v3')["rows"]})
# ladders
def ladder(t, st):
    d = load(t, st); out = {}
    for r in d["rows"]:
        out[r["id"]] = {rg: (v["v3"].get("onset_rel_landmark"), v.get("gate_v3", {}).get("pass")) for rg, v in r["rungs"].items()}
    return out
def ladder_summary(t, st):
    L = ladder(t, st); rels = set(); gates = 0; n = 0
    for rid, rg in L.items():
        for k, (rel, g) in rg.items():
            n += 1; gates += bool(g); 
            if rel is not None: rels.add((rid, rel))
    return f"gate pass {gates}/{n} · rel per row {sorted({(rid, rel) for rid, rel in rels})}"
add("Ladder l1: gate pass · rel", lambda t: ladder_summary(t, "cii_l1_v3"))
add("Ladder l2: gate pass · rel", lambda t: ladder_summary(t, "cii_l2_v3"))
add("Nulls: all>1000 · min · margin dex", lambda t: f"{load(t,'cii_nulls_v3')['summary']['all_gt_1000']} · {load(t,'cii_nulls_v3')['summary']['min_cell']} · {load(t,'cii_nulls_v3')['summary']['margin_dex']}")
print("| 量 | OLD(08-26・c73a32d1) | CTL(今日・c73a32d1) | NEW(今日・c6ee7d22) |\n|---|---|---|---|")
for name, vals in rows:
    print(f"| {name} | {vals['OLD']} | {vals['CTL']} | {vals['NEW']} |")
print("\n| stage | 葉 moved OLD→CTL(走行の乱れ) | 葉 moved CTL→NEW(lens) | 葉 moved OLD→NEW |\n|---|---|---|---|")
for st in ("arm_a_v3", "arm_c_v3", "cii_l1_v3", "cii_l2_v3", "cii_nulls_v3"):
    o, c, n = load("OLD", st), load("CTL", st), load("NEW", st)
    if not (o and c and n): print(f"| {st} | (pending) | | |"); continue
    pred = lambda k: True
    a, na = n_moved(o, c, pred); b, nb = n_moved(c, n, pred); cc, nc = n_moved(o, n, pred)
    # model-side leaves (p_word / p_ans / top1) separately
    pm = lambda k: ("p_word" in k or "p_ans" in k or "top1" in k)
    am, _ = n_moved(o, c, pm); bm, _ = n_moved(c, n, pm)
    print(f"| {st} | {a}/{na}(model 側 {am}) | {b}/{nb}(model 側 {bm}) | {cc}/{nc} |")
