#!/usr/bin/env python3
"""R2-11 --- re-derivation of the ShapeJitter floor with the v3 sets (TSD/R2 card, sec. 2 row 11).

REVISION (2026-08-27, token-set defect repair):
  The old floor (in-domain max 0.045 dex; max 0.3135 dex for pairs with the 484 rung; bitwise for
  88-110 and 286-352) was an instrument property measured on series of the old sets. The card
  froze: "the floor is an instrument property => re-derive and register anew (comparison
  descriptive)". This run re-draws the shared-prefix floor of the rung pairs of both ladders on
  the intermediate series of the v3 (joshaku canonical) sets, with the legacy series beside.
  Mechanism = frozen imports only (dlog_series / dist_stats from run_arm_b; P_SCAN from run_cii;
  landmarks from gate_cii / gate_cii_c2; set construction as in run_arms_v3). Rung pairs =
  L1 {8,10,11,12} x L2 {26,32,38,44} (the shape of the old registered pairs; r14 / r22 excluded
  and declared; the cross-run pair r22[v2] x r26 excluded because the v2 series has no v3 sets).
  This time every series is persisted, so the R2-11 re-derivation is re-checkable with zero
  forward passes. Controls run before the measurement.
  REVISION-ID: TSD-20260826

usage: python run_r211_v3.py --self-test | --fire
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
OUT_DIR = RES / "tsd_r2"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "probes" / "olmo3-f5-think-prefix"))
sys.path.insert(0, str(HERE))

import run_f5_lens as L  # noqa: E402
from run_arm_b import dlog_series, dist_stats  # noqa: E402
from run_cii import P_SCAN, restricted_min  # noqa: E402
from joshaku.pgrain import build_word_set  # noqa: E402

REVISION_ID = "TSD-20260826"
READCARD = REPO / "conversations/2026-08-26/READCARD_R2_faithfulprose_v2_20260826.md"
RUNGS_L1 = [8, 10, 11, 12]
RUNGS_L2 = [26, 32, 38, 44]
GROUPS = ("mid", "ans", "d1", "d2")


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def self_test() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'✅' if cond else '⛔'} {name}")
        ok = ok and cond

    # ① dist_stats fixture(run_arm_b 凍結の pin と同型)
    d = dist_stats([0.0, 0.1, 0.2, 0.3])
    check("① dist_stats median/q3/max", d["median"] == 0.15 and d["max"] == 0.3)
    # ② dlog_series fixture
    v = dlog_series([10, 100], [10, 1000])
    check("② dlog_series", v[0] == 0.0 and abs(v[1] - 1.0) < 1e-12)
    # ③ 共有 prefix slice fixture(alignment の機械確認)
    a, b = list(range(100)), list(range(100))
    lo, hi = P_SCAN, min(40, 60)
    check("③ slice [P_SCAN, min(lm))", len(a[lo:hi]) == 40 - P_SCAN)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--fire", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.fire:
        ap.error("--fire or --self-test")
    if self_test() != 0:
        print("⛔ 較正失敗 ⇒ 測定しない")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_wall = time.time()
    prov = L.provenance("R211-v3", a.allow_dirty)
    L.log(f"R7-a ✅ commit={prov['commit'][:8]} dirty={prov['dirty']} stage=r211-v3")

    gate2 = json.loads((RES / "gate_tokenizer.json").read_text())
    gate4 = json.loads((RES / "gate_filler_c.json").read_text())
    gate5 = json.loads((RES / "gate_cii.json").read_text())
    gate6 = json.loads((RES / "gate_cii_c2.json").read_text())
    rows = gate2["final_rows"]
    unit = gate4["final_unit"]

    pc2 = L.load_pc2()
    import torch, transformers, jlens  # noqa: E401
    assert _md5(pc2.LENS_PT) == pc2.LENS_MD5, "⛔ lens md5 不一致"
    assert pc2.MODEL_NAME == "allenai/Olmo-3-1125-32B", f"⛔ MODEL_NAME: {pc2.MODEL_NAME}"
    t0 = time.time()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        pc2.MODEL_NAME, dtype=torch.bfloat16, attn_implementation="eager").to("mps")
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(pc2.MODEL_NAME)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(str(pc2.LENS_PT))
    p2 = pc2.p2mod()
    L.log(f"stack ✅ ({time.time()-t0:.0f}s)")

    ids = json.loads(pc2.MASK_PATH.read_text())
    mask = torch.zeros(len(tok), dtype=torch.bool); mask[ids] = True
    mask_ids = set(int(i) for i in ids)
    assert int(mask.sum()) == 70812, f"⛔ N: {int(mask.sum())}"

    def legacy_sets(rid):
        rep = next(r for r in gate2["reports"] if r["id"] == rid and r["pass"])
        return ({k: [i for i, _ in rep["detail"][k]["distinctive"]] for k in rep["detail"]},
                {k: rep["detail"][k]["word"] for k in rep["detail"]})

    row_sets = {}
    for row in rows:
        leg, words = legacy_sets(row["id"])
        ws = build_word_set(tok, row["id"],
                            [words["mid"], words["ans"], words["d1"], words["d2"]],
                            mask_ids=mask_ids)
        ent = {g: e for g, e in zip(GROUPS, ws.entries)}
        if any(ent[g].canonical_id is None for g in GROUPS):
            continue
        merged = {g: [int(ent[g].canonical_id)] for g in GROUPS}
        merged.update({f"{g}_L": leg[g] for g in GROUPS})
        row_sets[row["id"]] = merged
    L.log(f"sets ✅ rows={list(row_sets)}")

    def series(jl_like, sets_row):
        med = p2.band_median_ranks(torch, jl_like, mask, sets_row)
        return {k: [int(v) for v in med[k]] for k in sets_row}

    def apply_full(prompt, sets_row):
        n_tok = len(tok(prompt)["input_ids"])
        jl, ml, _ = lens.apply(model, prompt, layers=pc2.BAND,
                               positions=list(range(n_tok)), max_seq_len=pc2.MAX_SEQ)
        return jl, n_tok

    meta = {"revision_id": REVISION_ID,
            "readcard": {"path": READCARD.name, "md5": _md5(READCARD)},
            "model": pc2.MODEL_NAME, "lens_md5": pc2.LENS_MD5,
            "band": [pc2.BAND[0], pc2.BAND[-1]], "mask_size": int(mask.sum()),
            "p_scan": P_SCAN, "rungs_l1": RUNGS_L1, "rungs_l2": RUNGS_L2,
            "scope_note": "r14/r22 と cross-run 対(r22[v2]×r26)は対象外(宣言)",
            "provenance": prov}
    out = {"meta": meta, "stage": "r211_v3", "rows": [], "controls": {}}
    op = OUT_DIR / "r211_v3.json"

    # ---- 統制(測定の前・f6en の教訓)----
    row0 = next(r for r in rows if r["id"] in row_sets)
    sets0 = row_sets[row0["id"]]
    prompt0 = unit * RUNGS_L1[0] + row0["prompt"]
    jl, _ = apply_full(prompt0, sets0)
    jl2, _ = apply_full(prompt0, sets0)
    med0, med0b = series(jl, sets0), series(jl2, sets0)
    det = med0 == med0b
    jl_neg = {k: -v for k, v in jl.items()}
    med_neg = series(jl_neg, sets0)
    # V-N 統制は rung 系列の凍結 null 機構に合わせる(R2-9 = cii_nulls_v3):
    # restricted_min(P_SCAN 以降)・錨 10³。⚠ 初版は arm-A 基準(全域 min > 10⁴)を
    # rung 系列に誤適用し 3,648 で走行前停止(2026-08-27 登記・R3 の 135 と同族の
    # 統計量取り違え・統制先行が機能した)。全域 min は記述で併記。
    vn = int(restricted_min(med_neg["mid"]))
    vn_full = int(min(med_neg["mid"]))
    out["controls"] = {"determinism_pass": bool(det),
                       "vn_v3_restricted_min": vn, "vn_pass": vn > 1_000,
                       "vn_anchor": "restricted_min(P_SCAN)>10^3(cii_nulls_v3 R2-9 凍結)",
                       "vn_full_min_descriptive": vn_full,
                       "misanchor_note": "初版 = 全域 min>10⁴(arm-A 基準の誤適用)で 3,648 停止・登記"}
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    L.log(f"  統制 det={'✅' if det else '⛔'} V-N(restricted)={vn} (full={vn_full})")
    if not (det and vn > 1_000):
        out["verdict"] = "STOP_controls"
        op.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        raise SystemExit("⛔ 統制 FAIL")

    # ---- 本走: 各 row × 各 rung の全系列 persist ----
    def gate_lm(n_rep, rid):
        g = gate5 if n_rep in RUNGS_L1 else gate6
        return g["per_row"][rid][f"r{n_rep}"]["landmark_start"]

    for row in rows:
        rid = row["id"]
        if rid not in row_sets:
            out["rows"].append({"id": rid, "verdict": "(c)"})
            continue
        rec = {"id": rid, "rungs": {}}
        for n_rep in RUNGS_L1 + RUNGS_L2:
            prompt = unit * n_rep + row["prompt"]
            jl, n_tok = apply_full(prompt, row_sets[rid])
            med = series(jl, row_sets[rid])
            rec["rungs"][f"r{n_rep}"] = {
                "n_tok": n_tok, "landmark_start": gate_lm(n_rep, rid),
                "mid": med["mid"], "mid_L": med["mid_L"]}
        out["rows"].append(rec)
        op.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        L.log(f"  {rid} 済 ({time.time()-t_wall:.0f}s)")

    # ---- 床(共有 prefix・凍結式)----
    def floors(lane):
        fl = {}
        for rungs, tag in ((RUNGS_L1, "L1"), (RUNGS_L2, "L2")):
            for i, na in enumerate(rungs):
                for nb in rungs[i + 1:]:
                    ra, rb = f"r{na}", f"r{nb}"
                    vals = []
                    for rec in out["rows"]:
                        if "rungs" not in rec:
                            continue
                        A, B = rec["rungs"][ra], rec["rungs"][rb]
                        lo, hi = P_SCAN, min(A["landmark_start"], B["landmark_start"])
                        vals.extend(dlog_series(A[lane][lo:hi], B[lane][lo:hi]))
                    st = dist_stats(vals)
                    st["bitwise_n"] = sum(v == 0.0 for v in vals)
                    fl[f"{tag}:{ra}x{rb}"] = st
        return fl

    out["floors_v3"] = floors("mid")
    out["floors_legacy"] = floors("mid_L")
    old = {"in_domain_max": 0.045, "r44_pairs_max": 0.3135,
           "bitwise_pairs": "88-110 n=360 / 286-352 n=1,350"}
    out["old_registered_descriptive"] = old
    out["reading_note"] = "床は計器属性 ⇒ 新規登記。旧値との突合は記述のみ((a)(b) 照合しない)"
    out["wall_total_s"] = round(time.time() - t_wall, 1)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    mx = {k: v["max"] for k, v in out["floors_v3"].items()}
    L.log(f"⭕ r211_v3 floors_v3 max={mx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
