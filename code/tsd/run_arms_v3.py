#!/usr/bin/env python3
"""Arms A / C and the MapFirst ladders, v3 --- the TSD/R2 re-acquisition (joshaku sets, legacy beside).

REVISION (2026-08-26, token-set defect repair):
  The old reading sets (first tokens plus a >= 2-character filter) contained fragments
  (sp / in / Sh), and the within-set minimum carried a cardinality bias (T7). v3 replaces them
  by P-grain canonical sets, one word one id (SPEC v1.1), and re-runs the old sets as a legacy
  column in every pass. The mechanism (emergence / tau98 / onset_records / behavior_gate /
  band_median_ranks) is imported verbatim from the frozen runners and is not rewritten.
  REVISION-ID: TSD-20260826
  Frozen card = READCARD R2 v1 (md5 a25e03fd..., see frozen/PROVENANCE_v3.md); SPEC v1.1 (85715fe0...)

Reading direction (card item 3): "the intermediate rises above the distractors" means its
band-median rank is strictly smaller (= higher); the strict < in emergence is that sentence.

dp (behavioral gate, card sec. 0): p_word = sum of softmax(p)[id] over the ACCEPTED variants
(= exp(wmass), the probability form of SPEC v1.1 rule 3-7). eps_B = 0.02 is kept and used only
for the between-condition difference of the same word.

Controls:
  * in-run sign-flip (v3 sets) and in-run determinism (both lanes) are stop conditions;
  * the legacy lane's sign-flip value against the old record (arm_a.json 59,337 etc.) is
    recorded only (the frozen runner itself declared that between-run determinism on MPS is not
    frozen and hence not a stop condition; the bitwise gate is carried by in-run determinism).

usage:  python run_arms_v3.py --self-test
        python run_arms_v3.py --fire [--stage a|c|l1|l2|all]
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
OUT_DIR = HERE / "results" / "tsd_r2"
sys.path.insert(0, str(REPO))                                     # joshaku
sys.path.insert(0, str(REPO / "probes" / "olmo3-f5-think-prefix"))
sys.path.insert(0, str(HERE))

import run_f5_lens as L  # noqa: E402
from run_arm_a import K_HOLD, emergence  # noqa: E402
from run_arm_b import EPS_B, behavior_gate  # noqa: E402
from run_arm_c import (CAL_ROWS, CONDS, onset_records,  # noqa: E402
                       tau_from_calibration)
from run_arm_c import coherent as cal_coherent  # noqa: E402

from joshaku.pgrain import build_word_set  # noqa: E402
from joshaku.masks import WORDLIKE_ALNUM, wordlike_mask_ids  # noqa: E402

REVISION_ID = "TSD-20260826"
READCARD = REPO / "conversations/2026-08-26/READCARD_R2_faithfulprose_v1_20260826.md"
SPEC11 = REPO / "SPEC/SPEC_tokenset_acceptance_v1_1.md"
RUNGS_L1 = [8, 10, 11, 12, 14, 22]      # run_cii.RUNGS の repeats(unit=11 token)
RUNGS_L2 = [26, 32, 38, 44]             # run_cii_c2.NEW_RUNGS
GROUPS = ("mid", "ans", "d1", "d2")


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _joshaku_md5() -> str:
    import joshaku
    root = Path(joshaku.__file__).parent
    mods = sorted(p for p in root.rglob("*.py") if "tests" not in str(p))
    return hashlib.md5(b"".join(p.read_bytes() for p in mods)).hexdigest()


def self_test() -> int:
    """v3 較正 = 凍結側 fixture(arm_a/arm_c)を通し、v3 固有 = 単集合等価のみ足す。"""
    import run_arm_a as A
    import run_arm_c as C
    print("== frozen fixtures (arm_a) =="); ra = A.self_test()
    print("== frozen fixtures (arm_c) =="); rc = C.self_test()
    print("== v3 fixtures ==")
    ok = True
    # 単集合の band_median_ranks が「その id の rank」に一致(min の消滅の機械確認)
    import torch
    V = 12
    mask = torch.zeros(V, dtype=torch.bool); mask[2:10] = True
    logits = torch.zeros(1, V)
    for k in range(V):
        logits[0, k] = float(V - k)                 # 単調降下: rank(id5, pool) = 4
    sys.path.insert(0, str(REPO / "probes" / "olmo3-pc2-seat-cue"))
    pc2 = L.load_pc2(); p2 = pc2.p2mod()
    med = p2.band_median_ranks(torch, {0: logits}, mask, {"w": [5]})
    cond = med["w"] == [4]
    print(f"  {'✅' if cond else '⛔'} singleton band rank == pool rank (got {med['w']})")
    ok = ok and cond
    print("PASS" if ok and ra == 0 and rc == 0 else "FAIL")
    return 0 if (ok and ra == 0 and rc == 0) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--fire", action="store_true")
    ap.add_argument("--stage", default="all",
                    choices=["a", "c", "l1", "l2", "nulls", "all"])
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--coherence-max-v3", type=float, default=4.0,
                    help="v3 lane の coherence 柵。既定 4(旧定数)。TSD/R2b 裁可後は"
                         "導出値 B_A(r2b_derivation.json)を渡す")
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
    prov = L.provenance("GT-v3", a.allow_dirty)
    L.log(f"R7-a ✅ commit={prov['commit'][:8]} dirty={prov['dirty']} stage=arms-v3")

    gate2 = json.loads((RES / "gate_tokenizer.json").read_text())
    gate4 = json.loads((RES / "gate_filler_c.json").read_text())
    gate5 = json.loads((RES / "gate_cii.json").read_text())
    gate6 = json.loads((RES / "gate_cii_c2.json").read_text())
    rows = gate2["final_rows"]
    unit = gate4["final_unit"]
    repeats = gate4["repeats"]
    old_arm_a = json.loads((RES / "arm_a.json").read_text())

    pc2 = L.load_pc2()
    import torch, transformers, jlens  # noqa: E401
    assert _md5(pc2.LENS_PT) == pc2.LENS_MD5, "⛔ lens md5 不一致"
    t0 = time.time()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        pc2.MODEL_NAME, dtype=torch.bfloat16, attn_implementation="eager").to("mps")
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(pc2.MODEL_NAME)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(str(pc2.LENS_PT))
    p2 = pc2.p2mod()
    L.log(f"stack ✅ ({time.time()-t0:.0f}s) BAND={pc2.BAND[0]}..{pc2.BAND[-1]}")

    ids = json.loads(pc2.MASK_PATH.read_text())
    mask = torch.zeros(len(tok), dtype=torch.bool); mask[ids] = True
    mask_ids = set(int(i) for i in ids)

    # ---- 集合構築: v3 = joshaku canonical 単集合 / legacy = 旧 gate 集合 ------
    def legacy_sets(rid):
        rep = next(r for r in gate2["reports"] if r["id"] == rid and r["pass"])
        return ({k: [i for i, _ in rep["detail"][k]["distinctive"]] for k in rep["detail"]},
                {k: rep["detail"][k]["word"] for k in rep["detail"]})

    row_sets, row_entries, c_rows = {}, {}, []
    for row in rows:
        leg, words = legacy_sets(row["id"])
        ws = build_word_set(tok, row["id"],
                            [words["mid"], words["ans"], words["d1"], words["d2"]],
                            mask_ids=mask_ids)
        ent = {g: e for g, e in zip(GROUPS, ws.entries)}
        if any(ent[g].canonical_id is None for g in GROUPS):
            c_rows.append(row["id"])                 # 札 §2b: (c) 宣言行(campaign 行では無いはず)
            continue
        merged = {g: [int(ent[g].canonical_id)] for g in GROUPS}
        merged.update({f"{g}_L": leg[g] for g in GROUPS})
        row_sets[row["id"]] = merged
        row_entries[row["id"]] = ent
    L.log(f"sets ✅ v3 canonical rows={list(row_sets)} (c)_rows={c_rows}")

    def series(jl_like, sets_row):
        med = p2.band_median_ranks(torch, jl_like, mask, sets_row)
        return {k: [int(v) for v in med[k]] for k in sets_row}

    def apply_full(prompt, sets_row):
        n_tok = len(tok(prompt)["input_ids"])
        jl, ml, _ = lens.apply(model, prompt, layers=pc2.BAND,
                               positions=list(range(n_tok)), max_seq_len=pc2.MAX_SEQ)
        return jl, ml, n_tok

    def p_word(ml_last, ids_list):
        p = torch.softmax(ml_last.float(), dim=-1)
        return float(sum(p[i] for i in ids_list))

    meta = {"revision_id": REVISION_ID,
            "readcard": {"path": READCARD.name, "md5": _md5(READCARD)},
            "spec_v1_1": {"path": SPEC11.name, "md5": _md5(SPEC11)},
            "joshaku_md5": _joshaku_md5(),
            "lens_md5": pc2.LENS_MD5, "model": pc2.MODEL_NAME,
            "band": [pc2.BAND[0], pc2.BAND[-1]], "k_hold": K_HOLD, "eps_b": EPS_B,
            "mask_size": int(mask.sum()), "wordlike": WORDLIKE_ALNUM.pattern,
            "gate_md5": {k: _md5(RES / f"{k}.json") for k in
                         ("gate_tokenizer", "gate_filler_c", "gate_cii", "gate_cii_c2")},
            "c_rows": c_rows, "provenance": prov}
    # 札 v2 の宇宙 assert(ADJ_olmo32b §2-①③: N は弱い宇宙検査 ⇒ pc2 凍結定数の逐語一致)
    assert pc2.MODEL_NAME == "allenai/Olmo-3-1125-32B", f"⛔ MODEL_NAME: {pc2.MODEL_NAME}"
    assert pc2.MASK_PATH.name == "wordlike_mask_olmo3_32b.json", f"⛔ MASK: {pc2.MASK_PATH.name}"
    assert pc2.LENS_PT.name == "Olmo-3-1125-32B_jacobian_lens.pt", f"⛔ LENS: {pc2.LENS_PT.name}"
    assert meta["mask_size"] == 70812, \
        f"⛔ N 不一致: {meta['mask_size']} (札 v2 ⑥ = 70,812)"
    meta["tokenizer_pin"] = "allenai/Olmo-3-1125-32B@c2b61dae89a1ad10e4ad5653d0e46b590902607b"

    stages = [a.stage] if a.stage != "all" else ["a", "c", "l1", "l2"]

    # ============ Arm A ============
    if "a" in stages:
        out = {"meta": meta, "stage": "arm_a_v3", "controls": {}, "rows": []}
        op = OUT_DIR / "arm_a_v3.json"
        for irow, row in enumerate(rows):
            rid = row["id"]
            if rid not in row_sets:
                out["rows"].append({"id": rid, "verdict": "(c)"}); continue
            sets_row = row_sets[rid]
            jl, ml, n_tok = apply_full(row["prompt"], sets_row)
            med = series(jl, sets_row)
            if irow == 0:
                jl_neg = {k: -v for k, v in jl.items()}
                med_neg = series(jl_neg, sets_row)
                vn_v3 = int(min(med_neg["mid"]))
                vn_leg = int(min(med_neg["mid_L"]))
                jl2, _, _ = apply_full(row["prompt"], sets_row)
                det_ok = series(jl2, sets_row) == med
                old_vn = old_arm_a["controls"]["vn_signflip_min_rank"]
                out["controls"] = {
                    "vn_v3_min": vn_v3, "vn_pass": vn_v3 > 10_000,
                    "vn_legacy_min": vn_leg,
                    "vn_legacy_vs_old_registered": {"old": old_vn, "new": vn_leg,
                                                    "match": vn_leg == old_vn,
                                                    "note": "登記のみ(run 間決定性は未凍結・凍結 runner の先例)"},
                    "determinism_pass": bool(det_ok)}
                L.log(f"  統制 V-N v3={vn_v3} legacy={vn_leg}(旧 {old_vn}) det={'✅' if det_ok else '⛔'}")
                if not (vn_v3 > 10_000 and det_ok):
                    out["verdict"] = "STOP_controls"
                    op.write_text(json.dumps(out, ensure_ascii=False, indent=1))
                    raise SystemExit("⛔ 走行時統制 FAIL")
            em_v3, w_v3 = emergence(med["mid"], [med["d1"], med["d2"]])
            em_lg, w_lg = emergence(med["mid_L"], [med["d1_L"], med["d2_L"]])
            out["rows"].append({"id": rid, "n_tok": n_tok, "lens": med,
                                "emerged_v3": bool(em_v3), "window_start_v3": w_v3,
                                "emerged_legacy": bool(em_lg), "window_start_legacy": w_lg})
            op.write_text(json.dumps(out, ensure_ascii=False, indent=1))
            L.log(f"  A {rid} v3={'⭕' if em_v3 else '—'}@{w_v3} legacy={'⭕' if em_lg else '—'}@{w_lg}")
        out["m_v3"] = sum(1 for r in out["rows"] if r.get("emerged_v3"))
        out["m_legacy"] = sum(1 for r in out["rows"] if r.get("emerged_legacy"))
        op.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        L.log(f"⭕ arm_a_v3: m_v3={out['m_v3']}/5 m_legacy={out['m_legacy']}/5")

    # ============ Arm C ============
    if "c" in stages:
        out = {"meta": meta, "stage": "arm_c_v3", "rows": []}
        op = OUT_DIR / "arm_c_v3.json"
        # 較正 τ98(v3 集合・凍結式)+ legacy 並記
        pooled_v3, pooled_lg, cal = [], [], {}
        for rid in CAL_ROWS:
            row = next(r for r in rows if r["id"] == rid)
            prompt = unit * repeats["fbig"] + row["prompt"]
            jl, _, _ = apply_full(prompt, row_sets[rid])
            jl_neg = {k: -v for k, v in jl.items()}
            med = series(jl_neg, row_sets[rid])
            pooled_v3.extend(med["mid"]); pooled_lg.extend(med["mid_L"])
            cal[rid] = {"min_v3": int(min(med["mid"])), "min_legacy": int(min(med["mid_L"]))}
        tau_v3 = tau_from_calibration(pooled_v3)
        tau_lg = tau_from_calibration(pooled_lg)
        mins_v3 = [c["min_v3"] for c in cal.values()]
        spread_v3 = max(mins_v3) / min(mins_v3)
        # TSD/R2b: v3 lane の柵 = 導出値 B(--coherence-max-v3)。legacy coherent は記述。
        ok = spread_v3 <= a.coherence_max_v3
        out["calibration"] = {"per_row": cal, "tau98_v3": tau_v3, "tau98_legacy": tau_lg,
                              "spread_v3": round(spread_v3, 4),
                              "coherence_bound_v3": a.coherence_max_v3,
                              "coherent_v3": bool(ok),
                              "coherent_legacy_descriptive": bool(
                                  cal_coherent([c["min_legacy"] for c in cal.values()]))}
        L.log(f"  較正 τ98 v3={tau_v3} legacy={tau_lg} "
              f"spread={spread_v3:.2f} vs B={a.coherence_max_v3} {'✅' if ok else '⛔'}")
        if not ok:
            out["verdict"] = "M7_STOP_calibration"
            op.write_text(json.dumps(out, ensure_ascii=False, indent=1))
            raise SystemExit("⛔ 較正整合 FAIL")
        for row in rows:
            rid = row["id"]
            if rid not in row_sets:
                out["rows"].append({"id": rid, "verdict": "(c)"}); continue
            conds = {}
            for cond in CONDS:
                prompt = unit * repeats[cond] + row["prompt"]
                lm0 = gate4["per_row"][rid][cond]["landmark_start"]
                jl, ml, n_tok = apply_full(prompt, row_sets[rid])
                med = series(jl, row_sets[rid])
                rec3 = onset_records(med["mid"], [med["d1"], med["d2"]], lm0)
                recL = onset_records(med["mid_L"], [med["d1_L"], med["d2_L"]], lm0)
                conds[cond] = {"n_tok": n_tok, "landmark_start": lm0,
                               "v3": rec3, "legacy": recL}
                op_dump = {"id": rid, "conds": conds}
            out["rows"].append({"id": rid, "conds": conds})
            op.write_text(json.dumps(out, ensure_ascii=False, indent=1))
            L.log(f"  C {rid} " + " ".join(
                f"{c}:v3@{conds[c]['v3']['onset_rel_landmark']}"
                f"/L@{conds[c]['legacy']['onset_rel_landmark']}" for c in CONDS))
        op.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        L.log("⭕ arm_c_v3 saved")

    # ============ MapFirst ============
    def ladder(stage_name, rung_reps, gate_lm, ref_store):
        out = {"meta": meta, "stage": stage_name, "rows": []}
        op = OUT_DIR / f"{stage_name}.json"
        for row in rows:
            rid = row["id"]
            if rid not in row_sets:
                out["rows"].append({"id": rid, "verdict": "(c)"}); continue
            ent_ans = row_entries[rid]["ans"]
            ans_v3_ids = [i for _v, i in ent_ans.accepted]
            leg_ans = row_sets[rid]["ans_L"]
            rec = {"id": rid, "rungs": {}}
            for n_rep in rung_reps:
                rk = f"r{n_rep}"
                prompt = unit * n_rep + row["prompt"]
                lm0 = gate_lm[rid][rk]["landmark_start"]
                jl, ml, n_tok = apply_full(prompt, row_sets[rid])
                med = series(jl, row_sets[rid])
                last = ml[-1]
                top1 = int(torch.argmax(last.float()))
                rec["rungs"][rk] = {
                    "n_tok": n_tok, "landmark_start": lm0,
                    "top1": top1, "top1_str": tok.decode([top1]),
                    "p_word_v3": round(p_word(last, ans_v3_ids), 6),
                    "p_ans_legacy": round(p_word(last, leg_ans), 6),
                    "v3": onset_records(med["mid"], [med["d1"], med["d2"]], lm0),
                    "legacy": onset_records(med["mid_L"], [med["d1_L"], med["d2_L"]], lm0)}
                op.write_text(json.dumps(out | {"rows": out["rows"] + [rec]},
                                         ensure_ascii=False, indent=1))
            # behavior gate vs ref(r8)
            ref = ref_store.setdefault(rid, rec["rungs"].get("r8"))
            for rk, rr in rec["rungs"].items():
                if ref is None or rk == "r8":
                    continue
                g3, why3 = behavior_gate(ref["top1"], ref["p_word_v3"],
                                         rr["top1"], rr["p_word_v3"])
                gl, whyl = behavior_gate(ref["top1"], ref["p_ans_legacy"],
                                         rr["top1"], rr["p_ans_legacy"])
                rr["gate_v3"] = {"pass": bool(g3), "why": why3}
                rr["gate_legacy"] = {"pass": bool(gl), "why": whyl}
            out["rows"].append(rec)
            op.write_text(json.dumps(out, ensure_ascii=False, indent=1))
            L.log(f"  {stage_name} {rid} " + " ".join(
                f"{rk}:{'⭕' if rr['v3']['emerged_landmark'] else '—'}"
                f"@{rr['v3']['onset_rel_landmark']}" for rk, rr in rec["rungs"].items()))
            if time.time() - t_wall > 7200:
                raise SystemExit("⛔ wall 2h 超過")
        op.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        L.log(f"⭕ {stage_name} saved")
        return out

    # ============ MapFirst null 統制(R2-9・rung ごと sign-flip・scan 域 min)=====
    if "nulls" in stages:
        from run_cii import P_SCAN, restricted_min
        out = {"meta": meta, "stage": "cii_nulls_v3", "p_scan": P_SCAN, "cells": []}
        op = OUT_DIR / "cii_nulls_v3.json"
        import math
        for row in rows:
            rid = row["id"]
            if rid not in row_sets:
                continue
            for n_rep, gl in [(n, gate5["per_row"]) for n in RUNGS_L1] + \
                             [(n, gate6["per_row"]) for n in RUNGS_L2]:
                rk = f"r{n_rep}"
                prompt = unit * n_rep + row["prompt"]
                jl, _, _ = apply_full(prompt, row_sets[rid])
                jl_neg = {k: -v for k, v in jl.items()}
                med = series(jl_neg, row_sets[rid])
                v3m = restricted_min(med["mid"])
                lgm = restricted_min(med["mid_L"])
                out["cells"].append({"id": rid, "rung": rk,
                                     "null_min_v3": int(v3m), "null_min_legacy": int(lgm),
                                     "pass_v3": v3m > 1000})
                op.write_text(json.dumps(out, ensure_ascii=False, indent=1))
            L.log(f"  nulls {rid} 済")
        vals = [c["null_min_v3"] for c in out["cells"]]
        out["summary"] = {"n_cells": len(vals), "all_gt_1000": all(v > 1000 for v in vals),
                          "min_cell": min(vals),
                          "margin_dex": round(math.log10(min(vals) / 1000), 4)}
        op.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        L.log(f"⭕ nulls: min={min(vals)} margin={out['summary']['margin_dex']} dex")

    ref_store = {}
    if "l1" in stages:
        ladder("cii_l1_v3", RUNGS_L1, gate5["per_row"], ref_store)
    if "l2" in stages:
        if not ref_store:  # L2 単独走行時は L1 出力から r8 参照を読む
            prev = json.loads((OUT_DIR / "cii_l1_v3.json").read_text())
            for r in prev["rows"]:
                if "rungs" in r:
                    ref_store[r["id"]] = r["rungs"].get("r8")
        ladder("cii_l2_v3", RUNGS_L2, gate6["per_row"], ref_store)

    L.log(f"⭕ arms v3 全段完了 wall={time.time()-t_wall:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
