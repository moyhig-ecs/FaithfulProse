#!/usr/bin/env python3
"""C-ii campaign 2 (512 級) —— 計器域外較正の実験・第二梯子。

凍結: conversations/2026-08-20/PREREG_cii_ladder_20260820.md v3 block
裁可: 先生「campaign2回しましょう。2時間後にトークン量リフレッシュなので混みそう、
      その前に」(2026-08-20・発火込み)。

最小差分 (v2 からの変更 = 梯子のみ):
  新 rung {r26:286, r32:352, r38:418, r44:484}。r8 (参照)・r22 は v2 走行から再利用
  (同 shape 決定論・INSTR verbatim・裁定 §2.3-(v) の先例・cross-run bitwise 3 例の実績)。
  述語は全て v2 verbatim import: τ_dec / 域述語 / 膝指標 (帯セット不変 = 比較可能性優先) /
  行動ゲート / 窓比較 / 分岐表。★ 追加は記述列のみ: 深部 placebo 対比 (膝述語の帯対比を
  深部帯対 {(256,288)v(288,320), (384,416)v(416,448)} でも計算・判定不使用・A-5)。
  予言 slot なし (P-C2 三本は campaign 1 で消費済・「静かな境界」は事前素材であって
  予言ではない = 検収便 verbatim)。

usage:  python run_cii_c2.py --self-test
        python run_cii_c2.py --fire
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
sys.path.insert(0, str(REPO / "probes" / "olmo3-f5-think-prefix"))
sys.path.insert(0, str(HERE))
import run_f5_lens as L  # noqa: E402
from run_arm_a import emergence, K_HOLD, W_BAND, ETA_BAND  # noqa: E402
from run_arm_b import behavior_gate, EPS_B, dlog_series, dist_stats  # noqa: E402
from run_arm_c import onset_records  # noqa: E402
from run_cii import (TAU_DEC, P_SCAN, W_FIT, R_SHARED, SHARED_FLAG_DEX, PERIOD,  # noqa: E402
                     KNEE_BANDS, KNEE_FACTOR, KNEE_MAJORITY,
                     phase_positions, knee_series, restricted_min, region_min,
                     shared_flag, branch)

NEW_RUNGS = ["r26", "r32", "r38", "r44"]
REF_RUNG = "r8"          # 参照 = v2 store から再利用 (cross-run・宣言済)
PLACEBO_BANDS = (((256, 288), (288, 320)), ((384, 416), (416, 448)))
# 深部 placebo 対比 (A-5 配置・記述列のみ・判定不使用。錨 = 膝述語の帯対比が
# 「境界 128 特異」かを深部の同型対比と並置するため・述語自体は verbatim 不変)


def placebo_contrasts(mid_series: list, lm0: int) -> list:
    """深部帯対の位相対比 (knee_series と同式・判定不使用・記述)。"""
    out = []
    for b_pre, b_post in PLACEBO_BANDS:
        votes = []
        for phi in range(PERIOD):
            pos = phase_positions(phi, lm0)
            m = []
            for b in (b_pre, b_post):
                vals = [math.log10(mid_series[p]) for p in pos if b[0] <= p < b[1]]
                m.append(statistics.median(vals) if vals else None)
            votes.append(None if any(v is None for v in m) else round(abs(m[1] - m[0]), 4))
        valid = [v for v in votes if v is not None]
        out.append({"bands": [b_pre, b_post],
                    "median_contrast": round(statistics.median(valid), 4) if valid else None,
                    "n_valid": len(valid)})
    return out


def self_test() -> int:
    fails = []

    def check(name, cond):
        print(f"  {'✅' if cond else '⛔'} {name}")
        if not cond:
            fails.append(name)

    check("① emergence import", emergence([9, 5, 5, 5, 9], [[8] * 5, [7] * 5]) == (True, 1))
    check("② tau_dec/region import", TAU_DEC == 1000 and region_min([1] * 16 + [50] * 84, 16, 88) == 50)
    # ③ placebo: 平坦系列 ⇒ 対比 ~0 / 深部段差 ⇒ 対比 >0
    flat = [1000] * 494
    step = [1000] * 288 + [100] * 206
    p_flat = placebo_contrasts(flat, 484)
    p_step = placebo_contrasts(step, 484)
    check("③ placebo flat ~0", p_flat[0]["median_contrast"] == 0.0)
    check("③ placebo step >0", p_step[0]["median_contrast"] > 0.5)
    # ④ knee 述語は verbatim (import 同一性・階段再検分)
    from run_cii import knee_phase
    check("④ knee import", knee_phase([1000] * 128 + [100] * 128) is True)
    # ⑤ v2 store の再利用 loader 形 (synthetic)
    fake = {"rows": [{"id": "X", "rungs": {"r8": {"lens": {"mid": [1, 2]}, "top1": 7,
            "p_ans": 0.5, "emerged_landmark": True, "onset_rel_landmark": 3,
            "onset_abs_landmark": 91, "landmark_start": 88}}}]}
    ref = fake["rows"][0]["rungs"]["r8"]
    check("⑤ v2 ref fields", ref["top1"] == 7 and ref["onset_rel_landmark"] == 3)
    print("PASS" if not fails else f"FAIL: {fails}")
    return 0 if not fails else 1


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

    t_wall = time.time()
    prov = L.provenance("C-ii-c2", a.allow_dirty)
    L.log(f"R7-a ✅ commit={prov['commit'][:8]} dirty={prov['dirty']} stage=C-ii-c2")

    gate2 = json.loads((RES / "gate_tokenizer.json").read_text())
    gate6 = json.loads((RES / "gate_cii_c2.json").read_text())
    if gate2["verdict"] != "PASS" or gate6["verdict"] != "PASS":
        raise SystemExit("⛔ gate v2/c2 が PASS でない")
    v2 = json.loads((RES / "cii.json").read_text())
    if v2.get("verdict") != "完走 (計器の地図+onset 登記・読みは判断層)":
        raise SystemExit("⛔ v2 store が完走 verdict でない")
    g2_md5 = hashlib.md5((RES / "gate_tokenizer.json").read_bytes()).hexdigest()
    g6_md5 = hashlib.md5((RES / "gate_cii_c2.json").read_bytes()).hexdigest()
    v2_md5 = hashlib.md5((RES / "cii.json").read_bytes()).hexdigest()
    self_md5 = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
    rows = gate2["final_rows"]
    unit = gate6["unit"]
    L.log(f"gate ✅ v2={g2_md5[:8]} c2={g6_md5[:8]} v2store={v2_md5[:8]} "
          f"rungs={gate6['rungs']}")

    pc2 = L.load_pc2()
    import torch, transformers, jlens  # noqa: E401
    got = hashlib.md5(pc2.LENS_PT.read_bytes()).hexdigest()
    if got != pc2.LENS_MD5:
        raise SystemExit(f"⛔ lens md5 不一致: {got}")
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

    def series(jl_like, sets_row):
        med = p2.band_median_ranks(torch, jl_like, mask, sets_row)
        return {k: [int(v) for v in med[k]] for k in sets_row}

    def sets_of(rid):
        rep = next(r for r in gate2["reports"] if r["id"] == rid and r["pass"])
        return {k: [i for i, _ in rep["detail"][k]["distinctive"]]
                for k in rep["detail"]}

    def build(rid, rk):
        row = next(r for r in rows if r["id"] == rid)
        return unit * gate6["per_row"][rid][rk]["repeats"] + row["prompt"]

    def apply_full(prompt, sets_row):
        n_tok = len(tok(prompt)["input_ids"])
        jl, ml, _ = lens.apply(model, prompt, layers=pc2.BAND,
                               positions=list(range(n_tok)),
                               max_seq_len=pc2.MAX_SEQ)
        return jl, ml, n_tok

    out = {"arm": "C-ii campaign 2 (512 級)", "gate_v2_md5": g2_md5,
           "gate_c2_md5": g6_md5, "v2_store_md5": v2_md5, "runner_md5": self_md5,
           "lens_md5": pc2.LENS_MD5, "band": [pc2.BAND[0], pc2.BAND[-1]],
           "w_fit": W_FIT, "p_scan": P_SCAN, "rungs": gate6["rungs"],
           "ref_rung": REF_RUNG, "ref_source": "v2 store (cross-run 再利用・INSTR)",
           "eps_b": EPS_B, "tau_dec": TAU_DEC, "k_hold": K_HOLD,
           "eta_band": ETA_BAND, "w_band_dex": W_BAND,
           "provenance": prov, "calibration": {}, "controls": {}, "rows": []}
    out_path = RES / "cii_c2.json"

    # ---- Phase 較正 (v2 述語 verbatim・新 rung のみ 20 forward) ---------------
    ledger = {}
    for row in rows:
        rid = row["id"]
        v2_led = v2["calibration"]["ledger"][rid]
        ledger[rid] = {}
        for rk in NEW_RUNGS:
            jl, _, _ = apply_full(build(rid, rk), sets_of(rid))
            jl_neg = {k: -v for k, v in jl.items()}
            s = series(jl_neg, sets_of(rid))["mid"]
            lm0 = gate6["per_row"][rid][rk]["landmark_start"]
            ent = {"scan_min": restricted_min(s),
                   "region_min": {"R_shared": region_min(s, *R_SHARED),
                                  "R_newfill": region_min(s, 88, lm0),
                                  "R_land": region_min(s, lm0, len(s))},
                   "flipped_series": s}
            ent["gate_pass"] = bool(ent["scan_min"] > TAU_DEC)
            ledger[rid][rk] = ent
            if not ent["gate_pass"]:
                out["calibration"] = {"ledger": ledger, "tau_dec": TAU_DEC}
                out["verdict"] = "STOP_tau_dec"
                out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
                raise SystemExit(f"⛔ 決定錨床 FAIL ({rid}/{rk}: "
                                 f"{ent['scan_min']} ≤ {TAU_DEC}) ⇒ 停止")
        L.log(f"  台帳 {rid}: " + " ".join(
            f"{rk}:{ledger[rid][rk]['scan_min']}" for rk in NEW_RUNGS)
            + " ✅ (> τ_dec)")
    # 共有域 sanity FLAG (vs v2 r8 R_shared min・cross-run 宣言済・停止機ではない)
    flags = []
    for rid in ledger:
        m8 = v2["calibration"]["ledger"][rid]["r8"]["region_min"]["R_shared"]
        for rk in NEW_RUNGS:
            mk = ledger[rid][rk]["region_min"]["R_shared"]
            if shared_flag(mk, m8):
                flags.append({"row": rid, "rung": rk, "min": mk, "r8": m8,
                              "d_dex": round(abs(math.log10(mk) - math.log10(m8)), 4)})
    out["calibration"] = {"ledger": ledger, "tau_dec": TAU_DEC,
                          "shared_sanity_flags": flags,
                          "spread_descriptive": {rk: round(
                              max(ledger[r][rk]["scan_min"] for r in ledger)
                              / min(ledger[r][rk]["scan_min"] for r in ledger), 2)
                              for rk in NEW_RUNGS}}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    L.log(f"  共有域 FLAG = {len(flags)} 件・spread(記述) = "
          f"{out['calibration']['spread_descriptive']}")
    # 決定性 (P1 r44 = 最大 shape)
    p1_sets = sets_of("P1")
    jl1, _, _ = apply_full(build("P1", "r44"), p1_sets)
    med1 = series(jl1, p1_sets)
    jl2, _, _ = apply_full(build("P1", "r44"), p1_sets)
    det_ok = series(jl2, p1_sets) == med1
    out["controls"]["determinism_r44_pass"] = bool(det_ok)
    L.log(f"  決定性 (P1 r44) {'✅' if det_ok else '⛔'}")
    if not det_ok:
        out["verdict"] = "STOP_controls"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        raise SystemExit("⛔ 決定性 FAIL ⇒ 停止 (値の報告をしない)")

    # ---- Phase 測定 (20 forward) --------------------------------------------
    for row in rows:
        rid = row["id"]
        sets_row = sets_of(rid)
        v2_row = next(r for r in v2["rows"] if r["id"] == rid)
        rec = {"id": rid, "rungs": {}}
        for rk in NEW_RUNGS:
            lm0 = gate6["per_row"][rid][rk]["landmark_start"]
            jl, ml, n_tok = apply_full(build(rid, rk), sets_row)
            med_l = series(jl, sets_row)
            logits = ml[-1].float()
            p = torch.softmax(logits, dim=-1)
            top1 = int(torch.argmax(logits))
            p_ans = float(sum(p[i] for i in sets_row["ans"]))
            dis = [k for k in sets_row if k.startswith("d")]
            onset = onset_records(med_l["mid"], [med_l[k] for k in dis], lm0)
            rec["rungs"][rk] = {"n_tok": n_tok, "landmark_start": lm0,
                                "lens": med_l, "top1": top1,
                                "top1_str": tok.decode([top1]),
                                "p_ans": round(p_ans, 6), **onset}
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        out["rows"].append(rec)
        L.log(f"  {rid} " + " ".join(
            f"{rk}:{'⭕' if rec['rungs'][rk]['emerged_landmark'] else '—'}"
            f"@rel{rec['rungs'][rk]['onset_rel_landmark']}" for rk in NEW_RUNGS))
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        if time.time() - t_wall > 7200:
            out["verdict"] = "STOP_wall"
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
            raise SystemExit("⛔ wall 2h 超過 ⇒ 停止して報告")

    # ---- Phase 後処理 (zero-forward・v2 述語 verbatim) -----------------------
    for rec in out["rows"]:
        v2_row = next(r for r in v2["rows"] if r["id"] == rec["id"])
        ref = v2_row["rungs"][REF_RUNG]
        bg, deltas = {}, {}
        for rk in NEW_RUNGS:
            v = rec["rungs"][rk]
            ok, why = behavior_gate(ref["top1"], ref["p_ans"], v["top1"], v["p_ans"])
            bg[rk] = {"pass": bool(ok), "reason": why,
                      "signed_dp_ans": round(v["p_ans"] - ref["p_ans"], 6)}
            if ok and ref["emerged_landmark"] and v["emerged_landmark"]:
                deltas[rk] = {"d_rel": v["onset_rel_landmark"]
                              - ref["onset_rel_landmark"],
                              "d_abs": v["onset_abs_landmark"]
                              - ref["onset_abs_landmark"]}
        rec["behavior_gate"] = bg
        rec["delta_onsets_vs_ref"] = deltas
        r44 = rec["rungs"]["r44"]
        rec["knee_r44"] = knee_series(r44["lens"]["mid"], r44["landmark_start"])
        rec["placebo_r44"] = placebo_contrasts(r44["lens"]["mid"],
                                               r44["landmark_start"])

    # 共有 prefix 床: 新 rung 対 (同 run) + r22(v2)×新 (cross-run 宣言)
    floors = {}
    for i, ra in enumerate(NEW_RUNGS):
        for rb in NEW_RUNGS[i + 1:]:
            vals = []
            for rec in out["rows"]:
                la = rec["rungs"][ra]["landmark_start"]
                lb = rec["rungs"][rb]["landmark_start"]
                lo, hi = P_SCAN, min(la, lb)
                vals.extend(dlog_series(rec["rungs"][ra]["lens"]["mid"][lo:hi],
                                        rec["rungs"][rb]["lens"]["mid"][lo:hi]))
            floors[f"{ra}x{rb}"] = dist_stats(vals)
    vals = []
    for rec in out["rows"]:
        v2_row = next(r for r in v2["rows"] if r["id"] == rec["id"])
        s22 = v2_row["rungs"]["r22"]["lens"]["mid"]
        s26 = rec["rungs"]["r26"]["lens"]["mid"]
        lo, hi = P_SCAN, min(242, 286)
        vals.extend(dlog_series(s22[lo:hi], s26[lo:hi]))
    floors["r22[v2]xr26"] = dist_stats(vals)
    out["shared_prefix_floors"] = floors

    # 分岐表 (全 rung out-of-fit・v2 述語 verbatim)
    verdicts = {}
    for rec in out["rows"]:
        v2_row = next(r for r in v2["rows"] if r["id"] == rec["id"])
        per = {}
        for rk in NEW_RUNGS:
            v = rec["rungs"][rk]
            if not (rec["behavior_gate"][rk]["pass"] and v["emerged_landmark"]
                    and v2_row["rungs"][REF_RUNG]["emerged_landmark"]):
                per[rk] = None
                continue
            degraded = rec["delta_onsets_vs_ref"].get(rk, {}).get("d_rel", 0) != 0
            per[rk] = branch(False, degraded, rec["knee_r44"]["knee"])
        verdicts[rec["id"]] = per
    out["branch_table"] = verdicts
    out["m_by_rung"] = {rk: sum(r["rungs"][rk]["emerged_landmark"]
                                for r in out["rows"]) for rk in NEW_RUNGS}
    out["verdict"] = "完走 (計器の地図 第二梯子・読みは判断層)"
    out["wall_total_s"] = round(time.time() - t_wall, 1)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    L.log(f"⭕ saved {out_path} m={out['m_by_rung']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
