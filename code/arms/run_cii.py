#!/usr/bin/env python3
"""C-ii (跨ぎ版 filler translation・仮称) —— 計器域外較正の実験。

凍結: conversations/2026-08-20/PREREG_cii_ladder_20260820.md (全 §)
裁可: 先生「凍結、予言三本採択で起草しましょう」(起草・凍結)。⛔ 発火 = 先生一言。

第一産物 = 計器の地図 (総評 verbatim)。走行順: 較正が先 (統制域 = 測定域 = p≥16)。
機構 ⛔ 無改変: emergence/K_HOLD = run_arm_a / behavior_gate/EPS_B/dlog_series/
dist_stats = run_arm_b / tau_from_calibration/coherent/onset_records = run_arm_c
(全て verbatim import・単一出所)。新規 = 位相集合・膝指標・比較域述語・分岐表
⇒ fixture ①-⑧ を走行前に通す。

usage:  python run_cii.py --self-test
        python run_cii.py --fire
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
from run_arm_c import onset_records  # noqa: E402  (v2: tau/coherent は裁定で撤去)

# ---- 凍結定数 (PREREG §1-§7 + v2 裁定 §2.3・A-5 は錨を札に登記) --------------
RUNGS = ["r8", "r10", "r11", "r12", "r14", "r22"]   # lm0 = 11×{8,10,11,12,14,22}
REF_RUNG = "r8"                                      # C-i fbig 錨
P_SCAN = 16                                          # 走査域下限 (= W_fit 下限)
W_FIT = (16, 128)                                    # 両側述語 (hygiene 5)
V1_CAL_ROWS = ["P2", "P3", "B1", "P5"]               # v1 r8 系列の再利用出所
PERIOD = 11
TAU_DEC = 1000       # 決定錨床 = 10·τ_A (τ_A=100 GT-A 凍結値・一 dex = 家の自然単位。
                     # 本日の null 値を参照しない錨・汚染開示は PREREG v2 に全数)
R_SHARED = (16, 88)  # 域述語 (構成定数): 共有 prefix 走査域。R_newfill=[88,lm0)/R_land=[lm0,n)
SHARED_FLAG_DEX = 0.135  # 共有域 sanity FLAG 幅 = 共有 prefix ゆらぎ max 0.045 × 係数 3
                         # (配置定数・FLAG は停止機ではない)
KNEE_BANDS = ((96, 128), (128, 160), (160, 192), (192, 224))  # 膝指標の帯 (§5)
KNEE_FACTOR = 2                                      # 配置定数 (§5・検分 = 判断層)
KNEE_MAJORITY = 6                                    # 位相 11 本中 ≥6


def phase_positions(phi: int, lm0: int, lo: int = P_SCAN) -> list:
    """位相集合述語 (§5): d = phi + 11k, lo <= d < lm0。"""
    return [d for d in range(phi, lm0, PERIOD) if d >= lo]


def band_median_log(series: list, band: tuple) -> float | None:
    vals = [math.log10(series[p]) for p in range(band[0], min(band[1], len(series)))]
    return statistics.median(vals) if vals else None


def knee_phase(series: list) -> bool | None:
    """膝指標 (§5・機械): 帯対比 > KNEE_FACTOR × 遠方対比。帯不足なら None。"""
    m = [band_median_log(series, b) for b in KNEE_BANDS]
    if any(v is None for v in m):
        return None
    boundary = abs(m[1] - m[0])
    distal = abs(m[3] - m[2])
    return boundary > KNEE_FACTOR * distal


def knee_series(mid_series: list, lm0: int) -> dict:
    """rung の位相曲線 11 本に膝指標を適用 (filler 域のみ・系列膝判 = ≥6/11)。"""
    votes = []
    for phi in range(PERIOD):
        pos = phase_positions(phi, lm0)
        if not pos:
            votes.append(None)
            continue
        # 位相曲線を position 順の series に展開して帯中央値を取る
        curve = {p: mid_series[p] for p in pos}
        m = []
        for b in KNEE_BANDS:
            vals = [math.log10(curve[p]) for p in pos if b[0] <= p < b[1]]
            m.append(statistics.median(vals) if vals else None)
        if any(v is None for v in m):
            votes.append(None)
        else:
            votes.append(abs(m[1] - m[0]) > KNEE_FACTOR * abs(m[3] - m[2]))
    n_yes = sum(1 for v in votes if v is True)
    n_valid = sum(1 for v in votes if v is not None)
    return {"votes": votes, "n_yes": n_yes, "n_valid": n_valid,
            "knee": bool(n_yes >= KNEE_MAJORITY)}


def restricted_min(series: list, lo: int = P_SCAN) -> int:
    """p≥16 制限 min (§2 (i)(ii))。"""
    return int(min(series[lo:]))


def region_min(series: list, lo: int, hi: int, floor: int = P_SCAN):
    """域述語 min (v2 裁定 §2.3-(i)): [max(lo, floor), hi)。空域は None。"""
    seg = series[max(lo, floor):hi]
    return int(min(seg)) if seg else None


def shared_flag(min_rung: int, min_r8: int, width: float = SHARED_FLAG_DEX) -> bool:
    """共有域 sanity (v2 裁定 §2.3-(iii)): |Δlog10| > width で FLAG (停止機ではない)。"""
    return abs(math.log10(min_rung) - math.log10(min_r8)) > width


def comparison_domain_ok(pos: int, lm0_pair: tuple) -> bool:
    """窓比較修正述語 (§4): 非共有域 = min(lm0) 以深。"""
    return pos >= min(lm0_pair)


def branch(in_fit_window: bool, degraded: bool, knee: bool | None) -> str:
    """分岐表 B1-B4 (§6・機械適用)。"""
    if in_fit_window:
        return "B1" if not degraded else "B3"  # in-fit 劣化は膝概念外 ⇒ 保留枝
    if not degraded:
        return "B4"
    return "B2" if knee else "B3"


def self_test() -> int:
    fails = []

    def check(name, cond):
        print(f"  {'✅' if cond else '⛔'} {name}")
        if not cond:
            fails.append(name)

    # ① import 同一性 (arm_a fixture 再検分)
    check("① emergence import", emergence([9, 5, 5, 5, 9], [[8] * 5, [7] * 5]) == (True, 1))
    # ② 位相集合述語
    check("② phase set", phase_positions(5, 33) == [16, 27])
    check("② phase set lo", phase_positions(0, 88) == [22, 33, 44, 55, 66, 77])
    # ③ 膝指標: 128 で 1 dex 落ちる階段 vs 平坦
    step = [1000] * 128 + [100] * 128
    flat = [1000] * 256
    check("③ knee step -> True", knee_phase(step) is True)
    check("③ knee flat -> False", knee_phase(flat) is False)
    # ④ p≥16 制限 min
    check("④ restricted min", restricted_min([1] * 16 + [50, 40, 60]) == 40)
    # ⑤ 比較域述語: (88,110) 対で pos 90 = 適格 / pos 20 = 分母外
    check("⑤ domain", comparison_domain_ok(90, (88, 110)) is True
          and comparison_domain_ok(20, (88, 110)) is False)
    # ⑥ v2 較正機構: 域述語 min / τ_dec ゲート (strict >) / 共有域 FLAG
    s = [1] * 16 + [50] * 72 + [30] * 22 + [70] * 10   # n=120, lm0=110 相当
    check("⑥ region mins", (region_min(s, *R_SHARED), region_min(s, 88, 110),
                            region_min(s, 110, 120)) == (50, 30, 70))
    check("⑥ region empty -> None", region_min([9] * 98, 88, 88) is None)
    check("⑥ tau_dec gate", (1001 > TAU_DEC) is True and (1000 > TAU_DEC) is False)
    check("⑥ shared flag", shared_flag(2000, 1000) is True
          and shared_flag(1100, 1000) is False)
    # ⑦ 分岐表
    check("⑦ branches", (branch(True, False, None), branch(False, True, True),
                         branch(False, True, False), branch(False, False, None))
          == ("B1", "B2", "B3", "B4"))
    # ⑧ P-C2-3 条件構造: 劣化なし ⇒ 前半は判定対象外 (None)・行動半は独立
    def pc23(degraded_any, knee_when_degraded, behavior_all_pass):
        first = None if not degraded_any else bool(knee_when_degraded)
        return first, bool(behavior_all_pass)
    check("⑧ P-C2-3 conditional", pc23(False, None, True) == (None, True)
          and pc23(True, True, True) == (True, True))
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
    prov = L.provenance("C-ii", a.allow_dirty)
    L.log(f"R7-a ✅ commit={prov['commit'][:8]} dirty={prov['dirty']} stage=C-ii")

    gate2 = json.loads((RES / "gate_tokenizer.json").read_text())
    gate5 = json.loads((RES / "gate_cii.json").read_text())
    if gate2["verdict"] != "PASS" or gate5["verdict"] != "PASS":
        raise SystemExit("⛔ gate v2/v5 が PASS でない")
    arm_c = json.loads((RES / "arm_c.json").read_text())
    g2_md5 = hashlib.md5((RES / "gate_tokenizer.json").read_bytes()).hexdigest()
    g5_md5 = hashlib.md5((RES / "gate_cii.json").read_bytes()).hexdigest()
    self_md5 = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
    rows = gate2["final_rows"]
    unit = gate5["unit"]
    L.log(f"gate ✅ v2={g2_md5[:8]} v5={g5_md5[:8]} rungs={gate5['rungs']}")

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
        return unit * gate5["per_row"][rid][rk]["repeats"] + row["prompt"]

    def apply_full(prompt, sets_row):
        n_tok = len(tok(prompt)["input_ids"])
        jl, ml, _ = lens.apply(model, prompt, layers=pc2.BAND,
                               positions=list(range(n_tok)),
                               max_seq_len=pc2.MAX_SEQ)
        return jl, ml, n_tok

    out = {"arm": "C-ii (跨ぎ版・仮称)", "gate_v2_md5": g2_md5, "gate_v5_md5": g5_md5,
           "runner_md5": self_md5, "lens_md5": pc2.LENS_MD5,
           "band": [pc2.BAND[0], pc2.BAND[-1]], "w_fit": W_FIT, "p_scan": P_SCAN,
           "rungs": gate5["rungs"], "ref_rung": REF_RUNG, "eps_b": EPS_B,
           "k_hold": K_HOLD, "eta_band": ETA_BAND, "w_band_dex": W_BAND,
           "provenance": prov, "calibration": {}, "controls": {}, "rows": []}
    out_path = RES / "cii.json"

    # ---- Phase 較正 v2 (裁定 §2.3・per-row×rung×域 台帳化+決定錨ゲート) ------
    # r8 の 4 系列は v1 走行から再利用 (同 shape 決定論・INSTR verbatim・出所 md5 登記)
    v1 = json.loads((RES / "cii_v1_m7.json").read_text())
    v1_r8 = v1["calibration"]["r8"]["series"]
    out["controls"]["v1_reuse_md5"] = hashlib.md5(
        (RES / "cii_v1_m7.json").read_bytes()).hexdigest()
    out["controls"]["contamination_disclosure"] = (
        "既見 null 値 = v1 r8 4 値 {16415,1666,3656,3522}・9.85・τ=1666(不使用)・"
        "P1 全系列 3648。τ_dec = 10·τ_A は本日の値非参照の錨 (既見 4 値が ≥0.22 dex で"
        "通ることは開示の上で凍結・PREREG v2)")
    ledger = {}
    for row in rows:
        rid = row["id"]
        ledger[rid] = {}
        for rk in RUNGS:
            if rk == "r8" and rid in v1_r8:
                s, reused = v1_r8[rid], True
            else:
                jl, _, _ = apply_full(build(rid, rk), sets_of(rid))
                jl_neg = {k: -v for k, v in jl.items()}
                s, reused = series(jl_neg, sets_of(rid))["mid"], False
            lm0 = gate5["per_row"][rid][rk]["landmark_start"]
            ent = {"reused_v1": reused, "scan_min": restricted_min(s),
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
                                 f"{ent['scan_min']} ≤ {TAU_DEC}) ⇒ 停止・"
                                 "値の報告をしない")
        L.log(f"  台帳 {rid}: scan_min = "
              + " ".join(f"{rk}:{ledger[rid][rk]['scan_min']}" for rk in RUNGS)
              + " ✅ (> τ_dec)")
    # 共有域 sanity FLAG (停止機ではない・裁定 §2.3-(iii))
    flags = []
    for rid in ledger:
        m8 = ledger[rid]["r8"]["region_min"]["R_shared"]
        for rk in RUNGS[1:]:
            mk = ledger[rid][rk]["region_min"]["R_shared"]
            if shared_flag(mk, m8):
                flags.append({"row": rid, "rung": rk, "min": mk, "r8": m8,
                              "d_dex": round(abs(math.log10(mk) - math.log10(m8)), 4)})
    out["calibration"] = {"ledger": ledger, "tau_dec": TAU_DEC,
                          "shared_sanity_flags": flags,
                          "spread_descriptive": {rk: max(ledger[r][rk]["scan_min"]
                                                         for r in ledger)
                                                 / min(ledger[r][rk]["scan_min"]
                                                       for r in ledger)
                                                 for rk in RUNGS}}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    L.log(f"  共有域 sanity FLAG = {len(flags)} 件 (停止機ではない)・"
          f"spread(記述列) = { {k: round(v,2) for k, v in out['calibration']['spread_descriptive'].items()} }")
    # 決定性 (P1 r22 = 最大 shape・二重適用)
    p1_sets = sets_of("P1")
    jl1, _, _ = apply_full(build("P1", "r22"), p1_sets)
    med1 = series(jl1, p1_sets)
    jl2, _, _ = apply_full(build("P1", "r22"), p1_sets)
    det_ok = series(jl2, p1_sets) == med1
    out["controls"]["determinism_r22_pass"] = bool(det_ok)
    L.log(f"  決定性 (P1 r22) {'✅' if det_ok else '⛔'}")
    if not det_ok:
        out["verdict"] = "STOP_controls"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        raise SystemExit("⛔ 決定性 FAIL ⇒ 停止 (値の報告をしない)")

    # ---- Phase 測定 (30 forward) --------------------------------------------
    for row in rows:
        rid = row["id"]
        sets_row = sets_of(rid)
        rec = {"id": rid, "words": {k: w for k, w in
                                    ((g, next(r for r in gate2["reports"]
                                              if r["id"] == rid and r["pass"])
                                      ["detail"][g]["word"]) for g in sets_row)},
               "rungs": {}}
        for rk in RUNGS:
            lm0 = gate5["per_row"][rid][rk]["landmark_start"]
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
            f"@rel{rec['rungs'][rk]['onset_rel_landmark']}" for rk in RUNGS))
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        if time.time() - t_wall > 7200:
            out["verdict"] = "STOP_wall"
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
            raise SystemExit("⛔ wall 2h 超過 ⇒ 停止して報告")

    # ---- Phase 後処理 (zero-forward・機械) ----------------------------------
    for rec in out["rows"]:
        ref = rec["rungs"][REF_RUNG]
        bg, deltas = {}, {}
        for rk in RUNGS:
            if rk == REF_RUNG:
                bg[rk] = {"pass": True, "reason": "reference"}
                continue
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
        # 位相プロファイル+膝 (r22)
        rec["knee_r22"] = knee_series(rec["rungs"]["r22"]["lens"]["mid"],
                                      rec["rungs"]["r22"]["landmark_start"])
        # bitwise-88 (vs arm_c fbig・記述・停止機ではない)
        a_row = next(r for r in arm_c["rows"] if r["id"] == rec["id"])
        rec["r8_matches_armc_fbig"] = bool(
            rec["rungs"]["r8"]["lens"] == a_row["conditions"]["fbig"]["lens"])

    # 共有 prefix 床 (rung 対・[16, min lm0)・語群 mid)
    floors = {}
    for i, ra in enumerate(RUNGS):
        for rb in RUNGS[i + 1:]:
            vals = []
            for rec in out["rows"]:
                la = rec["rungs"][ra]["landmark_start"]
                lo, hi = P_SCAN, min(la, rec["rungs"][rb]["landmark_start"])
                vals.extend(dlog_series(rec["rungs"][ra]["lens"]["mid"][lo:hi],
                                        rec["rungs"][rb]["lens"]["mid"][lo:hi]))
            floors[f"{ra}x{rb}"] = dist_stats(vals)
    out["shared_prefix_floors"] = floors

    # 分岐表+予言 (機械適用・読みは判断層)
    verdicts = {}
    for rec in out["rows"]:
        per = {}
        for rk in ("r10", "r11", "r12", "r14", "r22"):
            v = rec["rungs"][rk]
            if not (rec["behavior_gate"][rk]["pass"] and v["emerged_landmark"]
                    and rec["rungs"][REF_RUNG]["emerged_landmark"]):
                per[rk] = None
                continue
            on = v["onset_abs_landmark"]
            in_fit = W_FIT[0] <= on < W_FIT[1]
            degraded = rec["delta_onsets_vs_ref"].get(rk, {}).get("d_rel", 0) != 0
            per[rk] = branch(in_fit, degraded, rec["knee_r22"]["knee"])
        verdicts[rec["id"]] = per
    out["branch_table"] = verdicts
    # P-C2-1: r10 の適格行で d_rel = 0
    r10 = [rec["delta_onsets_vs_ref"].get("r10", {}).get("d_rel")
           for rec in out["rows"]
           if rec["behavior_gate"]["r10"]["pass"]
           and rec["rungs"]["r10"]["emerged_landmark"]
           and rec["rungs"][REF_RUNG]["emerged_landmark"]]
    out["P_C2_1"] = {"d_rels": r10, "hold": bool(r10 and all(d == 0 for d in r10))}
    # P-C2-3: 前半 = 劣化観測時のみ・後半 = 無条件
    degraded_any = any(d.get("d_rel", 0) != 0 for rec in out["rows"]
                       for d in rec["delta_onsets_vs_ref"].values())
    knees = [rec["knee_r22"]["knee"] for rec in out["rows"]]
    behavior_all = all(g["pass"] for rec in out["rows"]
                       for g in rec["behavior_gate"].values())
    out["P_C2_3"] = {"degraded_any": bool(degraded_any),
                     "first_half": (None if not degraded_any
                                    else bool(any(knees))),
                     "behavior_all_pass": bool(behavior_all)}
    out["m_by_rung"] = {rk: sum(r["rungs"][rk]["emerged_landmark"]
                                for r in out["rows"]) for rk in RUNGS}
    out["r8_bitwise_count"] = sum(r["r8_matches_armc_fbig"] for r in out["rows"])
    out["verdict"] = "完走 (計器の地図+onset 登記・読みは判断層)"
    out["wall_total_s"] = round(time.time() - t_wall, 1)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    L.log(f"⭕ saved {out_path} m={out['m_by_rung']} "
          f"P-C2-1={out['P_C2_1']['hold']} P-C2-3={out['P_C2_3']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
