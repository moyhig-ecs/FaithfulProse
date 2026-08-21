#!/usr/bin/env python3
"""GT-C (腕三・域内版) —— filler translation・content-locked か position-locked かの素材。

凍結: conversations/2026-08-19/DRAFT_PREREG_arms_graspthink_20260819.md §4 (v5 で具体化)
裁可: 先生 2026-08-20「arm-B/Cを進めます」(発火)。順序 A→B→C 不変 —— 本 runner の
      発火は GT-B 完走 (M-7 / 統制 FAIL なし) の後。
素材: results/gate_tokenizer.json (v2 PASS) + results/gate_filler_c.json (v4 PASS・
      unit 第一宣言・repeats {f0:0, fmid:3, fbig:8}・landmark_start {0, 33, 88})。

★ 記録は onset まで L1 (claim hygiene 4 条 verbatim 継承・DESIGN §3):
  onset まで L1 / 閾値横断 ≠ 計算時刻 (first-passage 罠) / 計算順序は patching /
  機序 (lens 外挿破綻 vs 表現側変化) は本系列単独で分離しない。
  ⇒ ⛔ 実行層は content-locked / position 由来 の読みをしない (読み = 判断層+先生)。

浮上述語 = run_arm_a.emergence の verbatim import (単一出所・k=3)。
  記録 (per 行 × 条件): 全系列窓 (絶対座標) と landmark 域限定窓 (絶対+相対座標) の両方。
  記述比較子 (凍結・記述のみ): 全条件浮上行の Δrel / Δabs (fmid−f0, fbig−f0)。

★ v6 統制 (β-2 較正版・G0 判 = β の帰結・式は 3,648 非参照で先凍結):
  v5 の V-N (GT-A の 10-position 宇宙で凍結した閾値 10⁴ の verbatim 継承) は 98-position
  系列で FAIL し凍結どおり停止 (FLAG_arm_c_vn_control_stop・測定値ゼロ)。裁定 = 判断層
  DRAFT_ADJUDICATION_redline_armBC §3.3 β-2 + 先生「3 提案通り実施」(2026-08-20)。
  較正 (R6・測定前・同一 process): 較正集合 = 凍結述語「v5 行集合 ∖ {P1}」= P2,P3,B1,P5
    (台帳順・⛔ P1 除外 = 汚染: v5 走行で flipped 値既見)。各 fbig 系列の符号反転
    mid rank 全系列を pooled 経験分布に。
  凍結式 (β-2・順序統計): τ_98 = F̂^{-1}(1-(1-Q_FLOOR)^(1/N_MIN)) (type-1 経験逆関数・
    分位 Q_FLOOR = 0.05 宣言込み)。sanity: 較正 4 行 min が因子 COHERENCE_MAX 以内 ⇒
    不成立なら M-7 停止・測定しない。
  走行時: (i) P1 fbig 符号反転 min > τ_98 —— ⚠ 劣化統制と登記 (P1 の flipped 値は
    v5 で既見・未知なのは閾値側のみ・汚染 flag 明示・FAIL なら候補3/4 へ = それも情報) /
    (ii) 決定性 (P1 fbig 二重適用)。
  ③ (記述・停止条件ではない): f0 系列 vs arm_a.json 登記系列の一致率 —— cross-run
  再現の登記 (MPS の run 間決定性は未凍結ゆえ停止機にしない・不一致は登記)。

usage:  python run_arm_c.py --self-test
        python run_arm_c.py --fire
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
sys.path.insert(0, str(REPO / "probes" / "olmo3-f5-think-prefix"))
sys.path.insert(0, str(HERE))
import run_f5_lens as L  # noqa: E402
from run_arm_a import emergence, K_HOLD, W_BAND, ETA_BAND  # noqa: E402  (単一出所)

CONDS = ("f0", "fmid", "fbig")
GROUPS = ("mid", "ans", "d1", "d2")

# ---- v6 較正統制の配置定数 (β-2・A-5 作法: 値+錨・3,648 非参照で先凍結) ------
CAL_ROWS = ["P2", "P3", "B1", "P5"]  # 凍結述語「v5 行集合 ∖ {P1}」台帳順。P1 = 汚染 (flipped 値既見)
COHERENCE_MAX = 4   # 較正 4 行 min が桁で割れないことの要求 (因子 4 < 1 桁・M-7 sanity)
Q_FLOOR = 0.05      # min-over-n 予測の下側分位 (宣言・慣用値。確率保証ではない —
                    # ρ = 0.12-0.42 実測 = 正相関ゆえ iid 近似は床を保守側(低め)に置く)
N_MIN = 98          # min-over-n の n = P1 fbig 系列長 (gate_filler_c.json 登記・token 構造のみ)


def tau_from_calibration(pooled: list, n: int = N_MIN, q: float = Q_FLOOR) -> int:
    """β-2 凍結式 (順序統計・iid 近似): min-over-n の下側分位 q に対応する
    per-position 分位 p* = 1-(1-q)^(1/n) を pooled 経験分布の type-1 逆関数で読む。
    ★ 走行前に導出可能な構造事実: len(pooled) ~4×100 では ceil(p*·N) = 1 に帰着し
    τ = pooled min となる (値不要・後知恵ではない)。"""
    import math
    p_star = 1 - (1 - q) ** (1 / n)
    s = sorted(pooled)
    k = max(1, math.ceil(p_star * len(s)))
    return int(s[k - 1])


def coherent(mins: list) -> bool:
    """較正 sanity: 4 行の per-row min が因子 COHERENCE_MAX 以内に束なる。"""
    return max(mins) / min(mins) <= COHERENCE_MAX


def onset_records(mid: list, distractors: list[list], lm0: int) -> dict:
    """全系列窓と landmark 域限定窓の両座標を機械記録。"""
    em_full, w_full = emergence(mid, distractors, K_HOLD)
    em_lm, w_lm = emergence(mid[lm0:], [d[lm0:] for d in distractors], K_HOLD)
    return {"emerged_full": bool(em_full),
            "onset_abs_full": None if w_full is None else int(w_full),
            "emerged_landmark": bool(em_lm),
            "onset_abs_landmark": None if w_lm is None else int(w_lm + lm0),
            "onset_rel_landmark": None if w_lm is None else int(w_lm)}


def self_test() -> int:
    fails = []

    def check(name, cond):
        print(f"  {'✅' if cond else '⛔'} {name}")
        if not cond:
            fails.append(name)

    # ① 浮上述語は run_arm_a の fixture ①⑤ を verbatim 再検分 (import 同一性)
    check("① emergence k=3 window",
          emergence([9, 5, 5, 5, 9], [[8] * 5, [7] * 5]) == (True, 1))
    check("① window at tail",
          emergence([9, 9, 5, 5, 5], [[8] * 5, [7] * 5]) == (True, 2))
    # ② 座標変換: lm0=2 で窓が landmark 域内 ⇒ abs = rel + lm0
    r = onset_records([9, 9, 5, 5, 5], [[8] * 5, [7] * 5], lm0=2)
    check("② abs = rel + lm0",
          (r["onset_abs_landmark"], r["onset_rel_landmark"]) == (2, 0))
    # ③ filler 域のみの窓: full は立つが landmark 域限定は立たない
    r = onset_records([5, 5, 5, 9, 9], [[8] * 5, [7] * 5], lm0=3)
    check("③ filler-only window split",
          r["emerged_full"] is True and r["emerged_landmark"] is False)
    # ④ 非浮上: 両方 None
    r = onset_records([9] * 5, [[8] * 5, [7] * 5], lm0=0)
    check("④ no window -> None",
          r["emerged_full"] is False and r["onset_rel_landmark"] is None)
    # ⑤ lm0=0 (f0) では full と landmark が一致
    r = onset_records([5, 5, 5, 9, 9], [[8] * 5, [7] * 5], lm0=0)
    check("⑤ lm0=0 identity",
          r["onset_abs_full"] == r["onset_abs_landmark"] == 0)
    # ⑥ v6 β-2 式: n=98・N~400 で pooled min に帰着 / n=1 では分位機構が生きる
    check("⑥ tau reduces to pooled min",
          tau_from_calibration([10, 20, 30, 40] * 100, n=98, q=0.05) == 10)
    # (n=1, q=0.045 ⇒ p*·N = 4.5 —— 整数境界を避けた検分。実走行域は p*·N ~0.2 で境界なし)
    check("⑥ tau quantile mechanics (n=1)",
          tau_from_calibration(list(range(1, 101)), n=1, q=0.045) == 5)
    # ⑦ 整合 sanity: 因子 > 4 ⇒ False / 以内 ⇒ True
    check("⑦ coherence", coherent([8000, 6000, 7000, 9000]) is True
          and coherent([1000, 8000, 7000, 9000]) is False)
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
    prov = L.provenance("GT-C", a.allow_dirty)
    L.log(f"R7-a ✅ commit={prov['commit'][:8]} dirty={prov['dirty']} stage=arm-C")

    gate2 = json.loads((RES / "gate_tokenizer.json").read_text())
    gate4 = json.loads((RES / "gate_filler_c.json").read_text())
    if gate2["verdict"] != "PASS" or gate4["verdict"] != "PASS":
        raise SystemExit("⛔ gate v2/v4 が PASS でない")
    arm_b = json.loads((RES / "arm_b.json").read_text())
    if arm_b.get("verdict") != "完走 (記録札・判定枝なし)":
        raise SystemExit("⛔ GT-B が完走していない (順序 A→B→C)")
    g2_md5 = hashlib.md5((RES / "gate_tokenizer.json").read_bytes()).hexdigest()
    g4_md5 = hashlib.md5((RES / "gate_filler_c.json").read_bytes()).hexdigest()
    self_md5 = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
    rows = gate2["final_rows"]
    unit = gate4["final_unit"]
    repeats = gate4["repeats"]
    arm_a = json.loads((RES / "arm_a.json").read_text())
    L.log(f"gate ✅ v2={g2_md5[:8]} v4={g4_md5[:8]} repeats={repeats}")

    pc2 = L.load_pc2()
    import torch, transformers, jlens  # noqa: E401
    got = hashlib.md5(pc2.LENS_PT.read_bytes()).hexdigest()
    if got != pc2.LENS_MD5:
        raise SystemExit(f"⛔ lens md5 不一致: {got}")
    L.log(f"lens md5 ✅ {pc2.LENS_MD5}")

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

    out = {"arm": "GT-C (腕三・域内版 filler translation)", "gate_v2_md5": g2_md5,
           "gate_v4_md5": g4_md5, "runner_md5": self_md5,
           "lens_md5": pc2.LENS_MD5, "band": [pc2.BAND[0], pc2.BAND[-1]],
           "k_hold": K_HOLD, "filler_unit": unit, "repeats": repeats,
           "eps_fill_from_arm_b": {g: arm_b["pooled_by_group"][g] for g in GROUPS},
           "eta_band": ETA_BAND, "w_band_dex": W_BAND,
           "provenance": prov, "controls": {}, "rows": []}
    out_path = RES / "arm_c.json"

    def series(jl_like, sets_row):
        med = p2.band_median_ranks(torch, jl_like, mask, sets_row)
        return {k: [int(v) for v in med[k]] for k in sets_row}

    def sets_of(rid):
        rep = next(r for r in gate2["reports"] if r["id"] == rid and r["pass"])
        return {k: [i for i, _ in rep["detail"][k]["distinctive"]]
                for k in rep["detail"]}

    def flipped_series(prompt, sets_row):
        n_tok = len(tok(prompt)["input_ids"])
        jl, _, _ = lens.apply(model, prompt, layers=pc2.BAND,
                              positions=list(range(n_tok)),
                              max_seq_len=pc2.MAX_SEQ)
        jl_neg = {k: -v for k, v in jl.items()}
        return series(jl_neg, sets_row)["mid"], jl

    # ---- v6 較正統制 (β-2・R6・測定前・凍結式・null 読みのみ) ----------------
    cal_series, cal_mins, pooled = {}, {}, []
    for rid in CAL_ROWS:
        row = next(r for r in rows if r["id"] == rid)
        s, _ = flipped_series(unit * repeats["fbig"] + row["prompt"], sets_of(rid))
        cal_series[rid] = s
        cal_mins[rid] = int(min(s))
        pooled.extend(s)
        L.log(f"  較正 {rid} fbig flipped: n={len(s)} min={min(s)}")
    ok = coherent(list(cal_mins.values()))
    tau98 = tau_from_calibration(pooled)
    out["calibration"] = {"rows": CAL_ROWS, "per_row_min": cal_mins,
                          "pooled_n": len(pooled), "q_floor": Q_FLOOR,
                          "n_min": N_MIN, "coherence_max": COHERENCE_MAX,
                          "coherent": bool(ok), "tau98": tau98,
                          "series": cal_series}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    if not ok:
        out["verdict"] = "M7_STOP_calibration"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        raise SystemExit("⛔ 較正整合 FAIL ⇒ M-7 停止・測定しない")
    L.log(f"  較正 ✅ τ_98 = {tau98} (β-2 式・pooled n={len(pooled)}・coherent)")
    # P1 fbig: 劣化統制 (flipped 値は v5 で既見・未知は閾値側のみ・汚染 flag) + 決定性
    p1 = next(r for r in rows if r["id"] == "P1")
    p1_prompt = unit * repeats["fbig"] + p1["prompt"]
    p1_sets = sets_of("P1")
    s_p1, jl_p1 = flipped_series(p1_prompt, p1_sets)
    v_p1 = int(min(s_p1))
    med_p1 = series(jl_p1, p1_sets)
    jl2, _, _ = lens.apply(model, p1_prompt, layers=pc2.BAND,
                           positions=list(range(len(tok(p1_prompt)["input_ids"]))),
                           max_seq_len=pc2.MAX_SEQ)
    det_ok = series(jl2, p1_sets) == med_p1
    vn_ok = v_p1 > tau98
    out["controls"] = {"vn_p1_fbig_flipped_min": v_p1, "tau98": tau98,
                       "vn_pass": bool(vn_ok), "determinism_pass": bool(det_ok),
                       "contamination_flag": "P1 flipped 値は v5 走行で既見・"
                       "本統制は閾値側の未知性のみに依存する劣化統制と登記"}
    L.log(f"  統制 V-N(P1・劣化/汚染登記)={'✅' if vn_ok else '⛔'} "
          f"({v_p1} vs τ {tau98}) 決定性={'✅' if det_ok else '⛔'}")
    if not (vn_ok and det_ok):
        out["verdict"] = "STOP_controls"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        raise SystemExit("⛔ 走行時統制 FAIL ⇒ 停止 (値の報告をしない)")

    for irow, row in enumerate(rows):
        rep = next(r for r in gate2["reports"] if r["id"] == row["id"] and r["pass"])
        sets_row = {k: [i for i, _ in rep["detail"][k]["distinctive"]]
                    for k in rep["detail"]}
        a_row = next(r for r in arm_a["rows"] if r["id"] == row["id"])
        conds = {}
        for cond in CONDS:
            n = repeats[cond]
            prompt = unit * n + row["prompt"]
            lm0 = gate4["per_row"][row["id"]][cond]["landmark_start"]
            n_tok = len(tok(prompt)["input_ids"])
            positions = list(range(n_tok))
            t1 = time.time()
            jl, ml, _ = lens.apply(model, prompt, layers=pc2.BAND,
                                   positions=positions, max_seq_len=pc2.MAX_SEQ)
            med_l = series(jl, sets_row)
            dis_keys = [k for k in sets_row if k.startswith("d")]
            rec = onset_records(med_l["mid"], [med_l[k] for k in dis_keys], lm0)
            conds[cond] = {"n_tok": n_tok, "landmark_start": lm0, "lens": med_l,
                           **rec, "wall_s": round(time.time() - t1, 1)}
        # 統制③ (記述): f0 vs arm_a 登記系列の一致 (停止機ではない)
        f0_match = conds["f0"]["lens"] == a_row["lens"]
        # 記述比較子 (凍結・記述のみ): landmark 域限定 onset の差
        deltas = {}
        if all(conds[c]["emerged_landmark"] for c in CONDS):
            r0 = conds["f0"]["onset_rel_landmark"]
            a0 = conds["f0"]["onset_abs_landmark"]
            deltas = {c: {"d_rel": conds[c]["onset_rel_landmark"] - r0,
                          "d_abs": conds[c]["onset_abs_landmark"] - a0}
                      for c in ("fmid", "fbig")}
        out["rows"].append({
            "id": row["id"],
            "words": {k: rep["detail"][k]["word"] for k in rep["detail"]},
            "sets": sets_row, "conditions": conds,
            "f0_matches_arm_a": bool(f0_match),
            "delta_onsets": deltas})
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        L.log(f"  {row['id']} " + " ".join(
            f"{c}:{'⭕' if conds[c]['emerged_landmark'] else '—'}"
            f"@rel{conds[c]['onset_rel_landmark']}" for c in CONDS)
            + f" f0=arm_a一致:{'✅' if f0_match else '⚠'}")
        if time.time() - t_wall > 7200:
            out["verdict"] = "STOP_wall"
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
            raise SystemExit("⛔ wall 2h 超過 ⇒ 停止して報告")

    out["m_by_cond"] = {c: sum(r["conditions"][c]["emerged_landmark"]
                               for r in out["rows"]) for c in CONDS}
    out["f0_match_count"] = sum(r["f0_matches_arm_a"] for r in out["rows"])
    out["verdict"] = "完走 (onset 登記・読みは判断層)"
    out["wall_total_s"] = round(time.time() - t_wall, 1)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    L.log(f"⭕ saved {out_path} m={out['m_by_cond']} "
          f"f0一致={out['f0_match_count']}/5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
