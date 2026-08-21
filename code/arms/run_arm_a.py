#!/usr/bin/env python3
"""GT-A (腕一) —— SilentHop 衛生支流・home-domain 再現 (陽性統制 R6)。

凍結: conversations/2026-08-19/DRAFT_PREREG_arms_graspthink_20260819.md (v2→v3 凍結)
      + RECORD_arms_q1_design_adjudication_20260819.md (k述語=定性宣言・k=配置定数)
裁可: 先生 2026-08-19「1済み、2発火🐥 でいいですか？」(発火) — 小 slot は A-5 作法で
      配置 (τ_A=100 記述用 / 32B 一撃 / P-A 予測は置かない申告 / ε_B,n は腕二へ繰延)。
素材: results/gate_tokenizer.json (gate v2 PASS・P1,P2,P3,B1,P5・P4→B1 差し替え登記)。

判定 (凍結・順序読み側):
  浮上(row) := prompt の position 系列上に、mid の lens band-median rank が
               全妨害語の同 rank より厳密に小さい 連続 K_HOLD position の窓が存在する。
  三枝: ⭕ m>=4 / ⚠ 1<=m<=3 / ⛔ m=0   (m = 浮上した row 数, 全 5 row)
  ★ 本述語は「η 帯 (0.643-0.902, 0819/corr・同一実物 lens・域外実測の下界引用) を随伴した
    頑健性の定性宣言」であり、計算済み誤り率つきの判定ではない (裁定 verbatim)。
  ⛔ 量的読み (浮上幅 dex, τ_A=100 横断) は記述のみ・判定不使用 (W 帯 1.57-1.81 dex 併記)。
  ⛔ 不再現時に モデル差 / lens 品質 / 単 token 語彙制約 を weld しない (DESIGN §2A)。

機構 ⛔ 無改変: lens.apply / band_median_ranks / mask は F5/corr の凍結経路 verbatim。
新規 = 浮上述語 (emergence) のみ ⇒ self-test fixture ①-⑤ を走行前に通す。
統制 (走行時・宣言済): V-N 符号反転 (row 1 の jl を負値化 → mid の rank が深底) /
決定性 (row 1 を二重適用 → 系列一致)。

usage:  python run_arm_a.py --self-test
        python run_arm_a.py --fire
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
import run_f5_lens as L  # noqa: E402

# ---- 配置定数 (A-5 作法: 値+錨) --------------------------------------------
K_HOLD = 3     # 頑健化 margin: 単発の順序 flip ((1-η) 帯 0.10-0.36) で判定が立たないため。
               # 独立仮定 (1-η)^k は使わない (ρ=0.12-0.42 実測・確率保証ではない)。
TAU_A = 100    # 記述用のみ (2 dex 級)。判定に不使用。
W_BAND = (1.57, 1.81)   # 32B 単群誤差帯 dex ((i) Step 2 転載・記述の但し書き用)
ETA_BAND = (0.643, 0.902)  # 0819/corr 実測 (worst は T=14)・域外実測の下界引用


def emergence(mid: list, distractors: list[list], k: int = K_HOLD):
    """浮上述語 (凍結): mid[t] < 全 distractor[t] が連続 k 個続く窓の存在。strict <。"""
    n = len(mid)
    run = 0
    for t in range(n):
        ok = all(mid[t] < d[t] for d in distractors)
        run = run + 1 if ok else 0
        if run >= k:
            return True, t - k + 1
    return False, None


def self_test() -> int:
    fails = []

    def check(name, cond):
        print(f"  {'✅' if cond else '⛔'} {name}")
        if not cond:
            fails.append(name)

    e = emergence
    # ① 明確な浮上 (k=3 の窓・窓開始 index も検分)
    r1 = e([9, 5, 5, 5, 9], [[8, 8, 8, 8, 8], [7, 7, 7, 7, 7]])
    check("① emergence k=3 window", r1 == (True, 1))
    # ② k=2 のみ ⇒ 立たない
    check("② k=2 only -> False",
          e([9, 5, 5, 9, 9], [[8, 8, 8, 8, 8], [7, 7, 7, 7, 7]])[0] is False)
    # ③ tie は窓を切る (strict <)
    check("③ tie breaks window",
          e([5, 7, 5, 9], [[7, 7, 7, 7], [8, 8, 8, 8]])[0] is False)
    # ④ 系列が k 未満 ⇒ False
    check("④ short series -> False", e([1, 1], [[2, 2]])[0] is False)
    # ⑤ 窓が末尾で閉じる場合
    check("⑤ window at tail", e([9, 9, 5, 5, 5], [[8] * 5, [7] * 5]) == (True, 2))
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
    prov = L.provenance("GT-A", a.allow_dirty)
    L.log(f"R7-a ✅ commit={prov['commit'][:8]} dirty={prov['dirty']} stage=arm-A")

    gate = json.loads((RES / "gate_tokenizer.json").read_text())
    if gate["verdict"] != "PASS":
        raise SystemExit("⛔ tokenizer gate が PASS でない")
    gate_md5 = hashlib.md5((RES / "gate_tokenizer.json").read_bytes()).hexdigest()
    self_md5 = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
    rows = gate["final_rows"]
    L.log(f"gate ✅ {gate['final_set']} md5={gate_md5[:8]}")

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

    out = {"arm": "GT-A (仮称・腕一)", "gate_md5": gate_md5, "runner_md5": self_md5,
           "lens_md5": pc2.LENS_MD5, "band": [pc2.BAND[0], pc2.BAND[-1]],
           "k_hold": K_HOLD, "tau_a_descriptive": TAU_A,
           "eta_band": ETA_BAND, "w_band_dex": W_BAND,
           "provenance": prov, "controls": {}, "rows": []}
    out_path = RES / "arm_a.json"

    def series(jl_like, sets_row, model_side=False):
        med = p2.band_median_ranks(torch, jl_like, mask, sets_row)
        return {k: [int(v) for v in med[k]] for k in sets_row}

    for irow, row in enumerate(rows):
        det = gate["reports"]
        rep = next(r for r in det if r["id"] == row["id"] and r["pass"])
        sets_row = {k: [i for i, _ in rep["detail"][k]["distinctive"]]
                    for k in rep["detail"]}
        n_tok = len(tok(row["prompt"])["input_ids"])
        positions = list(range(n_tok))
        t1 = time.time()
        jl, ml, _ = lens.apply(model, row["prompt"], layers=pc2.BAND,
                               positions=positions, max_seq_len=pc2.MAX_SEQ)
        med_l = series(jl, sets_row)
        med_m = series({0: ml}, sets_row)

        if irow == 0:
            # 統制① V-N 符号反転: mid の rank が深底へ
            jl_neg = {k: -v for k, v in jl.items()}
            med_neg = series(jl_neg, sets_row)
            vn_ok = min(med_neg["mid"]) > 10_000
            # 統制② 決定性: 二重適用で系列一致
            jl2, ml2, _ = lens.apply(model, row["prompt"], layers=pc2.BAND,
                                     positions=positions, max_seq_len=pc2.MAX_SEQ)
            det_ok = series(jl2, sets_row) == med_l
            out["controls"] = {"vn_signflip_min_rank": int(min(med_neg["mid"])),
                               "vn_pass": bool(vn_ok), "determinism_pass": bool(det_ok)}
            L.log(f"  統制 V-N={'✅' if vn_ok else '⛔'} (min {min(med_neg['mid'])}) "
                  f"決定性={'✅' if det_ok else '⛔'}")
            if not (vn_ok and det_ok):
                out["verdict"] = "STOP_controls"
                out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
                raise SystemExit("⛔ 走行時統制 FAIL ⇒ 停止 (値の報告をしない)")

        dis_keys = [k for k in sets_row if k.startswith("d")]
        emerged, w0 = emergence(med_l["mid"], [med_l[k] for k in dis_keys])
        out["rows"].append({
            "id": row["id"], "prompt": row["prompt"], "n_tok": n_tok,
            "words": {k: rep["detail"][k]["word"] for k in rep["detail"]},
            "sets": sets_row,
            "lens": med_l, "model_final": med_m,
            "emerged": bool(emerged), "window_start": w0,
            "descriptive": {
                "mid_min_rank_lens": int(min(med_l["mid"])),
                "mid_argmin_pos": int(med_l["mid"].index(min(med_l["mid"]))),
                "mid_min_rank_model": int(min(med_m["mid"])),
                "tau_a_crossed": bool(min(med_l["mid"]) <= TAU_A)},
            "wall_s": round(time.time() - t1, 1)})
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        L.log(f"  {row['id']} 浮上={'⭕' if emerged else '—'} "
              f"mid_min={min(med_l['mid'])} @pos{med_l['mid'].index(min(med_l['mid']))} "
              f"({time.time()-t1:.0f}s)")
        if time.time() - t_wall > 3600:
            out["verdict"] = "STOP_wall"
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
            raise SystemExit("⛔ wall 1h 超過 ⇒ 停止して報告")

    m = sum(r["emerged"] for r in out["rows"])
    out["m"] = m
    out["verdict"] = "⭕ 再現" if m >= 4 else ("⚠ 部分" if m >= 1 else "⛔ 不再現")
    out["wall_total_s"] = round(time.time() - t_wall, 1)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    L.log(f"⭕ saved {out_path}  m={m}/5 ⇒ {out['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
