#!/usr/bin/env python3
"""GT-B (腕二) —— filler rewrite・一槽 ADJ 差し替え = 操作の較正 (絶縁は検証するもの)。

凍結: conversations/2026-08-19/DRAFT_PREREG_arms_graspthink_20260819.md §3 (v4→v5 凍結)
裁可: 先生 2026-08-20「arm-B/Cを進めます」(発火)。小 slot は A-5 作法で v5 に配置
      (ε_B=0.02 / 読み位置 = landmark 域 / 参照 = 第一宣言 ADJ)。
素材: results/gate_tokenizer.json (v2 PASS) + results/gate_lengths_b.json (v3 PASS・
      5 ADJ 全生存・landmark_start=10 全行同一)。

★ 記録札・判定枝なし (v5 凍結): 本腕の出力 ε_fill が次腕 (GT-C) の計器仕様。
  ε_fill := 参照変種との per-position |Δlog10(band-median rank)| (landmark 域・
  pinned 4 語群 each) の分布 — median / q3(inclusive) / max を per-行・per-語群で報告。
  行動不変ゲート (末尾 position・model 最終層): (a) top-1 全変種一致 かつ
  (b) |Δp(ans 弁別集合)| ≤ ε_B = 0.02。ゲート落ち変種は ε_fill 集計から除外し登記
  (⚠ ゲート落ち自体が selectivity 所見の記述)。
  ⛔ ε_fill 実測前の比較主張をしない。⛔ 量的読みの判定不使用 (W 帯 1.57-1.81 dex 併記)。

機構 ⛔ 無改変: lens.apply / band_median_ranks / mask は F5/corr の凍結経路 verbatim。
新規 = ε_fill 集計・行動不変ゲート ⇒ self-test fixture ①-⑤ を走行前に通す。
統制 (走行時・宣言済): V-N 符号反転 (row 1 参照変種) / 決定性 (row 1 参照変種二重適用)。

usage:  python run_arm_b.py --self-test
        python run_arm_b.py --fire
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
import run_f5_lens as L  # noqa: E402

# ---- 配置定数 (A-5 作法: 値+錨・v5 登記) -----------------------------------
EPS_B = 0.02   # 行動不変ゲート: 答の p ~0.3-0.9 級に対する小摂動帯 (v4 配置候補のまま確定)
W_BAND = (1.57, 1.81)      # 32B 単群誤差帯 dex (記述の但し書き用・判定不使用)
ETA_BAND = (0.643, 0.902)  # 0819/corr 実測 (域外実測の下界引用・限定句つき)
GROUPS = ("mid", "ans", "d1", "d2")


def dlog_series(ref: list, var: list) -> list:
    """per-position |Δlog10(rank)|。rank ≥ 1 (band-median rank の定義域)。"""
    return [abs(math.log10(v) - math.log10(r)) for r, v in zip(ref, var)]


def dist_stats(vals: list) -> dict:
    """median / q3 (inclusive) / max。fixture ① で pin。"""
    if not vals:
        return {"n": 0, "median": None, "q3": None, "max": None}
    q3 = statistics.quantiles(vals, n=4, method="inclusive")[2] if len(vals) > 1 \
        else vals[0]
    return {"n": len(vals), "median": round(statistics.median(vals), 4),
            "q3": round(q3, 4), "max": round(max(vals), 4)}


def behavior_gate(top1_ref: int, p_ref: float, top1_v: int, p_v: float,
                  eps: float = EPS_B) -> tuple[bool, str]:
    if top1_v != top1_ref:
        return False, f"top1 mismatch ({top1_v} != {top1_ref})"
    if abs(p_v - p_ref) > eps:
        return False, f"|dp|={abs(p_v - p_ref):.4f} > {eps}"
    return True, ""


def self_test() -> int:
    fails = []

    def check(name, cond):
        print(f"  {'✅' if cond else '⛔'} {name}")
        if not cond:
            fails.append(name)

    # ① dlog + dist_stats (q3 inclusive を pin)
    d = dlog_series([10, 100], [100, 100])
    check("① dlog [1.0, 0.0]", [round(x, 6) for x in d] == [1.0, 0.0])
    s = dist_stats([0.0, 1.0])
    check("① stats median=0.5 q3=0.75 max=1.0",
          (s["median"], s["q3"], s["max"]) == (0.5, 0.75, 1.0))
    # ② 行動ゲート: 一致 + dp 小 ⇒ pass
    check("② gate pass", behavior_gate(7, 0.50, 7, 0.51)[0] is True)
    # ③ top1 不一致 ⇒ fail / dp 超過 ⇒ fail
    check("③ gate top1 fail", behavior_gate(7, 0.50, 8, 0.50)[0] is False)
    check("③ gate dp fail", behavior_gate(7, 0.50, 7, 0.53)[0] is False)
    # ④ 境界: dp = eps ちょうどは pass (≤)。二進で正確な値で検分 (0.53125-0.5 = 0.03125)
    check("④ dp == eps pass", behavior_gate(7, 0.5, 7, 0.53125, eps=0.03125)[0] is True)
    # ⑤ pooling: 二変種 × P=3 ⇒ n=6
    pool = dlog_series([1, 10, 100], [10, 10, 100]) + \
        dlog_series([1, 10, 100], [1, 100, 100])
    check("⑤ pool n=6", dist_stats(pool)["n"] == 6)
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
    prov = L.provenance("GT-B", a.allow_dirty)
    L.log(f"R7-a ✅ commit={prov['commit'][:8]} dirty={prov['dirty']} stage=arm-B")

    gate2 = json.loads((RES / "gate_tokenizer.json").read_text())
    gate3 = json.loads((RES / "gate_lengths_b.json").read_text())
    if gate2["verdict"] != "PASS" or gate3["verdict"] != "PASS":
        raise SystemExit("⛔ gate v2/v3 が PASS でない")
    g2_md5 = hashlib.md5((RES / "gate_tokenizer.json").read_bytes()).hexdigest()
    g3_md5 = hashlib.md5((RES / "gate_lengths_b.json").read_bytes()).hexdigest()
    self_md5 = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
    rows = gate2["final_rows"]
    adjs = gate3["final_adjs"]
    ref_adj = gate3["reference_adj"]
    preface = gate3["preface"]
    L.log(f"gate ✅ v2={g2_md5[:8]} v3={g3_md5[:8]} adjs={adjs} ref={ref_adj}")

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

    out = {"arm": "GT-B (腕二・filler rewrite)", "gate_v2_md5": g2_md5,
           "gate_v3_md5": g3_md5, "runner_md5": self_md5,
           "lens_md5": pc2.LENS_MD5, "band": [pc2.BAND[0], pc2.BAND[-1]],
           "eps_b": EPS_B, "reference_adj": ref_adj, "adjs": adjs,
           "eta_band": ETA_BAND, "w_band_dex": W_BAND,
           "provenance": prov, "controls": {}, "rows": [], "pooled_by_group": {}}
    out_path = RES / "arm_b.json"

    def series(jl_like, sets_row):
        med = p2.band_median_ranks(torch, jl_like, mask, sets_row)
        return {k: [int(v) for v in med[k]] for k in sets_row}

    def final_pos_behavior(ml, ans_ids):
        logits = ml[-1].float()
        p = torch.softmax(logits, dim=-1)
        top1 = int(torch.argmax(logits))
        p_ans = float(sum(p[i] for i in ans_ids))
        return top1, p_ans

    pooled = {g: [] for g in GROUPS}
    for irow, row in enumerate(rows):
        rep = next(r for r in gate2["reports"] if r["id"] == row["id"] and r["pass"])
        sets_row = {k: [i for i, _ in rep["detail"][k]["distinctive"]]
                    for k in rep["detail"]}
        lm0 = gate3["per_row"][row["id"]]["landmark_start"]
        variants = {}
        for adj in adjs:
            prompt = preface.format(adj=adj) + row["prompt"]
            n_tok = len(tok(prompt)["input_ids"])
            positions = list(range(n_tok))
            t1 = time.time()
            jl, ml, _ = lens.apply(model, prompt, layers=pc2.BAND,
                                   positions=positions, max_seq_len=pc2.MAX_SEQ)
            med_l = series(jl, sets_row)
            top1, p_ans = final_pos_behavior(ml, sets_row["ans"])
            variants[adj] = {"n_tok": n_tok, "lens": med_l,
                             "top1": top1, "top1_str": tok.decode([top1]),
                             "p_ans": round(p_ans, 6),
                             "wall_s": round(time.time() - t1, 1)}
            if irow == 0 and adj == ref_adj:
                jl_neg = {k: -v for k, v in jl.items()}
                med_neg = series(jl_neg, sets_row)
                vn_ok = min(med_neg["mid"]) > 10_000
                jl2, _, _ = lens.apply(model, prompt, layers=pc2.BAND,
                                       positions=positions, max_seq_len=pc2.MAX_SEQ)
                det_ok = series(jl2, sets_row) == med_l
                out["controls"] = {"vn_signflip_min_rank": int(min(med_neg["mid"])),
                                   "vn_pass": bool(vn_ok),
                                   "determinism_pass": bool(det_ok)}
                L.log(f"  統制 V-N={'✅' if vn_ok else '⛔'} "
                      f"(min {min(med_neg['mid'])}) 決定性={'✅' if det_ok else '⛔'}")
                if not (vn_ok and det_ok):
                    out["verdict"] = "STOP_controls"
                    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
                    raise SystemExit("⛔ 走行時統制 FAIL ⇒ 停止 (値の報告をしない)")

        ref = variants[ref_adj]
        gates, excl = {}, []
        for adj in adjs:
            if adj == ref_adj:
                gates[adj] = {"pass": True, "reason": "reference"}
                continue
            ok, why = behavior_gate(ref["top1"], ref["p_ans"],
                                    variants[adj]["top1"], variants[adj]["p_ans"])
            gates[adj] = {"pass": bool(ok), "reason": why,
                          "dp_ans": round(abs(variants[adj]["p_ans"]
                                              - ref["p_ans"]), 6)}
            if not ok:
                excl.append(adj)
        eps_fill = {}
        per_variant = {}
        for g in GROUPS:
            vals = []
            for adj in adjs:
                if adj == ref_adj or not gates[adj]["pass"]:
                    continue
                d = dlog_series(ref["lens"][g][lm0:], variants[adj]["lens"][g][lm0:])
                per_variant.setdefault(adj, {})[g] = dist_stats(d)
                vals.extend(d)
            eps_fill[g] = dist_stats(vals)
            pooled[g].extend(vals)
        out["rows"].append({
            "id": row["id"], "landmark_start": lm0,
            "words": {k: rep["detail"][k]["word"] for k in rep["detail"]},
            "sets": sets_row, "variants": variants, "behavior_gate": gates,
            "excluded": excl, "eps_fill_by_group": eps_fill,
            "eps_fill_per_variant": per_variant})
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        L.log(f"  {row['id']} gate落ち={excl or 'なし'} "
              f"ε_fill mid median={eps_fill['mid']['median']} "
              f"max={eps_fill['mid']['max']}")
        if time.time() - t_wall > 7200:
            out["verdict"] = "STOP_wall"
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
            raise SystemExit("⛔ wall 2h 超過 ⇒ 停止して報告")

    out["pooled_by_group"] = {g: dist_stats(v) for g, v in pooled.items()}
    out["n_gate_excluded_total"] = sum(len(r["excluded"]) for r in out["rows"])
    out["verdict"] = "完走 (記録札・判定枝なし)"
    out["wall_total_s"] = round(time.time() - t_wall, 1)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    L.log(f"⭕ saved {out_path} gate除外計={out['n_gate_excluded_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
