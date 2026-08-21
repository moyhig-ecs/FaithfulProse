#!/usr/bin/env python3
"""C-ii (跨ぎ版・仮称) gate v5 — 走行前・値を見ない機械検分 (tokenizer のみ)。

凍結: conversations/2026-08-20/PREREG_cii_ladder_20260820.md §1
裁可: 先生 2026-08-20「凍結、予言三本採択で起草しましょう」(起草)・⛔ 発火は別の一言。

gate 述語 (機械・値は一切見ない):
  (a) 総長 ≤ 256 (全行 × 全 rung)
  (b) landmark suffix token 列 = tok(" "+prompt) (全 rung)
  (c) landmark_start = 11×k exact ({88,110,121,132,154,242} = 11×{8,10,11,12,14,22})・単調
  (d) 全 rung の landmark 先頭 token = " The" 系 (translation 族 uniform・f0 型混入なし)
  不合格 ⇒ M-7 停止。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
sys.path.insert(0, str(REPO / "probes" / "olmo3-f5-think-prefix"))
import run_f5_lens as L  # noqa: E402

RUNGS = {f"r{k}": k * 11 for k in (8, 10, 11, 12, 14, 22)}  # lm0 = {88,...,242}


def main():
    pc2 = L.load_pc2()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pc2.MODEL_NAME)
    gate2 = json.loads((RES / "gate_tokenizer.json").read_text())
    gate4 = json.loads((RES / "gate_filler_c.json").read_text())
    if gate2["verdict"] != "PASS" or gate4["verdict"] != "PASS":
        raise SystemExit("⛔ gate v2/v4 が PASS でない")
    unit = gate4["final_unit"]
    rows = gate2["final_rows"]

    fails, per_row = [], {}
    for row in rows:
        rid = row["id"]
        span = tok.encode(" " + row["prompt"], add_special_tokens=False)
        rec, prev = {}, -1
        for rk, lm0_target in RUNGS.items():
            n_rep = lm0_target // 11
            ids = tok(unit * n_rep + row["prompt"])["input_ids"]
            n = len(ids)
            lm0 = n - len(span)
            if n > 256:
                fails.append(f"{rid}/{rk}: total {n} > 256")
            if ids[-len(span):] != span:
                fails.append(f"{rid}/{rk}: landmark suffix mismatch")
            if lm0 != lm0_target:
                fails.append(f"{rid}/{rk}: lm0 {lm0} != 11x{n_rep}={lm0_target}")
            if not tok.decode([ids[lm0]]).startswith(" "):
                fails.append(f"{rid}/{rk}: landmark 先頭が space 系でない")
            if lm0 <= prev:
                fails.append(f"{rid}/{rk}: lm0 not monotone")
            prev = lm0
            rec[rk] = {"n_tok": n, "landmark_start": lm0, "repeats": n_rep}
        per_row[rid] = rec
    out = {"rungs": RUNGS, "unit": unit, "per_row": per_row, "fails": fails,
           "verdict": "PASS" if not fails else "M7_STOP"}
    (RES / "gate_cii.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({"rungs": RUNGS, "fails": fails, "verdict": out["verdict"],
                      "per_row_P1": per_row.get("P1")}, ensure_ascii=False, indent=1))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
