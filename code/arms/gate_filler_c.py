#!/usr/bin/env python3
"""GT-C (腕三・域内版) gate v4 — 走行前・値を見ない機械検分 (tokenizer のみ)。

凍結: conversations/2026-08-19/DRAFT_PREREG_arms_graspthink_20260819.md §4 (v5 で具体化)
発火裁可: 先生 2026-08-20「arm-B/Cを進めます」。

gate 述語 (機械・値は一切見ない):
  (a) 総 token 長 ≤128 (域内版の定義・全行 × 全 filler 条件)。
  (b) filler 非衝突: filler unit の全 token id が当該行 pinned 弁別集合
      (gate_tokenizer.json v2 distinctive・単一出所) と非交差。
  (c) landmark suffix 同一: n≥1 変種の token 列が tok(" "+prompt) を接尾辞として含む。
      n=0 は素 prompt (GT-A verbatim)。★ 境界 token 登記: n=0 の landmark 先頭は
      裸 "The"・n≥1 は " The" — first-token id が異なる (gate v1 M-7 と同型の
      tokenizer 構造・値は見ていない・登記のみ)。
  (d) landmark_start 単調分離: f0 (0) < fmid < fbig。
  不合格 unit は予備から宣言順に差し替え・登記。差し替え後も不合格 ⇒ M-7 停止。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
sys.path.insert(0, str(REPO / "probes" / "olmo3-f5-think-prefix"))
import run_f5_lens as L  # noqa: E402

FILLER_UNITS = [   # 宣言順 (第一が本命・予備二本)
    "Please stay focused and keep reading this text with care. ",
    "There is nothing important in this sentence so just move along. ",
    "We will get to the main part of this message very soon now. ",
]
REPEATS = {"f0": 0, "fmid": 3, "fbig": 8}   # 配置定数 (錨は v5 登記)


def build(unit: str, n: int, prompt: str) -> str:
    return unit * n + prompt


def gate_unit(tok, unit, rows, distinctive_by_row):
    fails, per_row = [], {}
    unit_ids = set(tok.encode(unit, add_special_tokens=False)) | \
        set(tok.encode(" " + unit.strip(), add_special_tokens=False))
    for row in rows:
        rid = row["id"]
        span_bare = tok(row["prompt"])["input_ids"]
        span_sp = tok.encode(" " + row["prompt"], add_special_tokens=False)
        pinned = set().union(*distinctive_by_row[rid].values())
        if unit_ids & pinned:
            fails.append(f"{rid}: filler token in pinned set {sorted(unit_ids & pinned)}")
        rec = {}
        for cond, n in REPEATS.items():
            ids = tok(build(unit, n, row["prompt"]))["input_ids"]
            if len(ids) > 128:
                fails.append(f"{rid}/{cond}: total {len(ids)} > 128")
            if n == 0:
                if ids != span_bare:
                    fails.append(f"{rid}/f0: not bare prompt")
                rec[cond] = {"n_tok": len(ids), "landmark_start": 0}
            else:
                if ids[-len(span_sp):] != span_sp:
                    fails.append(f"{rid}/{cond}: landmark suffix mismatch")
                rec[cond] = {"n_tok": len(ids),
                             "landmark_start": len(ids) - len(span_sp)}
        if not (rec["f0"]["landmark_start"] < rec["fmid"]["landmark_start"]
                < rec["fbig"]["landmark_start"]):
            fails.append(f"{rid}: landmark_start not monotone")
        per_row[rid] = rec
    return {"unit": unit, "fails": fails, "per_row": per_row}


def main():
    pc2 = L.load_pc2()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pc2.MODEL_NAME)
    gate2 = json.loads((RES / "gate_tokenizer.json").read_text())
    if gate2["verdict"] != "PASS":
        raise SystemExit("⛔ tokenizer gate v2 が PASS でない")
    rows = gate2["final_rows"]
    distinctive_by_row = {}
    for row in rows:
        rep = next(r for r in gate2["reports"] if r["id"] == row["id"] and r["pass"])
        distinctive_by_row[row["id"]] = {
            k: {i for i, _ in rep["detail"][k]["distinctive"]} for k in rep["detail"]}

    swaps, reports, chosen = [], [], None
    for unit in FILLER_UNITS:
        rep = gate_unit(tok, unit, rows, distinctive_by_row)
        reports.append(rep)
        if not rep["fails"]:
            chosen = rep
            break
        swaps.append({"out": unit, "reason": rep["fails"]})
    out = {"units_declared": FILLER_UNITS, "repeats": REPEATS,
           "final_unit": chosen["unit"] if chosen else None,
           "per_row": chosen["per_row"] if chosen else {},
           "boundary_token_note": "f0 landmark 先頭は裸 'The'・fmid/fbig は ' The' "
           "(first-token id 相違・token 構造の登記のみ・値は未見)",
           "swaps": swaps, "reports": reports,
           "verdict": "PASS" if chosen else "M7_STOP"}
    (RES / "gate_filler_c.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in
                      ("final_unit", "repeats", "swaps", "verdict", "per_row")},
                     ensure_ascii=False, indent=1))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
