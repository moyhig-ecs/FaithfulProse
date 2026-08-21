#!/usr/bin/env python3
"""GT-B (腕二) gate v3 — 走行前・値を見ない機械検分 (tokenizer のみ・model 不 load)。

凍結: conversations/2026-08-19/DRAFT_PREREG_arms_graspthink_20260819.md §3 (v4→v5 凍結)
発火裁可: 先生 2026-08-20「arm-B/Cを進めます」(A-5 作法の配置は v5 に登記)。

gate 述語 (機械・v3 = 三 clause・値は一切見ない):
  (a) 総 token 長同一 (v4 verbatim): 各行で全 ADJ 変種の総 token 長が一致 (mode 比較)。
  (b) landmark suffix 同一 (強化・gate v2 先例 = 偽陽性の穴を塞ぐ側): 各変種の token 列が
      tok(" "+prompt) を接尾辞として含む — landmark 域の token 化と絶対 position が
      全変種で機械的に同一であることの保証。
  (c) ADJ 非衝突 (強化): " "+ADJ の first-token が当該行 pinned 弁別集合
      (gate_tokenizer.json v2 の distinctive ids・単一出所) と非交差。
  不合格 ADJ は予備から宣言順に差し替え・登記。長さ同一 5 種未満 ⇒ M-7 停止。
  ⚠ 総長 ≤128 (fit 域内) も機械検分。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
sys.path.insert(0, str(REPO / "probes" / "olmo3-f5-think-prefix"))
import run_f5_lens as L  # noqa: E402

PREFACE = "Here is a {adj} question for you to answer: "
ADJS = ["simple", "little", "tricky", "second", "final"]   # 宣言順 (v4 §3)
BACKUP_ADJS = ["quick", "small"]                           # 予備 (宣言順)


def full_prompt(adj: str, prompt: str) -> str:
    return PREFACE.format(adj=adj) + prompt


def gate_adj_on_rows(tok, adj, rows, distinctive_by_row):
    """1 ADJ を全行で検分。fail 理由 list と per-row 記録を返す (値は見ない)。"""
    fails, per_row = [], {}
    adj_ids = tok.encode(" " + adj, add_special_tokens=False)
    for row in rows:
        rid = row["id"]
        ids = tok(full_prompt(adj, row["prompt"]))["input_ids"]
        span = tok.encode(" " + row["prompt"], add_special_tokens=False)
        n = len(ids)
        if n > 128:
            fails.append(f"{rid}: total {n} > 128")
        if ids[-len(span):] != span:
            fails.append(f"{rid}: landmark suffix mismatch")
        pinned = set().union(*distinctive_by_row[rid].values())
        if adj_ids and adj_ids[0] in pinned:
            fails.append(f"{rid}: ADJ first-token {adj_ids[0]} in pinned set")
        per_row[rid] = {"n_tok": n, "landmark_start": n - len(span),
                        "n_landmark": len(span)}
    return {"adj": adj, "adj_ids": adj_ids, "fails": fails, "per_row": per_row}


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

    swaps, reports = [], []
    candidates, backups = list(ADJS), list(BACKUP_ADJS)
    surviving = []
    while candidates:
        adj = candidates.pop(0)
        rep = gate_adj_on_rows(tok, adj, rows, distinctive_by_row)
        reports.append(rep)
        if not rep["fails"]:
            surviving.append(rep)
        elif backups:
            b = backups.pop(0)
            swaps.append({"out": adj, "in": b, "reason": rep["fails"]})
            candidates.append(b)
        else:
            swaps.append({"out": adj, "in": None, "reason": rep["fails"]})
    # (a) 総 token 長同一: 行ごとに mode と比較し、外れ ADJ を落とす (値は見ない)
    from collections import Counter
    dropped = []
    for row in rows:
        rid = row["id"]
        lengths = Counter(r["per_row"][rid]["n_tok"] for r in surviving)
        mode = lengths.most_common(1)[0][0]
        for r in list(surviving):
            if r["per_row"][rid]["n_tok"] != mode:
                surviving.remove(r)
                dropped.append({"adj": r["adj"], "reason": f"{rid}: n_tok "
                                f"{r['per_row'][rid]['n_tok']} != mode {mode}"})
    out = {"preface": PREFACE, "adjs_declared": ADJS, "backups": BACKUP_ADJS,
           "final_adjs": [r["adj"] for r in surviving],
           "reference_adj": surviving[0]["adj"] if surviving else None,
           "per_row": {row["id"]: surviving[0]["per_row"][row["id"]]
                       for row in rows} if surviving else {},
           "swaps": swaps, "dropped_by_length": dropped, "reports": reports,
           "verdict": "PASS" if len(surviving) >= 5 else "M7_STOP"}
    (RES / "gate_lengths_b.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in
                      ("final_adjs", "reference_adj", "swaps",
                       "dropped_by_length", "verdict", "per_row")},
                     ensure_ascii=False, indent=1))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
