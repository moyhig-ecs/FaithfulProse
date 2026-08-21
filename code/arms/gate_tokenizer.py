#!/usr/bin/env python3
"""GT-A (腕一) tokenizer gate — 走行前・値を見ない機械検分。

凍結: conversations/2026-08-19/DRAFT_PREREG_arms_graspthink_20260819.md §2-1
      first-token 同定 = wgrd S 変種規約 verbatim ({w, " "+w} の first-tokens)。

gate 述語 (機械・v2 = 弁別部分集合則):
  各 pinned 語の 読み集合 := {w, " "+w} の first-tokens のうち decode 文字列 (strip 後) が
    2 文字以上のもののみ (単字 token は集合から落とす — 偽陽性の穴を塞ぐ側の強化)。
  行が FAIL := (a') いずれかの語の読み集合が空, または
              (b) 読み集合が同一行内の別語の読み集合と交差する。
  FAIL 行は BACKUPS から宣言順に差し替え (差し替えも本 gate を通す)。
  FAIL ≥ 3 行 (差し替え後も) ⇒ M-7 停止・FLAG。
  ⚠ v1 (全変種に単字禁止) は M-7 発火 (results/gate_tokenizer_v1_m7.json 保存) —
    裸変種の大文字固有名詞が系統的に単字先頭に割れるため。v2 は読み集合の側を
    強化する形の改訂 (Q 札の rank-preserving 変種の先例・値は一切見ていない)。
  ★ runner は本 gate の distinctive token ids をそのまま読み集合として使う (単一出所)。

⛔ 本 gate は model を load しない (tokenizer のみ)。値 (rank) を一切見ない。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "probes" / "olmo3-f5-think-prefix"))
import run_f5_lens as L  # noqa: E402  (load_pc2 経路の verbatim 流用)

# ---- 素材 (draft §2-1 の 5 本・凍結) ---------------------------------------
PROMPTS = [
    {"id": "P1", "prompt": "The color of the planet fourth from the sun is",
     "mid": "Mars", "ans": "red", "distractors": ["Jupiter", "blue"]},
    {"id": "P2", "prompt": "The number of legs on the animal that spins webs is",
     "mid": "spider", "ans": "eight", "distractors": ["insect", "six"]},
    {"id": "P3", "prompt": "The capital of the country whose flag has a red maple leaf is",
     "mid": "Canada", "ans": "Ottawa", "distractors": ["Toronto", "Washington"]},
    {"id": "P4", "prompt": "The chemical symbol of the metal used in ordinary thermometers is",
     "mid": "mercury", "ans": "Hg", "distractors": ["iron", "silver"]},
    {"id": "P5", "prompt": "The first name of the author of the play Hamlet is",
     "mid": "Shakespeare", "ans": "William", "distractors": ["Romeo", "Charles"]},
]
# ---- 予備 (gate 結果を見る前に宣言・宣言順に消費) ---------------------------
BACKUPS = [
    {"id": "B1", "prompt": "The capital city of the country famous for the Eiffel Tower is",
     "mid": "France", "ans": "Paris", "distractors": ["Italy", "London"]},
    {"id": "B2", "prompt": "The number of sides on the shape that has three angles is",
     "mid": "triangle", "ans": "three", "distractors": ["square", "four"]},
    {"id": "B3", "prompt": "The continent where the country of the pyramids of Giza is located is",
     "mid": "Egypt", "ans": "Africa", "distractors": ["Greece", "Asia"]},
    # 第二 tranche (v1 M-7 後・token 構造のみ見て宣言・rank 値は未見)
    {"id": "C1", "prompt": "The color of the fruit that keeps the doctor away is",
     "mid": "apple", "ans": "red", "distractors": ["banana", "green"]},
    {"id": "C2", "prompt": "The number of sides of the shape on a stop sign is",
     "mid": "octagon", "ans": "eight", "distractors": ["circle", "four"]},
    {"id": "C3", "prompt": "The first name of the author of the play Macbeth is",
     "mid": "Shakespeare", "ans": "William", "distractors": ["Hamlet", "Charles"]},
]


def first_tokens(tok, w: str):
    """wgrd S 変種規約: {w, ' '+w} の first-token id と文字列の組。"""
    out = []
    for v in (w, " " + w):
        ids = tok.encode(v, add_special_tokens=False)
        if ids:
            out.append((ids[0], tok.decode([ids[0]])))
    return out


def gate_row(tok, row):
    words = [("mid", row["mid"]), ("ans", row["ans"])] + [
        (f"d{i+1}", d) for i, d in enumerate(row["distractors"])]
    detail, fail = {}, []
    ftok = {}
    for key, w in words:
        fts = first_tokens(tok, w)
        distinctive = [(i, s) for i, s in fts if len(s.strip()) >= 2]
        ftok[key] = {i for i, _ in distinctive}
        detail[key] = {"word": w, "first_tokens": [(i, s) for i, s in fts],
                       "distinctive": [(i, s) for i, s in distinctive]}
        if not distinctive:
            fail.append(f"{key}:{w}: distinctive set empty")
    keys = list(ftok)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            inter = ftok[keys[a]] & ftok[keys[b]]
            if inter:
                fail.append(f"{keys[a]}x{keys[b]}: shared first-token {sorted(inter)}")
    # prompt 長 (≤128 token・fit 域内) も機械検分
    n_tok = len(tok.encode(row["prompt"], add_special_tokens=False))
    if n_tok > 128:
        fail.append(f"prompt too long: {n_tok} > 128")
    return {"id": row["id"], "prompt_tokens": n_tok, "detail": detail,
            "fail": fail, "pass": not fail}


def main():
    pc2 = L.load_pc2()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pc2.MODEL_NAME, revision=pc2.REVISION) \
        if hasattr(pc2, "REVISION") else AutoTokenizer.from_pretrained(pc2.MODEL_NAME)
    final, swaps, backups = [], [], list(BACKUPS)
    reports = []
    for row in PROMPTS:
        rep = gate_row(tok, row)
        reports.append(rep)
        if rep["pass"]:
            final.append(row)
            continue
        # 差し替え (宣言順・差し替え先も gate)
        replaced = False
        while backups:
            b = backups.pop(0)
            brep = gate_row(tok, b)
            reports.append(brep)
            if brep["pass"]:
                final.append(b)
                swaps.append({"out": row["id"], "in": b["id"], "reason": rep["fail"]})
                replaced = True
                break
            swaps.append({"out": b["id"], "in": None, "reason": brep["fail"]})
        if not replaced:
            swaps.append({"out": row["id"], "in": None, "reason": rep["fail"]})
    n_fail_final = 5 - len(final)
    out = {"model": pc2.MODEL_NAME, "final_set": [r["id"] for r in final],
           "swaps": swaps, "n_missing": n_fail_final, "reports": reports,
           "verdict": "M7_STOP" if n_fail_final >= 3 or len(final) < 5 else "PASS",
           "final_rows": final}
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    (res / "gate_tokenizer.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("final_set", "swaps", "n_missing", "verdict")},
                     ensure_ascii=False, indent=1))
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
