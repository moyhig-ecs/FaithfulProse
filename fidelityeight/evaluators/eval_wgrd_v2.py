#!/usr/bin/env python3
"""SFIT wgrd 評価器 v2 —— 束縛 v2 = **位置域を計器の宣言 fit 域に整合**(p ≥ 16)。

v1(eval_wgrd.py・md5 44e9c695…・⛔ 凍結不変)からの変更は **読み出し位置の定義域のみ**:

    v1: 有効位置 = 次トークンが word-like な全位置(p = 0 起点)
    v2: 有効位置 = 同条件 ∧ **p ≥ SKIP_FIRST_N_POSITIONS(= 16・上流から import)**

根拠(上流 jlens/fitting.py:41-42 逐語): "positions act as attention sinks and have
atypical residual statistics" —— fit() が除外する位置域を eval も除外し、計器が
fit された定義域でのみ読む。⛔ v1 の判定(P5′/P0/P0′)は束縛宣言つきで不変・v2 は並置
(ADJUDICATION_sfit2_roadmap_and_publication_20260818 §2-E)。

機構の継承(⛔ 再実装ではなく同一オブジェクト): WORDLIKE / wordlike_mask_ids /
variant_group_ids / group_rank / wgrd_position / lens_from_fit_checkpoint を
v1 module からそのまま import する。v1 の md5 は meta に併記。

R6 較正(--self-test・⛔ 実測前に必ず通す・落ちたら exit 2):
  ①〜⑥ v1 の self_test をそのまま実行(継承機構の較正)
  ⑦ 位置 filter fixture(v2 固有):
     (a) 上流定数 pin: SKIP_FIRST_N_POSITIONS == 16
     (b) 全位置 word-like の seq(len 20)⇒ v2 有効位置 = [16, 17, 18](手計算)
     (c) word-like が p<16 にしかない seq ⇒ 有効位置 [] = 除外経路(機械報告)
     (d) v1 定義との差集合 = {p < 16} のみ(v2 が落とすのは頭だけ・尾は不変)

使い方:
    python eval_wgrd_v2.py --self-test
    python eval_wgrd_v2.py --lens_checkpoint CKPT --items items.jsonl --out out.json \
        [--model allenai/Olmo-3-1025-7B --revision REV --device mps --max_seq_len 128]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

_V1_PATH = Path(__file__).resolve().parent / "eval_wgrd.py"
_spec = importlib.util.spec_from_file_location("eval_wgrd_v1", _V1_PATH)
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)


def skip_first() -> int:
    """上流の定数を import して返す(⛔ 再宣言しない)。"""
    from jlens.fitting import SKIP_FIRST_N_POSITIONS
    return int(SKIP_FIRST_N_POSITIONS)


def valid_positions_v2(seq, mask_bool, skip: int):
    """束縛 v2 の有効位置: 次トークンが word-like ∧ p ≥ skip。"""
    return [p for p in range(len(seq) - 1)
            if p >= skip and bool(mask_bool[seq[p + 1]])]


def eval_item_v2(torch, lens, model, tok, text, mask_bool, max_seq_len, skip):
    """item 一本(v1 と同一機構・位置域のみ v2)。返り値 (wgrd_item | None, n_valid)。"""
    ids = model.encode(text, max_length=max_seq_len)
    seq = ids[0].tolist() if hasattr(ids, "dim") else list(ids)
    valid = valid_positions_v2(seq, mask_bool, skip)
    if not valid:
        return None, 0
    lens_logits, model_logits, _ = lens.apply(
        model, text, positions=valid, max_seq_len=max_seq_len)
    per_pos = []
    for i, p in enumerate(valid):
        gids = v1.variant_group_ids(tok, seq[p + 1])
        per_pos.append(v1.wgrd_position(torch, lens_logits, model_logits[i], i,
                                        mask_bool, gids))
    return statistics.median(per_pos), len(valid)


def self_test() -> int:
    # ①〜⑥: v1 の較正をそのまま(継承機構)
    if v1.self_test() != 0:
        return 2

    import torch
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'⭕' if cond else '⛔'} {name}")
        ok = ok and cond

    print("\n  ── v2 固有(⑦ 位置 filter)──")
    sk = skip_first()
    check(f"⑦a 上流定数 pin  SKIP_FIRST_N_POSITIONS = {sk} == 16", sk == 16)

    V = 12
    mask = torch.zeros(V, dtype=torch.bool); mask[2:10] = True
    seq_all = [5] * 20                                    # 次トークン常に word-like
    vp = valid_positions_v2(seq_all, mask, sk)
    check(f"⑦b 全 word-like seq(len 20)⇒ 有効位置 {vp} == [16, 17, 18]",
          vp == [16, 17, 18])

    seq_head = [5] * 10 + [0] * 10                        # word-like は頭だけ
    vp2 = valid_positions_v2(seq_head, mask, sk)
    check(f"⑦c 頭のみ word-like ⇒ 有効位置 [] = 除外経路({vp2})", vp2 == [])

    seq_mix = [0, 5] * 15                                 # 交互
    v1_set = {p for p in range(len(seq_mix) - 1) if bool(mask[seq_mix[p + 1]])}
    v2_set = set(valid_positions_v2(seq_mix, mask, sk))
    check("⑦d v1 − v2 = {p < 16} のみ(尾は不変)",
          v1_set - v2_set == {p for p in v1_set if p < sk} and v2_set <= v1_set)

    print(f"\n  ⇒ v2 較正 {'⭕ PASS' if ok else '⛔ FAIL'}")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--lens_checkpoint")
    ap.add_argument("--items", help="JSONL with field 'text'")
    ap.add_argument("--out")
    ap.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max_seq_len", type=int, default=128)
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    if not (a.lens_checkpoint and a.items and a.out):
        ap.error("--lens_checkpoint/--items/--out are required (or --self-test)")

    # ⛔ 較正を通してからしか測定に入らない(R6)
    if self_test() != 0:
        print("⛔ 較正失敗 ⇒ 測定しない")
        return 2

    import torch, transformers, jlens
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, revision=a.revision).to(a.device)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = jlens.from_hf(hf, tok, compile=False)
    lens = v1.lens_from_fit_checkpoint(torch, jlens, a.lens_checkpoint)

    mask_ids = v1.wordlike_mask_ids(tok)
    mask_bool = torch.zeros(len(tok), dtype=torch.bool)
    mask_bool[mask_ids] = True

    sk = skip_first()
    items = [json.loads(l) for l in open(a.items) if l.strip()]
    out_items, skipped = [], 0
    for idx, it in enumerate(items):
        w, nv = eval_item_v2(torch, lens, model, tok, it["text"], mask_bool,
                             a.max_seq_len, sk)
        if w is None:
            skipped += 1
        out_items.append({"idx": idx, "n_valid": nv,
                          "wgrd_item": (None if w is None else round(w, 10))})
        if (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{len(items)}]", flush=True)

    meta = {
        "lens_checkpoint": a.lens_checkpoint,
        "items_file": a.items,
        "items_md5": v1._md5(Path(a.items)),
        "model": a.model, "revision": a.revision,
        "max_seq_len": a.max_seq_len,
        "n_items": len(items), "n_excluded_zero_valid": skipped,
        "mask_size": int(mask_bool.sum()),
        "mechanism_pins": {
            "eval_wgrd_v1_md5": v1._md5(_V1_PATH),
            "step3_pass2.py_md5": v1._md5(v1.UPSTREAM),
            "DECISION_TABLE.en.md_md5": v1._md5(v1.CANON),
            "WORDLIKE": v1.WORDLIKE.pattern,
            "variant_convention": '{w, " "+w} first-token (HEHI_WORDS 規約)',
            "skip_first_positions": sk,
            "skip_first_source": "jlens.fitting.SKIP_FIRST_N_POSITIONS (import)",
        },
        "binding": "v2 (p >= 16) — ADJUDICATION_sfit2_roadmap_and_publication_20260818 §2-E"
                   " on top of ADJUDICATION_sfit_wgrd_binding_20260813",
    }
    json.dump({"meta": meta, "items": out_items}, open(a.out, "w"), indent=1)
    print(f"⭕ saved {a.out}  (items {len(items)} / excluded {skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
