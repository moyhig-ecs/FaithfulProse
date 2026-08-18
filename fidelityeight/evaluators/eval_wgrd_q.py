#!/usr/bin/env python3
"""SFIT wgrd 評価器 Q 変種 —— **mask 幅を logits 幅に整合**(Qwen 系 tokenizer 対応)。

技術不能の出所(FLAG_sfit2_q_vocab_mask_20260818): Qwen/Qwen3-8B は
len(tokenizer) = 151,669 < config.vocab_size = 151,936(embedding padding 267 슬롯)。
v1/v2 は mask を torch.zeros(len(tok)) で作るため logits[mask] が幅不一致で落ちる。
OLMo 系は両者一致ゆえ露出しなかった経路。

変更点は **mask の器の幅のみ**: mask を max(len(tok), vocab_size) で作り、
padding 슬롯は False のまま。★ rank 保存の宣言:
  - group_rank の lw = logits[mask] は word-like id(全て < len(tok))しか選ばない
  - 群 id も実トークン(< len(tok))
  ⇒ padding 슬롯は rank 算術に一度も参加しない —— OLMo 系と同一の演算(bit 同値)。

束縛は --binding v1 | v2 で選ぶ(機構は v1/v2 module から同一オブジェクト import・
両 md5 を meta に pin)。⛔ v1/v2 file は無改変のまま。

R6 較正(--self-test): v1 ①〜⑥ + v2 ⑦ を両方実行 + ⑧ mask-padding fixture
(同一 logits に対し mask 幅 V と V+267(padding False)で rank が bit 一致)。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v1 = _load("eval_wgrd")
v2 = _load("eval_wgrd_v2")


def self_test() -> int:
    if v1.self_test() != 0:
        return 2
    if v2.self_test() != 0:
        return 2

    import torch
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'⭕' if cond else '⛔'} {name}")
        ok = ok and cond

    print("\n  ── Q 変種固有(⑧ mask-padding)──")
    V, PAD = 12, 267
    mask_v = torch.zeros(V, dtype=torch.bool); mask_v[2:10] = True
    mask_p = torch.zeros(V + PAD, dtype=torch.bool); mask_p[2:10] = True
    logits_v = torch.linspace(3, 0, V)
    logits_p = torch.cat([logits_v, torch.full((PAD,), -1e4)])   # padding 슬롯の値は不問
    for gid in ([2], [5], [5, 7]):
        r_v = v1.group_rank(torch, logits_v, mask_v, gid)
        r_p = v1.group_rank(torch, logits_p, mask_p, gid)
        check(f"⑧ 群 {gid}: rank {r_v} == {r_p}(幅 {V} vs {V+PAD})", r_v == r_p)

    print(f"\n  ⇒ Q 変種較正 {'⭕ PASS' if ok else '⛔ FAIL'}")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--binding", choices=["v1", "v2"], default=None)
    ap.add_argument("--lens_checkpoint")
    ap.add_argument("--items")
    ap.add_argument("--out")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max_seq_len", type=int, default=128)
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    if not (a.binding and a.lens_checkpoint and a.items and a.out):
        ap.error("--binding/--lens_checkpoint/--items/--out are required (or --self-test)")

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

    # ★ 唯一の変更点: mask の器を logits 幅へ(padding 슬롯 = False・rank 保存)
    width = max(len(tok), int(getattr(hf.config, "vocab_size", len(tok))))
    mask_ids = v1.wordlike_mask_ids(tok)
    mask_bool = torch.zeros(width, dtype=torch.bool)
    mask_bool[mask_ids] = True

    items = [json.loads(l) for l in open(a.items) if l.strip()]
    out_items, skipped = [], 0
    sk = v2.skip_first() if a.binding == "v2" else None
    for idx, it in enumerate(items):
        if a.binding == "v1":
            w, nv = v1.eval_item(torch, lens, model, tok, it["text"], mask_bool,
                                 a.max_seq_len)
        else:
            w, nv = v2.eval_item_v2(torch, lens, model, tok, it["text"], mask_bool,
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
        "mask_width": width, "len_tokenizer": len(tok),
        "binding_variant": a.binding,
        "mechanism_pins": {
            "eval_wgrd_v1_md5": v1._md5(_DIR / "eval_wgrd.py"),
            "eval_wgrd_v2_md5": v1._md5(_DIR / "eval_wgrd_v2.py"),
            "step3_pass2.py_md5": v1._md5(v1.UPSTREAM),
            "DECISION_TABLE.en.md_md5": v1._md5(v1.CANON),
            "WORDLIKE": v1.WORDLIKE.pattern,
            "variant_convention": '{w, " "+w} first-token (HEHI_WORDS 規約)',
            "mask_padding": "width = max(len(tok), config.vocab_size); padding slots False (rank-preserving)",
        },
        "binding": ("v1 (ADJUDICATION_sfit_wgrd_binding_20260813)" if a.binding == "v1"
                    else "v2 (p >= 16) — ADJUDICATION_sfit2_roadmap_and_publication_20260818 §2-E"),
    }
    json.dump({"meta": meta, "items": out_items}, open(a.out, "w"), indent=1)
    print(f"⭕ saved {a.out}  (items {len(items)} / excluded {skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
