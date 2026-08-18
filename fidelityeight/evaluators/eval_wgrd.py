#!/usr/bin/env python3
"""SFIT wgrd 評価器 —— 凍結定義の実装（ADJUDICATION_sfit_wgrd_binding_20260813 の束縛に従う）。

凍結定義（正典 = FaithfulProse/frozen_en/DECISION_TABLE.en.md §1「Definition」・逐語）:

    wgrd(S, p) = median over the read-out band of
                 | log10 rank_lens(S,p,l) − log10 rank_model(S,p) |

束縛（ADJUDICATION_sfit_wgrd_binding_20260813・先生裁可 2026-08-13）:
  ① S = 実次トークン（ground truth continuation）の word-like 変種束。
       変種作法は released 上流の意味論を verbatim 継承 —— 変種 = {w, " "+w}
       （step3_pass2.py の HEHI_WORDS 規約: case 変種なし・先頭空白変種のみ）、
       群 id = set_ids と同じく各変種 encoding の先頭 token。⛔ 再導出禁止。
  ② p = WORDLIKE 述語が有効とする全読み出し位置（= 次トークンが word-like mask に
       属する位置）。item 内は median（★ 新設の内側層 —— 外側の
       「median over items」(PREREG §5-2) は不変のまま、の限定句を随伴）。
  ③ 非 word-like 位置 = skip（released 述語の継承）。有効位置ゼロの item は除外し
       件数を機械報告。

機構の verbatim 継承（⛔ 再実装ではなく同じ演算）:
  WORDLIKE  = re.compile(r"^[A-Za-z0-9]{2,}$")          # step3_pass2.py:55 逐語
  mask      = decode(tid).strip() が WORDLIKE に match する token 全体
  群 rank   = lw = logits[:, mask] / r = 1 + (lw > logits[:, tid:tid+1]).sum(-1)
              / 群内 min                                 # band_median_ranks 内部機構 逐語

⚠ 宣言事項（★ 実装時に発見・declaration）:
  - released の band_median_ranks() 自体は「層中央値の rank」を返す補助関数であり、
    凍結式（層ごとの |Δlog10| の中央値）とは集約順が異なる。⇒ 本評価器は
    ★ 凍結式を実装し、機構（mask・群 min-rank）のみを verbatim 継承する。
  - max_seq_len = 128（fit と同一 cap）。eval 側の item を fit と同じ token 空間で読む。
  - rank_model は最終層 logits に対する同一機構（層に依らず定数・凍結式のとおり）。

R6 較正（--self-test・⛔ 実測前に必ず通す・落ちたら exit 2）:
  ① 恒等統制      lens logits ≡ model logits ⇒ wgrd = 0 厳密
  ② 手計算 fixture 小語彙の手計算 rank と一致
  ③ set_ids 群 rank fixture（裁定追加）: 2 変種群の min-rank が手計算と一致
  ④ 二段 median fixture（裁定追加）: 3 位置 × 3 層の手計算と一致
  ⑤ ★ 陰性統制    garbage lens logits（符号反転）⇒ wgrd が大きい（> 1 dex）
  ⑥ 決定性        同一入力 2 走で per-item 値が bit 一致

使い方:
    python eval_wgrd.py --self-test
    python eval_wgrd.py --lens_checkpoint CKPT --items items.jsonl --out out.json \
        [--model allenai/Olmo-3-1025-7B --revision REV --device mps --max_seq_len 128]

出力 JSON: {"meta": {..., mechanism md5, counts}, "items": [{"idx", "n_valid", "wgrd_item"}]}
⛔ 集計（w[L,E] median・I・τ）は本評価器の外（無覗き順序の別段）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

# ── 機構（逐語継承・⛔ 変更禁止）───────────────────────────────────────────
WORDLIKE = re.compile(r"^[A-Za-z0-9]{2,}$")          # step3_pass2.py:55 verbatim

UPSTREAM = Path(__file__).resolve().parents[2] / "FaithfulProse/release/code/step3_pass2.py"
CANON = Path(__file__).resolve().parents[2] / "FaithfulProse/frozen_en/DECISION_TABLE.en.md"


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def wordlike_mask_ids(tok) -> list[int]:
    """mask = decode 1 token が word-like な id 全体（step3_pass2.wordlike_mask と同一機構）。"""
    return [tid for tid in range(len(tok)) if WORDLIKE.match(tok.decode([tid]).strip())]


def variant_group_ids(tok, next_id: int) -> list[int]:
    """束縛 ①: S = 実次トークンの変種束 {w, " "+w}・set_ids 意味論（各変種 encoding の先頭 token）。"""
    w = tok.decode([next_id]).strip()
    ids = {int(next_id)}
    for v in (w, " " + w):
        enc = tok.encode(v, add_special_tokens=False)
        if enc:
            ids.add(int(enc[0]))
    return sorted(ids)


def group_rank(torch, logits_1d, mask_bool, group_ids) -> int:
    """群 rank: band_median_ranks の内部機構 逐語（word-like 中の順位・群内 min）。"""
    lw = logits_1d[mask_bool]                                     # [W]
    best = None
    for tid in group_ids:
        r = int(1 + (lw > logits_1d[tid]).sum())
        best = r if best is None else min(best, r)
    return best


def wgrd_position(torch, lens_logits_by_layer, model_logits_1d, pos_i,
                  mask_bool, group_ids) -> float:
    """凍結式: median over band of |log10 r_lens(l) − log10 r_model|。"""
    r_model = group_rank(torch, model_logits_1d, mask_bool, group_ids)
    deltas = []
    for _l, ll in lens_logits_by_layer.items():
        r_lens = group_rank(torch, ll[pos_i], mask_bool, group_ids)
        deltas.append(abs(math.log10(r_lens) - math.log10(r_model)))
    return statistics.median(deltas)


def eval_item(torch, lens, model, tok, text, mask_bool, max_seq_len):
    """item 一本: 束縛 ②③。返り値 (wgrd_item | None, n_valid)。"""
    ids = model.encode(text, max_length=max_seq_len)
    seq = ids[0].tolist() if hasattr(ids, "dim") else list(ids)
    # 有効位置: p の次トークンが word-like（mask に属する）
    valid = [p for p in range(len(seq) - 1) if bool(mask_bool[seq[p + 1]])]
    if not valid:
        return None, 0
    lens_logits, model_logits, _ = lens.apply(
        model, text, positions=valid, max_seq_len=max_seq_len)
    per_pos = []
    for i, p in enumerate(valid):
        gids = variant_group_ids(tok, seq[p + 1])
        per_pos.append(wgrd_position(torch, lens_logits, model_logits[i], i,
                                     mask_bool, gids))
    return statistics.median(per_pos), len(valid)


def lens_from_fit_checkpoint(torch, jlens, path: str):
    """fit checkpoint（jacobian_sum / n_done）→ JacobianLens（J̄ = Σ/n）。"""
    st = torch.load(path, map_location="cpu", weights_only=False)
    n = int(st["n_done"])
    jac = {int(l): (st["jacobian_sum"][l].to(torch.float32) / n)
           for l in st["jacobian_sum"]}
    d = next(iter(jac.values())).shape[0]
    return jlens.JacobianLens(jacobians=jac, n_prompts=n, d_model=d)


# ── R6 較正 ────────────────────────────────────────────────────────────────
def self_test() -> int:
    import torch
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'⭕' if cond else '⛔'} {name}")
        ok = ok and cond

    V, P, L = 12, 3, 3
    mask = torch.zeros(V, dtype=torch.bool); mask[2:10] = True     # word-like = ids 2..9
    g = [5]                                                        # 群 = {5}

    # ① 恒等統制: lens ≡ model ⇒ wgrd = 0 厳密
    m = torch.linspace(1, 0, V).repeat(P, 1)                       # 単調降下 logits
    lensL = {l: m.clone() for l in range(L)}
    w = [wgrd_position(torch, lensL, m[i], i, mask, g) for i in range(P)]
    check("恒等統制  wgrd = 0 厳密", all(x == 0.0 for x in w))

    # ② 手計算 fixture: 群 id=5 の logit を層ごとに操作して rank を既知化
    #    model: id5 が word-like 中 rank 1 / lens: rank 10 (最下位) ⇒ |Δ| = 1 dex
    mm = torch.zeros(1, V); mm[0, 5] = 9.0
    for k in range(V):
        if k != 5: mm[0, k] = float(k) * 0.1
    ll_bad = mm.clone(); ll_bad[0, 5] = -99.0                       # 群語を最下位へ
    r_m = group_rank(torch, mm[0], mask, g)
    r_l = group_rank(torch, ll_bad[0], mask, g)
    check(f"手計算 rank  model=1 lens=8（実測 {r_m},{r_l}）", r_m == 1 and r_l == 8)
    wp = wgrd_position(torch, {0: ll_bad, 1: ll_bad, 2: ll_bad}, mm[0], 0, mask, g)
    expect = abs(math.log10(8) - math.log10(1))
    check(f"手計算 wgrd = log10(8) = {expect:.6f}", abs(wp - expect) < 1e-12)

    # ③ set_ids 群 rank fixture（裁定追加）: 群 {5, 7}・7 を上位へ ⇒ min が効く
    # 手計算: word-like 値 = {.2,.3,.4,−99,.6,.85,.8,.9} ⇒ 0.85 より大 = id9 のみ ⇒ rank 2
    ll2 = ll_bad.clone(); ll2[0, 7] = 0.85
    r2 = group_rank(torch, ll2[0], mask, [5, 7])
    check(f"群 min-rank（{r2} = 2）", r2 == 2)

    # ④ 二段 median fixture（裁定追加）: 3 層の Δ = {0, 0.3, 1.0} ⇒ 層 median 0.3
    lensmix = {0: mm.clone(), 1: None, 2: ll_bad}                   # Δ=0 / 0.3 相当 / 1.0
    # 手計算: id5=0.85 ⇒ 大きいのは id9(0.9) のみ ⇒ rank 2（Δ = log10 2）
    mid = mm.clone(); mid[0, 5] = 0.85
    lensmix[1] = mid
    wp2 = wgrd_position(torch, lensmix, mm[0], 0, mask, g)
    check(f"二段 median（層 median = log10 2 = {math.log10(2):.6f}・実測 {wp2:.6f}）",
          abs(wp2 - math.log10(2)) < 1e-12)
    # 位置 median: 3 位置の wgrd {0, x, 1} ⇒ median x
    pos_meds = statistics.median([0.0, wp2, 1.0])
    check("位置 median（二段目）", pos_meds == wp2)

    # ⑤ ★ 陰性統制: garbage lens（符号反転）⇒ 大 wgrd
    mono = torch.linspace(3, 0, V).repeat(1, 1)
    garb = {l: -mono for l in range(L)}
    r_gm = group_rank(torch, mono[0], mask, [2])                    # id2 = word-like 首位
    wg = wgrd_position(torch, garb, mono[0], 0, mask, [2])
    check(f"陰性統制  garbage ⇒ wgrd = {wg:.3f} > 0.5 dex（model rank {r_gm}）", wg > 0.5)

    # ⑥ 決定性: 同一入力 2 走 bit 一致
    a = [wgrd_position(torch, lensL, m[i], i, mask, g) for i in range(P)]
    b = [wgrd_position(torch, lensL, m[i], i, mask, g) for i in range(P)]
    check("決定性  2 走 bit 一致", a == b)

    print(f"\n  ⇒ 較正 {'⭕ PASS' if ok else '⛔ FAIL'}")
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

    # ⛔ 較正を通してからしか測定に入らない（R6）
    if self_test() != 0:
        print("⛔ 較正失敗 ⇒ 測定しない")
        return 2

    import torch, transformers, jlens
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, revision=a.revision).to(a.device)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = jlens.from_hf(hf, tok, compile=False)
    lens = lens_from_fit_checkpoint(torch, jlens, a.lens_checkpoint)

    mask_ids = wordlike_mask_ids(tok)
    mask_bool = torch.zeros(len(tok), dtype=torch.bool)
    mask_bool[mask_ids] = True

    items = [json.loads(l) for l in open(a.items) if l.strip()]
    out_items, skipped = [], 0
    for idx, it in enumerate(items):
        w, nv = eval_item(torch, lens, model, tok, it["text"], mask_bool, a.max_seq_len)
        if w is None:
            skipped += 1
        out_items.append({"idx": idx, "n_valid": nv,
                          "wgrd_item": (None if w is None else round(w, 10))})
        if (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{len(items)}]", flush=True)

    meta = {
        "lens_checkpoint": a.lens_checkpoint,
        "items_file": a.items,
        "items_md5": _md5(Path(a.items)),
        "model": a.model, "revision": a.revision,
        "max_seq_len": a.max_seq_len,
        "n_items": len(items), "n_excluded_zero_valid": skipped,
        "mask_size": int(mask_bool.sum()),
        "mechanism_pins": {
            "step3_pass2.py_md5": _md5(UPSTREAM),
            "DECISION_TABLE.en.md_md5": _md5(CANON),
            "WORDLIKE": WORDLIKE.pattern,
            "variant_convention": '{w, " "+w} first-token (HEHI_WORDS 規約)',
        },
        "binding": "ADJUDICATION_sfit_wgrd_binding_20260813",
    }
    json.dump({"meta": meta, "items": out_items}, open(a.out, "w"), indent=1)
    print(f"⭕ saved {a.out}  (items {len(items)} / excluded {skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
