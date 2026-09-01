#!/usr/bin/env python3
"""wgrd 評価器 v3 —— joshaku 呼び出し版(TSD 改版プログラム・R1 の実行形)。

REVISION (2026-08-26, token-set defect repair):
  v1 の変種束(variant_group_ids = 各変種 encoding の先頭 token)は T1/T2 を含み、
  群 min(T7)と併せて r_lens を系統的に引き下げた。v3 は凍結読み札
  READCARD_R1_fidelityeight_v1_20260826(TSD/R1)の宇宙四点に従う:
    (i)   集約 = 三段 median(帯 median → item 内 median → item 間 median は外)
    (ii)  mask = WORDLIKE_ALNUM 逐語(joshaku.masks から import)
    (iii) rank = mask プール内の 1 始まり順位(v1 と同宇宙・再基底化なし)
    (iv)  対象 = **観測次トークン id そのもの**(⛔ canonical へ写像しない・
          変種束 min は legacy 並記のみ)
  REVISION-ID: TSD-20260826
  ⛔ v1/v2(md5 pin 済・公開物)は不変。本 file は新設。

R6 較正(--self-test・⛔ 実測前に必ず通す・落ちたら exit 2):
  ① 恒等統制 wgrd=0 厳密 / ② 手計算 fixture / ④ 二段 median fixture
  ⑤ 陰性統制(符号反転 ⇒ 大 wgrd) / ⑥ 決定性 bit 一致
  ⑦ ★ torch 経路と joshaku.ranks.word_rank(numpy)の rank bit 一致(乱数 200 本)
     —— run_group_audit の「複製には bit 一致 assert」の形を継承

使い方:
    python eval_wgrd_v3.py --self-test
    python eval_wgrd_v3.py --lens_checkpoint CKPT --items items.jsonl --out out.json \
        [--model allenai/Olmo-3-1025-7B --device mps --max_seq_len 128] [--sign_flip]

  --sign_flip: J̄ → −J̄(chance floor 用。OLMo は既存 garbage checkpoint を使えば不要・
               Qwen 地の新設床(READCARD §2)はこの flag で作る。meta に登記される)

出力 JSON: {"meta": {...}, "items": [{"idx","n_valid","wgrd_item","legacy_wgrd_item"}]}
⛔ 集計(cell 値 = median over items)は本評価器の外(v1 と同じ無覗き順序)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))                       # joshaku
sys.path.insert(0, os.environ.get("JLENS_SRC", "../jacobian-lens"))  # jlens (upstream, untouched); set JLENS_SRC

from joshaku.masks import WORDLIKE_ALNUM, wordlike_mask_ids  # noqa: E402
from joshaku.ranks import word_rank as np_word_rank          # noqa: E402

REVISION_ID = "TSD-20260826"
READCARD = REPO / "conversations/2026-08-26/READCARD_R1_fidelityeight_v1_20260826.md"
SPEC = REPO / "SPEC/SPEC_tokenset_acceptance_v1.md"


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _joshaku_md5() -> str:
    import joshaku
    root = Path(joshaku.__file__).parent
    mods = sorted(p for p in root.rglob("*.py") if "tests" not in str(p))
    return hashlib.md5(b"".join(p.read_bytes() for p in mods)).hexdigest()


def rank_torch(logits_1d, tid: int, mask_bool) -> int:
    """READCARD (iii): mask プール内の 1 始まり順位(機構は v1 逐語・torch 経路)。

    由来: eval_wgrd.py:87-94(group_rank)を単一 id に特殊化 / 塞ぐ欠陥: T1, T7
    REVISION-ID: TSD-20260826
    ⚠ joshaku.ranks.word_rank(numpy)との bit 一致は self-test ⑦ が担保。
    """
    assert bool(mask_bool[tid]), f"id {tid} は mask 外(READCARD (iv) の前提が破れた)"
    lw = logits_1d[mask_bool]
    return int(1 + (lw > logits_1d[tid]).sum())


def legacy_group_ids(tok, next_id: int) -> list[int]:
    """v1 の変種束(legacy 並記専用・READCARD (iv))。⛔ 判定に使わない。"""
    w = tok.decode([next_id]).strip()
    ids = {int(next_id)}
    for v in (w, " " + w):
        enc = tok.encode(v, add_special_tokens=False)
        if enc:
            ids.add(int(enc[0]))
    return sorted(ids)


def wgrd_position(lens_logits_by_layer, model_logits_1d, pos_i, mask_bool,
                  next_id: int, legacy_ids: list[int]) -> tuple[float, float]:
    """READCARD (i): 帯 median of |log10 r_lens − log10 r_model|。

    戻り = (primary, legacy)。primary = 観測 id 単独 / legacy = 変種束 min(並記)。
    """
    r_model = rank_torch(model_logits_1d, next_id, mask_bool)
    r_model_leg = min(rank_torch(model_logits_1d, t, mask_bool)
                      for t in legacy_ids if bool(mask_bool[t])) if legacy_ids else r_model
    d_pri, d_leg = [], []
    for _l, ll in lens_logits_by_layer.items():
        r_l = rank_torch(ll[pos_i], next_id, mask_bool)
        d_pri.append(abs(math.log10(r_l) - math.log10(r_model)))
        in_mask = [t for t in legacy_ids if bool(mask_bool[t])]
        r_ll = min(rank_torch(ll[pos_i], t, mask_bool) for t in in_mask) if in_mask else r_l
        d_leg.append(abs(math.log10(r_ll) - math.log10(r_model_leg)))
    return statistics.median(d_pri), statistics.median(d_leg)


def eval_item(lens, model, tok, text, mask_bool, max_seq_len, skip: int = 0):
    """item 一本: 有効位置 = 次トークンが mask に属する位置 ∧ p ≥ skip。

    skip=0 が束縛 v1(v1 逐語)・skip=SKIP_FIRST_N_POSITIONS が束縛 v2
    (eval_wgrd_v2.valid_positions_v2 逐語)。
    """
    ids = model.encode(text, max_length=max_seq_len)
    seq = ids[0].tolist() if hasattr(ids, "dim") else list(ids)
    valid = [p for p in range(len(seq) - 1)
             if p >= skip and bool(mask_bool[seq[p + 1]])]
    if not valid:
        return None, None, 0
    lens_logits, model_logits, _ = lens.apply(
        model, text, positions=valid, max_seq_len=max_seq_len)
    # Qwen 系: logits 幅(151,936)> len(tok)(151,669)。embedding padding 行は False。
    # 機構 = step3_pass2.py:85-86 逐語(band_median_ranks 冒頭の pad)。
    V = int(model_logits.shape[-1])
    if int(mask_bool.shape[0]) < V:
        import torch
        mask_bool = torch.cat(
            [mask_bool, torch.zeros(V - int(mask_bool.shape[0]), dtype=torch.bool)])
    pri, leg = [], []
    for i, p in enumerate(valid):
        nid = int(seq[p + 1])
        a, b = wgrd_position(lens_logits, model_logits[i], i, mask_bool,
                             nid, legacy_group_ids(tok, nid))
        pri.append(a); leg.append(b)
    return statistics.median(pri), statistics.median(leg), len(valid)


def lens_from_fit_checkpoint(torch, jlens, path: str, sign_flip: bool):
    st = torch.load(path, map_location="cpu", weights_only=False)
    n = int(st["n_done"])
    sgn = -1.0 if sign_flip else 1.0
    jac = {int(l): sgn * (st["jacobian_sum"][l].to(torch.float32) / n)
           for l in st["jacobian_sum"]}
    d = next(iter(jac.values())).shape[0]
    return jlens.JacobianLens(jacobians=jac, n_prompts=n, d_model=d)


# ── R6 較正 ────────────────────────────────────────────────────────────────
def self_test() -> int:
    import numpy as np
    import torch
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'⭕' if cond else '⛔'} {name}")
        ok = ok and cond

    V, P, L = 12, 3, 3
    mask = torch.zeros(V, dtype=torch.bool); mask[2:10] = True
    nid = 5

    # ① 恒等統制
    m = torch.linspace(1, 0, V).repeat(P, 1)
    lensL = {l: m.clone() for l in range(L)}
    w = [wgrd_position(lensL, m[i], i, mask, nid, [nid])[0] for i in range(P)]
    check("恒等統制  wgrd = 0 厳密", all(x == 0.0 for x in w))

    # ② 手計算 fixture: 観測 id=5 を model 首位・lens 最下位へ ⇒ log10(8)
    mm = torch.zeros(1, V); mm[0, 5] = 9.0
    for k in range(V):
        if k != 5: mm[0, k] = float(k) * 0.1
    ll_bad = mm.clone(); ll_bad[0, 5] = -99.0
    r_m = rank_torch(mm[0], 5, mask); r_l = rank_torch(ll_bad[0], 5, mask)
    check(f"手計算 rank  model=1 lens=8(実測 {r_m},{r_l})", r_m == 1 and r_l == 8)
    wp, _ = wgrd_position({0: ll_bad, 1: ll_bad, 2: ll_bad}, mm[0], 0, mask, 5, [5])
    check("手計算 wgrd = log10(8)", abs(wp - math.log10(8)) < 1e-12)

    # ④ 二段 median: 3 層 Δ = {0, log10 2, log10 8} ⇒ 帯 median = log10 2
    mid = mm.clone(); mid[0, 5] = 0.85
    wp2, _ = wgrd_position({0: mm.clone(), 1: mid, 2: ll_bad}, mm[0], 0, mask, 5, [5])
    check("二段 median = log10 2", abs(wp2 - math.log10(2)) < 1e-12)

    # ⑤ 陰性統制(符号反転)
    mono = torch.linspace(3, 0, V).repeat(1, 1)
    garb = {l: -mono for l in range(L)}
    wg, _ = wgrd_position(garb, mono[0], 0, mask, 2, [2])
    check(f"陰性統制  garbage ⇒ wgrd = {wg:.3f} > 0.5 dex", wg > 0.5)

    # ⑥ 決定性
    a = [wgrd_position(lensL, m[i], i, mask, nid, [nid])[0] for i in range(P)]
    b = [wgrd_position(lensL, m[i], i, mask, nid, [nid])[0] for i in range(P)]
    check("決定性  2 走 bit 一致", a == b)

    # ⑦ ★ torch 経路 ↔ joshaku.ranks.word_rank(numpy)bit 一致(乱数 200 本)
    rng = np.random.default_rng(20260826)
    agree = 0
    for _ in range(200):
        v = rng.normal(size=V).astype(np.float64)
        tid = int(rng.integers(2, 10))
        rt = rank_torch(torch.tensor(v), tid, mask)
        rn = np_word_rank(v, tid, mask.numpy())
        agree += int(rt == rn)
    check(f"torch↔joshaku rank bit 一致 200/200(実測 {agree})", agree == 200)

    print(f"\n  ⇒ 較正 {'⭕ PASS' if ok else '⛔ FAIL'}")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--lens_checkpoint")
    ap.add_argument("--items")
    ap.add_argument("--out")
    ap.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--sign_flip", action="store_true")
    ap.add_argument("--binding", choices=["v1", "v2"], default="v1",
                    help="v2 = p >= jlens.fitting.SKIP_FIRST_N_POSITIONS(eval_wgrd_v2 逐語)")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not (a.lens_checkpoint and a.items and a.out):
        ap.error("--lens_checkpoint/--items/--out required (or --self-test)")
    if self_test() != 0:
        print("⛔ 較正失敗 ⇒ 測定しない")
        return 2

    import torch
    import transformers

    import jlens
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, revision=a.revision).to(a.device)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = jlens.from_hf(hf, tok, compile=False)
    lens = lens_from_fit_checkpoint(torch, jlens, a.lens_checkpoint, a.sign_flip)

    mask_ids = wordlike_mask_ids(tok, WORDLIKE_ALNUM)      # joshaku 呼び出し
    mask_bool = torch.zeros(len(tok), dtype=torch.bool)
    mask_bool[sorted(mask_ids)] = True                      # set → 整列 list(torch index 用)

    skip = 0
    if a.binding == "v2":
        from jlens.fitting import SKIP_FIRST_N_POSITIONS
        skip = int(SKIP_FIRST_N_POSITIONS)

    items = [json.loads(l) for l in open(a.items) if l.strip()]
    out_items, skipped = [], 0
    for idx, it in enumerate(items):
        wp, wl, nv = eval_item(lens, model, tok, it["text"], mask_bool,
                               a.max_seq_len, skip=skip)
        if wp is None:
            skipped += 1
        out_items.append({"idx": idx, "n_valid": nv,
                          "wgrd_item": None if wp is None else round(wp, 10),
                          "legacy_wgrd_item": None if wl is None else round(wl, 10)})
        if (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{len(items)}]", flush=True)

    meta = {
        "revision_id": REVISION_ID,
        "readcard": {"path": str(READCARD.relative_to(REPO)), "md5": _md5(READCARD)},
        "spec": {"path": str(SPEC.relative_to(REPO)), "md5": _md5(SPEC)},
        "joshaku_md5": _joshaku_md5(),
        "target": "observed next-token id (READCARD (iv)); legacy variant-group min in parallel only",
        "rank_universe": "wordlike pool, 1-based (READCARD (iii))",
        "aggregation": "median over band -> median over positions; median over items is outside (READCARD (i))",
        "wordlike": WORDLIKE_ALNUM.pattern,
        "tokenizer_pin": f"{a.model}@{a.revision}",
        "sign_flip": bool(a.sign_flip),
        "binding": a.binding, "skip_first_n_positions": skip,
        "lens_checkpoint": a.lens_checkpoint,
        "items_file": a.items, "items_md5": _md5(Path(a.items)),
        "model": a.model, "revision": a.revision, "max_seq_len": a.max_seq_len,
        "n_items": len(items), "n_excluded_zero_valid": skipped,
        "mask_size": int(mask_bool.sum()),
    }
    json.dump({"meta": meta, "items": out_items}, open(a.out, "w"), indent=1)
    print(f"⭕ saved {a.out}  (items {len(items)} / excluded {skipped} / mask {meta['mask_size']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
