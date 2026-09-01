#!/usr/bin/env python3
"""`F5` order-agreement stage --- replace the optimistic side of a two-set verdict by a direct measurement.

Variant of run_f5_lens.py (2026-08-19). In the same forward pass, the model logits that
`lens.apply` returns (and that the earlier runner discarded) are kept, and for every position
the band-median group ranks of the two word groups (the label group and the apex group) are
recorded on BOTH the lens side and the model side.

Main quantity   * order agreement eta: the fraction of positions where
                sign(h_lens - a_lens) == sign(h_model - a_model); ties (h == a) are excluded
                and counted. This replaces the sandwich assumption (independent / perfectly
                correlated) by a direct reading.
Secondary       dh = log10(h_lens / h_model), da likewise; per-trace Pearson rho(dh, da), recorded only.
Mechanism       unchanged: band_median_ranks / set_ids are imported verbatim from step3_pass2;
                the model side passes {0: model_logits} through the same band_median_ranks
                (a single-layer median is the identity, i.e. the final-layer group rank).
Corpus          the 18 traces of F5 verbatim (f5_runs_{F5a-2,F5b-1}.json; prompt md5 asserted).

Calibration (--self-test; run before any measurement; exit 2 on failure):
  1. identity control   lens == model  =>  eta = 1 exactly; rho undefined is reported as N/A
  2. flip control       swap the two groups' logits on the lens side  =>  every separating position disagrees
  3. tie path           tie fixture  =>  excluded and counted
  4. determinism        two runs on identical input agree bitwise

usage:
  python run_f5_corr.py --self-test
  python run_f5_corr.py --variant F5a-2 [--allow-dirty]
output: results/f5_corr_{variant}.json (saved per seed; resumable)
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
sys.path.insert(0, str(HERE))
import run_f5_lens as L                     # 経路の verbatim 流用(provenance/load_pc2/RES/CUE)

RES = L.RES
ADJUDICATION = "adjudication record 2026-08-19 (branch c)"


def pair_series(torch, p2, jl, ml, mask, sets):
    """lens band-median と model(最終層)の二群 rank series を同一機構で返す。"""
    med_lens = p2.band_median_ranks(torch, jl, mask, sets)
    med_model = p2.band_median_ranks(torch, {0: ml}, mask, sets)
    return med_lens, med_model


def agreement(h_l, a_l, h_m, a_m):
    """順序一致率 η と tie 件数。tie(いずれかの側で h==a)は除外+報告。"""
    n_agree = n_used = n_tie = 0
    for hl, al, hm, am in zip(h_l, a_l, h_m, a_m):
        sl, sm = (hl > al) - (hl < al), (hm > am) - (hm < am)
        if sl == 0 or sm == 0:
            n_tie += 1
            continue
        n_used += 1
        if sl == sm:
            n_agree += 1
    eta = (n_agree / n_used) if n_used else None
    return eta, n_used, n_tie


def deltas_rho(h_l, a_l, h_m, a_m):
    """副量: Δh, Δa の Pearson ρ(登記のみ)。分散ゼロは None。"""
    dh = [math.log10(x) - math.log10(y) for x, y in zip(h_l, h_m)]
    da = [math.log10(x) - math.log10(y) for x, y in zip(a_l, a_m)]
    n = len(dh)
    if n < 2:
        return None
    mh, ma = sum(dh) / n, sum(da) / n
    cov = sum((x - mh) * (y - ma) for x, y in zip(dh, da))
    vh = sum((x - mh) ** 2 for x in dh)
    va = sum((y - ma) ** 2 for y in da)
    if vh == 0 or va == 0:
        return None
    return cov / math.sqrt(vh * va)


def self_test() -> int:
    import torch
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'⭕' if cond else '⛔'} {name}")
        ok = ok and cond

    # 小宇宙: 4 位置・h/a の既知 rank 系列で agreement を手計算照合
    # ① 恒等統制: lens ≡ model ⇒ η = 1
    h = [3, 5, 2, 9]; a = [7, 1, 6, 4]
    eta, used, tie = agreement(h, a, h, a)
    check(f"① 恒等統制 η = 1 厳密(used {used}/tie {tie})", eta == 1.0 and used == 4 and tie == 0)

    # ② 反転統制: lens 側で h/a を入替 ⇒ 全位置不一致 ⇒ η = 0
    eta2, used2, _ = agreement(a, h, h, a)
    check(f"② 反転統制 η = 0 厳密({eta2})", eta2 == 0.0 and used2 == 4)

    # ③ tie 経路: lens 側に tie を混ぜる ⇒ 除外 + 件数(pos0: 一致 / pos1: tie / pos2: 一致)
    eta3, used3, tie3 = agreement([3, 5, 5], [7, 5, 1], [3, 5, 6], [7, 1, 2])
    check(f"③ tie 除外(used {used3} = 2 / tie {tie3} = 1)・残り一致 η = {eta3}",
          used3 == 2 and tie3 == 1 and eta3 == 1.0)

    # ④ 決定性
    r1 = agreement(h, a, h, a); r2 = agreement(h, a, h, a)
    check("④ 決定性 2 走 bit 一致", r1 == r2)

    # ⑤ 副量 ρ の手計算 fixture: Δh = Δa = [−0.301, −0.602, −0.903](変動しつつ完全一致)⇒ ρ = 1
    #    (⚠ 初版 fixture は Δ が定数で ρ 定義不能 = code の正しい None を踏んだ ⇒ 較正が捕捉)
    rho = deltas_rho([10, 100, 1000], [1, 1, 1], [20, 400, 8000], [2, 4, 8])
    check(f"⑤ ρ 手計算(Δh = Δa 変動・完全一致 ⇒ ρ = 1・実測 {rho})", rho is not None and abs(rho - 1.0) < 1e-12)

    print(f"\n  ⇒ 較正 {'⭕ PASS' if ok else '⛔ FAIL'}")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--variant")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.variant:
        ap.error("--variant is required (or --self-test)")
    if self_test() != 0:
        print("⛔ 較正失敗 ⇒ 測定しない")
        return 2

    prov = L.provenance(a.variant, a.allow_dirty)
    L.log(f"R7-a ✅ commit={prov['commit'][:8]} dirty={prov['dirty']} stage=corr variant={a.variant}")

    pc2 = L.load_pc2()
    import screen_base as SB
    from run_f5 import build_prompt

    src = RES / f"f5_runs_{a.variant}.json"
    doc = json.loads(src.read_text())
    rows = {r["seed"]: r for r in doc["rows"]}

    base = SB.load_prompts()[L.CONDITION]
    newbase, _ = build_prompt(base, a.variant)
    prompt_text = newbase + L.CUE
    md5 = hashlib.md5(prompt_text.encode()).hexdigest()
    if md5 != doc["prompt_md5_new"]:
        raise SystemExit(f"⛔ prompt md5 不一致: {md5} != {doc['prompt_md5_new']}")
    L.log(f"prompt md5 ✅ {md5}")

    import torch, transformers, jlens
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
    L.log(f"stack ✅ ({time.time()-t0:.0f}s) BAND={pc2.BAND[0]}..{pc2.BAND[-1]} MAX_SEQ={pc2.MAX_SEQ}")

    ids = json.loads(pc2.MASK_PATH.read_text())
    mask = torch.zeros(len(tok), dtype=torch.bool); mask[ids] = True
    sets = {"hehi": p2.set_ids(tok, p2.HEHI_WORDS), "apex": p2.set_ids(tok, p2.APEX_WORDS)}

    out_path = RES / f"f5_corr_{a.variant}.json"
    doc_out = json.loads(out_path.read_text()) if out_path.exists() else {
        "adjudication": ADJUDICATION, "variant": a.variant,
        "prompt_md5_new": md5, "lens_md5": pc2.LENS_MD5,
        "band": [pc2.BAND[0], pc2.BAND[-1]], "max_seq": pc2.MAX_SEQ,
        "provenance": prov, "rows": []}
    have = {r["seed"] for r in doc_out["rows"]}

    n_prompt = len(tok(prompt_text)["input_ids"])
    for seed in sorted(rows):
        if seed in have:
            continue
        r = rows[seed]
        if not r["content"].strip() or r.get("label") is None:
            doc_out["rows"].append({"seed": seed, "excluded": "label=None or 空"})
            continue
        full = prompt_text + r["content"]
        T = len(tok(full)["input_ids"]) - n_prompt
        positions = [p for p in range(n_prompt, n_prompt + T) if p < pc2.MAX_SEQ]
        truncated = len(positions) < T
        t1 = time.time()
        hl_s, al_s, hm_s, am_s = [], [], [], []
        for i in range(0, len(positions), p2.CHUNK):
            chunk = positions[i:i + p2.CHUNK]
            jl, ml, _ = lens.apply(model, full, layers=pc2.BAND,
                                   positions=chunk, max_seq_len=pc2.MAX_SEQ)
            med_l, med_m = pair_series(torch, p2, jl, ml, mask, sets)
            hl_s.extend(med_l["hehi"]); al_s.extend(med_l["apex"])
            hm_s.extend(med_m["hehi"]); am_s.extend(med_m["apex"])
        eta, n_used, n_tie = agreement(hl_s, al_s, hm_s, am_s)
        rho = deltas_rho(hl_s, al_s, hm_s, am_s)
        doc_out["rows"].append({
            "seed": seed, "label": r["label"], "T": len(hl_s), "lens_truncated": truncated,
            "eta": (None if eta is None else round(eta, 6)),
            "n_used": n_used, "n_tie": n_tie,
            "rho_delta": (None if rho is None else round(rho, 6)),
            "wall_s": round(time.time() - t1, 1),
            "h_lens": hl_s, "a_lens": al_s, "h_model": hm_s, "a_model": am_s})
        out_path.write_text(json.dumps(doc_out, ensure_ascii=False, indent=1) + "\n")
        L.log(f"  s{seed} T={len(hl_s):5d} η={eta} tie={n_tie} ρ={rho} ({time.time()-t1:.0f}s)")

    L.log(f"⭕ saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
