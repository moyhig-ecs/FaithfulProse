#!/usr/bin/env python
"""Loader from a separate, unpublished research programme that uses the same
instrument.

For the present paper this module supplies only `load_prompts()`, which reads
the frozen prompt configuration and **asserts the md5 of each prompt** before
returning it. That assertion is why the measured texts can be rebuilt exactly:
if the configuration had drifted, the loader would refuse.

The remainder --- the behavioural screen it was written for --- belongs to that
other programme and is included unchanged so that the imports resolve.
"""
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path

REPO = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
CG = REPO / "ConstructGap"
sys.path.insert(0, str(CG)); sys.path.insert(0, str(CG / "exp"))
from run_moy1_concyclic_blocked import classify                  # noqa: E402

HERE = Path(__file__).parent
RES = HERE / "results"; RES.mkdir(parents=True, exist_ok=True)
FROZEN_CFG = REPO / "probes/moy1-claude5-replication/results/replication_config_frozen.json"

MODEL_NAME = "allenai/Olmo-3-1125-32B"        # ★ lens が張られている物体（base）
WINDOW_MD5 = {"open": "3c7a63dd0d2b4393820e7836458e33cd",
              "blocked": "616682f531557c56d1103738b8d9dbda"}
CONDITIONS = ["blocked", "open"]              # blocked 先行（決め手）
RUNS = [("temp0", 0.0, 0)] + [("temp0.7", 0.7, s) for s in range(1, 10)]
MAX_NEW = 32768                               # E3
CUE = "\n"                                    # E5: 最小・非誘導の continuation cue


def log(m): print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def load_prompts():
    cfg = json.loads(FROZEN_CFG.read_text())
    pr = {k: cfg["prompts"][k] for k in CONDITIONS}
    for k, v in pr.items():
        got = hashlib.md5(v.encode()).hexdigest()
        if got != WINDOW_MD5[k]:
            raise SystemExit(f"⛔ prompt md5 不一致 {k}: {got} != {WINDOW_MD5[k]}")
    return pr


def main():
    trial = len(sys.argv) > 1 and sys.argv[1] == "trial"
    prompts = load_prompts()
    out = RES / ("trial.json" if trial else "screen_olmo3_base_bf16.json")
    rows = json.loads(out.read_text())["rows"] if out.exists() else []
    have = {(r["condition"], r["tag"], r["seed"]) for r in rows}
    log(f"model={MODEL_NAME} (base bf16)  prompt md5 ✅ 窓内と一致  既存 {len(rows)} 行")

    import mlx.core as mx
    from mlx_lm import load as mlx_load, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    t0 = time.time()
    model, tok = mlx_load(MODEL_NAME)
    log(f"load 完了 ({time.time()-t0:.0f}s)")

    # E1: base に chat template は無い。あれば流儀の前提が崩れるので明示的に検査する
    has_ct = getattr(tok, "chat_template", None)
    log(f"chat_template: {'⚠ あり（前提と違う）' if has_ct else 'なし ✅（base として扱う）'}")

    plan = [(CONDITIONS[0], *RUNS[0])] if trial else [(c, *r) for c in CONDITIONS for r in RUNS]
    for cond, tag, temp, seed in plan:
        if (cond, tag, seed) in have:
            continue
        text = prompts[cond] + CUE             # E1: chat template なし / E5: cue を後置
        mx.random.seed(seed)
        t0 = time.time()
        try:
            raw = mlx_generate(model, tok, prompt=text, max_tokens=MAX_NEW,
                               sampler=make_sampler(temp=temp), verbose=False)
            err = None
        except Exception as e:
            raw, err = "", f"{type(e).__name__}: {str(e)[:300]}"
        gen = (raw or "").strip()
        lab, tgt = (classify(gen, "") if gen else (None, None))   # E2: 全文を分類
        rows.append({"model": MODEL_NAME, "flow": "base-bf16-no-chat-template",
                     "cue": CUE,                                   # E5: 明示記録
                     "condition": cond, "tag": tag, "seed": seed, "temp": temp,
                     "label": lab, "target": tgt, "err": err,
                     "content": gen, "n_chars": len(gen),
                     "hit_cap": len(tok.encode(gen)) >= MAX_NEW - 8 if gen else False,
                     "max_new": MAX_NEW,
                     "wall_s": round(time.time() - t0, 1),
                     "datetime_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        out.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=1) + "\n")
        log(f"  {cond:<8} {tag:<8} s={seed}: {lab}  ({rows[-1]['wall_s']}s, {len(gen)} 字)"
            + (" ⚠上限到達" if rows[-1]["hit_cap"] else "") + (f"  ERR {err}" if err else ""))
        if trial:
            print("\n=== 生成全文（先頭 3000 字）===\n")
            print(gen[:3000])
            print(f"\n=== 以上（全 {len(gen)} 字）===")
            print(f"\n所要 {rows[-1]['wall_s']}s/件 → 20 件で約 "
                  f"{rows[-1]['wall_s']*20/3600:.1f} 時間の見込み")

    if not trial:
        print(f"\n=== {MODEL_NAME} (base bf16) —— moy-1 label 分布 ===")
        for cond in CONDITIONS:
            s = [r for r in rows if r["condition"] == cond]
            if not s: continue
            nb = sum(1 for r in s if r["label"] == "B-bind")
            cap = sum(1 for r in s if r.get("hit_cap"))
            print(f"  {cond:<8} n={len(s):>2}  {dict(Counter(r['label'] for r in s))}"
                  f"   **bind {nb}/{len(s)}**   上限到達 {cap}")
        print("\n  ⛔ Q4 screen（Think 系・16384）とは数値比較しない —— 計器が違う")
        print("  出口: (i) bind 優勢/attractor≈0 → 部分代替 / (ii) 割れる → 完全代替候補 /"
              " (iii) 乗らない → 判断層へ差し戻し")
    print(f"\nartifact -> {out}")


if __name__ == "__main__":
    main()
