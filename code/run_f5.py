#!/usr/bin/env python3
"""Driver from a separate, unpublished research programme that uses the same
instrument.

For the present paper this module supplies only the prompt construction used to
rebuild the measured texts: the condition selector, the cue string, and
`build_prompt`, which applies an edit variant to the base prompt. The variant
definitions are retained because the measured prompts are exactly their output
and could not otherwise be reconstructed.

The remainder --- the generation loop, the verdict categories and the analysis
--- belongs to that other programme and is included unchanged so that the
imports resolve and the run reproduces byte-for-byte.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
OWN_RESULTS = "probes/olmo3-f5-think-prefix/results/"
sys.path.insert(0, str(REPO / "probes" / "olmo3-base-bf16-screen"))
sys.path.insert(0, str(REPO / "ConstructGap"))
sys.path.insert(0, str(REPO / "ConstructGap" / "exp"))

# ── 凍結諸元（v2 §3）⛔ 変更禁止 ─────────────────────────────────────
CUE, CUE_TAG, CONDITION, TEMP = "\n\n", "nlnl", "blocked", 0.7
SEEDS = list(range(1, 10))
ADJUDICATION = "ADJUDICATION_f5_v2_effective_20260801.md"

# ── ⑥ / ② / ④ / ⑤ の置換キー（★ いずれも prompt 内で一意・preflight が assert）──
K6 = "Respond with JSON only:"
K2 = ("# The four common tangent lines of circles K1, K2 meet pairwise. The four MIXED "
      "intersections (one external + one internal tangent) are X1,X2,X3,X4; the two "
      "same-type intersections are He, Hi.")
K4 = "The two circles are disjoint and have four common tangents;"
K5 = "as a relation (concyclic / equal length / collinear / right angle)"

# ── variant 定義（v2 §1-2 / §2・裁可済み英文 verbatim）────────────────
R2_SEP = ("# The four common tangent lines of circles K1, K2 meet pairwise. Two of these four "
          "lines separate K1 from K2 (the two circles lie on opposite sides of the line); the "
          "other two do not (both circles lie on the same side). The four intersections where a "
          "separating line meets a non-separating line are X1,X2,X3,X4; the two intersections "
          "formed by lines of the same kind are He, Hi.")
R2_FUNC = ("# Four lines each touch both K1 and K2, and they meet pairwise. Two of them separate "
           "K1 from K2 (the two circles lie on opposite sides of the line); the other two do not. "
           "The four intersections where a separating line meets a non-separating line are "
           "X1,X2,X3,X4; the two intersections formed by lines of the same kind are He, Hi.")

VARIANTS: dict[str, dict] = {
    "F5a-1": {"family": "F5a", "single_variable": "⑥ output-order instruction (step-by-step)",
              "edits": [(K6, "Think step by step, then respond with the JSON object:")]},
    "F5a-2": {"family": "F5a", "single_variable": "⑥ output-order instruction (reasoning first)",
              "edits": [(K6, "First write your reasoning as plain text.\n"
                             "After that, respond with the JSON object:")]},
    "F5b-1": {"family": "F5b", "single_variable": "② tangent-type classification → separation register",
              "edits": [(K2, R2_SEP)]},
    "F5b-2": {"family": "F5b", "single_variable": "F5b-1 ⊇ + 'common tangent' → functional description",
              "edits": [(K2, R2_FUNC),
                        (K4, "The two circles are disjoint and there are four such lines;")]},
    "F5b-3": {"family": "F5b", "single_variable": "F5b-2 ⊇ + ⑤ answer-type menu removed",
              "edits": [(K2, R2_FUNC),
                        (K4, "The two circles are disjoint and there are four such lines;"),
                        (K5, "as a relation")]},
}


def log(m: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def provenance(allow_dirty: bool, variant: str) -> dict:
    """Provenance guard: only untracked files under this probe's own results directory are exempt from the clean-tree requirement; the exempted lines and the rule are recorded."""
    def git(*a):
        return subprocess.check_output(["git", "-C", str(REPO), *a]).decode().strip()

    lines = git("status", "--porcelain").splitlines()
    excluded = [l for l in lines if l.startswith("??") and OWN_RESULTS in l]
    dirty_lines = [l for l in lines if l not in excluded]
    dirty = bool(dirty_lines)
    prov = {"commit": git("rev-parse", "HEAD"), "dirty": dirty, "dirty_files": dirty_lines[:20],
            "dirty_check_excluded": excluded,
            "dirty_check_exclusion_rule": f"untracked under {OWN_RESULTS} only",
            "resume": False, "run_kind": "main", "variant": variant,
            "adjudication": ADJUDICATION,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    from importlib import metadata as _md

    def ver(n):
        try:
            v = getattr(__import__(n), "__version__", None)
            if v:
                return str(v)
        except Exception as e:
            return f"<import:{type(e).__name__}>"
        for d in (n, n.replace("_", "-")):
            try:
                return _md.version(d)
            except Exception:
                continue
        return "<unknown>"

    prov["versions"] = {"python": sys.version.split()[0],
                        **{n: ver(n) for n in ("mlx", "mlx_lm", "transformers", "numpy")}}
    import platform
    prov["device"] = {"machine": platform.machine(), "platform": platform.platform(),
                      "node": platform.node()}
    if dirty and not allow_dirty:
        raise SystemExit("⛔ R7-a: dirty tree では本走行しない。\n   未コミット "
                         f"{len(dirty_lines)} 件:\n     " + "\n     ".join(dirty_lines[:10]))
    return prov


def build_prompt(base: str, variant: str) -> tuple[str, list[dict]]:
    """Keep the single-variable discipline: apply only the defined substitutions, asserting that each key is unique."""
    spec = VARIANTS[variant]
    out, applied = base, []
    for key, rep in spec["edits"]:
        n = out.count(key)
        if n != 1:
            raise SystemExit(f"⛔ 置換キーが一意でない（{n} 回）: {key[:60]!r}")
        out = out.replace(key, rep, 1)
        applied.append({"key": key, "replacement": rep})
    return out, applied


def analyze(gen: str, tok=None, n_prompt: int | None = None, full: str | None = None) -> dict:
    """Three label columns plus the two derived quantities. The classifier is not reimplemented."""
    from run_moy1_concyclic_blocked import classify, parse_json_field
    conj = parse_json_field(gen, "conjecture")
    reason = parse_json_field(gen, "reason")
    rec = {
        "label": classify(gen, "")[0],                       # ⛔ 公式・凍結計器
        "label_json_fields": classify(conj, reason)[0],
        "label_conj_only": classify(conj, "")[0],
        "target": classify(gen, "")[1],
        "n_brace": gen.count("{"),
    }
    # ★ 2026-08-02 訂正（ADJUDICATION_t_backfill_and_driver_fix §2）:
    #   T は τ の計算 block の内側にあったため、`{` が 1 個も無い行では未計算のまま
    #   落ちていた（F5a-2 s2 / F5b-1-ext s14 の 2 行）。T は brace の有無と独立に存在する。
    #   ⇒ T を前段へ出す。⛔ τ / n_brace / 3 列の挙動は不変。
    have_tok = tok is not None and full is not None and n_prompt is not None
    if have_tok:
        rec["T_tokens"] = len(tok(full)["input_ids"]) - n_prompt
    i = gen.find("{")
    if i >= 0 and have_tok:
        T = rec["T_tokens"]
        tpos = len(tok(full[:len(full) - len(gen) + i])["input_ids"]) - n_prompt
        rec["brace_token_pos"] = tpos
        rec["tau_brace"] = round(tpos / T, 6) if T else None
    else:
        rec["tau_brace"] = None
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="⛔ 本走行では使わない")
    # ★ seed 拡張（ADJUDICATION_closure_locate_read §3 指示 3）。⛔ prompt・計器・解析は一切変えない。
    #    認定済み 9 行の成果物を汚さないため、出力先は --out-tag で分ける。
    ap.add_argument("--seeds", default=None,
                    help="例 10-18 または 10,11,12。既定は SEEDS（1..9）")
    ap.add_argument("--out-tag", default=None,
                    help="成果物名 f5_runs_<tag>.json の tag。既定は variant")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    if args.seeds:
        if "-" in args.seeds and "," not in args.seeds:
            a, b = args.seeds.split("-")
            seeds = list(range(int(a), int(b) + 1))
        else:
            seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    else:
        seeds = list(SEEDS)
    out_tag = args.out_tag or args.variant
    prov = provenance(args.allow_dirty, args.variant)
    log(f"R7-a ✅ commit={prov['commit'][:8]} dirty={prov['dirty']} variant={args.variant} "
        f"seeds={seeds[0]}..{seeds[-1]} out_tag={out_tag}")

    import screen_base as SB
    base = SB.load_prompts()[CONDITION]                      # ★ md5 assert はこの中
    base_full = base + CUE
    new_base, applied = build_prompt(base, args.variant)
    new_full = new_base + CUE
    md5_old = hashlib.md5(base_full.encode()).hexdigest()
    md5_new = hashlib.md5(new_full.encode()).hexdigest()
    diff = "\n".join(difflib.unified_diff(base.splitlines(), new_base.splitlines(),
                                          "current", args.variant, lineterm="", n=1))
    log(f"payload md5 ✅ 窓内と一致  base_full={md5_old}")
    log(f"★ prompt_md5_new = {md5_new}   単一変数 = {VARIANTS[args.variant]['single_variable']}")
    log(f"置換 {len(applied)} 箇所（キーはいずれも一意）")

    meta = {"adjudication": ADJUDICATION, "variant": args.variant,
            "family": VARIANTS[args.variant]["family"],
            "single_variable": VARIANTS[args.variant]["single_variable"],
            "symbols_unchanged": True,
            "prompt_md5_current": md5_old, "prompt_md5_new": md5_new,
            "prompt_verbatim_diff": diff, "edits_applied": applied,
            "condition": CONDITION, "cue": CUE, "cue_tag": CUE_TAG,
            "temp": TEMP, "seeds": seeds, "max_new": SB.MAX_NEW,
            "model": SB.MODEL_NAME, "flow": "base-bf16-no-chat-template",
            "provenance": prov, "phase": 0}

    if args.preflight:
        (RES / f"f5_preflight_{args.variant}.json").write_text(
            json.dumps({**meta, "prompt_new": new_full}, ensure_ascii=False, indent=1) + "\n")
        print("\n--- prompt diff ---\n" + diff)
        log("preflight のみ ⇒ 生成せず終了")
        return 0

    import mlx.core as mx
    from mlx_lm import load as mlx_load, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    import transformers

    t0 = time.time()
    model, tok_mlx = mlx_load(SB.MODEL_NAME)
    tok = transformers.AutoTokenizer.from_pretrained(SB.MODEL_NAME)
    n_prompt = len(tok(new_full)["input_ids"])
    log(f"load 完了 ({time.time()-t0:.0f}s)  n_prompt={n_prompt}")

    out_path = RES / f"f5_runs_{out_tag}.json"
    doc = json.loads(out_path.read_text()) if out_path.exists() else {**meta, "rows": []}
    have = {r["seed"] for r in doc["rows"]}

    for seed in seeds:
        if seed in have:
            continue
        mx.random.seed(seed)
        t0 = time.time()
        try:
            raw = mlx_generate(model, tok_mlx, prompt=new_full, max_tokens=SB.MAX_NEW,
                               sampler=make_sampler(temp=TEMP), verbose=False)
            err = None
        except Exception as e:
            raw, err = "", f"{type(e).__name__}: {str(e)[:300]}"
        gen = (raw or "").strip()
        rec = analyze(gen, tok, n_prompt, new_full + gen) if gen else {"label": None}
        doc["rows"].append({
            "phase": 0, "family": meta["family"], "variant_id": args.variant,
            "single_variable": meta["single_variable"], "symbols_unchanged": True,
            "prompt_md5_new": md5_new,
            "cue_tag": CUE_TAG, "condition": CONDITION, "seed": seed, "temp": TEMP,
            "err": err, "content": gen, "n_chars": len(gen),
            "hit_cap": len(tok_mlx.encode(gen)) >= SB.MAX_NEW - 8 if gen else False,
            "max_new": SB.MAX_NEW, "wall_s": round(time.time() - t0, 1),
            "commit": prov["commit"], "dirty": prov["dirty"],
            "datetime_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **rec})
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
        r = doc["rows"][-1]
        log(f"  [s{seed}] {r['label']} | json={r.get('label_json_fields')} | "
            f"conj={r.get('label_conj_only')}  τ={r.get('tau_brace')} n_brace={r.get('n_brace')} "
            f"cap={r['hit_cap']} ({r['wall_s']:.0f}s, {len(gen)} chars)")

    # ── 中止条件の判定（v2 §5 + 裁定 §2）───────────────────────────
    from collections import Counter
    rows = doc["rows"]
    cols = {c: dict(Counter(r.get(c) for r in rows))
            for c in ("label", "label_json_fields", "label_conj_only")}
    flail = sum(1 for r in rows if r.get("label") == "B-flail")     # ★ 公式列で数える
    taus = [r.get("tau_brace") for r in rows if r.get("tau_brace") is not None]
    doc["wave_report"] = {
        "n": len(rows), "label_distributions": cols,
        "tau_brace": {"values": taus,
                      "n_zero": sum(1 for t in taus if t == 0.0),
                      "n_positive": sum(1 for t in taus if t and t > 0)},
        "n_brace": dict(Counter(r.get("n_brace") for r in rows)),
        "stop_conditions": {
            "c2_flail_ge_3of9": {"fired": flail >= 3, "count": flail,
                                 "column": "label (公式・classify(gen,\"\"))"},
            "c3_split_report": {"note": "tau>0 両側の split は即時報告。⚠ 適格判定は別裁定"
                                        "（結論部での He/Hi commit）"},
            "c4_established_group": {"note": "lens 段で判定（本 driver は生成段のみ）"},
        },
        "axis_reading_note": ("★ F5a では τ 移動 ≈ 指示追従（retrieval 仮説への証拠力は弱い）。"
                             "F5b でこそ τ=0 跳躍の除去が retrieval 仮説の検定になる（裁定 §2-3）"),
    }
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")

    w = doc["wave_report"]
    log(f"\n★ 波報告 n={w['n']}")
    for c, d in w["label_distributions"].items():
        log(f"  {c:18s} {d}")
    log(f"  tau_brace: τ=0 が {w['tau_brace']['n_zero']} / τ>0 が {w['tau_brace']['n_positive']}")
    log(f"  n_brace: {w['n_brace']}")
    log(f"  中止条件 2（flail>=3/9・公式列）: {'★ 発火' if flail >= 3 else '不発火'}（flail={flail}）")
    log(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
