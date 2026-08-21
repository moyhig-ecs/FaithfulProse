#!/usr/bin/env python3
"""`F5` lens 段 —— 生成済み variant の 9 行にword-group readout読み出しと `hehi_suffix_best` を掛ける。

指示: [ADJUDICATION_f5a2_wave1_received §3](
      ../../conversations/2026-08-01/ADJUDICATION_f5a2_wave1_received_20260801.md)（即実施・全 9 行）
的:   [DESIGN_phase0 §4](../../conversations/2026-08-01/DESIGN_phase0_hehi_positive_control_20260801.md)
      —— 主要観測量は **`hehi_suffix_best`（連続値）**。⛔ 二値だと失敗が情報を持たない。

⛔ 計器は書き直さない（family #11）:
  - `run_pc2.py` から MODEL_NAME / LENS_PT / LENS_MD5 / BAND / MAX_SEQ / MASK_PATH / p2mod
  - `p2`（`step3_pass2.py`）から set_ids / band_median_ranks / _establish
  - `tripwire.py` から **`spot_check_S1`**（= `seat_of` の verbatim 経路・selftest 済み）
  - `run_f5.py` から `build_prompt`（★ variant の prompt を同一手順で再構成する）

★ `established_group = (hehi_suffix_best >= 0.80) かつ (seat_S == "HEHI")` —— 二重条件
  （[INSTR_b3_v2 §0-9](../../conversations/2026-08-01/INSTR_b3_v2_gpair_tripwire_and_unified_p0_20260801.md)）

⚠ 汚染の扱い（[DESIGN_phase0 §2-2](../../conversations/2026-08-01/DESIGN_phase0_hehi_positive_control_20260801.md)）:
  `T < 200` と `lens_truncated` は **除外せず記録**し、集計時に**適格/非適格を並記**する。
  ⇒ ⛔ 黙って落とさない。

★ `R7-a` 適用。⛔ `phase = 0`・探索・非証拠。

usage:
    python3 run_f5_lens.py --variant F5a-2
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
PC2 = REPO / "probes/olmo3-pc2-seat-cue/run_pc2.py"
sys.path.insert(0, str(REPO / "probes" / "olmo3-base-bf16-screen"))
sys.path.insert(0, str(REPO / "probes" / "exploration-b"))
sys.path.insert(0, str(HERE))

CUE, CONDITION = "\n\n", "blocked"
ADJUDICATION = "ADJUDICATION_f5a2_wave1_received_20260801.md §3"


def log(m: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def load_pc2():
    spec = importlib.util.spec_from_file_location("pc2_for_f5", PC2)
    m = importlib.util.module_from_spec(spec)
    sys.modules["pc2_for_f5"] = m
    spec.loader.exec_module(m)
    return m


def provenance(variant: str, allow_dirty: bool) -> dict:
    def git(*a):
        return subprocess.check_output(["git", "-C", str(REPO), *a]).decode().strip()
    lines = git("status", "--porcelain").splitlines()
    excluded = [l for l in lines if l.startswith("??") and OWN_RESULTS in l]
    dirty_lines = [l for l in lines if l not in excluded]
    dirty = bool(dirty_lines)
    prov = {"commit": git("rev-parse", "HEAD"), "dirty": dirty, "dirty_files": dirty_lines[:20],
            "dirty_check_excluded": excluded,
            "dirty_check_exclusion_rule": f"untracked under {OWN_RESULTS} only",
            "resume": False, "run_kind": "main", "stage": "lens", "variant": variant,
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
                        **{n: ver(n) for n in ("torch", "transformers", "numpy")}}
    if dirty and not allow_dirty:
        raise SystemExit("⛔ R7-a: dirty tree では本走行しない。\n   未コミット "
                         f"{len(dirty_lines)} 件:\n     " + "\n     ".join(dirty_lines[:10]))
    return prov


def hehi_suffix_best(traj: dict) -> tuple[float, int | None, float]:
    """`_establish` が要求する形（`need = max(20, 0.1T)` 以上の suffix）での
    dominance 率の**最大値** = ★ 確立への肉薄度（DESIGN_phase0 §2-2）。"""
    T = traj["T"]
    h, a = traj["hehi"], traj["apex"]
    need = max(20, int(0.1 * T))

    def best(dom):
        b, at = 0.0, None
        for t in range(T):
            if not dom[t]:
                continue
            seg = dom[t:]
            if len(seg) < need:
                break
            f = sum(seg) / len(seg)
            if f > b:
                b, at = f, t
        return b, at

    bh, ath = best([h[t] < a[t] for t in range(T)])
    ba, _ = best([a[t] < h[t] for t in range(T)])
    return bh, ath, ba


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    prov = provenance(args.variant, args.allow_dirty)
    log(f"R7-a ✅ commit={prov['commit'][:8]} dirty={prov['dirty']} stage=lens variant={args.variant}")

    pc2 = load_pc2()
    import screen_base as SB
    from run_f5 import build_prompt
    from tripwire import spot_check_S1            # ★ seat_of の verbatim 経路（selftest 済み）

    src = RES / f"f5_runs_{args.variant}.json"
    doc = json.loads(src.read_text())
    rows = {r["seed"]: r for r in doc["rows"]}

    base = SB.load_prompts()[CONDITION]
    newbase, _ = build_prompt(base, args.variant)
    prompt_text = newbase + CUE
    md5 = hashlib.md5(prompt_text.encode()).hexdigest()
    if md5 != doc["prompt_md5_new"]:
        raise SystemExit(f"⛔ prompt md5 不一致: {md5} != {doc['prompt_md5_new']}")
    log(f"prompt md5 ✅ {md5}（生成時と同一）")

    import torch, transformers, jlens
    got = hashlib.md5(pc2.LENS_PT.read_bytes()).hexdigest()
    if got != pc2.LENS_MD5:
        raise SystemExit(f"⛔ lens md5 不一致: {got}")
    log(f"lens md5 ✅ {pc2.LENS_MD5}")

    t0 = time.time()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        pc2.MODEL_NAME, dtype=torch.bfloat16, attn_implementation="eager").to("mps")
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(pc2.MODEL_NAME)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(str(pc2.LENS_PT))
    p2 = pc2.p2mod()
    log(f"stack ✅ ({time.time()-t0:.0f}s) BAND={pc2.BAND[0]}..{pc2.BAND[-1]} MAX_SEQ={pc2.MAX_SEQ}")

    ids = json.loads(pc2.MASK_PATH.read_text())        # ⛔ 既存 mask を再利用（作り直さない）
    mask = torch.zeros(len(tok), dtype=torch.bool); mask[ids] = True
    sets = {"hehi": p2.set_ids(tok, p2.HEHI_WORDS),
            "apex": p2.set_ids(tok, p2.APEX_WORDS),
            "center": p2.set_ids(tok, p2.CENTER_WORDS)}

    out_path = RES / f"f5_lens_{args.variant}.json"
    doc_out = json.loads(out_path.read_text()) if out_path.exists() else {
        "adjudication": ADJUDICATION, "phase": 0, "variant": args.variant,
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
            doc_out["rows"].append({"seed": seed, "excluded": "label=None or 空",
                                    "label": r.get("label")})
            log(f"  skip s{seed}: 測定不成立")
            continue
        full = prompt_text + r["content"]
        T = len(tok(full)["input_ids"]) - n_prompt
        positions = [p for p in range(n_prompt, n_prompt + T) if p < pc2.MAX_SEQ]
        truncated = len(positions) < T
        t0 = time.time()
        traj = {s: [] for s in sets}
        for i in range(0, len(positions), p2.CHUNK):
            chunk = positions[i:i + p2.CHUNK]
            jl, _a, _b = lens.apply(model, full, layers=pc2.BAND,
                                    positions=chunk, max_seq_len=pc2.MAX_SEQ)
            med = p2.band_median_ranks(torch, jl, mask, sets)
            for s in sets:
                traj[s].extend(med[s])
        tr = {"T": len(traj["hehi"]), "hehi": traj["hehi"], "apex": traj["apex"],
              "center": traj["center"], "lens_truncated": truncated}
        seat = spot_check_S1(tr)                    # ★ seat_of の verbatim 経路
        bh, ath, ba = hehi_suffix_best(tr)
        est = (bh >= 0.80) and (seat["seat_S"] == "HEHI")
        doc_out["rows"].append({
            "seed": seed, "label": r["label"], "label_conj_only": r.get("label_conj_only"),
            "tau_brace": r.get("tau_brace"), "n_chars": r["n_chars"],
            "T": tr["T"], "T_tokens": T, "lens_truncated": truncated,
            "eligible": (tr["T"] >= 200 and not truncated),   # ⛔ 除外せず記録
            "seat_S": seat["seat_S"], "tau": seat["tau"],
            "t_hehi": seat["t_hehi"], "t_apex": seat["t_apex"], "need": seat["need"],
            "hehi_suffix_best": round(bh, 6), "hehi_suffix_at": ath,
            "apex_suffix_best": round(ba, 6),
            "established_group": est,
            "wall_s": round(time.time() - t0, 1),
            "hehi": traj["hehi"], "apex": traj["apex"], "center": traj["center"]})  # S3 併用
        out_path.write_text(json.dumps(doc_out, ensure_ascii=False, indent=1) + "\n")
        x = doc_out["rows"][-1]
        log(f"  s{seed} {r['label']:13s} T={tr['T']:5d} seat={x['seat_S']:5s} "
            f"hehi_best={bh:.3f} apex_best={ba:.3f} est={est} ({x['wall_s']:.0f}s)")

    # ── 集計（⛔ 適格/非適格を並記）─────────────────────────────────
    from collections import Counter
    ok = [r for r in doc_out["rows"] if "seat_S" in r]
    elig = [r for r in ok if r["eligible"]]
    rep = {"n_measured": len(ok), "n_eligible": len(elig),
           "seat_all": dict(Counter(r["seat_S"] for r in ok)),
           "seat_eligible": dict(Counter(r["seat_S"] for r in elig)),
           "established_group_count": sum(1 for r in ok if r["established_group"]),
           "hehi_suffix_best_all": sorted((r["seed"], r["hehi_suffix_best"]) for r in ok),
           "hehi_suffix_best_max_eligible": max((r["hehi_suffix_best"] for r in elig), default=None),
           "excluded_reason": {"T<200": sum(1 for r in ok if r["T"] < 200),
                               "lens_truncated": sum(1 for r in ok if r["lens_truncated"])}}
    doc_out["lens_report"] = rep
    out_path.write_text(json.dumps(doc_out, ensure_ascii=False, indent=1) + "\n")
    log(f"\n★ lens 報告  測定 {rep['n_measured']} / 適格 {rep['n_eligible']}")
    log(f"  word-group readout（全）    {rep['seat_all']}")
    log(f"  word-group readout（適格）  {rep['seat_eligible']}")
    log(f"  ★ established_group = {rep['established_group_count']}")
    log(f"  hehi_suffix_best（適格の最大）= {rep['hehi_suffix_best_max_eligible']}")
    log(f"  除外内訳 {rep['excluded_reason']}")
    log(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
