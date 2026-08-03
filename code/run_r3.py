#!/usr/bin/env python
"""Multi-prompt extension of the P axis (two prompts added).

Frozen pre-registration: ../frozen/PREREG_multi_prompt_p_axis.en.md
The frozen runner `run_c_lens_pos.py` is not edited by a single character;
its constants and helpers are imported. The certified results file is not
rewritten; output goes to a separate file.

That the instrument here is a faithful copy is **measured, not assumed** ---
two positive controls:
  * Control A (aggregator): recompute the relative-coordinate table from the
    main run and compare it against every published value.
  * Control B (measurement): re-measure an already certified trace on the
    deterministic attention path and compare against every stored value.
If either fails, the run stops before taking any new measurement
(pre-registered stopping condition 1).
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_c_lens_pos as C                                    # noqa: E402 ★ 凍結計器
import run_pc2                                                # noqa: E402
import run_f5                                                 # noqa: E402

REPO = C.REPO
RES = C.RES
SRC = RES / "c_lens_pos.json"                                 # ⛔ read only
SRC_R2 = RES / "c_lens_pos_r2.json"                           # ⛔ read only
DST = RES / "c_lens_pos_r3.json"
PREREG = "PREREG_multi_prompt_p_axis.en.md"

# ── PREREG §1 で凍結した標本（⛔ ここを走行中に変えない）────────────────────
PROMPTS = {                       # name -> (variant or None, 期待 md5, 期待 n_prompt)
    "C*":     (None,    "039389308493aa2f3580c2a0d7011f69", 458),
    "F5a-2":  ("F5a-2", "c79980dc3ece8144207f486e455c9496", 470),
}
SOURCES = {
    "C*":    REPO / "probes/olmo3-vg2-gpair-regen/results/vg2_runs.json",
    "F5a-2": REPO / "probes/olmo3-f5-think-prefix/results/f5_runs_F5a-2.json",
}
SEEDS_FROZEN = {"C*": [(1, 639), (2, 2351)], "F5a-2": [(1, 3145), (3, 595)]}
MIN_TOKENS = 512                  # ★ grid 要件（PREREG §1-2）
CONTROL_TEXT = "F5b-1:s6"         # ★ 統制 B（V-N）
GRID_B1 = [1, 19, 37, 55, 74, 92, 110, 128]
GRID_B2 = [129, 184, 238, 293, 348, 403, 457, 512]

# ── P / O bucket（PREREG §2: PREREG_r2 §1-1 から verbatim 継承）──────────────
P_EDGES = [("P1", 1, 128), ("P2", 129, 320), ("P3", 321, None)]     # None = n_prompt
O_EDGES = [("O1", 1, 128), ("O2", 129, 512), ("O3", 513, 2048),
           ("O4", 2049, 4096), ("O5", 4097, 8191), ("O6", 8192, 16384)]

log = C.log


def med(xs):
    xs = sorted(xs)
    return None if not xs else xs[len(xs) // 2]


def p_bucket(pos: int, n_prompt: int):
    if pos > n_prompt:
        return None
    for name, lo, hi in P_EDGES:
        if lo <= pos <= (n_prompt if hi is None else hi):
            return name
    return None


def o_bucket(pos: int, n_prompt: int):
    o = pos - n_prompt
    if o < 1:
        return None
    for name, lo, hi in O_EDGES:
        if lo <= o <= hi:
            return name
    return None


def aggregate(rows, axis: str):
    """The same aggregation (median) as the published re-aggregation. Reports n two ways --- rows and distinct cells --- as pre-registered."""
    order = [e[0] for e in (P_EDGES if axis == "P" else O_EDGES)]
    out = []
    for attn in ("eager", "sdpa"):
        base = {}
        for bname in order:
            rs = [r for r in rows if r["attn"] == attn and r["bkt"] == bname]
            if not rs:
                continue
            cell = {
                "axis": axis, "attn": attn, "bucket": bname,
                "n": len(rs),
                "n_distinct": len({(r["prompt_md5"], r["position"]) for r in rs}),
                "n_prompts": len({r["prompt_md5"] for r in rs}),
                "rankcorr": med([r["rankcorr"] for r in rs]),
                "seatrank": med([max(r["seatrank_hehi"], r["seatrank_apex"]) for r in rs]),
                "kl": med([r["kl"] for r in rs]), "top1": med([r["top1"] for r in rs]),
                "rankcorr_ll": med([r["rankcorr_ll"] for r in rs]),
                "rankcorr_full": med([r["rankcorr_full"] for r in rs]),
                "rankcorr_slid": med([r["rankcorr_slid"] for r in rs]),
            }
            if not base:
                base = {"rankcorr": cell["rankcorr"], "seatrank": cell["seatrank"]}
            cell["rankcorr_deg"] = 1 - cell["rankcorr"] / base["rankcorr"]
            cell["seatrank_deg"] = cell["seatrank"] - base["seatrank"]
            out.append(cell)
    return out


def control_a(stored) -> dict:
    """Control A: does the aggregator reproduce the published re-aggregation? (single prompt, length 501)"""
    saved = json.loads(SRC_R2.read_text())
    rows = []
    for m in stored:
        if m["group"] != "G":
            continue
        for axis, fn in (("P", p_bucket), ("O", o_bucket)):
            b = fn(m["position"], 501)
            if b:
                rows.append({**m, "bkt": b, "axis": axis,
                             "prompt_md5": "cc05d498c10ca9c4acf3b030a5884236"})
    got = (aggregate([r for r in rows if r["axis"] == "P"], "P")
           + aggregate([r for r in rows if r["axis"] == "O"], "O"))
    gi = {(c["axis"], c["attn"], c["bucket"]): c for c in got}
    fields = ["n", "rankcorr", "seatrank", "kl", "top1", "rankcorr_ll",
              "rankcorr_full", "rankcorr_slid", "rankcorr_deg", "seatrank_deg"]
    n_cmp, diffs = 0, []
    for c in saved["cells"]:
        k = (c["axis"], c["attn"], c["bucket"])
        g = gi.get(k)
        if g is None:
            diffs.append({"cell": k, "missing": True})
            continue
        for f in fields:
            n_cmp += 1
            a, b = c.get(f), g.get(f)
            if isinstance(a, float) and isinstance(b, float):
                if round(a, 5) != round(b, 5):        # ★ 保存値は 5 桁丸め
                    diffs.append({"cell": k, "field": f, "saved": a, "got": b})
            elif a != b:
                diffs.append({"cell": k, "field": f, "saved": a, "got": b})
    return {"control": "A（集約器）", "reference": "c_lens_pos_r2.json",
            "n_cells_compared": len(saved["cells"]), "n_values_compared": n_cmp,
            "n_differing": len(diffs), "diffs": diffs[:20],
            "verdict": "IDENTICAL" if not diffs else "MISMATCH"}


def build_texts(tok):
    """Build the pre-registered sample, asserting each prompt md5, prompt length, token count and position grid."""
    import screen_base as SB
    base = SB.load_prompts()[run_f5.CONDITION]
    out = []
    for name, (variant, md5_exp, npr_exp) in PROMPTS.items():
        full = (base if variant is None else run_f5.build_prompt(base, variant)[0]) + run_f5.CUE
        md5 = hashlib.md5(full.encode()).hexdigest()
        if md5 != md5_exp:                                    # ⛔ 中止条件 3
            raise SystemExit(f"⛔ 中止 3: {name} prompt md5 {md5} != {md5_exp}")
        npr = len(tok(full)["input_ids"])
        if npr != npr_exp:                                    # ⛔ 中止条件 3
            raise SystemExit(f"⛔ 中止 3: {name} n_prompt {npr} != {npr_exp}")
        doc = json.loads(SOURCES[name].read_text())
        by = {r["seed"]: r for r in doc["rows"]}
        for seed, nt_exp in SEEDS_FROZEN[name]:
            text = full + by[seed]["content"]
            nt = len(tok(text)["input_ids"])
            if nt != nt_exp:                                  # ⛔ 中止条件 4
                raise SystemExit(f"⛔ 中止 4: {name}:s{seed} n_tokens {nt} != {nt_exp}")
            if nt <= MIN_TOKENS:                              # ⛔ PREREG §1-2 の規則
                raise SystemExit(f"⛔ 中止: {name}:s{seed} n_tokens {nt} <= {MIN_TOKENS}")
            per = dict(C.positions_for(nt)[1])
            if per["B1"] != GRID_B1 or per["B2"] != GRID_B2:  # ⛔ 中止条件 5
                raise SystemExit(f"⛔ 中止 5: {name}:s{seed} grid 不整合 "
                                 f"B1={per['B1']} B2={per['B2']}")
            out.append({"group": "G", "name": f"{name}:s{seed}", "text": text,
                        "n_tokens": nt, "n_prompt": npr, "prompt": name,
                        "prompt_md5": md5, "seed": seed,
                        "source": str(SOURCES[name].relative_to(REPO))})
    return out


def measure(torch, transformers, jlens, tok, p2, mask, sets, texts, plan):
    """A verbatim copy of the frozen runner's measurement loop --- identical call form and identical aggregation. Its correctness is established by the two positive controls, not asserted."""
    SLIDING = [l for l in run_pc2.BAND if l not in C.FULL_LAYERS]
    bucket_of = {}
    for name, lo, hi in C.BUCKETS:
        for q in range(lo, hi + 1):
            bucket_of[q] = name

    def agg(mat, layers, how="median"):
        idx = [run_pc2.BAND.index(l) for l in layers]
        sub = mat[idx]
        v = sub.mean(0) if how == "mean" else sub.median(0).values
        return [float(x) for x in v]

    lens = jlens.JacobianLens.from_pretrained(str(run_pc2.LENS_PT))
    meas = []
    for attn, ms in plan:
        todo = [tx for tx in texts if attn in tx.get("attn_plan", ("eager", "sdpa"))]
        if not todo:
            continue
        log(f"=== attn={attn}  max_seq={ms} ===")
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            run_pc2.MODEL_NAME, dtype=torch.bfloat16, attn_implementation=attn).to("mps")
        hf.eval()
        model = jlens.from_hf(hf, tok)
        for tx in todo:
            pos = [q for q in C.positions_for(tx["n_tokens"])[0] if q < ms]
            if not pos:
                continue
            t0 = time.time()
            got = {}
            for use_j, key in ((True, "lens"), (False, "logitlens")):
                gc.collect(); torch.mps.empty_cache()
                jl, ml, _ = lens.apply(model, tx["text"], layers=run_pc2.BAND,
                                       positions=pos, max_seq_len=ms, use_jacobian=use_j)
                mlog = ml.float()
                mlp = torch.log_softmax(mlog, dim=-1)
                m_top1 = mlog.argmax(-1)
                m_topk = mlog.topk(C.TOPK, dim=-1).indices
                kls, t1s, rcs = [], [], []
                for layer in run_pc2.BAND:
                    L = jl[layer].float()
                    kls.append((mlp.exp() * (mlp - torch.log_softmax(L, dim=-1))).sum(-1))
                    t1s.append((L.argmax(-1) == m_top1).float())
                    rc = []
                    for q in range(mlog.shape[0]):
                        ids = m_topk[q]
                        lr = L[q, ids].argsort(descending=True).argsort().float()
                        mr = mlog[q, ids].argsort(descending=True).argsort().float()
                        lr = lr - lr.mean(); mr = mr - mr.mean()
                        rc.append((lr @ mr) / (lr.norm() * mr.norm() + 1e-12))
                    rcs.append(torch.stack(rc))
                K, T1, RC = torch.stack(kls), torch.stack(t1s), torch.stack(rcs)
                br = p2.band_median_ranks(torch, jl, mask, sets)
                mr_ = p2.band_median_ranks(torch, {run_pc2.BAND[-1]: mlog}, mask, sets)
                del jl, ml, mlog, mlp, m_topk, kls, t1s, rcs
                got[key] = {
                    "kl": agg(K, run_pc2.BAND), "top1": agg(T1, run_pc2.BAND, "mean"),
                    "rankcorr": agg(RC, run_pc2.BAND),
                    "kl_full": agg(K, C.FULL_LAYERS), "kl_slid": agg(K, SLIDING),
                    "rankcorr_full": agg(RC, C.FULL_LAYERS), "rankcorr_slid": agg(RC, SLIDING),
                    "seat": {s: [abs(math.log10(max(x, 1)) - math.log10(max(y, 1)))
                                 for x, y in zip(br[s], mr_[s])] for s in sets},
                }
            for i, q in enumerate(pos):
                meas.append({
                    "text": tx["name"], "group": tx["group"], "attn": attn,
                    "position": q, "bucket": bucket_of[q],
                    "kl": got["lens"]["kl"][i], "top1": got["lens"]["top1"][i],
                    "rankcorr": got["lens"]["rankcorr"][i],
                    "kl_full": got["lens"]["kl_full"][i], "kl_slid": got["lens"]["kl_slid"][i],
                    "rankcorr_full": got["lens"]["rankcorr_full"][i],
                    "rankcorr_slid": got["lens"]["rankcorr_slid"][i],
                    "seatrank_hehi": got["lens"]["seat"]["hehi"][i],
                    "seatrank_apex": got["lens"]["seat"]["apex"][i],
                    "seatrank_center": got["lens"]["seat"]["center"][i],
                    "kl_ll": got["logitlens"]["kl"][i],
                    "top1_ll": got["logitlens"]["top1"][i],
                    "rankcorr_ll": got["logitlens"]["rankcorr"][i],
                })
            del got
            gc.collect(); torch.mps.empty_cache()
            log(f"  {attn:<6} {tx['name']:<14} n_pos={len(pos):<3} ({time.time()-t0:.0f}s)  "
                f"mps={torch.mps.driver_allocated_memory()/2**30:.1f} GiB")
        del model, hf
        gc.collect(); torch.mps.empty_cache()
    return meas


def main() -> int:
    prov = C.provenance(allow_dirty=False)                    # ⛔ R7-a
    prov["probe"] = "R-3 (P 軸 多 prompt 拡張)"
    prov["prereg"] = PREREG
    log(f"R7-a ✅ commit={prov['commit'][:8]} dirty={prov['dirty']}  PREREG={PREREG}")

    stored_doc = json.loads(SRC.read_text())
    stored = stored_doc["measurements"]

    # ── ★ 統制 A（集約器）—— 走らせる前に片付く ──────────────────────────
    ca = control_a(stored)
    log(f"★ 統制 A（集約器 vs c_lens_pos_r2.json）: cell {ca['n_cells_compared']} × 指標 "
        f"= {ca['n_values_compared']} 値  差 {ca['n_differing']}  ⇒ ★ {ca['verdict']}")
    if ca["verdict"] != "IDENTICAL":                          # ⛔ 中止条件 1
        for d in ca["diffs"][:10]:
            log(f"   ⛔ {d}")
        raise SystemExit("⛔ 中止 1: 集約器が R-2 を再現しない")

    import torch, transformers, jlens                          # noqa: E401
    got_md5 = hashlib.md5(run_pc2.LENS_PT.read_bytes()).hexdigest()
    if got_md5 != run_pc2.LENS_MD5:                            # ⛔ 中止条件 2
        raise SystemExit(f"⛔ 中止 2: lens md5 {got_md5}")
    log(f"lens md5 ✅ {run_pc2.LENS_MD5}")

    tok = transformers.AutoTokenizer.from_pretrained(run_pc2.MODEL_NAME)
    p2 = run_pc2.p2mod()
    _ids = json.loads(run_pc2.MASK_PATH.read_text())
    mask = torch.zeros(len(tok), dtype=torch.bool); mask[_ids] = True
    sets = {"hehi": p2.set_ids(tok, p2.HEHI_WORDS), "apex": p2.set_ids(tok, p2.APEX_WORDS),
            "center": p2.set_ids(tok, p2.CENTER_WORDS)}
    log(f"mask/sets ✅ word-like={len(_ids)} hehi={len(sets['hehi'])} apex={len(sets['apex'])}")

    new_texts = build_texts(tok)
    for t in new_texts:
        log(f"   ⭕ {t['name']:<14} n_tokens={t['n_tokens']:<6} n_prompt={t['n_prompt']} "
            f"md5={t['prompt_md5'][:8]}")

    # ── ★ 統制 B（測定器・V-N）: 認定済み text を eager で再測 ────────────
    ctl = next(t for t in C.g_texts(tok) if t["name"] == CONTROL_TEXT)
    ctl = {**ctl, "prompt": "F5b-1", "seed": 6, "attn_plan": ("eager",),
           "prompt_md5": "cc05d498c10ca9c4acf3b030a5884236"}

    t_all = time.time()
    meas_ctl = measure(torch, transformers, jlens, tok, p2, mask, sets,
                       [ctl], [("eager", C.EAGER_MAX)])
    FIELDS = [k for k in meas_ctl[0] if isinstance(meas_ctl[0][k], float)]
    idx = {(m["attn"], m["text"], m["position"]): m for m in stored}
    n_cmp, cb_diffs = 0, []
    for m in meas_ctl:
        old = idx.get((m["attn"], m["text"], m["position"]))
        if old is None:
            cb_diffs.append({"position": m["position"], "missing_in_stored": True}); continue
        for f in FIELDS:
            n_cmp += 1
            if old.get(f) != m.get(f):
                cb_diffs.append({"position": m["position"], "field": f,
                                 "saved": old.get(f), "got": m.get(f)})
    cb = {"control": "B（測定器・V-N）", "reference": f"c_lens_pos.json / eager:{CONTROL_TEXT}",
          "n_positions": len(meas_ctl), "n_fields_per_position": len(FIELDS),
          "n_values_compared": n_cmp, "n_differing": len(cb_diffs), "diffs": cb_diffs[:20],
          "verdict": "IDENTICAL" if not cb_diffs else "MISMATCH"}
    log(f"★ 統制 B（V-N eager:{CONTROL_TEXT}）: 位置 {cb['n_positions']} × 指標 "
        f"{len(FIELDS)} = {n_cmp} 値  差 {len(cb_diffs)}  ⇒ ★ {cb['verdict']}")
    if cb["verdict"] != "IDENTICAL":                           # ⛔ 中止条件 1
        for d in cb_diffs[:10]:
            log(f"   ⛔ {d}")
        DST.write_text(json.dumps({"aborted": "中止条件 1（統制 B）", "control_a": ca,
                                   "control_b": cb, "provenance": prov},
                                  ensure_ascii=False, indent=1))
        raise SystemExit("⛔ 中止 1: 測定器が既存成果物を再現しない")

    # ── 本測定 ────────────────────────────────────────────────────────────
    meas_new = measure(torch, transformers, jlens, tok, p2, mask, sets, new_texts,
                       [("eager", C.EAGER_MAX), ("sdpa", C.MAX_SEQ_PROBE)])

    # ── P / O 軸の集約（新規 + 既存 F5b-1 の流用）──────────────────────────
    npr = {t["name"]: t["n_prompt"] for t in new_texts}
    pmd = {t["name"]: t["prompt_md5"] for t in new_texts}
    rows = []
    for m in meas_new:
        rows.append({**m, "prompt": m["text"].split(":")[0],
                     "prompt_md5": pmd[m["text"]], "n_prompt": npr[m["text"]]})
    for m in stored:                                   # ★ 既存 F5b-1（⛔ 再測しない）
        if m["group"] == "G":
            rows.append({**m, "prompt": "F5b-1", "n_prompt": 501,
                         "prompt_md5": "cc05d498c10ca9c4acf3b030a5884236"})

    axes = {}
    for axis, fn in (("P", p_bucket), ("O", o_bucket)):
        tagged = []
        for r in rows:
            b = fn(r["position"], r["n_prompt"])
            if b:
                tagged.append({**r, "bkt": b})
        axes[axis] = {"all": aggregate(tagged, axis),
                      "by_prompt": {p: aggregate([r for r in tagged if r["prompt"] == p], axis)
                                    for p in sorted({r["prompt"] for r in tagged})}}

    DST.write_text(json.dumps({
        "check": "R-3 P 軸の多 prompt 拡張（帰属診断・⛔ 判定は動かさない）",
        "prereg": PREREG,
        "adjudication": "ADJUDICATION_faithfulprose_results_framing_20260802.md §4",
        "ratified_by": "先生 2026-08-03",
        "source_existing": "c_lens_pos.json（F5b-1 4 本を流用・⛔ 再測なし）",
        "prompts": [{"name": k, "md5": v[1], "n_prompt": v[2],
                     "source": str(SOURCES[k].relative_to(REPO))} for k, v in PROMPTS.items()],
        "P_buckets": [[e[0], e[1], e[2] if e[2] else "n_prompt"] for e in P_EDGES],
        "O_buckets": O_EDGES, "seed_rule": f"seed 昇順・n_tokens > {MIN_TOKENS} の先頭 2 本",
        "seeds_frozen": SEEDS_FROZEN,
        "texts": [{k: v for k, v in t.items() if k != "text"} for t in new_texts],
        "control_a": ca, "control_b": cb,
        "axes": axes, "wall_s": round(time.time() - t_all, 1),
        "provenance": prov, "measurements": meas_new,
    }, ensure_ascii=False, indent=1))
    log(f"-> {DST}   新規測定 {len(meas_new)}  ({time.time()-t_all:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
