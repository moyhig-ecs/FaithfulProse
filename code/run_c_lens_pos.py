#!/usr/bin/env python3
"""`C-lens-pos` --- measure how lens fidelity varies with absolute token position.

Frozen design: ../frozen/DECISION_TABLE.en.md
  * `seatrank` is decided on an **absolute delta in dex** (thresholds 0.10 / 0.30).
    No ratio is applied to it.
  * `top1(B6) >= top1(B5) - 5 points (absolute)`.

The lens, the word groups, the rank computation and the token mask are
**imported verbatim** from the upstream implementation; none of them is
reimplemented here.
  - from `run_pc2`: MODEL_NAME / LENS_PT / LENS_MD5 / BAND / MASK_PATH / p2mod
  - from `p2` (= `step3_pass2`): WORDLIKE / set_ids / band_median_ranks /
    the two word groups
All buckets are read in a single forward pass (`positions` is an arbitrary set).

This script does not decide anything: the frozen decision table is applied
mechanically, afterwards, in the write-up.

usage:
    python3 run_c_lens_pos.py --preflight       # cost only; does not measure
    python3 run_c_lens_pos.py                   # main run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "probes" / "olmo3-pc2-seat-cue"))
sys.path.insert(0, str(REPO / "probes" / "olmo3-base-bf16-screen"))
sys.path.insert(0, str(REPO / "probes" / "olmo3-f5-think-prefix"))

import run_pc2                                   # noqa: E402  ★ 凍結定数の出所
import run_f5                                    # noqa: E402  ★ CUE / CONDITION / build_prompt

RES = HERE / "results"
DST = RES / "c_lens_pos.json"

# ── 凍結された設計定数（DESIGN v2 §3 / §4）────────────────────────────────
BUCKETS = [("B1", 1, 128), ("B2", 129, 512), ("B3", 513, 2048),
           ("B4", 2049, 4096), ("B5", 4097, 8191), ("B6", 8192, 16384)]
PER_BUCKET = 8                 # bucket 内で等間隔 8 点
MAX_SEQ_PROBE = 16384          # ★ B6 に届かせるため（model max_position_embeddings = 65,536）
TOPK = 1000                    # rankcorr の頭部 K
G_ROWS = [                     # (成果物 tag, prompt variant, seed) —— 実 trace（計器の実働域）
    ("F5b-1", "F5b-1", 6), ("F5b-1", "F5b-1", 2),
    ("F5b-1-ext", "F5b-1", 14), ("F5b-1-ext", "F5b-1", 18),
]
W_N = 4                        # wikitext から連結する長文の本数
EAGER_MAX = 8192               # ★ 裁定: eager 走行は B1–B5（≤8,191）に clip
# ★ 層種の分割（model config の layer_types 由来・BAND 26..56 内）
FULL_LAYERS = [27, 31, 35, 39, 43, 47, 51, 55]
BRIDGE_GUARD = 0.02            # ★ 凍結: bridge 項（worst-of 中央値）がこれを超えたら B6 認定不可


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def provenance(allow_dirty: bool) -> dict:
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    porcelain = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip().splitlines()
    keep, excl = [], []
    for ln in porcelain:                       # R7-a 改一: 自成果物パス配下の未追跡のみ除外可
        p = ln[3:].strip()
        if ln.startswith("??") and p.startswith("probes/c-lens-pos/results/"):
            excl.append(ln)
        else:
            keep.append(ln)
    if keep and not allow_dirty:
        raise SystemExit("⛔ R7-a: tree が dirty です\n" + "\n".join(keep))
    import transformers
    return {"commit": commit, "dirty": bool(keep), "dirty_files": keep,
            "dirty_check_excluded": excl,
            "dirty_check_exclusion_rule": "untracked under probes/c-lens-pos/results/ only",
            "run_kind": "main", "probe": "C-lens-pos",
            "design": "DECISION_TABLE.en.md",
            "adjudication": "ADJUDICATION_c_lens_pos_v2_effective_20260802.md",
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": {"python": sys.version.split()[0],
                         "transformers": transformers.__version__}}


# ── 標本テキスト ──────────────────────────────────────────────────────────
def g_texts(tok) -> list[dict]:
    """The reasoning-trace corpus: stored traces (prompt + generation). Nothing is regenerated and the content is not inspected."""
    import screen_base as SB
    base = SB.load_prompts()[run_f5.CONDITION]
    out = []
    for tag, variant, seed in G_ROWS:
        doc = json.loads((REPO / "probes" / "olmo3-f5-think-prefix" / "results"
                          / f"f5_runs_{tag}.json").read_text())
        new_base, _ = run_f5.build_prompt(base, variant)
        full_prompt = new_base + run_f5.CUE
        assert hashlib.md5(full_prompt.encode()).hexdigest() == doc["prompt_md5_new"], tag
        row = next(r for r in doc["rows"] if r["seed"] == seed)
        text = full_prompt + row["content"]
        out.append({"group": "G", "name": f"{tag}:s{seed}", "text": text,
                    "n_tokens": len(tok(text)["input_ids"]),
                    "n_prompt": len(tok(full_prompt)["input_ids"])})
    return out


def w_texts(tok) -> list[dict]:
    """The prose corpus: wikitext --- the lens's own fit dataset. Article boundaries are recorded in the provenance block."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
                      streaming=True)
    out, buf, bounds, took = [], [], [], 0
    for rec in ds:
        t = rec["text"]
        if not t.strip():
            continue
        bounds.append(sum(len(x) for x in buf))          # ★ 記事境界（文字位置）
        buf.append(t)
        if len(tok("".join(buf))["input_ids"]) >= MAX_SEQ_PROBE + 64:
            text = "".join(buf)
            out.append({"group": "W", "name": f"wikitext-{took+1}", "text": text,
                        "n_tokens": len(tok(text)["input_ids"]), "n_prompt": 0,
                        "article_boundaries_chars": bounds})
            took += 1
            buf, bounds = [], []
            if took >= W_N:
                break
    return out


def positions_for(n_tokens: int) -> tuple[list[int], list[tuple[str, list[int]]]]:
    """Evenly spaced sample points per frozen bucket; a bucket beyond the text length is empty."""
    per, allpos = [], []
    for name, lo, hi in BUCKETS:
        hi_eff = min(hi, n_tokens - 1)
        if hi_eff < lo:
            per.append((name, []))
            continue
        if hi_eff - lo + 1 <= PER_BUCKET:
            pos = list(range(lo, hi_eff + 1))
        else:
            step = (hi_eff - lo) / (PER_BUCKET - 1)
            pos = sorted({int(round(lo + i * step)) for i in range(PER_BUCKET)})
        per.append((name, pos))
        allpos.extend(pos)
    return sorted(set(allpos)), per


# ── 指標 ─────────────────────────────────────────────────────────────────
def metrics_for_text(torch, p2, lens, model, mask, sets, text: str, positions: list[int]):
    """Two forward passes per text (with and without the Jacobian lens)."""
    out = {}
    for use_j, key in ((True, "lens"), (False, "logitlens")):
        jl, model_logits, _ = lens.apply(model, text, layers=run_pc2.BAND,
                                         positions=positions, max_seq_len=MAX_SEQ_PROBE,
                                         use_jacobian=use_j)
        P = model_logits.shape[0]
        mlog = model_logits.float()
        mlp = torch.log_softmax(mlog, dim=-1)
        m_top1 = mlog.argmax(-1)
        m_topk = mlog.topk(TOPK, dim=-1).indices                       # [P, K]
        # ★ 語群の帯中央値順位（verbatim import の機構）
        band_rank = p2.band_median_ranks(torch, jl, mask, sets)        # {set: [P]}
        model_rank = p2.band_median_ranks(torch, {run_pc2.BAND[-1]: mlog}, mask, sets)

        per_layer_kl, per_layer_top1, per_layer_rc = [], [], []
        for layer in run_pc2.BAND:
            L = jl[layer].float()
            llp = torch.log_softmax(L, dim=-1)
            kl = (mlp.exp() * (mlp - llp)).sum(-1)                     # KL(model ‖ lens) [P]
            per_layer_kl.append(kl)
            per_layer_top1.append((L.argmax(-1) == m_top1).float())
            rc = []
            for p in range(P):                                          # Spearman on model top-K
                ids = m_topk[p]
                lv, mv = L[p, ids], mlog[p, ids]
                lr = lv.argsort(descending=True).argsort().float()
                mr = mv.argsort(descending=True).argsort().float()
                lr = lr - lr.mean(); mr = mr - mr.mean()
                rc.append((lr @ mr) / (lr.norm() * mr.norm() + 1e-12))
            per_layer_rc.append(torch.stack(rc))
        stack = lambda xs: torch.stack(xs)                              # [n_layers, P]
        out[key] = {
            "kl": stack(per_layer_kl), "top1": stack(per_layer_top1),
            "rankcorr": stack(per_layer_rc),
            "band_rank": band_rank, "model_rank": model_rank,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    # ★ P1 の切り分け用。⛔ 本走行の MAX_SEQ_PROBE は動かさない（凍結設計）。
    ap.add_argument("--p1-max-seq", type=int, default=None,
                    help="P1 のみ: この長さで forward して壊れる位置を実測する")
    # ★ P1 診断のみ。⛔ 本走行の attn_implementation は run_pc2 と同じ "eager" のまま
    #   （変更は凍結された呼び出し形からの逸脱であり、redline を要する）。
    ap.add_argument("--p1-attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--p1-compare-attn", action="store_true",
                    help="P1 診断: 同じ max_seq で eager と sdpa を両方走らせ、出力差を実測する")
    # ★ 2 つの model を同一プロセスに積むと MPS 上で解放されず kill される（実測）。
    #   ⇒ 走行をプロセスで分け、済んだ (attn, text) は成果物から resume する。
    #   ⛔ 設計は不変（二重走行・同一 positions・同一指標）。分けたのは実行の器だけ。
    ap.add_argument("--attn-only", default=None, choices=["eager", "sdpa"],
                    help="この attn の pass だけ走らせる（もう一方は成果物から resume）")
    # ★ V-N（中立性検証・ADJUDICATION_runner_fix_and_vn §1）:
    #   修正“前”に測った text を修正“後”の runner で再測し、保存済み per-position 値の
    #   全値を突合する。完全一致なら「数値不変」が仮定でなく証明になる（V-G3 と同じ手）。
    #   ⛔ 測定ループ本体には触れない（本走行と同一の inline 経路をそのまま使う）。
    #   ⛔ 認定済み成果物 c_lens_pos.json は書き換えない（別ファイルへ出す）。
    ap.add_argument("--vn-attn", default=None, choices=["eager", "sdpa"])
    ap.add_argument("--vn-text", default=None, help="V-N で再測する text 名")
    # ★ D-CHECK（決定性検証）: 同一プロセス・同一 code・同一設定で apply を 2 回行い突合する。
    #   V-N で eager=IDENTICAL / sdpa=MISMATCH という非対称が出たため、
    #   「差はメモリ修正によるのか、attn 実装自体の非決定性か」を分離する。
    #   ⛔ 主線ではない診断。⛔ 成果物 c_lens_pos.json には触れない。
    ap.add_argument("--det-check", default=None, choices=["eager", "sdpa"])
    ap.add_argument("--det-text", default="F5b-1:s2")
    # ★ 凍結動作の実行（ADJUDICATION_runner_fix_and_vn §1 の MISMATCH 枝 = 修正前 text は全再測）。
    #   --redo は、--attn-only / --redo-text で指定した (attn, text) 組を成果物から外してから
    #   走る。⇒ 全 16 組が修正後になり、併合の問題自体が消える。
    #   ⛔ 既存成果物は事前に別名で保全すること（本 script は保全しない）。
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--redo-text", default=None, help="この text だけ redo（既定は attn 全 text）")
    # ★ R-1（統制再測・ADJUDICATION_c_lens_pos_verdict 発注 1）:
    #   sdpa を max_seq=8192 で B1–B5 測り、eager（同 max_seq）と同一位置で突合する。
    #   ⇒ bridge から系列長を落とし、★ attn 単独項を分離する。
    #   ⛔ c_lens_pos.json には触れない（c_lens_pos_r1.json へ出す）。
    ap.add_argument("--r1", action="store_true")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    prov = provenance(args.allow_dirty)
    log(f"R7-a ✅ commit={prov['commit'][:8]} dirty={prov['dirty']} probe=C-lens-pos")

    import torch, transformers, jlens                                   # noqa: E401
    got = hashlib.md5(run_pc2.LENS_PT.read_bytes()).hexdigest()
    if got != run_pc2.LENS_MD5:
        raise SystemExit(f"⛔ lens md5 不一致: {got}")
    log(f"P2 lens md5 ✅ {run_pc2.LENS_MD5}")

    tok = transformers.AutoTokenizer.from_pretrained(run_pc2.MODEL_NAME)
    p2 = run_pc2.p2mod()

    if run_pc2.MASK_PATH.exists():
        _ids = json.loads(run_pc2.MASK_PATH.read_text())
    else:
        _ids = [q for q in range(len(tok)) if p2.WORDLIKE.match(tok.decode([q]).strip())]
        run_pc2.MASK_PATH.write_text(json.dumps(_ids))
    mask = torch.zeros(len(tok), dtype=torch.bool); mask[_ids] = True
    sets = {"hehi": p2.set_ids(tok, p2.HEHI_WORDS), "apex": p2.set_ids(tok, p2.APEX_WORDS),
            "center": p2.set_ids(tok, p2.CENTER_WORDS)}
    log(f"mask/sets ✅ word-like={len(_ids)}  hehi={len(sets['hehi'])} apex={len(sets['apex'])}")

    log("P3 G 群の token 長を確定")
    G = g_texts(tok)
    for g in G:
        log(f"   {g['name']:<18} n_tokens={g['n_tokens']:<7} n_prompt={g['n_prompt']}")
    log("P4 W 群（wikitext）を連結")
    W = w_texts(tok)
    for w in W:
        log(f"   {w['name']:<18} n_tokens={w['n_tokens']:<7} "
            f"articles={len(w['article_boundaries_chars'])}")
    texts = W + G

    cover = {}
    for t in texts:
        _, per = positions_for(t["n_tokens"])
        cover[t["name"]] = {b: len(p) for b, p in per}
    log("bucket cell の実 n:")
    for name, c in cover.items():
        log(f"   {name:<18} " + "  ".join(f"{b}={c[b]}" for b, _, _ in BUCKETS))

    if args.preflight:
        log("P1 16,384 token forward の実測（1 本のみ・本走行はしない）")
        lens = jlens.JacobianLens.from_pretrained(str(run_pc2.LENS_PT))

        def build(attn):
            hf = transformers.AutoModelForCausalLM.from_pretrained(
                run_pc2.MODEL_NAME, dtype=torch.bfloat16,
                attn_implementation=attn).to("mps")
            hf.eval()
            return jlens.from_hf(hf, tok)

        if args.p1_compare_attn:      # ★ 診断: 同一 max_seq で eager / sdpa の出力差を測る
            ms = args.p1_max_seq or 8192
            probe = max(texts, key=lambda t: t["n_tokens"])
            pos = [q for q in positions_for(probe["n_tokens"])[0] if q < ms]
            res = {}
            for attn in ("eager", "sdpa"):
                m = build(attn)
                t0 = time.time()
                jl, ml, _ = lens.apply(m, probe["text"], layers=run_pc2.BAND,
                                       positions=pos, max_seq_len=ms)
                # ★ 決定表が消費する量（語群の帯中央値順位）でも比べる —— redline §1 の原則
                br = p2.band_median_ranks(torch, jl, mask, sets)
                res[attn] = {"model": ml.float(), "lens": jl[run_pc2.BAND[-1]].float(),
                             "band_rank": {k: list(map(float, v)) for k, v in br.items()},
                             "wall": time.time() - t0,
                             "mem": torch.mps.driver_allocated_memory() / 2**30}
                log(f"   {attn:<6} max_seq={ms}  wall={res[attn]['wall']:.1f}s  "
                    f"mps={res[attn]['mem']:.1f} GiB")
                del m
                torch.mps.empty_cache()
            dm = (res["eager"]["model"] - res["sdpa"]["model"]).abs().max().item()
            dl = (res["eager"]["lens"] - res["sdpa"]["lens"]).abs().max().item()
            t1a = res["eager"]["model"].argmax(-1)
            t1b = res["sdpa"]["model"].argmax(-1)
            log(f"   ★ eager vs sdpa  max|Δmodel_logits|={dm:.4g}  "
                f"max|Δlens_logits|={dl:.4g}  top1 一致={int((t1a==t1b).sum())}/{len(t1a)}")
            import math
            seat_delta = {}
            for s in ("hehi", "apex"):
                a = res["eager"]["band_rank"][s]; b = res["sdpa"]["band_rank"][s]
                d = [abs(math.log10(max(x, 1)) - math.log10(max(y, 1))) for x, y in zip(a, b)]
                d.sort()
                seat_delta[s] = {"median_delta_log10": d[len(d) // 2], "max_delta_log10": d[-1],
                                 "n": len(d), "identical_rank": sum(1 for x, y in zip(a, b)
                                                                    if x == y)}
                log(f"   ★ seatrank({s})  |Δlog10 rank| median={d[len(d)//2]:.4f} "
                    f"max={d[-1]:.4f}   順位完全一致={seat_delta[s]['identical_rank']}/{len(d)}")
            log("   ⚠ 判定閾値は 0.10 / 0.30 桁 —— 上の median/max がこれに対しどの水準か")
            (RES / "preflight_attn_compare.json").write_text(json.dumps(
                {"provenance": prov, "max_seq": ms, "n_positions": len(pos),
                 "text": probe["name"],
                 "max_abs_delta_model_logits": dm, "max_abs_delta_lens_logits": dl,
                 "top1_agreement": f"{int((t1a==t1b).sum())}/{len(t1a)}",
                 "seatrank_delta_eager_vs_sdpa": seat_delta,
                 "wall_s": {k: round(v["wall"], 1) for k, v in res.items()},
                 "mps_gib": {k: round(v["mem"], 2) for k, v in res.items()},
                 "note_mps": ("mps_gib は driver 累積であり 2 つ目の値は 1 つ目のモデル解放前を"
                              "含みうる。⛔ 単独走行の値と比較しない。")},
                ensure_ascii=False, indent=2))
            log(f"-> {RES / 'preflight_attn_compare.json'}   ⛔ 本走行はしていない")
            return 0

        model = build(args.p1_attn)
        probe = max(texts, key=lambda t: t["n_tokens"])
        ms = args.p1_max_seq or MAX_SEQ_PROBE
        pos = [p for p in positions_for(probe["n_tokens"])[0] if p < ms]
        log(f"   P1 max_seq={ms}  n_pos={len(pos)}  text={probe['name']}")
        t0 = time.time()
        jl, ml, _ = lens.apply(model, probe["text"], layers=run_pc2.BAND, positions=pos,
                               max_seq_len=ms)
        wall = time.time() - t0
        peak = torch.mps.driver_allocated_memory() / 2**30
        log(f"   ★ OK  attn={args.p1_attn}  max_seq={ms}  n_pos={len(pos)}  wall={wall:.1f}s  "
            f"mps_allocated={peak:.1f} GiB  lens_logits={len(jl)} 層 × {tuple(ml.shape)}")
        (RES / "preflight.json").write_text(json.dumps(
            {"provenance": prov, "coverage": cover,
             "texts": [{k: v for k, v in t.items() if k != "text"} for t in texts],
             "P1": {"text": probe["name"], "attn": args.p1_attn, "max_seq": ms,
                   "n_positions": len(pos),
                    "wall_s": round(wall, 1), "mps_allocated_gib": round(peak, 2),
                    "forecast_s": round(0.666 + 0.007982 * ms, 1)}},
            ensure_ascii=False, indent=2))
        log(f"-> {RES / 'preflight.json'}   ⛔ 本走行はしていない")
        return 0

    if args.r1:                            # ── R-1（統制再測: sdpa @ max_seq=8192）──────
        import gc, math                    # noqa: E401
        ms = EAGER_MAX
        stored = json.loads(DST.read_text())["measurements"]
        E = {(m["text"], m["position"]): m for m in stored if m["attn"] == "eager"}
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            run_pc2.MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa").to("mps")
        hf.eval()
        model = jlens.from_hf(hf, tok)
        lens = jlens.JacobianLens.from_pretrained(str(run_pc2.LENS_PT))
        log(f"R-1: sdpa @ max_seq={ms}（eager と同一 max_seq・同一位置）")
        rows = []
        for tx in texts:
            pos = [q for q in positions_for(tx["n_tokens"])[0] if q < ms]
            gc.collect(); torch.mps.empty_cache()
            t0 = time.time()
            jl, ml, _ = lens.apply(model, tx["text"], layers=run_pc2.BAND,
                                   positions=pos, max_seq_len=ms)
            mlog = ml.float()
            br = p2.band_median_ranks(torch, jl, mask, sets)
            mr_ = p2.band_median_ranks(torch, {run_pc2.BAND[-1]: mlog}, mask, sets)
            seat = {s: [abs(math.log10(max(x, 1)) - math.log10(max(y, 1)))
                        for x, y in zip(br[s], mr_[s])] for s in sets}
            for i, q in enumerate(pos):
                rows.append({"text": tx["name"], "group": tx["group"], "position": q,
                             "seatrank_hehi": seat["hehi"][i],
                             "seatrank_apex": seat["apex"][i]})
            del jl, ml, mlog
            log(f"  sdpa@8192 {tx['name']:<18} n_pos={len(pos):<3} ({time.time()-t0:.0f}s)")
        # ★ attn 単独の bridge（同一 max_seq・同一位置）
        bucket_of = {}
        for name, lo, hi in BUCKETS:
            for q in range(lo, hi + 1):
                bucket_of[q] = name
        def med(xs):
            xs = sorted(xs); return None if not xs else xs[len(xs) // 2]
        per = {}
        for r in rows:
            e = E.get((r["text"], r["position"]))
            if e is None:
                continue
            d_ = abs(max(r["seatrank_hehi"], r["seatrank_apex"])
                     - max(e["seatrank_hehi"], e["seatrank_apex"]))
            per.setdefault(bucket_of[r["position"]], []).append(d_)
        bridge = [{"bucket": b, "n": len(v), "delta_seatrank_median": med(v),
                   "delta_seatrank_max": max(v)} for b, v in sorted(per.items())]
        worst = max((b["delta_seatrank_median"] for b in bridge), default=None)
        (RES / "c_lens_pos_r1.json").write_text(json.dumps(
            {"check": "R-1 統制再測（sdpa @ max_seq=8192・attn 単独項の分離）",
             "adjudication": "ADJUDICATION_c_lens_pos_verdict_20260802.md 発注 1",
             "max_seq": ms, "n_rows": len(rows), "bridge_attn_only": bridge,
             "bridge_attn_only_worst_median": worst,
             "reference": "c_lens_pos.json の eager 測定（同一 max_seq=8192・同一位置）",
             "guard": BRIDGE_GUARD, "provenance": prov, "rows": rows},
            ensure_ascii=False, indent=1))
        log(f"★ R-1 attn 単独 bridge worst median = {worst}   (guard {BRIDGE_GUARD})")
        for b in bridge:
            log(f"   {b['bucket']}  n={b['n']}  中央 {b['delta_seatrank_median']:.5f}  "
                f"最大 {b['delta_seatrank_max']:.5f}")
        return 0

    if args.det_check:                     # ── D-CHECK（決定性検証・診断）──────────
        import gc
        attn = args.det_check
        ms = EAGER_MAX if attn == "eager" else MAX_SEQ_PROBE
        tx = next(x for x in texts if x["name"] == args.det_text)
        pos = [q for q in positions_for(tx["n_tokens"])[0] if q < ms]
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            run_pc2.MODEL_NAME, dtype=torch.bfloat16, attn_implementation=attn).to("mps")
        hf.eval()
        model = jlens.from_hf(hf, tok)
        lens = jlens.JacobianLens.from_pretrained(str(run_pc2.LENS_PT))
        log(f"D-CHECK attn={attn} text={tx['name']} n_pos={len(pos)} max_seq={ms}")
        runs = []
        for k in (1, 2):
            gc.collect(); torch.mps.empty_cache()
            t0 = time.time()
            jl, ml, _ = lens.apply(model, tx["text"], layers=run_pc2.BAND,
                                   positions=pos, max_seq_len=ms)
            br = p2.band_median_ranks(torch, jl, mask, sets)
            runs.append({"model_logits": ml.float().clone(),
                         "lens_top": jl[run_pc2.BAND[-1]].float().clone(),
                         "band_rank": {s: [float(v) for v in br[s]] for s in sets}})
            log(f"   run {k}: {time.time()-t0:.0f}s")
            del jl, ml
        a, b = runs
        dm = float((a["model_logits"] - b["model_logits"]).abs().max())
        dl = float((a["lens_top"] - b["lens_top"]).abs().max())
        same_rank = {s: sum(1 for x, y in zip(a["band_rank"][s], b["band_rank"][s]) if x == y)
                     for s in sets}
        t1 = int((a["model_logits"].argmax(-1) == b["model_logits"].argmax(-1)).sum())
        verdict = ("DETERMINISTIC" if dm == 0 and dl == 0
                   and all(v == len(pos) for v in same_rank.values()) else "NON-DETERMINISTIC")
        log(f"   ★ max|Δmodel_logits|={dm:.4g}  max|Δlens_logits|={dl:.4g}  "
            f"top1 一致={t1}/{len(pos)}")
        log(f"   ★ 帯中央順位の完全一致: " + " ".join(f"{s}={same_rank[s]}/{len(pos)}" for s in sets))
        log(f"   ★★ {attn} は {verdict}")
        (RES / f"det_check_{attn}.json").write_text(json.dumps(
            {"check": "D-CHECK（同一プロセス・同一設定で apply 2 回）",
             "attn": attn, "text": tx["name"], "n_positions": len(pos), "max_seq": ms,
             "max_abs_delta_model_logits": dm, "max_abs_delta_lens_logits": dl,
             "top1_agreement": f"{t1}/{len(pos)}",
             "band_rank_identical": {s: f"{same_rank[s]}/{len(pos)}" for s in sets},
             "verdict": verdict, "provenance": prov}, ensure_ascii=False, indent=2))
        log(f"-> {RES / f'det_check_{attn}.json'}")
        return 0

    # ── 本走行（★ 裁定 ADJUDICATION_c_lens_pos_attn_ruling: 二重走行） ──────────
    import gc
    import math
    SLIDING = [l for l in run_pc2.BAND if l not in FULL_LAYERS]
    bucket_of = {}
    for name, lo, hi in BUCKETS:
        for q in range(lo, hi + 1):
            bucket_of[q] = name

    def agg(mat, layers, how="median"):
        """Aggregate [n_layers, P] over a subset of layers, per position -> [P]."""
        idx = [run_pc2.BAND.index(l) for l in layers]
        sub = mat[idx]
        v = sub.mean(0) if how == "mean" else sub.median(0).values
        return [float(x) for x in v]

    lens = jlens.JacobianLens.from_pretrained(str(run_pc2.LENS_PT))

    def build(attn):
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            run_pc2.MODEL_NAME, dtype=torch.bfloat16, attn_implementation=attn).to("mps")
        hf.eval()
        return jlens.from_hf(hf, tok)

    VN = bool(args.vn_attn)
    stored = json.loads(DST.read_text()).get("measurements", []) if DST.exists() else []
    meas = []
    if VN:
        if not stored:
            raise SystemExit("⛔ V-N には既存成果物が要る")
        done_pairs = set()                 # ★ V-N は skip しない（再測が目的）
        log(f"V-N: 既存 {len(stored)} 測定と突合する（⛔ 上書きしない）"
            f"  対象 = {args.vn_attn}:{args.vn_text}")
    elif stored:                           # ★ resume: 済んだ (attn, text) は再計算しない
        if args.redo:                      # ★ 指定組を外してから resume する
            drop = {(m["attn"], m["text"]) for m in stored
                    if (args.attn_only is None or m["attn"] == args.attn_only)
                    and (args.redo_text is None or m["text"] == args.redo_text)}
            stored = [m for m in stored if (m["attn"], m["text"]) not in drop]
            log(f"★ redo: {len(drop)} 組を外した -> {sorted(drop)}")
        meas.extend(stored)
        done_pairs = {(m["attn"], m["text"]) for m in stored}
        log(f"resume: 既存 {len(stored)} 測定 / 済み {len(done_pairs)} (attn,text) 組")
    else:
        done_pairs = set()

    plan = [("eager", EAGER_MAX), ("sdpa", MAX_SEQ_PROBE)]
    if args.attn_only:
        plan = [x for x in plan if x[0] == args.attn_only]
    if VN:
        plan = [x for x in plan if x[0] == args.vn_attn]
        texts = [tx for tx in texts if tx["name"] == args.vn_text]
        if not texts:
            raise SystemExit(f"⛔ text が見つからない: {args.vn_text}")
    for attn, ms in plan:
        if all((attn, tx["name"]) in done_pairs for tx in texts):
            log(f"=== attn={attn} は全 text 済み ⇒ skip ===")
            continue
        log(f"=== attn={attn}  max_seq={ms} ===")
        model = build(attn)
        for tx in texts:
            if (attn, tx["name"]) in done_pairs:
                continue
            pos = [q for q in positions_for(tx["n_tokens"])[0] if q < ms]
            if not pos:
                continue
            t0 = time.time()
            got = {}
            for use_j, key in ((True, "lens"), (False, "logitlens")):
                # ★ 2026-08-02: 1 テキスト内で apply を 2 回呼ぶ間に MPS の解放が起きず、
                #   model 64 GB + 16,384 位置の activation が二重に載って swap を張り付かせた
                #   （実測 sdpa wikitext-1 = 1,641s ≒ 期待 520s の 3.2 倍・swap 空き 1.76 GB）。
                #   ⇒ apply の前後で明示解放する。⛔ 計器にも呼び出し形にも触れない（数値不変）。
                gc.collect(); torch.mps.empty_cache()
                jl, ml, _ = lens.apply(model, tx["text"], layers=run_pc2.BAND,
                                       positions=pos, max_seq_len=ms, use_jacobian=use_j)
                mlog = ml.float()
                mlp = torch.log_softmax(mlog, dim=-1)
                m_top1 = mlog.argmax(-1)
                m_topk = mlog.topk(TOPK, dim=-1).indices
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
                    "kl_full": agg(K, FULL_LAYERS), "kl_slid": agg(K, SLIDING),
                    "rankcorr_full": agg(RC, FULL_LAYERS), "rankcorr_slid": agg(RC, SLIDING),
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
            log(f"  {attn:<6} {tx['name']:<18} n_pos={len(pos):<3} ({time.time()-t0:.0f}s)  "
                f"mps={torch.mps.driver_allocated_memory()/2**30:.1f} GiB")
            if not VN:                     # ⛔ V-N は認定済み成果物に触れない
                DST.write_text(json.dumps({"partial": True, "measurements": meas},
                                          ensure_ascii=False))
        del model
        gc.collect()
        torch.mps.empty_cache()

    if VN:                                 # ── V-N 突合（⛔ 集約も判定もしない）──────────
        FIELDS = [k for k in meas[0] if isinstance(meas[0][k], float)] if meas else []
        idx = {(m["attn"], m["text"], m["position"]): m for m in stored}
        cmp_rows, n_cmp, n_diff = [], 0, 0
        for m in meas:
            old = idx.get((m["attn"], m["text"], m["position"]))
            if old is None:
                cmp_rows.append({"position": m["position"], "missing_in_stored": True})
                n_diff += 1
                continue
            diffs = {f: [old.get(f), m.get(f)] for f in FIELDS if old.get(f) != m.get(f)}
            n_cmp += len(FIELDS)
            if diffs:
                n_diff += len(diffs)
                cmp_rows.append({"position": m["position"], "diffs": diffs})
        verdict = "IDENTICAL" if n_diff == 0 else "MISMATCH"
        out = {"check": "V-N（中立性検証）",
               "adjudication": "ADJUDICATION_runner_fix_and_vn_20260802.md §1",
               "target": {"attn": args.vn_attn, "text": args.vn_text},
               "n_positions": len(meas), "n_fields_per_position": len(FIELDS),
               "n_values_compared": n_cmp, "n_values_differing": n_diff,
               "verdict": verdict, "fields": FIELDS,
               "mismatches": cmp_rows[:50], "provenance": prov,
               "note": "⛔ c_lens_pos.json は書き換えていない。測定ループ本体は本走行と同一 inline 経路。"}
        (RES / f"vn_{args.vn_attn}_{args.vn_text.replace(':','_')}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2))
        log(f"★ V-N {args.vn_attn}:{args.vn_text}  位置 {len(meas)} × 指標 {len(FIELDS)} "
            f"= {n_cmp} 値を突合  ⇒ 差 {n_diff} 値  ⇒ ★ {verdict}")
        return 0 if verdict == "IDENTICAL" else 3

    have = {(m["attn"], m["text"]) for m in meas}
    if not all((a, tx["name"]) in have for a in ("eager", "sdpa") for tx in texts):
        DST.write_text(json.dumps({"partial": True, "measurements": meas}, ensure_ascii=False))
        miss = [(a, tx["name"]) for a in ("eager", "sdpa") for tx in texts
                if (a, tx["name"]) not in have]
        log(f"⚠ 未完（{len(miss)} 組）: {miss[:4]}{' …' if len(miss) > 4 else ''}")
        log(f"-> {DST}（partial）  ⛔ 集約しない —— 両 pass が揃ってから")
        return 0

    # ── 集約（⛔ 判定はしない・凍結表の適用は RESULT 側） ──────────────────
    def med(xs):
        xs = sorted(xs)
        return None if not xs else xs[len(xs) // 2]

    cells = []
    for attn in ("eager", "sdpa"):
        for grp in ("G", "W"):
            for bname, _, _ in BUCKETS:
                rows = [m for m in meas if m["attn"] == attn and m["group"] == grp
                        and m["bucket"] == bname]
                if not rows:
                    continue
                cells.append({
                    "attn": attn, "group": grp, "bucket": bname, "n": len(rows),
                    "rankcorr": med([r["rankcorr"] for r in rows]),
                    "seatrank": med([max(r["seatrank_hehi"], r["seatrank_apex"]) for r in rows]),
                    "seatrank_hehi": med([r["seatrank_hehi"] for r in rows]),
                    "seatrank_apex": med([r["seatrank_apex"] for r in rows]),
                    "seatrank_center": med([r["seatrank_center"] for r in rows]),
                    "kl": med([r["kl"] for r in rows]), "top1": med([r["top1"] for r in rows]),
                    "kl_ll": med([r["kl_ll"] for r in rows]),
                    "rankcorr_ll": med([r["rankcorr_ll"] for r in rows]),
                    "rankcorr_full": med([r["rankcorr_full"] for r in rows]),
                    "rankcorr_slid": med([r["rankcorr_slid"] for r in rows]),
                    "kl_full": med([r["kl_full"] for r in rows]),
                    "kl_slid": med([r["kl_slid"] for r in rows]),
                })

    # ★ bridge 項: B1–B5 の同一位置における eager ↔ sdpa の差（実装差の実測）
    key = lambda m: (m["text"], m["position"])
    E = {key(m): m for m in meas if m["attn"] == "eager"}
    S = {key(m): m for m in meas if m["attn"] == "sdpa"}
    bridge = []
    for bname, _, _ in BUCKETS:
        ks = [k for k in E if k in S and E[k]["bucket"] == bname]
        if not ks:
            continue
        dsr = [abs(max(E[k]["seatrank_hehi"], E[k]["seatrank_apex"])
                   - max(S[k]["seatrank_hehi"], S[k]["seatrank_apex"])) for k in ks]
        drc = [abs(E[k]["rankcorr"] - S[k]["rankcorr"]) for k in ks]
        bridge.append({"bucket": bname, "n": len(ks),
                       "delta_seatrank_median": med(dsr), "delta_seatrank_max": max(dsr),
                       "delta_rankcorr_median": med(drc), "delta_rankcorr_max": max(drc),
                       "worst_median": max(med(dsr), med(drc))})
    bridge_worst_median = max([b["worst_median"] for b in bridge], default=None)

    DST.write_text(json.dumps({
        "design": "DECISION_TABLE.en.md",
        "ruling": "ADJUDICATION_c_lens_pos_attn_ruling_20260802.md",
        "provenance": prov, "buckets": BUCKETS, "per_bucket": PER_BUCKET, "topk": TOPK,
        "band": [run_pc2.BAND[0], run_pc2.BAND[-1]], "full_layers": FULL_LAYERS,
        "thresholds": {"rankcorr": [0.10, 0.30], "seatrank_digits": [0.10, 0.30],
                       "top1_b6_vs_b5_points": 5, "bridge_guard_digits": BRIDGE_GUARD,
                       "b6_min_n": 16},
        "attn_plan": {"eager": f"B1-B5 clip <{EAGER_MAX}", "sdpa": f"B1-B6 <{MAX_SEQ_PROBE}"},
        "texts": [{k: v for k, v in tx.items() if k != "text"} for tx in texts],
        "cells": cells, "bridge": bridge,
        "bridge_worst_median": bridge_worst_median,
        "measurements": meas,
    }, ensure_ascii=False, indent=1))
    log(f"★ bridge worst median = {bridge_worst_median}   (guard {BRIDGE_GUARD})")
    log(f"-> {DST}   cells={len(cells)} measurements={len(meas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
