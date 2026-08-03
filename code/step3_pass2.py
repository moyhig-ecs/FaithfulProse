#!/usr/bin/env python
"""J-lens pass 2 --- 2a: in-text position re-read / 2b: post-game review
(frozen pre-registration of 2026-07-17).

- position rule: the final token of the first-occurrence match of the marker
  regex (\b-bounded) --- the position at which the symbol has been fully read.
- rank: rank within the word-like (alnum>=2) mask. The full vocabulary mask is
  precomputed and vectorised. A set's representative rank is the best rank
  within the set; sets are compared by the band (L26-57) median, lower being
  dominant.
- P9 aggregation (config frozen): the primary position is the first occurrence
  of O1 (O2 is recorded as a replication). Majority vote across 3 degrees
  (>=2/3).
- 2b: the 20 gate-1 traces are re-forwarded by concatenating prompt+thinking
  (no new generation). Losing move / reversal point / establishment are the
  frozen definitions of the pre-registration; the verdict is only a
  cross-check against the final label, which is already deterministically
  classified.
- the third set is tracked only and is not used in any verdict.

usage: step3_pass2.py config | run2a | run2b | verdict
"""

import json
import re
import statistics
import sys
import time
from datetime import datetime
import os
from pathlib import Path

JL = Path(os.environ.get("JACOBIAN_LENS_ROOT", "jacobian-lens"))
CG = Path(os.environ.get("CONSTRUCTGAP_ROOT", "ConstructGap"))
for p in (JL, CG, CG / "exp", Path(os.environ.get("GALILEO_ROOT", "experiments/galileo-screening"))):
    sys.path.insert(0, str(p))

HERE = Path(__file__).parent
RESULTS = HERE / "results"
LENS_PT = HERE / "lenses" / "Qwen3-32B_jacobian_lens_n80_from_hub_ckpt.pt"
CFG_PATH = RESULTS / "pass2_config_frozen.json"
MASK_PATH = RESULTS / "wordlike_mask_qwen3_32b.json"
R2A_PATH = RESULTS / "pass2a_readouts.json"
R2B_PATH = RESULTS / "pass2b_trajectories.json"

MODEL_NAME = "Qwen/Qwen3-32B"
BAND = list(range(26, 58))
DEGS = [23, 37, 59]
CHUNK = 96

HEHI_WORDS = ["He", " He", "Hi", " Hi"]
APEX_WORDS = ["O1O2", " O1O2", "O1", " O1", "O2", " O2",
              "diameter", " diameter", "midpoint", " midpoint"]
CENTER_WORDS = ["center", " center", "centre", " centre"]  # 追跡単独集合 (判定不使用)
WORDLIKE = re.compile(r"^[A-Za-z0-9]{2,}$")


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def load_stack():
    import torch
    import transformers
    import jlens
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, attn_implementation="eager").to("mps")
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(str(LENS_PT))
    return torch, tok, model, lens


def wordlike_mask(tok):
    import torch
    if MASK_PATH.exists():
        ids = json.loads(MASK_PATH.read_text())
    else:
        ids = [tid for tid in range(len(tok)) if WORDLIKE.match(tok.decode([tid]).strip())]
        MASK_PATH.write_text(json.dumps(ids))
    m = torch.zeros(len(tok), dtype=torch.bool)
    m[ids] = True
    return m


def set_ids(tok, words):
    return sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})


def band_median_ranks(torch, logits_by_layer, mask, sets):
    """Band-median word-like rank of each set. logits_by_layer: {layer: [P, vocab]}"""
    first = next(iter(logits_by_layer.values()))
    V = first.shape[1]
    if mask.shape[0] < V:  # embedding padding 分 (実トークン外) は False
        mask = torch.cat([mask, torch.zeros(V - mask.shape[0], dtype=torch.bool)])
    out = {name: [] for name in sets}
    P = first.shape[0]
    per_pos = {name: [[] for _ in range(P)] for name in sets}
    for layer, logits in logits_by_layer.items():
        lw = logits[:, mask]                                   # [P, W]
        for name, ids in sets.items():
            best = None
            for tid in ids:
                r = 1 + (lw > logits[:, tid:tid + 1]).sum(-1)  # [P]
                best = r if best is None else torch.minimum(best, r)
            for p in range(P):
                per_pos[name][p].append(int(best[p]))
    return {name: [statistics.median(v) for v in per_pos[name]] for name in per_pos}


def find_pos(tok, text, pattern):
    """Index of the final token of the first regex match, or None."""
    m = re.search(pattern, text)
    if not m:
        return None
    enc = tok(text, return_offsets_mapping=True)
    last = None
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if a < m.end() and b > m.start():
            last = i
    return last


def build_moy1_texts(tok, cfg):
    import prompts as P
    texts = {}
    for deg in cfg["degs"]:
        base = P.build("higashida_circle", "A", float(deg))["prompt"]
        for cond in ("misbind", "supply"):
            append = cfg["appends"][cond]
            texts[(deg, cond)] = tok.apply_chat_template(
                [{"role": "user", "content": base + append}],
                tokenize=False, add_generation_prompt=True)
    return texts


def cmd_config():
    import transformers
    audit = json.loads((RESULTS / "arm_ab_manipulanda_audit_frozen_config.json").read_text())
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    traces = json.loads((RESULTS / "bf16_moy1_runs.json").read_text())["rows"]
    by_label = {}
    for r in traces:
        by_label.setdefault((r["arm"], r["label"]), []).append((r["temp"], r["seed"]))
    cfg = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_NAME, "lens": LENS_PT.name + " (n=80 caveat 帯同)",
        "band": BAND, "degs": DEGS,
        "appends": {"misbind": audit["table"]["closure-forcing"]["append_verbatim"],
                     "supply": audit["table"]["selection-supply"]["append_verbatim"]},
        "sets": {"hehi": HEHI_WORDS, "apex": APEX_WORDS, "center_追跡単独": CENTER_WORDS},
        "position_rule": "marker 正規表現(\\b境界)初出 match の末尾 token",
        "markers_2a_moy1": {"He": r"\bHe\b", "Hi": r"\bHi\b", "X1": r"\bX1\b", "X2": r"\bX2\b",
                             "X3": r"\bX3\b", "X4": r"\bX4\b", "O1": r"\bO1\b", "O2": r"\bO2\b",
                             "role_cue(supplyのみ)": r"\bDIAMETER\b"},
        "markers_2a_armC": {"JA": "ガリレオ|ケプラー|リッチョーリ", "EN": r"\bGalileo\b|\bKepler\b|\bRiccioli\b"},
        "p9_rule": "主位置=O1初出(O2=複製記録)。supply vs open: rank比>=2 または (supply top-20 かつ open 圏外)。3 deg の多数決(>=2/3)",
        "p2b_traces": {f"{k[0]}|{k[1]}": v for k, v in by_label.items()},
        "p2b_bind_count": len(by_label.get(("B-blocked", "B-bind"), [])),
        "p2b_misbind_count": len(by_label.get(("B-blocked", "B-attractor"), [])),
        "p2b_open_stuckflail_count": len(by_label.get(("A-open", "B-stuck-L1"), [])) + len(by_label.get(("A-open", "B-flail"), [])),
        "確立定義": "dominant となる t* 以後 [t*,T] の >=80% で保持、かつ T-t* >= max(20tok, 0.1T)",
        "blind_note": "2b の 20 trace は本 config 凍結時点で lens 未適用",
    }
    CFG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    log(f"config frozen -> {CFG_PATH} (bind={cfg['p2b_bind_count']} misbind={cfg['p2b_misbind_count']} open_sf={cfg['p2b_open_stuckflail_count']})")


def cmd_run2a():
    from screening import PROBES
    torch, tok, model, lens = load_stack()
    cfg = json.loads(CFG_PATH.read_text())
    mask = wordlike_mask(tok)
    sets = {"hehi": set_ids(tok, HEHI_WORDS), "apex": set_ids(tok, APEX_WORDS),
            "center": set_ids(tok, CENTER_WORDS)}
    out = {"meta": {"date": datetime.now().isoformat(timespec="seconds")}, "moy1": {}, "armC": {}}

    # open 条件 (P9 の対照) も必要 — open = append なし
    import prompts as P
    for deg in cfg["degs"]:
        base = P.build("higashida_circle", "A", float(deg))["prompt"]
        for cond, append in [("open", ""), ("misbind", cfg["appends"]["misbind"]),
                              ("supply", cfg["appends"]["supply"])]:
            text = tok.apply_chat_template([{"role": "user", "content": base + append}],
                                           tokenize=False, add_generation_prompt=True)
            marks = dict(cfg["markers_2a_moy1"])
            if cond != "supply":
                marks.pop("role_cue(supplyのみ)")
            pos = {k: find_pos(tok, text, pat) for k, pat in marks.items()}
            pos = {k: v for k, v in pos.items() if v is not None}
            jl_logits, _, _ = lens.apply(model, text, layers=BAND, positions=list(pos.values()), max_seq_len=2048)
            med = band_median_ranks(torch, jl_logits, mask, sets)
            out["moy1"][f"deg{deg}|{cond}"] = {
                k: {"token_pos": v, **{s: med[s][i] for s in sets}}
                for i, (k, v) in enumerate(pos.items())}
            log(f"2a moy1 deg{deg} {cond}: {len(pos)} 位置")

    for (probe, lang) in [(p, l) for p in ("PR-1", "PR-1C", "PR-2", "PR-C") for l in ("JA", "EN")]:
        text = tok.apply_chat_template([{"role": "user", "content": PROBES[(probe, lang)]}],
                                       tokenize=False, add_generation_prompt=True)
        pat = cfg["markers_2a_armC"][lang]
        p = find_pos(tok, text, pat)
        if p is None:
            continue
        jl_logits, _, _ = lens.apply(model, text, layers=BAND, positions=[p], max_seq_len=2048)
        # Arm C は緩和集合の質量 + top-20 可視性 (exploratory 記述読み)
        ts = json.loads((RESULTS / "armC_tokenset_frozen_config.json").read_text())
        ids = sorted({v["first_token_id"] for v in ts["table"].values()})
        masses, top20flags = [], []
        for layer in BAND:
            logits = jl_logits[layer][0]
            probs = torch.softmax(logits.float(), dim=-1)
            masses.append(float(probs[ids].sum()))
            mk = mask
            if mk.shape[0] < logits.shape[0]:
                mk = torch.cat([mk, torch.zeros(logits.shape[0] - mk.shape[0], dtype=torch.bool)])
            lw = logits[mk]
            best = min(1 + int((lw > logits[t]).sum()) for t in ids)
            top20flags.append(best <= 20)
        out["armC"][f"{probe}|{lang}"] = {
            "token_pos": p, "mass_band中央値": statistics.median(masses),
            "top20_layers": sum(top20flags)}
        log(f"2a armC {probe}-{lang}: mass_med={statistics.median(masses):.2e} top20層数={sum(top20flags)}")
    R2A_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log(f"2a readouts -> {R2A_PATH}")


def cmd_run2b():
    torch, tok, model, lens = load_stack()
    cfg = json.loads(CFG_PATH.read_text())
    mask = wordlike_mask(tok)
    sets = {"hehi": set_ids(tok, HEHI_WORDS), "apex": set_ids(tok, APEX_WORDS),
            "center": set_ids(tok, CENTER_WORDS)}
    import prompts as P
    base = P.build("higashida_circle", "A", 37.0)["prompt"]
    wt = RESULTS / "bf16_moy1_runs_with_thinking.json"   # 同一性 oracle 検証済み棋譜 (再生成)
    if not wt.exists():
        raise SystemExit("検証済み棋譜なし — pass2_regen_traces.py を先に")
    traces = json.loads(wt.read_text())["rows"]
    done = json.loads(R2B_PATH.read_text()) if R2B_PATH.exists() else {"trajectories": {}}
    for r in traces:
        key = f"{r['arm']}|t{r['temp']}|s{r['seed']}"
        if key in done["trajectories"]:
            continue
        # gate① と同一再構成: A-open は append なし / B-blocked は closure-forcing
        append = cfg["appends"]["misbind"] if r["arm"] == "B-blocked" else ""
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": base + append}],
            tokenize=False, add_generation_prompt=True)
        gen = (r.get("thinking") or "") + "</think>" + (r.get("content") or "")
        full = prompt_text + gen
        n_prompt = len(tok(prompt_text)["input_ids"])
        n_think = len(tok((r.get("thinking") or "") + "</think>")["input_ids"])
        think_range = list(range(n_prompt, n_prompt + n_think))
        t0 = time.time()
        traj = {s: [] for s in sets}
        for i in range(0, len(think_range), CHUNK):
            chunk = think_range[i:i + CHUNK]
            jl_logits, _, _ = lens.apply(model, full, layers=BAND,
                                         positions=chunk, max_seq_len=8192)
            med = band_median_ranks(torch, jl_logits, mask, sets)
            for s in sets:
                traj[s].extend(med[s])
        done["trajectories"][key] = {
            "label": r["label"], "T": len(think_range),
            "hehi": traj["hehi"], "apex": traj["apex"], "center": traj["center"],
            "wall_s": round(time.time() - t0, 1)}
        R2B_PATH.write_text(json.dumps(done, ensure_ascii=False, indent=1))
        log(f"2b {key} ({r['label']}): T={len(think_range)} ({time.time()-t0:.0f}s)")
    log(f"2b trajectories -> {R2B_PATH}")


def _establish(dom, T):
    """Return t* --- 'establishment' under the frozen definition --- for a boolean dominance sequence, or None."""
    need = max(20, int(0.1 * T))
    for t in range(T):
        if not dom[t]:
            continue
        seg = dom[t:]
        if len(seg) >= need and sum(seg) / len(seg) >= 0.8:
            return t
    return None


def cmd_verdict():
    cfg = json.loads(CFG_PATH.read_text())
    trajs = json.loads(R2B_PATH.read_text())["trajectories"]
    r2a = json.loads(R2A_PATH.read_text())

    # --- P9 (2a): O1 主位置・supply vs open・3 deg 多数決 ---
    p9_rows, p9_hits = {}, 0
    for deg in cfg["degs"]:
        o = r2a["moy1"][f"deg{deg}|open"].get("O1")
        s = r2a["moy1"][f"deg{deg}|supply"].get("O1")
        if not (o and s):
            p9_rows[deg] = "位置欠損"
            continue
        ratio = o["apex"] / max(s["apex"], 1)
        hit = (ratio >= 2) or (s["apex"] <= 20 and o["apex"] > 20)
        p9_hits += hit
        p9_rows[deg] = {"open_apex": o["apex"], "supply_apex": s["apex"],
                        "ratio": round(ratio, 2), "hit": hit,
                        "O2複製": {"open": r2a["moy1"][f"deg{deg}|open"].get("O2", {}).get("apex"),
                                   "supply": r2a["moy1"][f"deg{deg}|supply"].get("O2", {}).get("apex")}}
    p9 = p9_hits >= 2

    # --- 2b: P6/P7/P8 ---
    def dom_series(tr, a, b):
        return [tr[a][t] < tr[b][t] for t in range(tr["T"])]

    p6_rows, p7_rows, p8_rows = {}, {}, {}
    n6 = n6_ok = n7 = n7_ok = n8 = n8_ok = 0
    for key, tr in trajs.items():
        lab, T = tr["label"], tr["T"]
        if key.startswith("B-blocked") and lab == "B-attractor":
            n6 += 1
            t_star = _establish(dom_series(tr, "hehi", "apex"), T)
            ok = t_star is not None and t_star / T <= 0.5
            n6_ok += ok
            p6_rows[key] = {"敗着t*": t_star, "T": T, "t*/T": round(t_star / T, 2) if t_star is not None else None, "ok": ok}
        elif key.startswith("B-blocked") and lab == "B-bind":
            n7 += 1
            hd = dom_series(tr, "hehi", "apex")
            t_star = _establish(dom_series(tr, "apex", "hehi"), T)
            prior_hehi = t_star is not None and any(hd[:t_star])
            ok = t_star is not None and prior_hehi
            n7_ok += ok
            p7_rows[key] = {"逆転点t*": t_star, "先行He/Hi区間": prior_hehi, "T": T, "ok": ok}
        elif key.startswith("A-open") and lab in ("B-stuck-L1", "B-flail"):
            n8 += 1
            e1 = _establish(dom_series(tr, "hehi", "apex"), T)
            e2 = _establish(dom_series(tr, "apex", "hehi"), T)
            ok = e1 is None and e2 is None
            n8_ok += ok
            p8_rows[key] = {"hehi確立": e1, "apex確立": e2, "彷徨": ok}
    p6 = n6 > 0 and n6_ok / n6 >= 0.7
    p7 = n7 > 0 and n7_ok == n7
    p8 = n8 > 0 and n8_ok / n8 >= 0.7

    drinks_new = sum(not x for x in (p6, p7, p8, p9))
    verdict = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "P9(2a O1位置 supply vs open)": {"per_deg": p9_rows, "hits": f"{p9_hits}/3", "pass": p9},
        "P6(敗着 >=70% かつ前半)": {"rows": p6_rows, "rate": f"{n6_ok}/{n6}", "pass": p6},
        "P7(逆転点 全bind trace)": {"rows": p7_rows, "rate": f"{n7_ok}/{n7}", "pass": p7},
        "P8(彷徨 >=70%)": {"rows": p8_rows, "rate": f"{n8_ok}/{n8}", "pass": p8},
        "drinks": {"本文書": drinks_new, "通算": 7 + drinks_new},
    }
    out = RESULTS / f"pass2_verdict_{datetime.now():%Y%m%d_%H%M}.json"
    out.write_text(json.dumps(verdict, ensure_ascii=False, indent=1))
    log(f"P9={p9}({p9_hits}/3) P6={p6}({n6_ok}/{n6}) P7={p7}({n7_ok}/{n7}) P8={p8}({n8_ok}/{n8})"
        f" | drinks +{drinks_new} -> 通算 {7+drinks_new}")
    log(f"-> {out}")


if __name__ == "__main__":
    {"config": cmd_config, "run2a": cmd_run2a, "run2b": cmd_run2b,
     "verdict": cmd_verdict}[sys.argv[1] if len(sys.argv) > 1 else "config"]()
