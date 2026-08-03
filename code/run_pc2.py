#!/usr/bin/env python
"""Driver from a separate, unpublished research programme that uses the same
instrument.

For the present paper this module supplies only frozen constants --- the model
name, the lens path and its md5, the read-out band, the token-mask path, and
the loader for the upstream rank/word-group implementation. The rest of the
module belongs to that other programme and is included unchanged so that the
imports resolve and the run reproduces byte-for-byte.

Nothing in it is reimplemented here: the word groups, the rank computation and
the mask are imported verbatim from `step3_pass2`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
import os
from pathlib import Path

REPO = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))
JL = Path(os.environ.get("JACOBIAN_LENS_ROOT", "jacobian-lens"))
PILOT = REPO / "experiments/jlens-pilot"
for p in (JL, REPO / "probes/olmo3-base-bf16-screen"):
    sys.path.insert(0, str(p))

HERE = Path(__file__).parent
RES = HERE / "results"; RES.mkdir(parents=True, exist_ok=True)
TRAJ = RES / "pc2_trajectories.json"
SEATS = RES / "pc2_seats.json"
MASK_PATH = RES / "wordlike_mask_olmo3_32b.json"

# ---- 凍結パラメータ（PREREG §2・実走後に変更しない） -------------------------
PREREG = "PREREG_pc2_seat_cue_invariance_20260731.md"
FROZEN_COMMIT = "35f3ec1"                       # ★ PREREG 凍結 commit
MODEL_NAME = "allenai/Olmo-3-1125-32B"
LENS_PT = REPO / "probes/jlens-identity-distance/provenance/lens_metadata/olmo-3-1125-32b/Olmo-3-1125-32B_jacobian_lens.pt"
LENS_MD5 = "c73a32d1f72968bd73c104c06445a482"
BAND = list(range(26, 57))                      # 層 26..56 = 深さ比 0.4127-0.8889（63 層）
CUE_TAG = {" ": "space", "\n\n": "nlnl", "\n": "nl"}
T_MIN = 200                                     # ★ P2 (A): -a の定義域
WINDOW = 0.15                                   # ★ -a の窓（= 1.5 x 分解能 0.10）
MARGIN = 2                                      # ★ -b の margin（規律 R3）
MAX_SEQ = 8192                                  # 凍結計器 lens.apply の窓（先例 run_8b_p8 と同値）

SOURCES = [
    (REPO / "probes/olmo3-base-bf16-screen/results/screen_olmo3_base_bf16.json", "\n"),
    (REPO / "probes/olmo3-pc1-cue-invariance/results/pc1_runs.json", None),  # cue は行が持つ
]


def log(m): print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def p2mod():
    spec = importlib.util.spec_from_file_location("p2", PILOT / "step3_pass2.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["p2"] = m
    spec.loader.exec_module(m)
    return m


def load_rows():
    """Return the stored rows as (key, cue, condition, label, content, hit_cap). Nothing is re-scored."""
    out = []
    for path, fixed_cue in SOURCES:
        rows = json.loads(path.read_text())["rows"]
        for r in rows:
            cue = fixed_cue if fixed_cue is not None else r["cue"]
            tag = CUE_TAG[cue]
            out.append({"key": f"{tag}|{r['condition']}|s{r['seed']}", "cue": cue, "cue_tag": tag,
                        "condition": r["condition"], "label": r["label"],
                        "content": r["content"] or "", "hit_cap": bool(r.get("hit_cap"))})
    return out


# ------------------------------------------------------------------ lens pass
def cmd_lens():
    import torch, transformers, jlens
    import screen_base as SB                                     # 凍結 screen（import のみ）

    got = hashlib.md5(LENS_PT.read_bytes()).hexdigest()
    if got != LENS_MD5:
        raise SystemExit(f"⛔ lens md5 不一致: {got} != {LENS_MD5}")
    log(f"lens md5 ✅ {LENS_MD5}")

    prompts = SB.load_prompts()                                  # ★ payload md5 assert はこの中
    log("payload md5 ✅ 窓内と一致")

    p2 = p2mod()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, attn_implementation="eager").to("mps")
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(str(LENS_PT))
    if lens.d_model != hf.config.hidden_size:
        raise SystemExit(f"⛔ d_model 不一致: lens {lens.d_model} != model {hf.config.hidden_size}")
    log(f"stack ✅ layers={hf.config.num_hidden_layers} d_model={lens.d_model} "
        f"n_prompts={lens.n_prompts} BAND={BAND[0]}..{BAND[-1]}")

    if MASK_PATH.exists():
        ids = json.loads(MASK_PATH.read_text())
    else:
        ids = [t for t in range(len(tok)) if p2.WORDLIKE.match(tok.decode([t]).strip())]
        MASK_PATH.write_text(json.dumps(ids))
    mask = torch.zeros(len(tok), dtype=torch.bool); mask[ids] = True
    sets = {"hehi": p2.set_ids(tok, p2.HEHI_WORDS),
            "apex": p2.set_ids(tok, p2.APEX_WORDS),
            "center": p2.set_ids(tok, p2.CENTER_WORDS)}

    done = json.loads(TRAJ.read_text()) if TRAJ.exists() else {
        "prereg": PREREG, "frozen_commit": FROZEN_COMMIT, "lens_md5": LENS_MD5,
        "BAND": [BAND[0], BAND[-1]], "trajectories": {}}

    rows = load_rows()
    log(f"素材 {len(rows)} 行（新規生成ゼロ）  既存 traj {len(done['trajectories'])}")
    for r in rows:
        if r["key"] in done["trajectories"]:
            continue
        meta = {k: r[k] for k in ("cue_tag", "condition", "label", "hit_cap")}
        if r["label"] is None or not r["content"].strip():
            done["trajectories"][r["key"]] = {**meta, "excluded": "label=None or 空"}
            TRAJ.write_text(json.dumps(done, ensure_ascii=False, indent=1) + "\n")
            log(f"  skip {r['key']}: 測定不成立（分母外）")
            continue

        # ★ PREREG §2-2: base 腕の T = 生成継続のトークン数
        prompt_text = prompts[r["condition"]] + r["cue"]         # E1: chat template なし
        n_prompt = len(tok(prompt_text)["input_ids"])
        full = prompt_text + r["content"]
        T = len(tok(full)["input_ids"]) - n_prompt
        # ⚠ 凍結計器の窓 max_seq_len=8192 に当たる行がある（実測 6/60）。
        #   計器側の制限であって判定条件の変更ではない。⇒ clamp して事実を記録する。
        positions = [p for p in range(n_prompt, n_prompt + T) if p < MAX_SEQ]
        lens_truncated = len(positions) < T

        t0 = time.time()
        traj = {s: [] for s in sets}
        for i in range(0, len(positions), p2.CHUNK):
            chunk = positions[i:i + p2.CHUNK]
            jl_logits, _, _ = lens.apply(model, full, layers=BAND,
                                         positions=chunk, max_seq_len=MAX_SEQ)
            med = p2.band_median_ranks(torch, jl_logits, mask, sets)
            for s in sets:
                traj[s].extend(med[s])
        T_eff = len(traj["hehi"])                                # max_seq_len 打ち切り後の実長
        done["trajectories"][r["key"]] = {**meta, "T": T_eff, "T_tokens": T,
                                          "lens_truncated": lens_truncated,
                                          "hehi": traj["hehi"], "apex": traj["apex"],
                                          "center": traj["center"],
                                          "wall_s": round(time.time() - t0, 1)}
        TRAJ.write_text(json.dumps(done, ensure_ascii=False, indent=1) + "\n")
        log(f"  lens {r['key']} ({r['label']}): T={T_eff} ({time.time()-t0:.0f}s)")
    log(f"lens done -> {TRAJ}")


# ------------------------------------------------------------------- verdict
def seat_of(tr, p2):
    """Return the pre-registered readout category (four values) and the derived position ratio."""
    T = tr["T"]
    dom_h = [tr["hehi"][t] < tr["apex"][t] for t in range(T)]
    dom_a = [tr["apex"][t] < tr["hehi"][t] for t in range(T)]
    th, ta = p2._establish(dom_h, T), p2._establish(dom_a, T)
    need = max(20, int(0.1 * T))
    if th is not None and ta is None:
        return "HEHI", th / T, th, ta, need
    if ta is not None and th is None:
        return "APEX", ta / T, th, ta, need
    if th is None and ta is None:
        return "彷徨", None, th, ta, need
    if abs(th - ta) <= need:                                    # ★ P3
        return "両立", None, th, ta, need
    return ("HEHI", th / T, th, ta, need) if th < ta else ("APEX", ta / T, th, ta, need)


def fisher_or_chi2(table):
    """3xK contingency table: Fisher if scipy is available, chi-squared otherwise; both are returned."""
    out = {}
    try:
        from scipy import stats
        try:
            out["fisher_p"] = float(stats.fisher_exact(table)[1]) if len(table[0]) == 2 else None
        except Exception:
            out["fisher_p"] = None
        if out.get("fisher_p") is None:
            try:
                res = stats.fisher_exact(table) if len(table[0]) == 2 else None
                out["fisher_p"] = float(res[1]) if res else None
            except Exception:
                out["fisher_p"] = None
        try:
            nz = [row for row in table if sum(row) > 0]
            cols = [j for j in range(len(table[0])) if sum(r[j] for r in nz) > 0]
            red = [[r[j] for j in cols] for r in nz]
            out["chi2_p"] = float(stats.chi2_contingency(red)[1]) if len(red) > 1 and len(red[0]) > 1 else None
        except Exception:
            out["chi2_p"] = None
        # 多次元 Fisher（scipy>=1.16 の monte carlo）
        if out.get("fisher_p") is None:
            try:
                out["fisher_p"] = float(stats.fisher_exact(red, method="monte-carlo")[1])
                out["fisher_note"] = "monte-carlo（3xK）"
            except Exception:
                pass
    except ModuleNotFoundError:
        out["error"] = "scipy 不在"
    return out


def cmd_verdict():
    p2 = p2mod()
    d = json.loads(TRAJ.read_text())
    trajs = d["trajectories"]

    rows = []
    for key, tr in trajs.items():
        if tr.get("excluded"):
            rows.append({"key": key, **{k: tr.get(k) for k in ("cue_tag", "condition", "label")},
                         "excluded": tr["excluded"]})
            continue
        S, tau, th, ta, need = seat_of(tr, p2)
        rows.append({"key": key, "cue_tag": tr["cue_tag"], "condition": tr["condition"],
                     "label": tr["label"], "hit_cap": tr["hit_cap"], "T": tr["T"],
                     "S": S, "tau": tau, "t_hehi": th, "t_apex": ta, "need": need,
                     "T_ge_200": tr["T"] >= T_MIN})

    CUES = ["nl", "space", "nlnl"]
    SEATS_ALL = ["HEHI", "APEX", "両立", "彷徨"]
    verdict = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "prereg": PREREG, "frozen_commit": FROZEN_COMMIT,
               "lens_md5": LENS_MD5, "BAND": [BAND[0], BAND[-1]],
               "window": WINDOW, "T_MIN": T_MIN, "MARGIN": MARGIN,
               "rows": rows, "by_condition": {}}

    for cond in ("blocked", "open"):
        live = [r for r in rows if r.get("condition") == cond and "excluded" not in r]
        out = {"n": len(live), "excluded": len([r for r in rows if r.get("condition") == cond]) - len(live)}

        # ---- -a: word-group readoutごと・T>=200 限定（P1 P2） ----
        a = {}
        for S in ("HEHI", "APEX"):
            per = {c: [r["tau"] for r in live if r["cue_tag"] == c and r["S"] == S and r["T_ge_200"]]
                   for c in CUES}
            if any(len(v) == 0 for v in per.values()):
                a[S] = {"n_per_cue": {c: len(v) for c, v in per.items()}, "verdict": "判定不能",
                        "reason": "適格行 0 の cue がある"}
                continue
            med = {c: statistics.median(v) for c, v in per.items()}
            spread = max(med.values()) - min(med.values())
            a[S] = {"n_per_cue": {c: len(v) for c, v in per.items()},
                    "median_tau": {c: round(m, 4) for c, m in med.items()},
                    "spread": round(spread, 4), "verdict": "成立" if spread <= WINDOW else "不成立"}
        decided = [v["verdict"] for v in a.values() if v["verdict"] != "判定不能"]
        out["-a"] = {"by_seat": a,
                     "verdict": ("判定不能" if not decided else
                                 "成立" if all(v == "成立" for v in decided) else "不成立"),
                     "T<200 除外数": {c: len([r for r in live if r["cue_tag"] == c and not r["T_ge_200"]])
                                      for c in CUES}}

        # ---- -b: modal readout category・両立除外・margin（P3 R3） ----
        b = {}
        for c in CUES:
            votes = Counter(r["S"] for r in live if r["cue_tag"] == c and r["S"] != "両立")
            top = votes.most_common()
            margin = (top[0][1] - top[1][1]) if len(top) > 1 else (top[0][1] if top else 0)
            b[c] = {"votes": dict(votes), "modal": top[0][0] if top else None, "margin": margin,
                    "両立除外": len([r for r in live if r["cue_tag"] == c and r["S"] == "両立"])}
        modals = [v["modal"] for v in b.values()]
        if any(v["margin"] < MARGIN for v in b.values()):
            bv = "判定不能"
        elif len(set(modals)) == 1:
            bv = "成立"
        else:
            bv = "不成立（確定的な非同一）"
        out["-b"] = {"by_cue": b, "verdict": bv}

        # ---- -c: cue x S の 3x4 ----
        table = [[len([r for r in live if r["cue_tag"] == c and r["S"] == S]) for S in SEATS_ALL]
                 for c in CUES]
        st = fisher_or_chi2(table)
        p = st.get("fisher_p") if st.get("fisher_p") is not None else st.get("chi2_p")
        out["-c"] = {"table": {"cues": CUES, "seats": SEATS_ALL, "counts": table},
                     **st, "verdict": ("判定不能" if p is None else "成立" if p > 0.05 else "不成立")}

        vs = [out["-a"]["verdict"], out["-b"]["verdict"], out["-c"]["verdict"]]
        out["不変"] = all(v == "成立" for v in vs)
        verdict["by_condition"][cond] = out

    # ---- drink（★ P4 非対称） ----
    fail = []
    for cond, o in verdict["by_condition"].items():
        if o["-a"]["verdict"] == "不成立": fail.append(f"{cond}/-a")
        if o["-c"]["verdict"] == "不成立": fail.append(f"{cond}/-c")
        if o["-b"]["verdict"].startswith("不成立"): fail.append(f"{cond}/-b(確定的非同一)")
    debt = [f"{cond}/-b" for cond, o in verdict["by_condition"].items()
            if o["-b"]["verdict"] == "判定不能"]
    verdict["drink"] = {"実行層 drink": bool(fail), "根拠": fail,
                        "設計負債(判定不能・台帳外)": debt}

    SEATS.write_text(json.dumps(verdict, ensure_ascii=False, indent=1) + "\n")

    for cond, o in verdict["by_condition"].items():
        print(f"\n=== {cond} ===  n={o['n']}（測定不成立 {o['excluded']}）")
        print(f"  -a {o['-a']['verdict']}   T<200 除外 {o['-a']['T<200 除外数']}")
        for S, v in o["-a"]["by_seat"].items():
            print(f"     [{S}] {v['verdict']}  n={v['n_per_cue']}"
                  + (f"  median_tau={v['median_tau']}  spread={v['spread']}" if "spread" in v else ""))
        print(f"  -b {o['-b']['verdict']}")
        for c, v in o["-b"]["by_cue"].items():
            print(f"     {c:6} modal={v['modal']:<6} margin={v['margin']}  votes={v['votes']}  両立除外={v['両立除外']}")
        print(f"  -c {o['-c']['verdict']}  fisher_p={o['-c'].get('fisher_p')} chi2_p={o['-c'].get('chi2_p')}")
        print(f"     {o['-c']['table']['seats']}")
        for c, row in zip(CUES, o["-c"]["table"]["counts"]):
            print(f"     {c:6} {row}")
        print(f"  ★ 不変 = {o['不変']}")
    print(f"\n★ drink: {verdict['drink']}")
    print(f"artifact -> {SEATS}")


if __name__ == "__main__":
    {"lens": cmd_lens, "verdict": cmd_verdict}[sys.argv[1]]()
