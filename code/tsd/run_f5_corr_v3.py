#!/usr/bin/env python3
"""F5 order agreement, v3 --- the TSD/R4-c re-acquisition (READCARD R4, sec. 3).

REVISION (2026-08-28, token-set defect repair):
  The old sets were built as encode(word)[0], i.e. a multi-token word represented by its first
  token ("O1O2" -> 'O', "diameter" -> 'd'). v3 builds the sets with joshaku (P-grain canonical
  variants, SPEC v1.2 rule 3-8: alias best-of over {He, Hi} and {diameter, midpoint}; cardinality
  2 vs 2). The point labels O1O2/O1/O2 have no single-token variant and are recorded as missing
  by design. The eta definition, pair_series and agreement are imported verbatim from
  run_f5_corr.py. The legacy sets are re-run in the same pass and printed beside (per-trace
  replication control |d eta| <= 0.005; a failure is registered as environment dependence and
  diagnosed, not silenced). In-run determinism and the sign-flip control are stop conditions,
  run before measurement.
  REVISION-ID: TSD-20260826

REVISION 3.1 (2026-08-31, three standing fixes from the adjudication of the control ledger):
  1. a resume never continues from a persisted file whose verdict is STOP_* (a new file, or an
     explicit --override-stop "<reason>" that writes the reason into the file);
  2. a replaced control is stamped as CONTROLS_VERSION in every variant's controls dict, and a
     version mismatch re-runs the control block (a commit message is not evidence; the output is);
  3. a set-level control must first state whether it holds under the set aggregation: best-of
     (min) turns into max under negation, so the sign-flip identity eta_flip = 1 - eta is
     well-posed only for single-id sets. The stop condition is therefore the identity on every
     single-id pairing; the two-id best-of form is printed as descriptive only.
  This revision does not rewrite the R4-c persisted files (f5_corr_v3_F5a-2/F5b-1.json); the
  calibration record is a separate file (controls_rerun) plus the written record.

usage: python run_f5_corr_v3.py --variant F5a-2|F5b-1 --prereg-commit <hash> [--allow-dirty] [--override-stop "<reason>"]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import run_f5_lens as L  # noqa: E402
import run_f5_corr as C  # noqa: E402  pair_series / agreement / deltas_rho / self_test verbatim
from joshaku.pgrain import build_word_set  # noqa: E402
from joshaku.ranks import alias_pair  # noqa: E402

RES = L.RES
OUT_DIR = RES / "tsd_r4"
READCARD = REPO / "conversations/2026-08-28/READCARD_R4_sal_series_v1_20260828.md"
REP_TOL = 0.005   # 凍結(札 §3)
CONTROLS_VERSION = "v3.1-single-id-flip"   # ② 統制の版(差し替え時は上げる・全 variant で再走)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["F5a-2", "F5b-1"])
    ap.add_argument("--prereg-commit", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--override-stop", default="",
                    help="① STOP_* の persist を継ぐときの理由(persist に刻む)。無指定なら停止")
    a = ap.parse_args()
    if C.self_test() != 0:
        print("⛔ 較正失敗 ⇒ 測定しない")
        return 2
    rel = str(READCARD.relative_to(REPO))
    got = subprocess.run(["git", "log", "-1", "--format=%H", "--", rel],
                         capture_output=True, text=True, cwd=REPO).stdout.strip()
    if got != a.prereg_commit:
        raise SystemExit(f"⛔ 札 hash 不一致: {got[:12]} ≠ {a.prereg_commit[:12]}")
    prov = L.provenance(f"{a.variant}-v3", a.allow_dirty)
    L.log(f"R7-a ✅ commit={prov['commit'][:8]} dirty={prov['dirty']} stage=corr-v3 {a.variant}")

    pc2 = L.load_pc2()
    import screen_base as SB  # noqa: E402
    from run_f5 import build_prompt  # noqa: E402
    src = RES / f"f5_runs_{a.variant}.json"
    doc = json.loads(src.read_text())
    rows = {r["seed"]: r for r in doc["rows"]}
    old_corr = json.loads((RES / f"f5_corr_{a.variant}.json").read_text())
    old_eta = {r["seed"]: r.get("eta") for r in old_corr["rows"] if "eta" in r}

    base = SB.load_prompts()[L.CONDITION]
    newbase, _ = build_prompt(base, a.variant)
    prompt_text = newbase + L.CUE
    md5 = hashlib.md5(prompt_text.encode()).hexdigest()
    assert md5 == doc["prompt_md5_new"], f"⛔ prompt md5: {md5}"

    import torch, transformers, jlens  # noqa: E401
    assert hashlib.md5(pc2.LENS_PT.read_bytes()).hexdigest() == pc2.LENS_MD5
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        pc2.MODEL_NAME, dtype=torch.bfloat16, attn_implementation="eager").to("mps")
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(pc2.MODEL_NAME)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(str(pc2.LENS_PT))
    p2 = pc2.p2mod()
    ids = json.loads(pc2.MASK_PATH.read_text())
    mask = torch.zeros(len(tok), dtype=torch.bool); mask[ids] = True
    mask_ids = set(int(i) for i in ids)
    assert int(mask.sum()) == 70812, f"⛔ N: {int(mask.sum())}"
    L.log(f"stack ✅ BAND={pc2.BAND[0]}..{pc2.BAND[-1]} N={int(mask.sum())}")

    # ---- v3 集合(canonical・両 mask 構築 bit 一致 assert・落ち登記)----
    def canon(words, name):
        ws_n = build_word_set(tok, name, words, mask_ids=None)
        ws_m = build_word_set(tok, name, words, mask_ids=mask_ids)
        for en, em in zip(ws_n.entries, ws_m.entries):
            assert set(i for _, i in en.accepted) == set(i for _, i in em.accepted), \
                f"⛔ mask 二重構築不一致: {en.word}"
        cids = [int(e.canonical_id) for e in ws_n.entries if e.canonical_id is not None]
        dropped = [{"word": e.word, "variant": d.variant, "pieces": d.pieces,
                    "reason": d.reason} for e in ws_n.entries for d in e.dropped]
        missing = list(ws_n.missing)
        return cids, dropped, missing

    hehi_ids, drop_h, miss_h = canon(["He", "Hi"], "hehi")
    apex_ids, drop_a, miss_a = canon(["diameter", "midpoint"], "apex_rank")
    lab_ids, drop_l, miss_l = canon(["O1O2", "O1", "O2"], "labels_by_design")
    alias_pair(apex_ids, hehi_ids)   # 濃度 2 対 2 assert(SPEC v1.2 3-8)
    sets = {"hehi": hehi_ids, "apex": apex_ids,
            "hehi_L": p2.set_ids(tok, p2.HEHI_WORDS),
            "apex_L": p2.set_ids(tok, p2.APEX_WORDS)}
    L.log(f"sets ✅ v3 hehi={hehi_ids} apex={apex_ids} labels_missing={miss_l}"
          f"(by-design 欠測) dropped={len(drop_h)+len(drop_a)+len(drop_l)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"f5_corr_v3_{a.variant}.json"
    doc_out = json.loads(out_path.read_text()) if out_path.exists() else {
        "stage": "f5_corr_v3", "revision_id": "TSD-20260826",
        "readcard_md5": hashlib.md5(READCARD.read_bytes()).hexdigest(),
        "variant": a.variant, "prompt_md5_new": md5, "lens_md5": pc2.LENS_MD5,
        "band": [pc2.BAND[0], pc2.BAND[-1]], "mask_n": int(mask.sum()),
        "sets_v3": {"hehi": hehi_ids, "apex_rank": apex_ids},
        "labels_by_design_missing": miss_l,
        "dropped_variants": drop_h + drop_a + drop_l,
        "rep_tol": REP_TOL, "provenance": prov, "controls": {}, "rows": []}
    # ---- ① resume は STOP_* を継がない ----
    if str(doc_out.get("verdict", "")).startswith("STOP"):
        if not a.override_stop:
            raise SystemExit(f"⛔ persist の verdict = {doc_out['verdict']}: resume しない"
                             f"(新 file にするか --override-stop \"<理由>\" で理由を刻む)")
        doc_out.setdefault("verdict_history", []).append(
            {"verdict": doc_out["verdict"], "controls_as_found": doc_out.get("controls", {}),
             "override_reason": a.override_stop, "override_commit": prov["commit"]})
        del doc_out["verdict"]
        doc_out["controls"] = {}   # 統制は必ず再走
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
        s = {k: [] for k in sets}
        sm = {k: [] for k in sets}
        for i in range(0, len(positions), p2.CHUNK):
            chunk = positions[i:i + p2.CHUNK]
            jl, ml, _ = lens.apply(model, full, layers=pc2.BAND,
                                   positions=chunk, max_seq_len=pc2.MAX_SEQ)
            # ---- 統制先行(初回 chunk のみ・停止機)----
            # ⚠ 錨修正の登記(2026-08-28): 初版は top-k V-N(min > 10⁴)を置き 562 で
            #   走行前停止。η は順序(符号)レーンで top-k 錨は読む統計と対でない
            #   (562 = プール最深部の鏡像・順序読みに異常でない)= R3 135・R2-11 3,648 と
            #   同族の統計量-錨対応違い・三例目・いずれも統制先行が測定前に捕捉。
            #   lane 正しい flip 統制に差し替え: 符号反転で per-position の順序は全反転
            #   ⇒ η_flip = 1 − η(tie 数同一)の機械恒等を assert(較正 ② の run 内形)。
            if doc_out["controls"].get("controls_version") != CONTROLS_VERSION:   # ② 版が違えば再走
                jl2, ml2, _ = lens.apply(model, full, layers=pc2.BAND,
                                         positions=chunk, max_seq_len=pc2.MAX_SEQ)
                det = all(torch.equal(jl[l], jl2[l]) for l in pc2.BAND) and \
                    torch.equal(ml, ml2)
                jl_neg = {k: -v for k, v in jl.items()}

                def flip_pair(hs, as_):
                    ss = {"hehi": hs, "apex": as_}
                    mc, mcm = C.pair_series(torch, p2, jl, ml, mask, ss)
                    mf = p2.band_median_ranks(torch, jl_neg, mask, ss)
                    ec, uc, tc = C.agreement(mc["hehi"], mc["apex"], mcm["hehi"], mcm["apex"])
                    ef, uf, tf = C.agreement(mf["hehi"], mf["apex"], mcm["hehi"], mcm["apex"])
                    ok = (uc == uf and tc == tf and (ec is None or abs((1.0 - ec) - ef) < 1e-12))
                    return {"ids": [hs, as_], "eta": ec, "eta_flip": ef, "n_used": uc,
                            "n_tie": tc, "identity_holds": bool(ok)}, mf
                # ③ 集約に対する恒等: best-of(min)は負号で max に化けるので恒等は単一 id 集合でのみ well-posed。
                #    停止機 = 単一 id の全 pairing(4 通り)が厳密成立。2 id best-of は記述並記(非情報)。
                single = {}
                for nm, hs, as_ in (("single_0", sets["hehi"][:1], sets["apex"][:1]),
                                    ("single_1", sets["hehi"][1:], sets["apex"][1:]),
                                    ("single_x", sets["hehi"][:1], sets["apex"][1:]),
                                    ("single_y", sets["hehi"][1:], sets["apex"][:1])):
                    single[nm], _ = flip_pair(hs, as_)
                flip_ok = all(v["identity_holds"] for v in single.values())
                bestof, med_f = flip_pair(sets["hehi"], sets["apex"])
                vn_desc = int(min(min(med_f["hehi"]), min(med_f["apex"])))
                doc_out["controls"] = {
                    "controls_version": CONTROLS_VERSION,
                    "determinism_pass": bool(det),
                    "aggregation_note": "flip 恒等 η_flip = 1 − η は単一 id 集合でのみ厳密(best-of の min は負号で max)。"
                                        "停止機 = 単一 id 4 pairing 全成立。2 id best-of は記述並記(非情報)",
                    "flip_identity_single": {**single, "pass": bool(flip_ok)},
                    "flip_bestof_descriptive": {k: v for k, v in bestof.items() if k != "ids"},
                    "vn_min_descriptive": vn_desc,
                    "anchor_note": "初版 top-k V-N(>10⁴)は lane 不適合で 562 停止(登記)→ flip 恒等へ差し替え"
                                   "(順序レーンの読む統計と同対)→ v3.1 で単一 id 集合に定義し直し(ADJ 08-31 夜)"}
                single_txt = ", ".join(f"{k}:{v['eta']}->{v['eta_flip']}" for k, v in single.items())
                L.log(f"  統制 det={'✅' if det else '⛔'} 単一id flip恒等={'✅' if flip_ok else '⛔'}({single_txt}) "
                      f"best-of記述 η={bestof['eta']}->{bestof['eta_flip']} [vn記述={vn_desc}]")
                if not (det and flip_ok):
                    doc_out["verdict"] = "STOP_controls"
                    out_path.write_text(json.dumps(doc_out, ensure_ascii=False, indent=1))
                    raise SystemExit("⛔ 統制 FAIL")
            med_l, med_m = C.pair_series(torch, p2, jl, ml, mask, sets)
            for k in sets:
                s[k].extend(med_l[k]); sm[k].extend(med_m[k])
        eta3, used3, tie3 = C.agreement(s["hehi"], s["apex"], sm["hehi"], sm["apex"])
        etaL, usedL, tieL = C.agreement(s["hehi_L"], s["apex_L"], sm["hehi_L"], sm["apex_L"])
        rep = None
        if seed in old_eta and old_eta[seed] is not None and etaL is not None:
            rep = round(abs(etaL - old_eta[seed]), 6)
        row = {"seed": seed, "label": r["label"], "T": len(s["hehi"]),
               "lens_truncated": truncated,
               "eta_v3": None if eta3 is None else round(eta3, 6),
               "n_used_v3": used3, "n_tie_v3": tie3,
               "eta_legacy": None if etaL is None else round(etaL, 6),
               "n_used_legacy": usedL, "n_tie_legacy": tieL,
               "eta_legacy_old": old_eta.get(seed),
               "replication_abs_delta": rep,
               "replication_pass": None if rep is None else rep <= REP_TOL,
               "wall_s": round(time.time() - t1, 1),
               "series": {k: s[k] for k in sets},
               "series_model": {k: sm[k] for k in sets}}
        doc_out["rows"].append(row)
        out_path.write_text(json.dumps(doc_out, ensure_ascii=False, indent=1) + "\n")
        L.log(f"  s{seed} T={row['T']:5d} η_v3={row['eta_v3']} (tie {tie3}) "
              f"η_L={row['eta_legacy']} 旧Δ={rep} ({time.time()-t1:.0f}s)")
    # ② 統制が本 variant で実際に走ったことを実出力で示す(message は証拠でない)
    assert doc_out["controls"].get("controls_version") == CONTROLS_VERSION, "⛔ controls 未走(版不一致)"
    # 集計(記述)
    vals = [r["eta_v3"] for r in doc_out["rows"] if r.get("eta_v3") is not None]
    if vals:
        import statistics
        doc_out["summary_v3"] = {"n": len(vals), "min": min(vals), "max": max(vals),
                                 "median": round(statistics.median(vals), 6)}
        rep_fails = [r["seed"] for r in doc_out["rows"]
                     if r.get("replication_pass") is False]
        doc_out["replication_summary"] = {
            "fails": rep_fails,
            "reading_if_fail": "停止でなく環境依存の登記 + 診断走で切り分け(札 §3・F6 の型)"}
    out_path.write_text(json.dumps(doc_out, ensure_ascii=False, indent=1) + "\n")
    L.log(f"⭕ saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
