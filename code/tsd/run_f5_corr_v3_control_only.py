#!/usr/bin/env python3
"""F5 order agreement, v3 --- controls-only re-run (2026-08-31, control ledger).

Finding (2026-08-31, while red-lining the revision note): the persisted file
tsd_r4/f5_corr_v3_F5a-2.json still carried controls = {determinism_pass, vn_signflip_min 562,
vn_pass false} and verdict = STOP_controls while holding nine eta_v3 rows. The control block of
run_f5_corr_v3.py ran only when "determinism_pass" was absent, so after the sign-flip identity
control replaced the top-k control, the re-launch on F5a-2 did not run the new control (and the
old verdict remained). The commit message ("PASS after switching to the identity control")
disagreed with the persisted output. F5b-1 does carry flip_identity pass = true.

This script re-runs, for one variant, the same controls as the runner (determinism, sign-flip
identity, and the descriptive top-k value) on the SAME first chunk, and additionally checks the
persisted series against that chunk (|d| = 0). It also reports the identity on every single-id
pairing (the diagnosis that the two-id best-of form is not an identity under negation). The
original persisted file is not touched; the result is written to a separate file. No measurement.
REVISION-ID: TSD-20260826

usage: python run_f5_corr_v3_control_only.py --variant F5a-2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import run_f5_lens as L  # noqa: E402
import run_f5_corr as C  # noqa: E402
from joshaku.pgrain import build_word_set  # noqa: E402
from joshaku.ranks import alias_pair  # noqa: E402

RES = L.RES
OUT_DIR = RES / "tsd_r4"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["F5a-2", "F5b-1"])
    a = ap.parse_args()
    if C.self_test() != 0:
        print("⛔ 較正失敗")
        return 2
    pc2 = L.load_pc2()
    import screen_base as SB  # noqa: E402
    from run_f5 import build_prompt  # noqa: E402
    src = RES / f"f5_runs_{a.variant}.json"
    doc = json.loads(src.read_text())
    rows = {r["seed"]: r for r in doc["rows"]}
    base = SB.load_prompts()[L.CONDITION]
    newbase, _ = build_prompt(base, a.variant)
    prompt_text = newbase + L.CUE
    md5 = hashlib.md5(prompt_text.encode()).hexdigest()
    assert md5 == doc["prompt_md5_new"], f"⛔ prompt md5: {md5}"
    persisted = json.loads((OUT_DIR / f"f5_corr_v3_{a.variant}.json").read_text())
    assert persisted["prompt_md5_new"] == md5

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
    assert int(mask.sum()) == 70812

    def canon(words, name):
        ws_n = build_word_set(tok, name, words, mask_ids=None)
        return [int(e.canonical_id) for e in ws_n.entries if e.canonical_id is not None]
    hehi_ids = canon(["He", "Hi"], "hehi")
    apex_ids = canon(["diameter", "midpoint"], "apex_rank")
    alias_pair(apex_ids, hehi_ids)
    assert hehi_ids == persisted["sets_v3"]["hehi"] and apex_ids == persisted["sets_v3"]["apex_rank"], "⛔ sets_v3 不一致"
    sets = {"hehi": hehi_ids, "apex": apex_ids,
            "hehi_L": p2.set_ids(tok, p2.HEHI_WORDS), "apex_L": p2.set_ids(tok, p2.APEX_WORDS)}

    # runner と同じ「最初に処理される seed の初回 chunk」= sorted(rows) の先頭で content あり
    seed = next(s for s in sorted(rows) if rows[s]["content"].strip() and rows[s].get("label") is not None)
    r = rows[seed]
    n_prompt = len(tok(prompt_text)["input_ids"])
    full = prompt_text + r["content"]
    T = len(tok(full)["input_ids"]) - n_prompt
    positions = [p for p in range(n_prompt, n_prompt + T) if p < pc2.MAX_SEQ]
    chunk = positions[:p2.CHUNK]
    t0 = time.time()
    jl, ml, _ = lens.apply(model, full, layers=pc2.BAND, positions=chunk, max_seq_len=pc2.MAX_SEQ)
    jl2, ml2, _ = lens.apply(model, full, layers=pc2.BAND, positions=chunk, max_seq_len=pc2.MAX_SEQ)
    det = all(torch.equal(jl[l], jl2[l]) for l in pc2.BAND) and torch.equal(ml, ml2)
    v3sets = {"hehi": sets["hehi"], "apex": sets["apex"]}
    med_c, med_cm = C.pair_series(torch, p2, jl, ml, mask, v3sets)
    jl_neg = {k: -v for k, v in jl.items()}
    med_f = p2.band_median_ranks(torch, jl_neg, mask, v3sets)
    e_c, u_c, t_c = C.agreement(med_c["hehi"], med_c["apex"], med_cm["hehi"], med_cm["apex"])
    e_f, u_f, t_f = C.agreement(med_f["hehi"], med_f["apex"], med_cm["hehi"], med_cm["apex"])
    flip_ok = (u_c == u_f and t_c == t_f and (e_c is None or abs((1.0 - e_c) - e_f) < 1e-12))
    vn_desc = int(min(min(med_f["hehi"]), min(med_f["apex"])))
    # 診断(2026-08-31): flip 恒等は単一 id 集合でのみ厳密(best-of = min は負号で max に化ける)。
    # 同 chunk で単一 id 集合(各 2 通り)と、best-of を max に置いた読みで恒等が立つかを並記。
    diag = {}
    for nm, hs, as_ in (("single_0", hehi_ids[:1], apex_ids[:1]), ("single_1", hehi_ids[1:], apex_ids[1:]),
                        ("single_x", hehi_ids[:1], apex_ids[1:]), ("single_y", hehi_ids[1:], apex_ids[:1])):
        ss = {"hehi": hs, "apex": as_}
        mc, mcm = C.pair_series(torch, p2, jl, ml, mask, ss)
        mf = p2.band_median_ranks(torch, jl_neg, mask, ss)
        ec, uc, tc = C.agreement(mc["hehi"], mc["apex"], mcm["hehi"], mcm["apex"])
        ef, uf, tf = C.agreement(mf["hehi"], mf["apex"], mcm["hehi"], mcm["apex"])
        diag[nm] = {"ids": [hs, as_], "eta": ec, "eta_flip": ef, "n_used": [uc, uf], "n_tie": [tc, tf],
                    "identity_holds": bool(uc == uf and tc == tf and (ec is None or abs((1.0 - ec) - ef) < 1e-12))}
    # persist 済 series との再現(同 chunk・全 set)
    med_l, med_m = C.pair_series(torch, p2, jl, ml, mask, sets)
    prow = next(x for x in persisted["rows"] if x.get("seed") == seed)
    rep = {}
    for k in sets:
        ps = prow["series"][k][:len(chunk)]
        rep[k] = max(abs(float(x) - float(y)) for x, y in zip(med_l[k], ps))
    ok = bool(det and flip_ok)
    out = {"stage": "f5_corr_v3_controls_rerun", "date": "2026-08-31", "variant": a.variant,
           "reason": "persist の controls に flip_identity 無し・verdict STOP_controls 残存(runner の統制 skip 条件)",
           "seed_used": seed, "chunk_len": len(chunk), "band": [pc2.BAND[0], pc2.BAND[-1]],
           "controls": {"determinism_pass": bool(det),
                        "flip_identity": {"eta_chunk": e_c, "eta_flip": e_f, "n_used": u_c, "n_tie": t_c,
                                          "pass": bool(flip_ok)},
                        "vn_min_descriptive": vn_desc},
           "replication_vs_persisted_series_maxabs": rep,
           "diagnosis_singleton_sets": diag,
           "persisted_controls_as_found": persisted["controls"],
           "persisted_verdict_as_found": persisted.get("verdict"),
           "pass": ok, "wall_s": round(time.time() - t0, 1),
           "started": datetime.now().isoformat(timespec="seconds")}
    op = OUT_DIR / f"f5_corr_v3_{a.variant}_controls_rerun_20260831.json"
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    L.log(f"統制 det={'✅' if det else '⛔'} flip恒等={'✅' if flip_ok else '⛔'}(η={e_c} → flip {e_f}; n={u_c} tie={t_c}) "
          f"vn記述={vn_desc} 再現Δ={rep} -> {op}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
