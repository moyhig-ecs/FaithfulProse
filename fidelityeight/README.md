# FidelityEight — companion artifacts

Companion artifacts for **"FidelityEight: Frozen-Protocol Fidelity Measurements of the
Distributed Jacobian-Lens Checkpoints"** — Zenodo, FaithfulProse track:
**DOI [10.5281/zenodo.21990390](https://doi.org/10.5281/zenodo.21990390)** (v1, published
2026-08-18; all versions [10.5281/zenodo.21990389](https://doi.org/10.5281/zenodo.21990389);
the record links back to this tree via `isSupplementedBy`).

**v2.0.0 (2026-09-01, REVISION-ID: TSD-20260826)** accompanies note v2 — **DOI
[10.5281/zenodo.22220541](https://doi.org/10.5281/zenodo.22220541)**: the eight cells were
re-acquired under a repaired token-set acceptance and a ninth measurement was added in a
separate frame. Nothing from v1.0.0 was changed or removed (the three v1 evaluators are
checked byte-identical at build time); the additions live under `evaluators/tsd/` and
`outputs/tsd/`. See "What is new in v2.0.0" below.

Eight `wgrd` measurement cells (v1; re-acquired in v2.0.0, see below) over the pre-fitted Jacobian-lens checkpoints
distributed at `neuronpedia/jacobian-lens`: three distributed lenses (OLMo-3-7B,
OLMo-3-32B, Qwen3-8B) under two position-domain bindings, plus two measured floors
(chance = sign-flipped lens; baseline = vanilla logit-lens via J = I).

## Contents

- `evaluators/` — the three evaluator sources, **byte-identical** to the versions whose
  md5s are printed in the note (`eval_wgrd.py` = binding v1; `eval_wgrd_v2.py` =
  binding v2, p ≥ 16; `eval_wgrd_q.py` = mask-width variant for tokenizers whose
  length differs from the logits width, e.g. Qwen).
- `outputs/` — the eight per-item output JSONs. Per this repository's release
  convention, absolute source-tree paths in the JSON metadata were relativized;
  `MANIFEST.json` records the transformation and **both** checksums (`as_run_md5` =
  the value registered at measurement time, `released_md5` = the file here).
  The Zenodo attachments are the as-run files.
- `MANIFEST.json` — per-file transform record and checksums.

## Quick check (no downloads needed)

```
python evaluators/eval_wgrd.py --self-test
python evaluators/eval_wgrd_v2.py --self-test
python evaluators/eval_wgrd_q.py --self-test
```

Each evaluator refuses to measure unless its calibration battery (identity control,
hand-computed fixtures, negative control, determinism) passes in the same invocation.

## Full reproduction

Requires the upstream `jlens` package (github.com/anthropics/jacobian-lens), the
distributed lens checkpoints, and the models:

| Object | Identity |
|---|---|
| Lens OLMo-3-7B (n=616) | md5 `d21b82aee9ac2f66f357e01b6b988b51` |
| Lens OLMo-3-32B (n=470) | md5 `c73a32d1f72968bd73c104c06445a482` |
| Lens Qwen3-8B (n=461) | md5 `710e83bc455e9f2760f86befb639b691` |
| allenai/Olmo-3-1025-7B | revision `a81bae42db3975be1671e27b9c9a56da1a9f980f` |
| allenai/Olmo-3-1125-32B | revision `c2b61dae89a1ad10e4ad5653d0e46b590902607b` |
| Qwen/Qwen3-8B | revision `b968826d9c46dd6066d109eabc6255188de91218` |

The 266-item prose set is identified in the note by md5. Distributed per-layer lens
dicts are wrapped as `{jacobian_sum: J, n_done: 1}` (exact identity; the evaluator
divides by n).

Layout note: at measurement time the v1/v2 evaluators additionally pin two canon files
by md5 through relative paths (`FaithfulProse/release/code/step3_pass2.py` and
`FaithfulProse/frozen_en/DECISION_TABLE.en.md`, resolved two directories above the
evaluator). Those files are this repository's `code/step3_pass2.py` and
`frozen/DECISION_TABLE.en.md`; to reproduce the metadata pins, place them at the
source-tree locations above. `--self-test` needs no layout.

## What is new in v2.0.0 (REVISION-ID: TSD-20260826)

The v1 rule "a word that splits into sub-word pieces is represented by its first token"
(the `set_ids` semantics imported verbatim, note §2) admits fragments into the variant
group — e.g. `'sp'` for *spider* — and the min-over-group then reads the fragment's rank.
The rule is registered as defect T1. In v2 the measured object is the emitted next-token
id itself (no variant bundle, no canonical mapping); the v1 variant-bundle minimum is
computed beside it in the same run as a legacy juxtaposition only. All eight cells were
re-acquired; the chance floors are the calibration gate (arithmetic prediction
log10(N/2) with a band drawn before the run: OLMo 4.5488 vs 4.5491 ± 0.053, in band;
Qwen, new, 4.5105 vs 4.5437 ± 0.061, in band on the low side at 1.6σ, flagged). The v1
values are superseded and are not counted as evidence; they remain in the note, marked.

- `evaluators/tsd/eval_wgrd_v3.py` — the v2 evaluator (both bindings via `--binding`,
  `--sign_flip` for the floors). It calls the instrument library **joshaku** for the
  word-like mask and, in its self-test, for a 200/200 rank bit-agreement check
  (`joshaku.masks`, `joshaku.ranks`). joshaku is **not vendored** here: it is archived
  as its own record, v1.2.0, [10.5281/zenodo.22218669](https://doi.org/10.5281/zenodo.22218669)
  (place the `joshaku/` package on `sys.path`, e.g. two directories above this file, as
  the source layout does). The runs recorded joshaku at combined md5 `61356a4a…` (first
  frozen state; SPEC v1); the archived v1.2.0 is `89b194be…` — of the seven modules the
  evaluator imports two, `masks.py` (byte-identical) and `ranks.py` (`word_rank`
  unchanged; v1.2 adds alias functions not used here). One line of the evaluator is
  transformed for release: the upstream `jlens` source path is read from the `JLENS_SRC`
  environment variable instead of an absolute path (`MANIFEST.json` records both md5s;
  the note prints the as-run one, `b6be3947…`).
- `outputs/tsd/*.json` — the nine re-acquired runs (`c_floor_7b`, `c_baseline_7b`,
  `p_7b_v1`, `p_7b_v2`, `p_32b_v1`, `p_32b_v2`, `q_qwen_v1`, `q_qwen_v2`, `c_floor_qwen`),
  as-run (no absolute paths; `transform: none`). Each `meta` records the reading-card
  md5, the SPEC md5, the joshaku fingerprint, sign flip, and model pin, and — in eight of
  the nine — the binding: the first run (`c_floor_7b`, the OLMo chance floor) preceded
  the `--binding` option and ran on all positions (the v1 binding); the evaluator state
  of that first run was not separately pinned (the note, §9 Instrument). Each
  item carries `wgrd_item` (v2, the emitted token) and `legacy_wgrd_item` (v1 semantics,
  same run). Cell value = median over items, computed outside the evaluator.
- `outputs/tsd/r3fid_v1.json` — the ninth measurement (note §9 Part III): the distributed
  Qwen3-32B lens read on 20 reasoning traces, in-mask positions only (36% dropped and
  registered), pooled median 1.7324 dex. Per-trace series and controls (determinism,
  sign flip) are included; the runner belongs to the CommitStage code record
  ([10.5281/zenodo.21617993](https://doi.org/10.5281/zenodo.21617993), all versions).

Quick check for the v2 evaluator (needs joshaku on `sys.path`; no downloads):

```
python evaluators/tsd/eval_wgrd_v3.py --self-test
```

## Status

Archival companion to the note; not actively maintained. Questions and replication
reports are welcome via issues.
