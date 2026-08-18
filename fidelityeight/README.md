# FidelityEight — companion artifacts

Companion artifacts for **"FidelityEight: Frozen-Protocol Fidelity Measurements of the
Distributed Jacobian-Lens Checkpoints"** (Zenodo, FaithfulProse track; the record links
back to this tree via `isSupplementedBy`).

Eight `wgrd` measurement cells over the pre-fitted Jacobian-lens checkpoints
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

## Status

Archival companion to the note; not actively maintained. Questions and replication
reports are welcome via issues.
