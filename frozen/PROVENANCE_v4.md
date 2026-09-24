# PROVENANCE_v4 --- the instrument provenance note of paper v3.2 (2026-09-24)

Every OLMo-3-32B reading of v1.0.0--v3.0.0 used the lens artifact distributed on 2026-06-16; upstream refitted it on
2026-09-21. The two artifacts, the forward they were read on, and the records of the re-read campaign:

| | June artifact (all previous readings) | refitted artifact (v3.1.0 re-reads) |
|---|---|---|
| Hub revision (neuronpedia/jacobian-lens) | `a4114d7752d11eb546e6cf372213d7e75526d3a1` | `496cf110a6efbf6db491c916e47ba74bc894ef8d` |
| md5 / sha256 | `c73a32d1f72968bd73c104c06445a482` / `4d9e440da655490e23c52cb1bc343dd85fe1271131d746ff595c5e825f028292` | `b76610b9e0e989cd689cc5c0c7f6d354` / `c6ee7d22c3a224fd1a6f9bb66ee9d23c7c4ac678f2f554e0f3815bbe0540a508` |
| bytes / prompts fitted / final identity distance | 3,303,033,940 / 470 / 0.130071 | 3,303,034,580 / 456 / 0.129729 |
| fitted under | transformers 5.11.0 (huggingface/transformers#39847: YaRN rope_scaling applied to the sliding-window layers; corrected in 5.13.0 by #46911) | transformers 5.17.0 (recorded in the artifact's provenance field) |
| read out on | transformers 5.14.1, torch 2.11.0, python 3.14.3, MPS (both lenses; same forward) | same |

The June artifact therefore encoded a forward that the library had already corrected when the read-outs ran (the lens
hash and the transformers version were pinned; the fit environment of the lens was not). `data/lens/` holds the re-reads
with the refitted artifact; `data/lens/control/` the June artifact re-read the same day (platform drift is read between
the June frozen files and these controls; the lens effect between the controls and `data/lens/`). The wrapper
`code/lens/lens_swap.py` runs each frozen runner unchanged, swaps the lens in memory and redirects every write under
`results/` to `results/lens_<tag>/`; the runners' own provenance blocks (commit, dirty flag, versions, lens md5) are kept,
with the variant suffixed. Invocations (repo-relative): `python3 lens_swap.py --runner <frozen runner> --tag c6ee7d22 -- <runner args>`
(control: `--tag c73a32d1-rerun --no-swap`); Arm C: `run_arms_v3_lensdelta.py --fire --stage all --coherence-max-v3 10482.15 --allow-dirty`,
then `--stage nulls`; the position calibration: `run_c_lens_pos.py --attn-only eager` then `--attn-only sdpa` under the wrapper with `--fresh-read c_lens_pos.json`.

| document (Japanese original, not reproduced) | role | md5 (first 8) | sha256 | last commit (author date) |
|---|---|---|---|---|
| `RECORD_jlens_olmo3_yarn_misfit_and_refit_20260921.md` | RECORD: the misfit (transformers 5.11.0 YaRN) and the upstream refit; inventory of affected artifacts | `c2440c22` | `a7fec37769d4a33773661f55406afeafd80c62fc9fce04bc41ea5b8fe75525d4` | bcea7b1b719027f8e23ee2c16cc5374b603d099c 2026-09-21T18:27:11+09:00 |
| `RESULT_lensdelta_arms_v3_20260921.md` | RESULT: three-way comparison (June frozen / June re-read / refitted) for every arm, ladder and series | `fa5e64ae` | `26ff67736da880c26dee0ed557dc34cc7ebef8e2745178c86b8c936391d3d5e3` | ee74faf0705b024fa4680dda695cb1ac92dfe334 2026-09-23T16:36:56+09:00 |
| `TABLE_lens_oldfit_vs_newfit_20260921.md` | TABLE: the two lens files compared layer by layer (float64) | `61bc41dd` | `a8acc534d563c9d14e12aa2e5a8573809dc29d3101834ea70fb9b13a995af448` | f8f55357d087d7be6cab771e5623566b2eb97bd0 2026-09-22T12:18:00+09:00 |
| `RECORD_lens_c_chain5_launch_20260922.md` | RECORD: the full re-read campaign (order, wrapper semantics, exclusions) | `99fa14ca` | `30ef035cf1808ffd52245ea79bdbca17f4d727a07e33f050c507ac618b89016c` | ad0bcee6ea68bfc8098cecf892e1301942ebd5cd 2026-09-23T10:54:36+09:00 |

What moved (paper v3.2, Sections 2.7 and the coordinate table): no verdict of the frozen decision table changes; the
levels shift (band KL +0.17 nats median; word-group rank deviation +0.1 to +0.3 dex on the reasoning-trace cells; the
Arm C null threshold 1,666 -> 1,310). The June values stand as measurements of the June artifact.
