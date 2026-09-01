# Faithful on Prose, Unanchored on Reasoning — data and code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21768365.svg)](https://doi.org/10.5281/zenodo.21768365)

Companion release for the preprint

> **Faithful on Prose, Unanchored on Reasoning:
> A Position-Domain Calibration of the Jacobian Lens**
> Manabu Higashida, D3 Center, The University of Osaka

The paper calibrates the fidelity of a released Jacobian lens
([Gurnee et al., 2026](https://transformer-circuits.pub/2026/workspace/))
against token position, on OLMo-3-1125-32B. Its methodological commitment is
that the decision table, the thresholds, and both figure frames were frozen
and committed **before** any measurement existed. This package is what makes
that claim checkable.

## What is new in v3.0.0 (2026-09-01) --- REVISION-ID: TSD-20260826

Version 3 of the paper re-acquires every reading-set measurement of the second campaign under a
repaired token-set acceptance, and this release adds the artifacts of that re-acquisition.
Nothing from v1.0.0 or v2.0.0 was changed or removed (both are checked byte for byte at build
time); the additions are:

* `data/tsd/` --- the re-acquired read-outs: Arms A and C (`arm_a_v3.json`, `arm_c_v3.json`), both
  MapFirst ladders (`cii_l1_v3.json`, `cii_l2_v3.json`) with the re-derived null ledgers
  (`cii_nulls_v3.json`) and the population-null bound of the SameSeat two-stage reading
  (`r2b_derivation.json`), the ShapeJitter floor re-derived with every series persisted
  (`r211_v3.json`), and the order-agreement lanes (`f5_corr_v3_F5a-2.json`, `f5_corr_v3_F5b-1.json`)
  with the controls re-run after the fact (`*_controls_rerun_20260831.json`). Every file carries
  the legacy (version-2) lane beside the repaired one. Data is never transformed.
* `code/tsd/` --- the v3 runners (`run_arms_v3.py`, `run_r211_v3.py`, `run_f5_corr_v3.py`, its
  controls-only companion) and the frozen order-agreement stage they import verbatim
  (`run_f5_corr.py`). The rise predicate, the gate and the floor statistics are still imported
  from `code/arms/` (single source). The runners require the acceptance instrument **joshaku
  v1.2.0**, which is *not* vendored here: it is its own record,
  [`10.5281/zenodo.22218669`](https://doi.org/10.5281/zenodo.22218669) (seven modules, combined
  md5 `89b194be...`, 20 tests; SPEC v1.2 included). Unpack it next to this package or install it
  from the archive; the runners import `joshaku.pgrain` and `joshaku.ranks`.
* `frozen/PROVENANCE_v3.md` --- the hashes and commit timestamps of the three frozen reading cards
  of the re-acquisition (Japanese originals, not reproduced) and of SPEC v1.2.

**What the defect was.** The version-2 reading sets represented a multi-token word by its first
token (spider -> `sp`, Shakespeare -> `Sh`, insect -> `in`), a rule that had been pre-registered
and is now registered as defect T1. The repaired acceptance admits a surface variant only if it
encodes as exactly one token and records every drop. Under it, Arm A is (a)-type (the same
integers newly observed), the SameSeat onset (3, 2, 3, 3) is newly observed under a
population-null bound, the order-agreement clause of the paper is converted from a supporting
clause to a limitation (0.456--0.692, median 0.544; 0.5 is the sign-independent reference), and
the ShapeJitter floor is re-registered (max 0.3372 dex at the deepest rung). The version-2
values are marked in the paper, not deleted, and are not counted as evidence. The paper's
Revision History section is the full statement.

**Calibration record (order agreement).** The controls of the F5a-2 half were re-run after the
fact (determinism; per-trace replication d = 0.0; sign-flip identity on single-id sets, exact for
both variants). The two-id identity control originally specified is non-informative by
construction (best-of over two ids turns a minimum into a maximum under negation) and is
replaced; the values are unchanged. `run_f5_corr_v3.py` v3.1 carries the three standing fixes
(no resume from a STOP file; versioned controls; set-level controls stated for the aggregation).

## What is new in v2.0.0 (2026-08-21)

Version 2 of the paper reports a second campaign, and this package release adds its
artifacts. Nothing from v1.0.0 was changed or removed; the additions are:

* `data/arms/` — the per-cell read-outs of **Arm A, Arm B and Arm C** (`arm_a.json`,
  `arm_b.json`, `arm_c.json`) and of the two **MapFirst** ladders (`cii.json` = ladder 1,
  corridors 88–242 tokens; `cii_c2.json` = ladder 2, corridors 286–484), the **halted** first
  attempt of ladder 1 (`cii_v1_m7.json`, stopped in its calibration phase — no measurement), and
  the pre-run gate artifacts (`gate_*.json`). Data is never transformed.
* `code/arms/` — the runners and gates of the second campaign, plus the shared helper
  (`run_f5_lens.py`) they import for provenance, logging and the frozen lens / band constants.
  The rise predicate, the behavioral-invariance gate and the floor statistics live in
  `run_arm_a.py` / `run_arm_b.py` / `run_arm_c.py` and are imported verbatim by the MapFirst
  runners (single source).
* `frozen/PREREG_arms_a_b_c.en.md`, `frozen/PREREG_mapfirst_ladder.en.md` — English renderings
  of the two pre-registrations (each frozen in successive states), and rows 4–9 of
  `frozen/PROVENANCE.md` with the commit timestamps and `sha256` of the Japanese originals.

**Names.** Internal identifiers are kept in the code and data: `GT-A/GT-B/GT-C` = Arm A/B/C;
`C-ii` = MapFirst; rung ids `r8 … r44` = corridor length / 11; row ids `P1/P2/P3/B1/P5` =
Mars / spider / Canada / France / Shakespeare (the intermediate entity of each prompt). The
paper uses the external names.

**Two cautions specific to v2.** (i) The fitting script of the lens excludes positions below
16 as well as capping the sequence at 128; every read-out of Arm A lies inside that excluded
band, and its validity rests on the controls that passed in the same run, not on the fit.
(ii) Beyond 128, an unchanged onset is recorded under the frozen reading rule (branch B4) as
descriptive only — "it did not break out of domain" is not a calibration.

## What is here

```
  data/     measurements, per-position values retained (never transformed)
  code/     the measurement runners and the imported upstream instrument
  figs/     the two figure scripts and their output
  frozen/   the frozen decision table and pre-registrations, and the evidence
            that they preceded the measurements
  MANIFEST.json   sha256 of every file, in both as-run and released form
  LICENSE   MIT
```

### `frozen/` — read this first if you care about the pre-registration claim

`PROVENANCE.md` holds the checkable part. Every frozen document's commit
timestamp is paired with the commit timestamp of the measurement it governs:

| frozen | committed (JST) | measurement committed | interval |
|---|---|---|---|
| decision table | 2026-08-02 10:08:50 | 2026-08-02 14:45:59 | **+4 h 37 m** |
| pre-registration, relative coordinates | 2026-08-02 15:02:07 | 2026-08-02 15:16:25 | **+14 m** |
| pre-registration, multi-prompt extension | 2026-08-03 11:47:19 | 2026-08-03 12:01:24 | **+14 m** |

For the third, the run log records the process starting **seven seconds** after
the pre-registration commit.

⚠ The originals are in Japanese; `frozen/` contains **English renderings
prepared after the fact**, and the decision table is an **extract** — scope
clauses belonging to a separate, unpublished programme that used the same
instrument are omitted. Nothing in the provenance claim rests on the
renderings: what fixes *when* is the commit timestamp, and what fixes *what* is
the `sha256` of the original, both recorded in `PROVENANCE.md`. The originals
can be supplied for a byte-level audit.

### `data/`

| file | contents |
|---|---|
| `c_lens_pos.json` | main run — **700 measurements** (`W` prose 348 + `G` reasoning traces 352) |
| `c_lens_pos_r1.json` | control: attention implementation isolated at equal sequence length |
| `c_lens_pos_r2.json` | relative-coordinate re-aggregation — **no new model run** |
| `c_lens_pos_r3.json` | multi-prompt extension — **224 measurements** + controls A and B |
| `det_check_sdpa.json` | determinism check (two applications, identical settings) |
| `vn_*.json` | neutrality checks |
| `preflight*.json` | preflight measurements |
| `c_lens_pos_prefix_mixed.json` | prefix-mixed diagnostic |
| `wordlike_mask_olmo3_32b.json` | word-like token mask derived from the tokenizer |

Every measurement retains its per-position values; the aggregated cells in
each file are reproducible from them.

### `code/` and `figs/`

`run_c_lens_pos.py` is the frozen measurement runner. `run_r3.py` performs the
multi-prompt extension: it **imports the frozen runner rather than
reimplementing it**, and verifies that it is a faithful copy with two positive
controls before taking any measurement —

* **Control A** — the aggregator reproduces the published re-aggregation
  (170 values, 0 differing);
* **Control B** — the measurement path reproduces a previously certified trace
  (520 values, 0 differing).

A mismatch in either was a pre-registered stopping condition.

The word groups, the rank computation, the token mask and the lens interface
are imported verbatim from the upstream implementation (`step3_pass2.py`).

The figure scripts draw an **empty frame** when no data file is present. That
is the mechanism by which the axes, bucket boundaries and decision lines could
be committed before the data existed.

## Reproducing

Requires the model weights and the fitted lens, which are not redistributed
here:

* model: `allenai/Olmo-3-1125-32B`
* lens: the released Jacobian lens artifact, md5 `c73a32d1f72968bd73c104c06445a482`

Paths are resolved from environment variables:

```sh
export REPO_ROOT=/path/to/this/checkout
export JACOBIAN_LENS_ROOT=/path/to/jacobian-lens
python code/run_c_lens_pos.py --preflight     # measure nothing; report cost
python code/run_c_lens_pos.py                 # main run
python code/run_r3.py                         # extension (controls run first)
python figs/fig_C1_fidelity_vs_position.py    # empty frame if data absent
python figs/fig_C2_relative_coordinates.py
```

The runners refuse to start on a dirty working tree, and record the commit,
the library versions and the start time in each output file's `provenance`
block.

## What was transformed for release

`MANIFEST.json` records, for every file, the sha256 of the file **as it ran**
and the sha256 **as released**, together with an enumeration of every
substitution. The policy:

| | |
|---|---|
| measurements | verbatim — never transformed |
| absolute paths in code | replaced by environment-variable resolution |
| docstrings | translated to English |
| internal vocabulary in comments | replaced by the external register used in the paper |
| identifiers naming a separate, unpublished programme | replaced by neutral names |
| inline comments | left in Japanese where they were |

The two hashes let anyone holding the original verify that nothing else moved.

## Caveats worth reading before reusing these numbers

1. **The reasoning-trace traces are not independent samples.** They cover 3
   prompts over 8 traces, but the prompts are edit variants sharing prefixes
   (divergence at tokens 86 and 354), and in a causal language model the state
   at a prompt position depends only on tokens up to that position. Before a
   divergence point, different prompts return *exactly* the same value. The
   correct unit for counting independent values is `(prefix, position)`.
2. **Readings depend slightly on run configuration.** Agreement is exact only
   when the set of positions read together and the sequence length also match
   (median 0.009 dex, max 0.031 dex — far below the 0.30 dex decision
   threshold, but not zero).
3. **The generation-side offset distribution is not an unbiased sample**: the
   extension traces were selected by a length rule. We do not read it, and
   neither should you.
4. The prose corpus is the lens's own fit dataset, which favours it.

## Licensing

| | |
|---|---|
| this package — code, data, frozen documents | **MIT** (see `LICENSE`; the Zenodo deposit carries the same) |
| the paper, as a preprint on Zenodo | **CC BY 4.0** |

MIT is a software licence, and it is what the Zenodo deposit inherited from the
repository. To be unambiguous about the parts that are not software: **the
measurements and the frozen documents in this package are released on the same
permissive terms as the code** — reuse them, redistribute them, build on them;
please cite the paper or the DOI.

The paper argues that a calibration should be checkable by anyone who cares to
check it. Licensing that gets in the way of checking would be a strange way to
end it.

## Archive

| | |
|---|---|
| this release (v3.0.0) | [`10.5281/zenodo.22219123`](https://doi.org/10.5281/zenodo.22219123) --- **version DOI**, pins these exact bytes; the paper v3 cites it |
| v2.0.0 | [`10.5281/zenodo.22041132`](https://doi.org/10.5281/zenodo.22041132) --- **version DOI**, pins the bytes that accompanied paper v2 |
| v1.0.0 | [`10.5281/zenodo.21768365`](https://doi.org/10.5281/zenodo.21768365) --- **version DOI**, pins the bytes that accompanied paper v1.0 / v1.1 |
| all versions | [`10.5281/zenodo.21768364`](https://doi.org/10.5281/zenodo.21768364) --- concept DOI, always resolves to the latest |
| the paper (v3) | [`10.5281/zenodo.22219346`](https://doi.org/10.5281/zenodo.22219346) --- version DOI of the v3.0 record (2026-09-01). Cite this one |
| the paper (v2) | [`10.5281/zenodo.22041724`](https://doi.org/10.5281/zenodo.22041724) --- **version DOI**, pins that exact PDF (16 pages, 2026-08-21); its readout-side values are superseded by v3 |
| the paper (v1.1) | [`10.5281/zenodo.21867046`](https://doi.org/10.5281/zenodo.21867046) --- **version DOI**, superseded by v2 |
| the paper (v1) | [`10.5281/zenodo.21800315`](https://doi.org/10.5281/zenodo.21800315) --- **version DOI**, superseded by v1.1 |
| the paper, all versions | [`10.5281/zenodo.21800314`](https://doi.org/10.5281/zenodo.21800314) --- concept DOI, always resolves to the latest |

v1.1 revises the abstract and one sentence of Section 7-2. The results,
figures, frozen artifacts and verdicts are unchanged from v1.0, and no claim
was withdrawn; the code and data in this release are untouched by it.

v2 (paper) and v2.0.0 add the second campaign --- see "What is new in v2.0.0"
above. v3 (paper) and v3.0.0 (this package) add the TSD-20260826 re-acquisition
--- see "What is new in v3.0.0" above. Everything from v1.0.0 and v2.0.0 is still
here, byte for byte; the v3 additions are under `data/tsd/`, `code/tsd/` and
`frozen/PROVENANCE_v3.md`.

The paper cites the version DOI, because what it needs to point at is the
snapshot the numbers came from. Cite the paper the same way, for the same
reason.

## Citation

```bibtex
@misc{higashida2026faithfulprose,
  author = {Manabu Higashida},
  title  = {Faithful on Prose, Unanchored on Reasoning:
            A Position-Domain Calibration of the Jacobian Lens},
  year   = {2026},
  note   = {Preprint (v2), Zenodo.
            \url{https://doi.org/10.5281/zenodo.22041724}.
            Data and code archived at
            \url{https://doi.org/10.5281/zenodo.22041132}}
}
```

This work was supported by JSPS KAKENHI Grant Number JP26K06399.
