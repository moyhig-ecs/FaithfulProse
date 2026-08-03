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
| internal vocabulary in comments | replaced by the external register used in the paper |
| Japanese comments | retained |

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

## Archive

| | |
|---|---|
| this release (v1.0.0) | [`10.5281/zenodo.21768365`](https://doi.org/10.5281/zenodo.21768365) --- **version DOI**, pins these exact bytes |
| all versions | [`10.5281/zenodo.21768364`](https://doi.org/10.5281/zenodo.21768364) --- concept DOI, always resolves to the latest |

The paper cites the version DOI, because what it needs to point at is the
snapshot the numbers came from.

## Citation

```bibtex
@misc{higashida2026faithfulprose,
  author = {Manabu Higashida},
  title  = {Faithful on Prose, Unanchored on Reasoning:
            A Position-Domain Calibration of the Jacobian Lens},
  year   = {2026},
  note   = {arXiv preprint. Data and code archived at
            \url{https://doi.org/10.5281/zenodo.21768365}}
}
```

This work was supported by JSPS KAKENHI Grant Number JP26K06399.
