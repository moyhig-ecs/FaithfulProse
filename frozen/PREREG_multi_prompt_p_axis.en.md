# Pre-registration: multi-prompt extension of the P axis (English rendering)

> **Original**: `PREREG_r3_multi_prompt_p_axis_20260803.md` (Japanese)
> **Frozen at**: commit `ddecebc`, 2026-08-03 11:47:19 JST
> **sha256**: `890da859c10c97ebe0b408e1487c8820…`
> **Governs**: the extension run — process start **11:47:26** (7 s after the
> freeze), results committed `fd19133`, 2026-08-03 12:01:24 JST
>
> ⚠ English rendering prepared 2026-08-03 — see `PROVENANCE.md`. Not itself a
> frozen artifact.

**Status of the original: frozen.** ⛔ Nothing was run until it was committed.

**Purpose.** The deep-prompt degradation was observed on a single prompt read
four times, which licenses nothing about prompts in general. Extend the P axis
to two further prompts and decide whether it reproduces. ⛔ Thresholds,
buckets and quantities are inherited unchanged; none are invented here.

---

## 0. Two corrections established before the run

### 0-1. The identifier of one prompt was cited wrongly upstream of this document

The commissioning document identified one prompt by a hash that turns out to be
the hash of the prompt **without** the cue string appended. The hash of the
string actually fed to the model differs. This pre-registration anchors on the
latter. ⭕ No data is affected — the stored artifact was always correct — and
the discrepancy was caught **before** the run.

### 0-2. ⚠ The n on the P axis is not a count of independent samples

Established from stored data only, with **no new computation**:

```
  The four traces of the initial measurement all share one prompt. In a causal
  language model the state at a prompt position depends only on the tokens up
  to that position, so the generated continuation cannot affect it.

  Measured:
    eager   15 prompt-side positions × 4 traces  ⇒ all 15 identical
    sdpa    the same 15 positions                ⇒ three distinct values
            ⚠ but the determinism check reports sdpa as deterministic within a
              process ⇒ the spread is a numerical difference across processes /
              sequence lengths. ⛔ No mechanism is claimed.
```

⇒ The previously reported counts of 32 / 16 / 12 are, under eager, **8 / 4 / 3
distinct values counted four times each**.
⇒ Therefore the value of this extension lies in the **number of prompts**, not
the number of rows.
⇒ **This pre-registration reports n two ways** (§4).

## 1. The sample (frozen — ⛔ not chosen after looking)

### 1-1. Prompts (anchored against the tokenizer, measured 2026-08-03)

| prompt | md5 (**cue included** — the string actually fed to the model) | length |
|---|---|---|
| 1 | `039389308493aa2f3580c2a0d7011f69` | **458** |
| 2 | `c79980dc3ece8144207f486e455c9496` | **470** |
| 3 (already measured) | `cc05d498c10ca9c4acf3b030a5884236` | **501** |

⛔ The prompt construction rule is unchanged: base prompt → edit variant → cue
appended. Prompt 1 is the base with no edit.

### 1-2. Seed rule (⛔ mechanical; content is not inspected)

```
  ★ Rule: scanning seeds in ascending order, take the first two traces with
          more than 512 tokens.

  ⚠ Why the length cutoff is a requirement and not a preference:
     bucket ceilings are clipped at (n_tokens − 1). A trace of 512 tokens or
     fewer therefore has its second bucket's sample points shifted, and
     ★ the same absolute positions can no longer be compared across prompts.
     Observed: one candidate has 484 tokens and its grid shifts accordingly
     ⇒ ⛔ disqualified by the rule.

  ⚠ Selecting on length introduces no bias on the P axis — by §0-2 the prompt
     side is trace-independent.
  ⛔ But the generation-offset axis of this run must NOT be read as an
     unbiased sample.
```

**Result of applying the rule (fixed before the run):** prompt 1 → seeds 1, 2
(639 / 2351 tokens); prompt 2 → seeds 1, **3** (3145 / 595 tokens; seed 2
disqualified at 484 tokens). ⇒ **four new texts.**

### 1-3. Positive control (⛔ the measurement is not read without it)

```
  Re-measure an already certified trace with the instrument used here, and
  compare against the stored values at every position and every quantity.
  ⇒ ★ This turns "the instrument is a verbatim copy" from an assumption into
       a measurement.
  ⛔ The control uses the deterministic attention path (§0-2).
```

## 2. Inherited and unchanged

Bucket definitions (absolute, and the P / O relative buckets), sampling
points, quantities, word groups and rank computation (imported verbatim), the
read-out band, the lens hash assertion, the attention plan, and **all
thresholds** — rank correlation 10% / 30%, word-group rank deviation 0.10 /
0.30 dex.

⛔ No new quantity. ⛔ No threshold moved. ⛔ The certified results file is not
rewritten; output goes to a new file.

## 3. Instrument

```
  ★ Constants and helpers are imported from the frozen runner; the frozen
    runner itself is not edited by a single character.
  ★ The measurement block is a verbatim copy, and the correctness of that copy
    is established by the controls in §1-3 rather than asserted.
  ⛔ The run refuses to start on a dirty working tree.
```

## 4. How n is reported (frozen)

```
  ★ Always report both:

    rows      the number of measurement rows (comparable with the earlier report)
    distinct  the number of distinct (prompt, position) cells
              ⇒ under eager this is the number of independent values

  ⛔ Never one without the other. ⛔ Never call the row count "the sample size".
```

Expected values, fixed before the run (they follow mechanically from the grid,
and are not results): prompt-internal buckets, rows / distinct = 64 / 24,
32 / 12, 24 / 9.

> ⚠ **Note added after the run** (disclosed in the paper's limitations): the
> sharper predicate is `(prefix, position)`, not `(prompt, position)`, because
> the three prompts share prefixes. Under that predicate the counts are
> 11 / 8 / 8. This was identified **after** the run and is not part of the
> freeze.

## 5. The reading table (frozen before measurement)

**Question:** is the deep-prompt degradation a property of prompts in general,
or of the one prompt measured first?

Decided per prompt, on the decision group, with the shallowest bucket as
baseline. **"Degraded" = rank-correlation drop ≥ 30% ∨ word-group rank
deviation ≥ +0.30 dex** (⛔ the existing thresholds verbatim).

| result | reading |
|---|---|
| ★ **All three prompts degraded** | ★★ the single-prompt limitation is resolved; the finding is promoted from suggestive to a reported cell |
| ★ **Two of three degraded, one flat** | ⚠ prompt-dependent. ⛔ Not promoted; state that the shape differs by prompt and keep it suggestive |
| ★ **Both new prompts flat** | ★★★ the degradation is specific to the first prompt ⇒ ⛔ withdraw the claim to an appendix observation ⇒ report immediately |
| ★ **Direction reversed in a new prompt** | **not anticipated**. Report on the spot; the reading goes to review |

⛔ All are observations. ⛔ The verdicts of the main decision table are not
moved. ⛔ The generation-offset axis is reported but not read (§1-2).

## 6. Stopping conditions (frozen — chosen to measure what they are meant to measure)

| # | condition | action |
|---|---|---|
| **1** | the positive control does not reproduce the stored values exactly | ⛔ **stop**; the copy is not verbatim; report the difference |
| **2** | the lens hash does not match | ⛔ stop |
| **3** | a prompt hash does not match §1-1 | ⛔ stop |
| **4** | a selected trace's token count does not match §1-2 | ⛔ stop (the tokenizer environment has changed) |
| **5** | the sample points of a new text do not match the canonical grid | ⛔ stop (comparison would be meaningless) |

⚠ **What condition 1 compares is specified**: per-position, per-quantity float
values, exactly. ⛔ Not a count of rows — a count can agree while the values do
not.

---

**Scope tag**: four new texts (two prompts × two seeds) under both attention
implementations; the four existing traces of prompt 3 are reused, not
re-measured; output to a separate file; ⛔ not used for any verdict.
