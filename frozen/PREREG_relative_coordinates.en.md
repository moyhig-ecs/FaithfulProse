# Pre-registration: re-aggregation in relative coordinates (English rendering)

> **Original**: `PREREG_r2_relative_coordinates_20260802.md` (Japanese)
> **Frozen at**: commit `0b7ccaf`, 2026-08-02 15:02:07 JST
> **sha256**: `febe6dd0f0cb0b307c256398b7b04297…`
> **Governs**: the re-aggregation, committed `67b481f`, 2026-08-02 15:16:25 JST (+14 m)
>
> ⚠ English rendering prepared 2026-08-03 — see `PROVENANCE.md`. Not itself a
> frozen artifact.

**Status of the original: frozen.** The instruction under which it was
written:

> Freeze the bucket definitions for the re-aggregation (prompt side and
> generation-offset side) and the empty frame of the second figure **before
> computing anything**. **Do not cut buckets after seeing the data** — apply to
> the next figure exactly the discipline the first one just demonstrated.

⛔ **No new model run.** This re-slices the coordinates of the 700 stored
measurements. ⛔ The verdicts of the main decision table are not touched; this
is an attribution diagnostic.

**The reason it was needed:** the reasoning-trace buckets straddle the
prompt/generation boundary (the prompt is 501 tokens). Absolute position
therefore confounds "how deep" with "prompt or generation".

---

## 1. The bucket definitions (frozen before computation)

### 1-1. Reasoning traces (prompt + generation, prompt length 501)

```
  ★ Coordinate P — depth within the prompt:   p        (1 ≤ p ≤ n_prompt)
     P1   1 – 128        ← same width as the fit domain
     P2 129 – 320
     P3 321 – 501        ← to the end of the prompt

  ★ Coordinate O — offset from the first generated token:  o = p − n_prompt   (o ≥ 1)
     O1    1 –  128      ← the first 128 generated tokens (same width as the fit domain)
     O2  129 –  512
     O3  513 – 2048
     O4 2049 – 4096
     O5 4097 – 8191
     O6 8192 – 16384

  ⛔ The boundaries are the existing absolute buckets moved onto the offset
     axis, unchanged. They are not re-cut to taste.
  ⛔ The three-way split on the P side follows a mechanical rule: cut at the
     fit-domain width (128), then divide the remainder into two.
```

### 1-2. The prose corpus has no corresponding coordinate

```
  The prose corpus is concatenated text with no prompt/generation distinction
  (n_prompt = 0).
  ⇒ ★ It stays in absolute coordinates.
  ⇒ ⛔ Prose and reasoning traces are not compared under the same bucket name:
       the coordinate systems differ.
  ⚠ This does not discard the prose corpus. It makes explicit, in the
    coordinates themselves, that prose answers a different question (the effect
    of position as such) from the one the traces answer (the effect of
    prompt versus generation).
```

## 2. Quantities (identical to the existing report — ⛔ no new metric)

```
  rank correlation / word-group rank deviation (= max over the two groups) /
  KL / top-1 / baseline-lens rank correlation / baseline-lens KL
  + the split into full-attention and sliding-attention layers

  ★ State the actual n in every cell (cells with n < 8 are not hidden)
  ★ The baseline is the first cell of each coordinate system (P1 / O1) —
    stated explicitly as being a different baseline from the absolute one
```

## 3. The reading table (frozen before computation)

| result | reading |
|---|---|
| ★ **Monotone degradation along O, flat along P** | degradation accompanies **the progress of generation**, not absolute position. The straddling confound was the main cause |
| ★ **Degradation along both P and O** | the confound does not explain it — an **absolute-position contribution remains**. The scope clause stands more strongly |
| **Flat along both** | ⚠ the degradation seen in absolute coordinates may be **an artifact of how the buckets were cut** ⇒ report immediately |
| **Degradation along P, flat along O** | ⇒ **not anticipated**. Report on the spot; the reading goes to review |

⛔ All four are observations, not verdicts. ⛔ The verdicts of the main
decision table are not moved: this is an attribution diagnostic, not material
for a decision.

## 4. The empty frame of the second figure (frozen in this commit)

```
  left panel   x = P1..P3 (depth within the prompt)     y = degradation vs P1
  right panel  x = O1..O6 (generation offset)           y = degradation vs O1
  ★ The decision lines (10% / 0.10 dex, 30% / 0.30 dex) are drawn at the same
    positions as in the first figure
     ⚠ but ⛔ the values in this figure are not used for a decision — the lines
       are for reference only
  ★ The generation boundary (o = 1) is marked
  ★ Points beyond the limit are shown with triangles and their printed values
    (same discipline as the first figure — ⛔ the axis is not widened)
  ★ The prose corpus is not plotted here (different coordinate system, §1-2)
```

---

**Scope tag**: the 700 measurements of the main run (no new run); coordinate
transform applied to the reasoning traces only (prompt length 501); the prose
corpus stays in absolute coordinates; ⛔ not used for any verdict.
