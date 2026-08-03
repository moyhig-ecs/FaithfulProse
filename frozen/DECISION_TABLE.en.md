# Frozen decision table (English rendering)

> **Original**: `DESIGN_c_lens_pos_v2_20260802.md` (Japanese)
> **Frozen at**: commit `5023560`, 2026-08-02 10:08:50 JST
> **sha256**: `75aea6a2391a06573c533adc5454031d…`
> **Governs**: the main run, committed `f0a7787`, 2026-08-02 14:45:59 JST (+4 h 37 m)
>
> ⚠ This is an **extract, rendered into English on 2026-08-03** — see
> `PROVENANCE.md`. It is not itself a frozen artifact. Scope clauses concerning
> a separate unpublished programme are omitted; everything load-bearing for
> this paper is rendered in full.

---

## 1. The fifth quantity, and why it was added

The instrument whose calibration is at stake consumes the **ranks of two word
groups**, and in practice those ranks lie two to five orders of magnitude deep
in the distribution.

> Rank correlation over the model's top-1000 measures fidelity at the **head**
> of the distribution. It does not guarantee fidelity at the depth the
> instrument reads. **A lens whose head is faithful and whose flank is
> distorted could pass a decision table built on rank correlation alone.**

Accordingly a fifth quantity was added before measurement, and the decision
was made the **worst of** {rank correlation, word-group rank deviation}.

### Definition (the mechanism is imported verbatim from the upstream implementation)

```
  Ranks are taken over word-like tokens only (the frozen token mask), and
  within a word group the minimum (best) rank is taken — this is the upstream
  implementation, not a reimplementation.

  rank_model(S, p)      rank of word group S under the model's final-layer logits
  rank_lens (S, p, l)   the same under the lens logits at layer l

  wgrd(S, p) = median over the read-out band of
               | log10 rank_lens(S,p,l) − log10 rank_model(S,p) |

  wgrd(p)    = max( wgrd(group A, p), wgrd(group B, p) )      ← worst across groups too
```

The worst is taken across the two groups because the downstream reading is a
**comparison between them**: if only one group is faithful, the comparison is
still distorted. Averaging is prohibited.

**Unit note.** `wgrd` is an absolute difference in `log10` rank; 0.3 means
"off by about a factor of two". Rank correlation is reported as a **ratio** to
the baseline cell, `wgrd` as an **absolute increase**; §4 fixes the direction
of each explicitly.

A third word group is computed but **not used in any decision** — reported
only.

## 2. Layers

```
  All 31 layers of the read-out band.

  Reason: not memory, but estimator identity. The downstream reading is a
  band median; computing it over a subset would make it a different estimator
  from the one being calibrated.

  A representative 8-layer subset is retained only as a fallback if the
  preflight cost is prohibitive — and if it is invoked, the run does not
  proceed: the design goes back for review, because the estimator changed.
```

## 3. Common decision conventions (frozen before the run)

| | |
|---|---|
| **(a)** | The decision group is the **reasoning-trace corpus**. The prose corpus is an **attribution diagnostic**: even the interaction of domain with position is decided on the reasoning traces. |
| **(b)** | Rank correlation: **degradation = 1 − bucket/baseline** (a ratio). Word-group rank deviation: **absolute Δ = bucket − baseline** (in dex). ⛔ No ratio is applied to the absolute quantity, and no floor is applied. |
| **(c)** | The decision is the **worst of** {rank correlation, word-group rank deviation}. |
| **(d)** | Treating the deepest bucket as in-domain additionally requires **n ≥ 16** in that bucket. If n is short, the branch is automatically "hold" — a thin cell does not certify. |
| **(e)** | "Sign reversal against the baseline lens" is decided on the band-median `KL`. |
| **(f)** | The >30% branch carries a re-adjudication of a downstream criterion that depends on these readings. |

**Thresholds (frozen):** rank correlation **10% / 30%**; word-group rank
deviation **0.10 / 0.30 dex**; deepest-bucket top-1 agreement must satisfy
**top1(deepest) ≥ top1(previous) − 5 points (absolute)**; implementation-bridge
guard **0.02 dex**; minimum n for in-domain certification **16**.

The 0.30 dex threshold was anchored, before measurement, against the margin
the downstream reading actually consumes (+0.80 to +1.32 dex): **0.30 dex is
about one third of that margin.**

## 4. The decision branches

### 4-1. Whether to commission measurement in the extrapolation region (deepest bucket)

| worst-of {rank corr., wgrd}, decision group, vs baseline | verdict | action |
|---|---|---|
| rank corr. **< 10%** ∧ wgrd **< 0.10 dex** ∧ **top1 ≥ prev − 5 pt** ∧ **n ≥ 16** | **in domain** | commission the deeper measurement; record the region as provisionally calibrated |
| rank corr. **10–30%** or wgrd **0.10–0.30 dex**, or **n < 16** | **hold** | do not commission; existing readings keep their measurement-window clause |
| rank corr. **> 30%** or wgrd **> 0.30 dex**, or **`KL` sign reversal** | **out of domain** | ⛔ never commission; the position-domain clause becomes standing |

### 4-2. Scope clause for readings already taken in the intermediate region

| worst-of {rank corr., wgrd} | verdict |
|---|---|
| rank corr. **< 10%** ∧ wgrd **< 0.10 dex** | lift the provisional clause — deep readings are outside the fit domain but at **calibrated** positions; the caveats on existing figures come off |
| rank corr. **10–30%** or wgrd **0.10–0.30 dex** | provisional **stands**, with the measured value stated in the caveat |
| rank corr. **> 30%** or wgrd **> 0.30 dex** | **re-interpretation required**; report immediately; the downstream criterion is re-adjudicated |

### 4-3. Separating the confound

The boundary between the two intermediate buckets coincides with the model's
sliding-window size. Full-attention and sliding-attention layers are therefore
reported **separately**, so that position and receptive field can be told
apart. ⛔ If they cannot be told apart, that is reported as such.

## 5. Reporting requirements

```
  Report three ways: corpus × position bucket × {band median, full-attention
  layers, sliding-attention layers}.

  ★ State the actual n in every cell. Cells with n < 8 are not hidden.
  ★ Record the article boundaries of the concatenated prose texts in the
    provenance block (so boundary-straddling positions can be identified).
  Quantities: KL / top-1 / rank correlation / word-group rank deviation
    (two groups deciding, a third reported only) / baseline-lens contrast.
```

## 6. Figure freezing

> The empty frame is committed **in this same commit** — the axes, the bucket
> boundaries, the decision lines and the footer are fixed before any data
> exists.
> ⇒ This closes "draw it after seeing it" structurally.
> Implementation: the plotting script **draws an empty frame and exits** if the
> results file is absent.

## 7. Order of operations

```
  1. ratify this table
  2. freeze the empty figure frame (done in the same commit)
  3. preflight — if the cost exceeds the budget, do not run; return for review
  4. main run → results → mechanical application of the frozen table
```
