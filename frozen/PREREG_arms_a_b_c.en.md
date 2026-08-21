# Pre-registration: Arms A, B and C — instrument checks inside the fit domain (English rendering)

> **Original**: `DRAFT_PREREG_arms_graspthink_20260819.md` (Japanese; one document frozen in three successive states, each state governing one run)
>
> | state | frozen at (commit, JST) | sha256 of the original at that commit | governs | run started (JST) | results committed |
> |---|---|---|---|---|---|
> | v3 | `fd06b032`, 2026-08-19 16:50:53 | `35b0cbb2d717e85b614c99815b25d0f8…` | Arm A | 16:51:02 (+9 s) | `ce5a4032`, 2026-08-19 16:54:02 |
> | v5 | `2be30a73`, 2026-08-20 10:37:08 | `6aaa0408bacbe790ca819d2408041e30…` | Arm B; first attempt of Arm C | 10:37:15 (+7 s) | Arm B `1cbecffd`, 10:40:19. Arm C attempt **halted** by its run-time control (sign-flipped minimum 3,648 < 10^4); halt committed `8f9cee1e`, 10:43:47; no value was reported |
> | v6 | `a865958c`, 2026-08-20 11:09:17 | `db9faf6945b398ac420cb1772e071dae…` | Arm C (re-run with the calibrated control) | 11:09:22 (+5 s) | `4fdf1861`, 2026-08-20 11:13:18 |
>
> Full hashes are in `MANIFEST.json` / `PROVENANCE.md`. Each run's own JSON carries the freeze commit and a `dirty = False` flag in its `provenance` block.
>
> ⚠ English rendering prepared 2026-08-21 — see `PROVENANCE.md`. Not itself a frozen artifact. Internal identifiers are kept as they appear in the code and data (GT-A/GT-B/GT-C = Arm A/B/C; rows P1/P2/P3/B1/P5 = Mars/spider/Canada/France/Shakespeare); the paper uses the external names.

**Status of the originals: frozen.** ⛔ Nothing was run until the governing state was committed. The order of the arms (A → B → C) was frozen and not reordered.

**Purpose.** The arms are instrument checks on the same distributed lens (md5 `c73a32d1…`), the same read-out band (layers 26–56) and the same model as the main paper. Arm A asks whether the lens reproduces a known large signal at all; Arm B asks whether a content-free change to a prompt is behaviorally neutral (it is not assumed); Arm C holds content fixed and moves it in depth inside the fit domain. ⛔ None of the arms tests a claim about the model; the program they belong to is not reported in the paper.

---

## 0. Position and non-interference (frozen text, as it stood)

The original states that findings of this series do not enter the FaithfulProse documents "until gate ⑦ (preprint publication) is complete". That gate closed on 2026-08-21 (the preprint had been public on Zenodo since 2026-08-05), which is why the arms appear in version 2. The frozen text is unchanged.

## 1. Questions

- **Q-A (Arm A)**: does the distributed lens reproduce the reported behaviour (an unverbalized intermediate concept rising in the read-out band) on short two-hop prompts — a positive control with a known answer (house rule `R6`: calibrate before measuring)?
- **Q-B (Arms B, C)**: is the prompt-side read-out locked to content or to position? Content is held fixed and depth is manipulated.

## 2. Arm A — positive control (state v3)

### 2-1. Materials (five two-hop prompts; the tokenizer gate ran before any value was seen)

| # | prompt (≤128 tokens, two hops) | intermediate | answer | distractors |
|---|---|---|---|---|
| 1 | The color of the planet fourth from the sun is | Mars | red | Jupiter / blue |
| 2 | The number of legs on the animal that spins webs is | spider | eight | insect / six |
| 3 | The capital of the country whose flag has a red maple leaf is | Canada | Ottawa | Toronto / Washington |
| 4 | *(replaced before the run)* The chemical symbol of the metal used in ordinary thermometers is | mercury | Hg | iron / silver |
| 5 | The first name of the author of the play Hamlet is | Shakespeare | William | Romeo / Charles |

```
  Tokenizer gate (before the run; no rank value seen): the first token of each intermediate,
  answer and distractor is identified with the released word-group convention (first tokens of
  {w, " "+w}). A row whose reading set is empty fails the gate and is replaced, and the
  replacement is recorded. Row 4 (mercury / Hg) failed — its discriminating set was empty —
  and was replaced by  "France / Paris / Italy, London"  (row id B1).  The first gate version
  (no single-character variants anywhere) was itself revised before the run to the
  "discriminating-subset" rule (reading set = multi-character first tokens only; empty => FAIL);
  the first version's artifact was kept. Only token structure was inspected; no rank value.
  ⛔ Upstream evidence is from a different model family. Non-reproduction is information.
  ⛔ No-weld: on non-reproduction, do not attribute to model difference / lens quality /
     single-token vocabulary constraints.
```

### 2-2. Instrument (all frozen artifacts imported verbatim)

```
  lens      distributed 32B checkpoint (md5 c73a32d1…, the same file as in the main paper)
  band      layers 26..56 (frozen constant, imported)
  read-out  band-median rank of each pinned token at every prompt position
  controls  sign-flipped-lens control (reproduces the chance floor) + fixed fixtures, passed
            in the same invocation; md5-pinned instrument; one cell (32B only)
```

### 2-3. Readability declaration (two kinds, per the house rule for observation windows)

```
  Ordinal reading (primary).  A row "rises" when the intermediate ranks above every distractor
    for k consecutive positions, k = 3 (a placed constant: a margin against single order flips,
    not a probability guarantee).  The order-agreement band of this lens with its model,
    eta = 0.643–0.902 (median 0.801), travels with every ordinal statement as a qualitative
    robustness declaration.  ⛔ (1−eta)^k is not used: consecutive positions are correlated
    (rho = 0.12–0.42 measured), so an independence-based error rate would be false precision.
  Quantitative reading (descriptive only).  Signal scale ~2–4 dex (rank 10^4 baseline to 10^2
    threshold) against the measured single-group error band W = 1.57–1.81 dex: the ratio
    (~1.1–2.5) does not always satisfy a "much greater than" margin, so rise magnitudes are
    recorded, not judged.
  Transfer caveat.  eta was measured at positions 501..8191 (outside the fit domain); Arm A reads
    at ≤128.  The out-of-domain value is cited as a lower bound.
```

### 2-4. Verdict (three branches, frozen)

```
  REPRODUCED     m ≥ 4 of 5 rows rise (ordinal form, k held)
  PARTIAL        1 ≤ m ≤ 3
  NOT REPRODUCED m = 0  (the no-weld clause applies; recorded either way)
  Prediction slot P-A: deliberately not placed (no defensible prior from a different model family).
```

### 2-5. Configuration frozen at v3

```
  tau_A = 100 (descriptive threshold only; never used in the verdict) / 32B single shot (no 7B)
  run-time controls (declared): sign-flipped lens, row 1, intermediate's deep floor > 10^4;
    determinism (two applications identical).  FAIL => stop, report no values.  wall ≤ 1 h.
  pins: run_arm_a.py md5 b43ebfbf… (fixtures 1–5 pass) / gate_tokenizer.py md5 4caa8d63… /
    gate artifact md5 7ff8c6f1… (first gate version's artifact 05b73d14… kept)
  incremental saving; names GT-A/B/C provisional (repository-wide collision check: 0 hits)
```

## 3. Arm B — is the filler inert? (state v5)

```
  operation   a fixed preface  "Here is a {ADJ} question for you to answer: "  is prepended to
              each of the five Arm A prompts; ADJ in {simple, little, tricky, second, final}
              (reserves {quick, small}, declared order).  Reference = simple.
              => the landmark sits at the same absolute position in every variant.
  gate v3     (before the run; no value seen) (a) identical total token length across variants
              (b) identical landmark suffix token sequence (c) ADJ first token disjoint from
              every pinned discriminating set.  Fewer than five surviving ADJ => M-7 stop.
              Facts seen before values: preface 10 tokens, landmark_start = 10 in every row,
              total length 20–24 (all ≤128).
  measurement rows × ADJ = 25 forward passes.  eps_fill := per-position |Δ log10 band-median
              rank| against the reference, landmark region, per row × word group
              (intermediate / answer / two distractors): median, q3 (inclusive), max; and pooled.
  gate        behavioral-invariance gate: at the model's output, final position,
              (a) top-1 token identical to the reference and (b) |Δp(answer)| ≤ eps_B = 0.02
              (placed: a small perturbation band for answer probabilities of order 0.3–0.9).
              Variants failing the gate are excluded from the eps_fill statistics; the exclusions
              are themselves reported (a gate failure is a finding about selectivity).
  status      recording card — no verdict branch; the output is the instrument specification for
              Arm C.  wall ≤ 2 h.  Controls as in Arm A (row 1, reference variant).
  pins        gate_lengths_b.py md5 9d15b33e… / run_arm_b.py md5 25f3dcee… (fixtures 1–5 pass;
              fixture 4 checks the eps_B boundary at exact binary values) / gate artifact dbc690a8…
```

## 4. Arm C — same content, two depths inside the fit domain (states v5 → v6)

```
  operation   a filler corridor is prefixed to each prompt: unit = "Please stay focused and keep
              reading this text with care. " (first declared; two reserves registered),
              repeated {f0: 0, fmid: 3, fbig: 8} times => landmark_start {0, 33, 88}.
              gate v4 (before the run): unit survives, maximum total length 102 ≤ 128, monotone
              separation — PASS.
  read-out    rise predicate = run_arm_a.emergence imported verbatim (k = 3).  Onset recorded in
              both coordinates (absolute; landmark-relative), over the full sequence and over the
              landmark region.  Descriptive comparators: Δrel / Δabs (fmid − f0, fbig − f0).
  boundary    registered before the run, token structure only: the f0 landmark begins with bare
              "The"; fmid / fbig with " The" (different first-token id).
  hygiene     (four clauses, inherited verbatim) up to the onset everything is L1 / a threshold
              crossing is not a computation time / computation order needs patching, not a lens /
              the mechanism (lens extrapolation vs representation change) is not separated here.
  order       B → C; an M-7 or control FAIL in B blocks C (machine assert in the runner).
  pins (v5)   gate_filler_c.py md5 6d6dab0b… / run_arm_c.py md5 fd38b01f… / gate artifact 4e84ac5f…
```

### 4-1. State v6 — the calibrated control (why there are two states)

The v5 run of Arm C was halted by its own run-time control: the sign-flipped minimum of row 1
(3,648) fell below the fixed floor 10^4 that Arm A had used. The judgment layer's review found
that the fixed floor had been inherited without checking the sequence length it was fitted for;
the fix was not to lower the floor by hand but to derive it from the null series before measuring.

```
  calibration (before measurement, same process): calibration set = rows minus {P1} =
    {spider, Canada, France, Shakespeare}; the sign-flipped intermediate-rank series of each fbig
    condition over all positions, pooled (~4 × 100 positions) into an empirical distribution F.
  frozen formula (order statistics, iid approximation): tau_98 = F^{-1}(1 − (1 − q)^{1/n}),
    q = 0.05 (declared), n = 98 (length of row 1's fbig series, from the gate's token count),
    type-1 empirical inverse.  Derivable before the run: with pooled N ~ 400, p*·N ≈ 0.21 < 1,
    so the formula reduces to the pooled minimum.  Caveat: positive correlation (rho = 0.12–0.42)
    makes the true minimum shallower than iid, so the floor sits on the conservative (low) side —
    not a probability guarantee.
  sanity: the four per-row minima lie within a factor of 4 of each other, else M-7 stop (no
    measurement).
  run-time: row 1's fbig sign-flipped minimum > tau_98 (declared a degraded control: row 1's
    flipped value had already been seen in the halted run; only the threshold side is new; the
    contamination flag is written into the JSON) + determinism (row 1 fbig, two applications).
  ⛔ the agreement of three arms (n × min within a factor of 2) is recorded as a design
    consistency check and is not used to rescue anything after the fact.
  pin: run_arm_c.py md5 458052df… (fixtures 1–7 = 9 checks pass; 6 = reduction of the formula and
    the quantile mechanism, 7 = sanity).
  registered qualifier (approved 2026-08-20): because eps_fill has a tail (max ~0.8 dex), Arm C's
    onset reading uses no single-position predicate; the k = 3 window is consistent with this.
```

## 5. Stop rules, calendar, names

```
  stop    Arm A: wall ≤ 1 h; ≥3 rows failing the tokenizer gate => M-7 stop (no values seen).
          Arms B / C: ≤ 2 h each.  Incremental saving; session-independent.
  order   A → B → C, frozen; no reordering during the walk.
  names   GT-A/GT-B/GT-C provisional (collision check 2026-08-19: 0 hits repository-wide;
          single letters A/B/C avoided).  External names Arm A/B/C fixed 2026-08-20.
```

## What this rendering leaves out

The original also carries a draft of a separate prediction program (§4-bis) and the list of
questions put to the review (§6). Neither governs the measurements reported in the paper, and
they are omitted here. The Japanese original, fixed by its sha256, is the authority.
