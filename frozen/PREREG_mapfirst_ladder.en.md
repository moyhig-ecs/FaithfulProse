# Pre-registration: MapFirst — the depth ladder across the fit boundary (English rendering)

> **Original**: `PREREG_cii_ladder_20260820.md` (Japanese; "C-ii" is the internal name of the series renamed MapFirst on 2026-08-20; one document frozen in three successive states)
>
> | state | frozen at (commit, JST) | sha256 of the original at that commit | governs | run started (JST) | results committed |
> |---|---|---|---|---|---|
> | v1 | `df889be7`, 2026-08-20 12:30:36 | `9f21e9f4006fe570c6655fa3dd161188…` | ladder 1, first attempt — **halted at calibration** (M-7), no measurement | 12:32:25 | halt committed `60edb995`, 12:35:19 (artifact `cii_v1_m7.json`) |
> | v2 | `35f0d072`, 2026-08-20 12:50:17 | `0a8111c997f0e0f3ae20be1764887dc3…` | ladder 1 (corridors 88–242) | 12:50:25 (+8 s) | `7634c452`, 2026-08-20 12:58:01 (`cii.json`) |
> | v3 | `a6beb711`, 2026-08-20 13:22:25 | `185d5bb0bad19a418abb8d27274d62c7…` | ladder 2 (corridors 286–484) | 13:22:32 (+7 s) | `dba48230`, 2026-08-20 13:31:28 (`cii_c2.json`) |
>
> ⚠ English rendering prepared 2026-08-21 — see `PROVENANCE.md`. Not itself a frozen artifact. Row ids P1/P2/P3/B1/P5 = Mars/spider/Canada/France/Shakespeare; rung ids r8…r44 = corridor length / 11.

**Status of the originals: frozen.** ⛔ Nothing was run until the governing state was committed. The v1 attempt stopped inside its calibration phase, before any measurement, because a premise of the calibration failed; v2 re-froze the calibration phase only, with the ladder, gates, predicates, branch table and predictions unchanged; v3 changed the ladder only.

**Purpose (verbatim in substance).** This is an experiment in calibrating the instrument outside its fitted domain. Its first product is a map of the instrument; any statement about the model can only be placed on top of that map.

---

## 1. Materials and ladder (frozen)

```
  rows      the five Arm A prompts (gate v2) with the Arm C filler unit, imported verbatim.
  ladder    corridor lengths lm0 in {88, 110, 121, 132, 154, 242} tokens (= 11 × {8,10,11,12,14,22});
            repeats = lm0 / 11; total length ≤ 256 (242 + 14).  Every rung begins with " The"
            (uniform translation family; no bare-"The" boundary token; asserted by gate v5).
            88 = anchor (same-shape re-run of the Arm C fbig condition; bitwise agreement is
            expected within a shape only — recorded, not a stop rule).  110 = new in-domain depth
            (target of prediction P-C2-1).  121 = the landmark straddles 128 in some rows
            (separated by a per-position fit-domain tag).  132 / 154 / 242 = out of the fit domain.
            The 512-class ladder is campaign 2 (after the profile's shape is seen).
  W_fit     two-sided predicate W_fit = [16, 128) tagged on every read-out position.
  gate v5   (before the run; no value seen) (a) total length ≤ 256 (b) landmark suffix token
            sequence = tok(" " + prompt) at every rung (c) landmark_start = 11 × k exactly, monotone
            (d) every rung begins with " The".  Failure => M-7 stop.
```

## 2. Run order (frozen) — calibration first; control region = measurement region = scan region p ≥ 16

```
  v1 calibration phase (all before measurement; null readings only; 24 + 6 forward passes):
   (i)   sign-flipped null floor per rung from the calibration rows {P2, P3, B1, P5} (P1 excluded as
         contaminated: its flipped value had been seen), pooled over p ≥ 16; frozen formula =
         tau_from_calibration imported verbatim (q = 0.05, type-1, n = P1's series length at that
         rung).  Sanity: the four row minima within a factor of 4 per rung, else M-7.
   (ii)  gate = P1 flipped (per rung, p ≥ 16 minimum) > tau  => FAIL = stop, report no values.
   (iii) determinism: P1 at rung 242 (largest shape), two applications identical.
  measurement phase (30 forward passes): rows × rungs, every position read; the behavioral
   reading (top-1, p(answer) at the final position) is taken from the same forward pass.
   Incremental saving; wall ≤ 2 h.  Post-processing: zero forward passes (§3–§6).
```

## 3. The behavioral-invariance gate as rung eligibility (frozen)

```
  behavior_gate imported verbatim from run_arm_b (eps_B = 0.02).  Reference = rung 88, per rung ×
  row.  A failing rung × row is excluded from the onset comparison and the exclusion is recorded
  (a selectivity finding in its own right).  The signed Δp(answer) is recorded at every rung.
```

## 4. Onset and window comparison (frozen)

```
  rise predicate = run_arm_a.emergence verbatim (k = 3); both the full-sequence window and the
  landmark-region window are recorded.
  window-comparison predicate: rungs may be compared only (i) over the non-shared region (deeper
  than the shorter corridor) or (ii) over the phase-aligned sets of §5.  Agreement over the shared
  region is a (near-)consequence of construction and is recorded but not counted.  Landmark
  windows compared against rung 88 are non-shared by construction => eligible (machine-checked).
  descriptive comparators: Δrel / Δabs against 88, over gate-passing rising rows.  No
  single-position predicate.  The order-agreement band eta = 0.643–0.902 travels with every
  ordinal statement (with the out-of-domain lower-bound qualifier).
```

## 5. Phase-aligned filler depth profiles and the shared-prefix floor (frozen; zero forward passes)

```
  phase sets: for each series and phase phi in [0, 11), the filler positions d = phi + 11k
  (16 ≤ d < lm0) give a reading curve of constant content and varying depth.  Rung 242 crosses
  128 (the main data for the knee statistic).  Four qualifiers travel with these curves: deep-rank
  regime; depth is confounded with the amount of preceding filler; p < 16 is outside the scan
  region; comparisons across shapes carry the measured floor.
  knee statistic (placed constants; review = judgment layer): at rung 242, for each phase curve of
  the intermediate, the band contrast |median log10 rank over [128,160) − median over [96,128)|
  exceeds twice the far contrast |median over [192,224) − median over [160,192)| => "knee" for
  that phase; ≥ 6 of 11 phases => series knee.  Mechanical.
  shared-prefix floor (no extra forward pass): for rung pairs i < j, per-position |Δ log10 rank|
  over the shared region [16, lm0_i), by depth band; the pairs (132,154), (132,242), (154,242)
  give the floor across 128.  Formula = dlog_series / dist_stats imported verbatim (run_arm_b).
```

## 6. Branch table B1–B4 (the reading rule, frozen; prediction P-C2-2)

```
  premise: the rung × row passes the behavioral gate.
  B1  in-domain window rungs (110; 121 on the in-domain side): Δrel = 0 continues
      => in-domain extension of "content-locked" (candidate for an L2 statement)
  B2  out-of-domain window: degradation (Δrel ≠ 0) AND knee present (§5)
      => a finding about the instrument's domain ("instrument ineligibility is not refutation") — strong
  B3  degradation AND no knee => recorded with attribution withheld — weak (three-way hygiene kept)
  B4  no degradation (Δrel = 0 even out of domain) => recorded side by side, L1 only
      ("it did not break out of domain" does not substitute for calibration — qualifier mandatory)
  Every branch is material for FaithfulProse v2 — no empty outcome.  Branch application is
  mechanical; the reading belongs to the judgment layer.
```

## 7. Three predictions (adopted by the author; wording frozen)

| # | prediction (frozen) | stake |
|---|---|---|
| P-C2-1 | at the new in-domain rung 110, the gate-passing rising rows keep Δrel (vs 88) = 0 | the author's |
| P-C2-2 | the branch table of §6 itself (a decision table; not scored) | — |
| P-C2-3 | if degradation appears in an out-of-domain window it is boundary-locked (knee present) **and** the behavioral gate passes at every rung × row. Scoring: the first half only when degradation is observed; the second half unconditionally | **the judgment layer's** (a miss is charged to it) |

## 8. Claim ceiling (six clauses, inherited verbatim)

```
  1 up to the onset everything is L1; a threshold crossing is not a computation time; computation
    order needs patching / 2 no three-way attribution (lens extrapolation / representation side /
    positional encoding) — scale note: the ladder (≤ 256) is far below the context length the
    model supports (65,536, registered) / 3 no "latent computation" vocabulary / 4 B4 does not
    substitute for calibration / 5 the two-sided W_fit tag on every read-out position /
    6 FaithfulProse non-interference (as it stood; see the arms rendering, §0)
```

## 9. Stops, pins, cost (v1)

```
  stop   wall ≤ 2 h; gate v5 M-7; calibration sanity M-7; gate / determinism FAIL = stop (report no
         values).  Bitwise agreement at 88 with the Arm C fbig series is recorded / flagged, not a stop.
  cost   60 forward passes (calibration 24 + gate 6 + measurement 30) + zero-forward post-processing.
  pins   gate_cii.py md5 86c73c53… (gate v5 PASS; artifact 64b3e564…; all rungs, all rows; lm0 = 11k
         exactly; max total 256) / run_cii.py md5 2d4c83b7… (fixtures 1–8 = 10 checks pass) /
         lens c73a32d1… / band 26..56 / max_seq 8192 (no truncation; checked) / mechanisms imported
         verbatim from run_arm_a / b / c (single source).
```

---

## State v2 — the calibration phase re-frozen after the M-7 halt (minimal diff)

The v1 attempt stopped in its calibration phase: the p ≥ 16-restricted null floors of the four
calibration rows at rung 88 were {P2 16,415 / P3 1,666 / B1 3,656 / P5 3,522} — a spread of 9.85,
outside the factor-4 sanity bound. What failed was not a constant but a premise ("the four rows
share one null"). The review rejected re-placing the constant and rejected excising the outlying
row; the calibration phase was redesigned and re-frozen; everything else was left verbatim.

```
  (i)   a ledger of flipped minima per row × rung × region, with the regions as construction
        constants: R_shared = [16, 88), R_newfill = [88, lm0), R_land = [lm0, n).  All five rows
        (P1 included; the "gate" concept dissolves into the per-row gate) × 6 rungs.  The full
        flipped series is kept — the null side of the phase profiles comes for free.  The
        calibration phase is itself part of the first product, the map.
  (ii)  gate = decision-anchor floor: for every row × rung, the flipped minimum over the scan region
        (p ≥ 16) must exceed tau_dec = 10 · tau_A = 10^3, anchored to Arm A's frozen tau_A = 100
        times one dex (the house's natural unit) — ⛔ not to the day's null values.  Disclosure:
        the already-seen values (the four v1 minima, 9.85, tau = 1,666 (unused), P1's full-series
        3,648) all clear tau_dec by ≥ 0.22 dex; this is disclosed and then frozen.
  (iii) shared-region sanity = FLAG (not a stop): for rungs > 88 the R_shared minimum is expected
        within ±0.135 dex of the rung-88 value (= the in-domain shape floor 0.045 dex × a placed
        factor 3) — a defensive use of a near-consequence.  Deviation = FLAG and first data for the
        shape-floor question.
  (iv)  row spread (max / min per rung) demoted to a descriptive column.
  (v)   the four rung-88 series are reused from the v1 run (same-shape determinism, declared;
        source cii_v1_m7.json md5 c52e7761…).  New flipped passes = 26.
  unchanged: ladder, gate v5, behavioral gate, window-comparison predicate, phase profiles, branch
  table, claim ceiling, the three predictions (unconsumed; P-C2-3's stake stays with the judgment
  layer), determinism (P1 at rung 242), wall ≤ 2 h.
  pin v2: run_cii.py md5 38c2ff21… (fixtures: 13 checks pass).  The diff between the two freeze
  commits is the authority for "minimal".
```

## State v3 — campaign 2 (the 512-class ladder; change = ladder only)

```
  ladder   new rungs lm0 in {286, 352, 418, 484} (= 11 × {26,32,38,44}); maximum total length 498
           ≤ 512, far below max_seq 8192 (no truncation).  gate c2 = the gate v5 predicates verbatim
           (≤ 512) => PASS for every row and rung; lm0 = 11k exactly.
  reuse    rung 88 (reference) and rung 242 are reused from the v2 store (cii.json, md5 recorded at
           run time) across runs — same-shape determinism, declared in advance.
  predicates  all v2 verbatim: tau_dec = 10^3 / regions / knee statistic (bands [96, 224) unchanged,
           for comparability across campaigns) / behavioral gate (reference = rung 88 of v2) /
           window-comparison predicate / branch table (every new rung is out of domain => only
           B2 / B3 / B4 can occur).
  added    descriptive columns only: a deep placebo contrast — the knee formula applied to the
           deep band pairs {(256,288) vs (288,320)} and {(384,416) vs (416,448)} — as material
           for whether the contrast at 128 is "a contrast that appears everywhere".  No verdict.
  predictions  none (the three were consumed in campaign 1; "a quiet boundary" is prior material,
           not a prediction).  Recording card.
  cost     calibration 20 + determinism 2 (P1 at rung 484) + measurement 20 = 42 forward passes.
  pins v3  gate_cii_c2.py md5 02258355… (artifact 2ca8fa99…) / run_cii_c2.py md5 6c096a5b…
           (fixtures: 6 checks pass; 3 = placebo flat / step, 4 = knee import identity).
```

## What this rendering leaves out

The original's cross-references to internal review documents, the cost-estimate memo and the
naming history are omitted. The Japanese originals, fixed by their sha256, are the authority.
