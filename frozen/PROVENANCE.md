# Freeze provenance

The methodological claim of this paper is that the decision table, the
thresholds, and both figure frames were committed **before** the measurements
they govern. This file is the evidence for that claim, and it is checkable
without trusting anything written here.

## What is the evidence

The evidence is **not** the prose of the frozen documents. It is the pair

* the commit timestamp of the frozen document, and
* the commit timestamp of the measurement it governs,

together with the `sha256` of the frozen document, which fixes *which* text
was frozen at that time.

## The record

| # | frozen document | commit | committed (JST) | governs | measurement committed | interval |
|---|---|---|---|---|---|---|
| 1 | decision table (`DESIGN_c_lens_pos_v2`) | `5023560` | 2026-08-02 10:08:50 | main run | `f0a7787` 2026-08-02 14:45:59 | **+4 h 37 m** |
| 2 | pre-registration, relative coordinates | `0b7ccaf` | 2026-08-02 15:02:07 | re-aggregation | `67b481f` 2026-08-02 15:16:25 | **+14 m** |
| 3 | pre-registration, multi-prompt extension | `ddecebc` | 2026-08-03 11:47:19 | extension run | `fd19133` 2026-08-03 12:01:24 | **+14 m** |

For (3) the run log records the process start at **11:47:26**, seven seconds
after the pre-registration commit: the freeze was committed and the run was
started immediately afterwards, in that order.

### `sha256` of the frozen originals

```
75aea6a2391a06573c533adc5454031d…   decision table          (DESIGN_c_lens_pos_v2)
febe6dd0f0cb0b307c256398b7b04297…   pre-registration, relative coordinates
890da859c10c97ebe0b408e1487c8820…   pre-registration, multi-prompt extension
```

Full hashes and the full commit SHAs are in `MANIFEST.json`.

## Status of the documents in this directory

⚠ **The originals are in Japanese.** The documents in this directory are
**English renderings prepared for release on 2026-08-03**, after the
measurements. They are *not* themselves frozen artifacts, and nothing in the
paper's provenance claim rests on them.

What they are for: a reader who wants to know *what* was frozen — the
thresholds, the decision branches, the reading tables, the stopping conditions
— without reading Japanese. What fixes *when* it was frozen is the table
above.

Two further points, stated because they bear on how much these renderings can
be trusted:

1. **The decision table is an extract, not a translation of the whole
   document.** The Japanese original also carries scope clauses for a separate,
   unpublished research programme that used the same instrument. Those clauses
   are omitted here: they are not load-bearing for this paper, and the
   programme they concern has not been published. The parts that *are*
   load-bearing — the measured quantities, the thresholds, the decision
   branches, the reporting requirements, and the figure-freezing rule — are
   rendered in full.
2. **Internal terminology has been mapped to the register used in the paper**
   (for instance, the internal name for the word-group readout instrument
   becomes "word-group rank deviation"). The mapping is the one given in the
   paper's Methods.

If the byte-level originals matter for a particular question — for a
replication audit, say — they exist, they are hashed above, and they can be
supplied.

---

## Appendix: the search behind the paper's negative claim

The paper states that the canonical write-up's **main text** does not give the
sequence length used to fit the lens. A negative claim of that kind should
travel with its search, so here it is.

```
  target   https://transformer-circuits.pub/2026/workspace/
  date     2026-08-03
  method   automated retrieval of the published page, then enumeration of the
           predicates below over the retrieved text

  predicates
      max_seq            max_seq_len        "sequence length"
      "context length"   "context window"   "token limit"
      "prompt length"    truncat            "window size"
      "128 tokens"

  result   none of the predicates occurs in the retrieved text
  found    the only statement about the fitting corpus is
           "a corpus of one thousand prompts sampled from a pretraining-like
            distribution" — a count of prompts, not a length
```

⚠ **The limitation, stated plainly.** The retrieval did **not** reach the
appendices: the returned text ends mid-section and the document references
appendix figures that were not retrieved. The sections that were retrieved are
Introduction, Methods, and the two main results sections.

⇒ **This is why the paper's claim is scoped to the main text.** We could have
written "nowhere in the write-up" and it might well be true; we cannot show it,
so we do not write it. If the appendices do state a fitting sequence length,
the paper's scoped claim remains correct and the point it supports — that a
user meets the parameter first in a configuration file — is unaffected, since
the configuration files are where we ourselves found it.

---

## The second campaign (package v2.0.0, paper v2 — added 2026-08-21)

The same evidence, for the arms and the MapFirst ladder. Two documents were frozen, each in
successive states; each state governs one run. A state that governs a **halted** run is listed
too: a halt is a measurement that did not happen, and the stop is part of the record.

| # | frozen document (state) | commit | committed (JST) | governs | run started (JST) | measurement committed | interval |
|---|---|---|---|---|---|---|---|
| 4 | pre-registration, Arms A–C (v3) | `fd06b032` | 2026-08-19 16:50:53 | Arm A | 16:51:02 (+9 s) | `ce5a4032` 2026-08-19 16:54:02 | **+3 m 09 s** |
| 5 | the same document (v5) | `2be30a73` | 2026-08-20 10:37:08 | Arm B | 10:37:15 (+7 s) | `1cbecffd` 2026-08-20 10:40:19 | **+3 m 11 s** |
| 5′ | the same state (v5) | `2be30a73` | 2026-08-20 10:37:08 | Arm C, first attempt — **halted** by its run-time control (sign-flipped minimum 3,648 < 10^4); no value reported | — | halt committed `8f9cee1e` 10:43:47 | — |
| 6 | the same document (v6: calibrated control) | `a865958c` | 2026-08-20 11:09:17 | Arm C | 11:09:22 (+5 s) | `4fdf1861` 2026-08-20 11:13:18 | **+4 m 01 s** |
| 7 | pre-registration, MapFirst ladder (v1) | `df889be7` | 2026-08-20 12:30:36 | ladder 1, first attempt — **halted** in its calibration phase (M-7); no measurement | 12:32:25 | halt committed `60edb995` 12:35:19 | — |
| 8 | the same document (v2: calibration re-frozen) | `35f0d072` | 2026-08-20 12:50:17 | ladder 1 (corridors 88–242) | 12:50:25 (+8 s) | `7634c452` 2026-08-20 12:58:01 | **+7 m 44 s** |
| 9 | the same document (v3: ladder 2) | `a6beb711` | 2026-08-20 13:22:25 | ladder 2 (corridors 286–484) | 13:22:32 (+7 s) | `dba48230` 2026-08-20 13:31:28 | **+9 m 03 s** |

The "run started" times are the `started_utc` field of each run's own JSON (converted to JST);
every run also records the freeze commit it ran under and `dirty = False`.

### `sha256` of the frozen originals (second campaign)

```
35b0cbb2d717e85b614c99815b25d0f854c54dc3eeec06de42a9c903ad78caf6   Arms A–C, state v3   (DRAFT_PREREG_arms_graspthink_20260819.md @ fd06b032)
6aaa0408bacbe790ca819d2408041e30fef63a138a00f7ee383352f32f97577f   Arms A–C, state v5   (same file @ 2be30a73)
db9faf6945b398ac420cb1772e071daeef6210271093097df4890e819132e76f   Arms A–C, state v6   (same file @ a865958c)
9f21e9f4006fe570c6655fa3dd161188932fd0e9a879db05fbb3d0432063a1b3   MapFirst ladder, v1  (PREREG_cii_ladder_20260820.md @ df889be7)
0a8111c997f0e0f3ae20be1764887dc3beb9b8c8a82e2075d85579ecefdb103a   MapFirst ladder, v2  (same file @ 35f0d072)
185d5bb0bad19a418abb8d27274d62c79dc7f18f5246c09c3fc0034bf0563990   MapFirst ladder, v3  (same file @ a6beb711)
```

The English renderings `PREREG_arms_a_b_c.en.md` and `PREREG_mapfirst_ladder.en.md` were
prepared on 2026-08-21, after all runs, and are not themselves frozen artifacts; the Japanese
originals, fixed by the hashes above, are the authority. A reader who wants to check the freeze
claim needs only the commit timestamps and the hashes, not the prose.
