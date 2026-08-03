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
