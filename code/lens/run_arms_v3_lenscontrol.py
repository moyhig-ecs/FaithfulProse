#!/usr/bin/env python3
"""B control (2026-09-21): the SAME frozen lens (c73a32d1) re-run today, so that "old lens today vs old lens 08-26" measures
run-to-run forward noise and "old lens today vs new lens today" isolates the lens. Outputs: results/tsd_r2_lens_c73a32d1_rerun/.
(neuronpedia/jacobian-lens @496cf110, fitted on transformers 5.17.0; sha256 c6ee7d22…, md5 b76610b9…) instead of the frozen
lens c73a32d1 (fitted on transformers 5.11.0, YaRN applied to sliding-window layers). Same model, same forward (5.14.1),
same rows, same gates; only the lens artifact differs. Outputs go to results/tsd_r2_lens_c6ee7d22/ so the frozen tsd_r2/
outputs are untouched. Nothing in run_arms_v3.py / run_f5_lens.py / run_pc2.py is modified: the lens path and md5 are
swapped in memory after load_pc2(), and the provenance variant is suffixed."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_arms_v3 as R  # noqa: E402  (sets up sys.path for joshaku / f5 / pc2)
L = R.L
NEW_LENS = None  # control: keep the FROZEN lens (c73a32d1); only the run date differs
NEW_MD5 = None
# control run: no lens swap
_load = L.load_pc2
def load_pc2_newlens():
    m = _load()
    pass  # control: unchanged lens
    return m
L.load_pc2 = load_pc2_newlens
_prov = L.provenance
L.provenance = lambda variant, allow_dirty: _prov(variant + "-lenscontrol-c73a32d1-rerun", allow_dirty)
R.OUT_DIR = HERE / "results" / "tsd_r2_lens_c73a32d1_rerun"
if __name__ == "__main__":
    sys.exit(R.main())
