#!/usr/bin/env python3
"""B (2026-09-21, the PI: "A の前に B"): the frozen v3 arms runner, unchanged, pointed at the RE-FITTED OLMo-3-32B lens
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
NEW_LENS = Path.home() / ".cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/496cf110a6efbf6db491c916e47ba74bc894ef8d/olmo-3-1125-32b/jlens/Salesforce-wikitext/Olmo-3-1125-32B_jacobian_lens.pt"
NEW_MD5 = "b76610b9e0e989cd689cc5c0c7f6d354"
assert NEW_LENS.exists() and NEW_LENS.name == "Olmo-3-1125-32B_jacobian_lens.pt"
_load = L.load_pc2
def load_pc2_newlens():
    m = _load()
    m.LENS_PT, m.LENS_MD5 = NEW_LENS, NEW_MD5
    return m
L.load_pc2 = load_pc2_newlens
_prov = L.provenance
L.provenance = lambda variant, allow_dirty: _prov(variant + "-lensdelta-c6ee7d22", allow_dirty)
R.OUT_DIR = HERE / "results" / "tsd_r2_lens_c6ee7d22"
if __name__ == "__main__":
    sys.exit(R.main())
