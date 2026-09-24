#!/usr/bin/env python3
"""Run a FROZEN lens runner unchanged, with (a) the lens artifact swapped in memory, (b) every write under <runner_dir>/results
redirected to <runner_dir>/results/lens_<tag>/, (c) the provenance variant suffixed. Nothing in the frozen files is modified.

  python3 lens_swap.py --runner ../graspthink-arms/run_arm_b.py --tag c6ee7d22 -- --fire --allow-dirty
  python3 lens_swap.py --runner ../graspthink-arms/run_arm_b.py --tag c73a32d1-rerun --no-swap -- --fire --allow-dirty   (control)
  python3 lens_swap.py --runner ../olmo3-f5-think-prefix/tsd_r5/run_r5_v3.py --tag c6ee7d22 --fresh-read "s0_lens_v3.json,prompt_lens_v3.json,rta_lens_v3.json,f5_lens_v3_*.json" \
          --ignore-dirty-under papers/,conversations/,TODO.md -- --fire --stage all --prereg-commit <hash> --allow-dirty          (C, chain_c_rest.sh)

Lens: neuronpedia/jacobian-lens @496cf110 (refit on transformers 5.17.0), sha256 c6ee7d22…, md5 b76610b9…
(2026-09-21, B 拡張; RECORD_jlens_olmo3_yarn_misfit_and_refit_20260921)."""
import argparse, builtins, fnmatch, hashlib, io, runpy, sys
import importlib.util as _iu
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NEW_LENS = Path.home() / ".cache/huggingface/hub/models--neuronpedia--jacobian-lens/snapshots/496cf110a6efbf6db491c916e47ba74bc894ef8d/olmo-3-1125-32b/jlens/Salesforce-wikitext/Olmo-3-1125-32B_jacobian_lens.pt"
NEW_MD5 = "b76610b9e0e989cd689cc5c0c7f6d354"

ap = argparse.ArgumentParser()
ap.add_argument("--runner", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--no-swap", action="store_true", help="control: keep the frozen lens, only redirect outputs + suffix provenance")
ap.add_argument("--ignore-dirty-under", default="", help="comma-separated repo-relative prefixes (docs: papers/,conversations/,TODO.md) whose modified/untracked entries are dropped from the R7-a dirty listing; every dropped line is recorded in results/lens_<tag>/dirty_ignored_<runner>.txt so the provenance stays auditable")
ap.add_argument("--fresh-read", default="", help="comma-separated file names under results/ whose READS are also redirected to results/lens_<tag>/ (so a runner that resumes from its own frozen output starts fresh, or reads this campaign's version)")
ap.add_argument("rest", nargs=argparse.REMAINDER)
a = ap.parse_args()
runner = Path(a.runner).resolve(); rdir = runner.parent
results_root = rdir / "results"                     # the probe's results dir: the runner's own dir, or the nearest ancestor that has one (tsd_r5/ runners)
for _d in (rdir, *rdir.parents):
    if _d == REPO: break
    if (_d / "results").is_dir(): results_root = _d / "results"; break
out_root = results_root / f"lens_{a.tag}"
out_root.mkdir(parents=True, exist_ok=True)

# 1. import the shared modules FIRST so the runner sees the patched objects
for p in (REPO, REPO / "probes" / "olmo3-f5-think-prefix", REPO / "probes" / "olmo3-pc2-seat-cue", rdir):
    sys.path.insert(0, str(p))
import run_f5_lens as L   # noqa: E402
import run_pc2            # noqa: E402  (module object cached in sys.modules -> the runner's `import run_pc2` gets this one)

if not a.no_swap:
    assert NEW_LENS.exists() and NEW_LENS.name == "Olmo-3-1125-32B_jacobian_lens.pt"
    assert hashlib.md5(NEW_LENS.read_bytes()).hexdigest() == NEW_MD5
    run_pc2.LENS_PT, run_pc2.LENS_MD5 = NEW_LENS, NEW_MD5
    _load = L.load_pc2
    def _load_pc2_swapped():
        m = _load(); m.LENS_PT, m.LENS_MD5 = NEW_LENS, NEW_MD5; return m
    L.load_pc2 = _load_pc2_swapped
    _sffl = _iu.spec_from_file_location
    def _sffl_swapped(name, location=None, *x, **k):      # run_f5_lens.load_pc2() / tripwire load run_pc2.py by PATH (fresh module) -> patch after exec
        spec = _sffl(name, location, *x, **k)
        if location is not None and Path(str(location)).name == "run_pc2.py" and spec is not None and spec.loader is not None:
            _exec = spec.loader.exec_module
            def _exec_swapped(m):
                _exec(m); m.LENS_PT, m.LENS_MD5 = NEW_LENS, NEW_MD5
            spec.loader.exec_module = _exec_swapped
        return spec
    _iu.spec_from_file_location = _sffl_swapped
_prov = L.provenance
L.provenance = lambda variant, allow_dirty: _prov(f"{variant}-lens-{a.tag}", allow_dirty)

# 1b. R7-a dirty check: the frozen runners read `git status --porcelain`; this campaign's own transient outputs
#     (untracked files under */results/lens_*/ and */results/tsd_r2_lens_*/) are filtered out of that listing, in the
#     same spirit as each runner's "own untracked results are excluded" rule. Everything else still counts as dirty.
import subprocess as _sp, re as _re
_LENSOUT = _re.compile(r"^.{2} .*results/(lens_|tsd_r2_lens_)")   # any status code (??/ M/A): campaign outputs only
IGN = [x.strip() for x in a.ignore_dirty_under.split(",") if x.strip()]
def _filter_status(text):
    keep, dropped = [], []
    for l in text.splitlines():
        path = l[3:].strip()
        if _LENSOUT.match(l) or any(path.startswith(pref) for pref in IGN): dropped.append(l)
        else: keep.append(l)
    if dropped:
        side = out_root / f"dirty_ignored_{runner.stem}.txt"
        side.write_text("# lines removed from `git status --porcelain` before the runner's R7-a check (campaign outputs + --ignore-dirty-under)\n" + "\n".join(dropped) + "\n")
    return "\n".join(keep)
_run, _co = _sp.run, _sp.check_output
def _is_status(args): return isinstance(args, (list, tuple)) and "status" in args and "--porcelain" in args
def _run_f(args, *x, **k):
    r = _run(args, *x, **k)
    if _is_status(args) and isinstance(getattr(r, "stdout", None), str): r.stdout = _filter_status(r.stdout)
    return r
def _co_f(args, *x, **k):
    out = _co(args, *x, **k)
    if _is_status(args):
        return _filter_status(out.decode()).encode() if isinstance(out, bytes) else _filter_status(out)
    return out
_sp.run, _sp.check_output = _run_f, _co_f

# 2. redirect: every WRITE whose path has a `results` directory component goes to <that>/results/lens_<tag>/<rest> (any probe:
#    the tsd_r5/ runners write to ../results/tsd_r5b, derive_r5_v3 reads probes/olmo3-rta/results, ...). READS are redirected the
#    same way only for names matching a --fresh-read pattern (fnmatch; e.g. "*.npz", "f5_lens_v3_*.json"), so a runner that
#    resumes from its own frozen output starts fresh / reads this campaign's version, and everything else (frozen behaviour
#    outputs, prompts, masks) is read from the frozen tree. A missing campaign copy fails loudly (FileNotFoundError), never
#    falls back to the frozen file.
def _lensify(p: Path):
    q = p if p.is_absolute() else Path.cwd() / p
    try: q = q.resolve()
    except Exception: pass
    parts = q.parts
    idx = [i for i, s_ in enumerate(parts) if s_ == "results"]
    if not idx: return None
    i = idx[-1]
    if len(parts) <= i + 1 or parts[i + 1].startswith(("lens_", "tsd_r2_lens_")): return None
    return Path(*parts[:i + 1]) / f"lens_{a.tag}" / Path(*parts[i + 1:])
def _redirect(p: Path) -> Path:                          # writes
    q = _lensify(Path(p))
    if q is None: return Path(p)
    q.parent.mkdir(parents=True, exist_ok=True); return q
FRESH = [n.strip() for n in a.fresh_read.split(",") if n.strip()]
def _rredirect(p: Path) -> Path:                         # reads of the listed names/patterns -> this campaign's copy
    p = Path(p)
    if FRESH and any(fnmatch.fnmatch(p.name, pat) for pat in FRESH):
        q = _lensify(p)
        if q is not None: return q
    return p
_rt = Path.read_text; _rb = Path.read_bytes; _ex = Path.exists
Path.read_text = lambda self, *x, **k: _rt(_rredirect(self), *x, **k)
Path.read_bytes = lambda self, *x, **k: _rb(_rredirect(self), *x, **k)
Path.exists = lambda self: _ex(_rredirect(self))
_wt = Path.write_text; _wb = Path.write_bytes; _open = builtins.open
Path.write_text = lambda self, *x, **k: _wt(_redirect(self), *x, **k)
Path.write_bytes = lambda self, *x, **k: _wb(_redirect(self), *x, **k)
def _open_redirect(file, mode="r", *x, **k):
    if isinstance(file, (str, Path)) and any(c in mode for c in "wax"): file = _redirect(Path(file))
    elif isinstance(file, (str, Path)): file = _rredirect(Path(file))
    return _open(file, mode, *x, **k)
builtins.open = _open_redirect
io.open = _open_redirect

print(f"[lens_swap] runner={runner.name} tag={a.tag} swap={'no (control)' if a.no_swap else 'yes -> ' + NEW_MD5[:8]} out={out_root} fresh={FRESH} ignore_dirty_under={IGN}", flush=True)
sys.argv = [str(runner)] + a.rest[1:] if a.rest and a.rest[0] == "--" else [str(runner)] + a.rest
runpy.run_path(str(runner), run_name="__main__")
