"""Online 1-D bin packing task (FunSearch protocol).

A candidate program defines  priority(item: float, bins: np.ndarray) -> np.ndarray.
The evaluator packs each item into the feasible bin with the highest priority.
Search score = -(mean bins used - mean L1 lower bound) over 40 synthetic instances (200 items, sizes uniform
1..100, capacity 150, fixed seed; higher is better, 0 = at bound). Held-out generalisation is measured on 20
instances of 500 items from the same distribution with a different seed. The OR-Library sets are kept in
data/ for reference (the FunSearch heuristic is tuned to them, which left no room for a 7B agent to improve).
"""
from __future__ import annotations

import atexit
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from .sandbox import RUNNER_PREAMBLE, static_check

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "binpack"
TRAIN_SPEC = dict(seed=10, n_inst=40, n_items=200, cap=150, lo=1, hi=100)
HELDOUT_SPEC = dict(seed=777, n_inst=20, n_items=500, cap=150, lo=1, hi=100)

TASK_DESCRIPTION = """\
## Task: online 1-D bin packing heuristic

Items arrive one at a time and must be placed immediately into a bin (capacity 150); a bin can never be
reopened or repacked. You write the placement heuristic:

    def priority(item: float, bins: np.ndarray) -> np.ndarray

`bins` holds the REMAINING capacity of every bin the item fits into (bins[i] - item >= 0 for all i).
Unused bins have remaining capacity == 150. Return an array of the same shape as `bins`; the item goes
into the bin with the highest priority (ties -> lowest index). numpy is imported as `np`.

Evaluation: 40 instances of 200 items, item sizes uniform in 1..100, capacity 150. Score =
-(mean bins used - mean L1 lower bound), so higher is better and 0 would mean every instance hits the
lower bound. Reference heuristics on this set: first-fit = {ff:.3f}, best-fit (return -(bins - item)) = {bf:.3f},
the piecewise gap-threshold heuristic published by FunSearch (DeepMind, 2023) for OR-Library data = {fs:.3f}.

Interface facts (getting these wrong scores catastrophically):
- Every bin passed in is already feasible; do NOT filter, mask or re-check feasibility.
- Return one float per bin (same shape as `bins`): a preference score. The evaluator takes argmax. Do NOT
  return a one-hot vector, an index, a boolean mask, or a modified copy of `bins`.
- Higher priority for LARGER remaining capacity means "put the item in the emptiest bin", which spreads items
  over many bins and is very bad. Best-fit does the opposite: it prefers the smallest leftover gap.
- Only numpy/math may be imported; deterministic; must not raise; values must be finite. Vectorize with numpy.
"""


def load_orlib(path: Path) -> list[dict]:
    toks = path.read_text().split()
    i = 0
    n = int(toks[i]); i += 1
    out = []
    for _ in range(n):
        name = toks[i]; i += 1
        cap, m, best = int(toks[i]), int(toks[i + 1]), int(toks[i + 2]); i += 3
        items = [int(x) for x in toks[i:i + m]]; i += m
        out.append({"name": name, "capacity": cap, "items": items, "best_known": best})
    return out


def gen_uniform(seed: int, n_inst: int, n_items: int, cap: int, lo: int, hi: int) -> list[dict]:
    import numpy as np
    rng = np.random.default_rng(seed)
    return [{"name": f"unif{n_items}_{i:02d}", "capacity": cap,
             "items": rng.integers(lo, hi + 1, n_items).tolist(), "best_known": 0} for i in range(n_inst)]


def l1_bound(items, capacity) -> int:
    import math
    return math.ceil(sum(items) / capacity)


EVAL_RUNNER = RUNNER_PREAMBLE + r'''
priority = load_candidate(sys.argv[1])
inst = json.load(open(sys.argv[2]))

def pack(items, capacity):
    bins = _full(len(items), float(capacity))
    for item in items:
        valid = _nonzero(bins - item >= 0)[0]
        pr = _asarray(priority(float(item), bins[valid]), dtype=float)
        if pr.shape != valid.shape:
            raise ValueError(f"priority returned shape {pr.shape}, expected {valid.shape}")
        if not _all(_isfinite(pr)):
            raise ValueError("priority returned NaN or inf values")
        bins[valid[int(_argmax(pr))]] -= item
    return int((bins != capacity).sum())

# Behavioural diagnostic on a probe: remaining capacities 20..150, item 50.
probe = _arange(20.0, 151.0)
try:
    pp = _asarray(priority(50.0, probe[probe >= 50]), dtype=float)
    feas = probe[probe >= 50]
    if _allclose(pp, pp[0]):
        diag = "constant priority: behaves like first-fit (always the lowest-index feasible bin)"
    elif int(_argmax(pp)) == len(pp) - 1:
        diag = "prefers the EMPTIEST bin (worst-fit behaviour): this is why the score collapsed"
    elif int(_argmax(pp)) == 0:
        diag = "prefers the tightest fit (best-fit-like)"
    else:
        diag = f"argmax on probe at remaining capacity {feas[int(_argmax(pp))]:.0f} (item 50): neither pure best-fit nor worst-fit"
except Exception as e:
    diag = f"probe failed: {type(e).__name__}"

used, lbs = [], []
for ins in inst:
    used.append(pack(ins["items"], ins["capacity"]))
    lbs.append(_ceil(_sum(ins["items"]) / ins["capacity"]))
print(json.dumps({"mean_bins": float(_mean(used)), "mean_l1": float(_mean(lbs)),
                  "score": float(-(_mean(used) - _mean(lbs))), "bins": used, "diagnostic": diag}))
'''


# Heuristic reported in the FunSearch paper/notebook for the OR datasets (Romera-Paredes et al., 2024).
FUNSEARCH_OR = """def priority(item, bins):
    def s(b, item):
        g = b - item
        if g <= 2: return 4
        elif g <= 3: return 3
        elif g <= 5: return 2
        elif g <= 7: return 1
        elif g <= 9: return 0.9
        elif g <= 12: return 0.95
        elif g <= 15: return 0.97
        elif g <= 21: return 0.98
        else: return 0.99
    return np.array([s(b, item) for b in bins])
"""


class BinPackTask:
    name = "binpack_uniform_c150"
    signature = "priority(item: float, bins: np.ndarray) -> np.ndarray"
    metric_label = "best score so far: -(mean bins - L1 bound)"
    regression_delta = 1.0  # a drop this far below the seed earns an explicit diagnostic for the agent
    train_spec = TRAIN_SPEC
    heldout_spec = HELDOUT_SPEC
    default_beta1 = 0.005
    default_beta2 = 0.01

    def __init__(self, data_dir: Path = DATA_DIR):
        self.train = gen_uniform(**TRAIN_SPEC)
        self.heldout = gen_uniform(**HELDOUT_SPEC)
        for fname, data in (("uniform_c150_train.json", self.train), ("uniform_c150_heldout.json", self.heldout)):
            f = data_dir / fname
            if not f.exists():
                f.write_text(json.dumps(data))
        self._tmp = Path(tempfile.mkdtemp(prefix="binpack_"))
        atexit.register(shutil.rmtree, self._tmp, True)
        self.train_json = self._tmp / "train.json"
        self.heldout_json = self._tmp / "heldout.json"
        self.train_json.write_text(json.dumps(self.train))
        self.heldout_json.write_text(json.dumps(self.heldout))
        self.runner = self._tmp / "runner.py"
        self.runner.write_text(EVAL_RUNNER)
        self.baselines = {
            "first_fit": "def priority(item, bins):\n    return -np.arange(len(bins), dtype=float)\n",
            "best_fit": "def priority(item, bins):\n    return -(bins - item)\n",
            "funsearch_or": FUNSEARCH_OR,
        }
        self.baseline_scores = {k: self.evaluate(v)["score"] for k, v in self.baselines.items()}
        # The program in the initial workspace that every new branch starts from (the paper's $baseline_dir).
        self.seed_name = "funsearch_or"
        self.seed_code = self.baselines[self.seed_name]
        self.seed_score = self.baseline_scores[self.seed_name]

    def description(self) -> str:
        return TASK_DESCRIPTION.format(ff=self.baseline_scores["first_fit"], bf=self.baseline_scores["best_fit"],
                                       fs=self.baseline_scores["funsearch_or"])

    def evaluate(self, src: str, heldout: bool = False, timeout: float = 90.0) -> dict:
        """Returns {"ok": bool, "score": float|None, "error": str|None, ...metrics}."""
        err = static_check(src)
        if err:
            return {"ok": False, "score": None, "error": err}
        prog = self._tmp / f"cand_{uuid.uuid4().hex}.py"
        prog.write_text(src)
        data = self.heldout_json if heldout else self.train_json
        try:
            r = subprocess.run([sys.executable, str(self.runner), str(prog), str(data)],
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "score": None, "error": f"timeout after {timeout}s"}
        finally:
            prog.unlink(missing_ok=True)
        if r.returncode != 0:
            lines = [l.strip() for l in r.stderr.strip().splitlines() if l.strip()]
            keep = lines[-1:] + [l for l in lines if "candidate.py" in l and "Warning" not in l][-1:]
            return {"ok": False, "score": None, "error": "runtime error: " + " | ".join(keep)[:400]}
        out = json.loads(r.stdout.strip().splitlines()[-1])
        out.update({"ok": True, "error": None})
        return out
