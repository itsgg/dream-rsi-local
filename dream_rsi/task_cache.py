"""Cache eviction task. The agent writes `priority(now, last_used, freq, inserted)` and the lowest-scoring
entry is evicted on a miss; LRU is `return last_used`, LFU `return freq`, FIFO `return inserted`.

Score is the mean hit rate over traces from several workload regimes. TASK_DESCRIPTION below is what the
agent sees.
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

# Swept until a frequency-plus-recency hybrid beat every textbook policy, the ordering survived to held-out,
# and the per-regime winners disagreed. An earlier version had drift 10x higher, which erased the frequency
# signal and left plain LRU unbeatable.
BASE = dict(n_traces=2, n_requests=10000, n_keys=6000, capacity=300,
            zipf_a=1.0, locality=0.15, drift=0.004, scan_every=700, scan_len=80)

TRAIN_SPEC = {
    "mixed":       dict(BASE, seed=20),
    "scan_heavy":  dict(BASE, seed=21, scan_every=500, scan_len=150, drift=0.0),
    "drift_heavy": dict(BASE, seed=22, drift=0.02, scan_len=0),
    "stationary":  dict(BASE, seed=23, drift=0.0, scan_len=0, zipf_a=1.1),
    "small_cache": dict(BASE, seed=24, capacity=120, n_keys=3000),
}
# Held-out shifts every parameter as well as the seed, so constants tuned to the search set do not transfer.
HELD = dict(BASE, n_requests=14000, n_keys=8000, capacity=400, zipf_a=0.95, locality=0.20,
            drift=0.003, scan_every=900, scan_len=100)
HELDOUT_SPEC = {
    "mixed":       dict(HELD, seed=920),
    "scan_heavy":  dict(HELD, seed=921, scan_every=600, scan_len=200, drift=0.0),
    "drift_heavy": dict(HELD, seed=922, drift=0.015, scan_len=0),
    "stationary":  dict(HELD, seed=923, drift=0.0, scan_len=0, zipf_a=1.05),
    "small_cache": dict(HELD, seed=924, capacity=150, n_keys=4000),
}


def build_traces(regimes: dict[str, dict]) -> list[dict]:
    out = []
    for name, spec in regimes.items():
        for t in gen_traces(**spec):
            t["regime"] = name
            out.append(t)
    return out


TASK_DESCRIPTION = """\
## Task: cache eviction policy

A fixed-size cache sits in front of a slow store. Requests arrive one at a time. On a hit, the entry is
served. On a miss with a full cache, exactly one entry must be evicted to make room. You write the rule that
decides which one:

    def priority(now: float, last_used: np.ndarray, freq: np.ndarray, inserted: np.ndarray) -> np.ndarray

All three arrays describe the entries currently in the cache, one element per entry, same length and order:

- `now`        current request number (a scalar float, increases by 1 per request)
- `last_used`  request number at which this entry was last hit (its insertion time if never hit again)
- `freq`       how many times this entry has been requested since it was inserted (starts at 1)
- `inserted`   request number at which this entry was put into the cache

Return one float per entry: how much you want to KEEP it. The entry with the LOWEST value is evicted (ties go
to the lowest index). numpy is imported as `np`.

Evaluation: mean hit rate over {ntraces} traces drawn from {nreg} DIFFERENT workload regimes, scored
together. Higher is better. Reference policies over the whole set:
LRU (`return last_used`) = {lru:.4f}, LFU (`return freq`) = {lfu:.4f}, FIFO (`return inserted`) = {fifo:.4f}.
Offline-optimal (Belady, evict whatever is used farthest in the future; not reachable online) = {opt:.4f}.

The regimes are: {regimes}. They differ in how fast the popular set drifts, how much of the traffic is
scans of keys never requested again, how skewed popularity is, and how big the cache is. A rule tuned with
constants that suit one regime loses in another, so thresholds that adapt to what you can observe (for
example scaling with the spread of `now - last_used`, or with the number of entries) travel better than
hard-coded numbers. Your per-regime hit rates are reported back to you in the diagnostic.

Interface facts (getting these wrong scores catastrophically):
- Return one float per entry, the SAME shape as `last_used`. Do NOT return an index, a boolean mask, or a
  single number.
- The LOWEST value is evicted. Higher means "keep this". Returning `-freq` evicts your most popular entry.
- `now - last_used` is the age since last use, always >= 0. `now - inserted` is the total age.
- Only numpy/math may be imported; deterministic; must not raise; values must be finite. Vectorize with numpy.
"""


def gen_traces(seed: int, n_traces: int, n_requests: int, n_keys: int, capacity: int,
               zipf_a: float, locality: float, drift: float, scan_every: int, scan_len: int) -> list[dict]:
    """Synthetic request traces. See the module docstring for what each ingredient is there to punish."""
    import numpy as np
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_keys + 1, dtype=float)
    p = ranks ** -zipf_a
    p /= p.sum()
    traces = []
    for t in range(n_traces):
        base = rng.choice(n_keys, size=n_requests, p=p)
        coin = rng.random(n_requests)
        pick = rng.integers(0, 32, size=n_requests)
        req: list[int] = []
        recent: list[int] = []
        scan_key = n_keys  # scan keys live outside the popular key space and are never repeated
        for i in range(n_requests):
            if scan_every and (i % scan_every) < scan_len:
                key = scan_key
                scan_key += 1
            elif recent and coin[i] < locality:
                key = recent[int(pick[i]) % len(recent)]
            else:
                key = int((base[i] + int(drift * i)) % n_keys)
            req.append(key)
            recent.append(key)
            if len(recent) > 32:
                recent.pop(0)
        traces.append({"name": f"trace{t:02d}", "capacity": capacity, "requests": req})
    return traces


EVAL_RUNNER = RUNNER_PREAMBLE + r'''
priority = load_candidate(sys.argv[1])
data = json.load(open(sys.argv[2]))

# Cheap gate first: a synthetic 4-entry cache, so a broken or degenerate policy fails before any trace runs.
#   A = just inserted, cold   B = old but hot   C = recent and warm   D = stale and cold
_now = 1000.0
_last = _asarray([999.0, 500.0, 990.0, 100.0])
_freq = _asarray([1.0, 50.0, 20.0, 2.0])
_ins = _asarray([999.0, 0.0, 100.0, 50.0])
pr = _asarray(priority(_now, _last.copy(), _freq.copy(), _ins.copy()), dtype=float)
if pr.shape != _last.shape:
    raise ValueError(f"priority returned shape {pr.shape}, expected {_last.shape}")
if not _all(_isfinite(pr)):
    raise ValueError("priority returned NaN or inf values")
if _allclose(pr, pr[0]):
    diag = "constant keep-score: always evicts entry 0, which is close to FIFO on this simulator"
else:
    _evicts = _int(_argmin(pr))
    diag = {
        0: "evicts the NEWEST entry (just inserted, never re-hit): every miss immediately undoes itself",
        1: "evicts the old-but-hot entry: pure recency, so scans and drift will flush your popular keys",
        2: "evicts a recent, warm entry: check the sign, higher should mean keep",
        3: "evicts the stale cold entry: sensible on this probe",
    }[_evicts]

regimes, rates = [], []
for tr in data:
    reqs = tr["requests"]
    cap = tr["capacity"]
    keys = _full(cap, -1, dtype=np.int64)
    last_used = _zeros(cap)
    freq = _zeros(cap)
    inserted = _zeros(cap)
    # Read-only views over the simulator's own state. They track the arrays as they are updated, but a
    # candidate that writes into them (last_used[:] = 0) cannot corrupt the simulation it is being scored on.
    ro = []
    for arr in (last_used, freq, inserted):
        v = arr.view()
        v.flags.writeable = False
        ro.append(v)
    lu_ro, fr_ro, in_ro = ro
    index = {}
    n_filled = 0
    hits = 0
    for now, key in _enumerate(reqs):
        slot = index.get(key)
        if slot is not None:
            hits += 1
            last_used[slot] = now
            freq[slot] += 1.0
            continue
        if n_filled < cap:
            slot = n_filled
            n_filled += 1
        else:
            keep = _asarray(priority(_float(now), lu_ro, fr_ro, in_ro), dtype=float)
            if keep.shape != last_used.shape:
                raise ValueError(f"priority returned shape {keep.shape}, expected {last_used.shape}")
            if not _all(_isfinite(keep)):
                raise ValueError("priority returned NaN or inf values")
            slot = _int(_argmin(keep))
            del index[_int(keys[slot])]
        keys[slot] = key
        index[key] = slot
        last_used[slot] = now
        freq[slot] = 1.0
        inserted[slot] = now
    regimes.append(tr.get("regime", "?"))
    rates.append(hits / _len(reqs))

by_regime = {}
for name, r in zip(regimes, rates):
    by_regime.setdefault(name, []).append(r)
by_regime = {k: round(_float(_mean(v)), 4) for k, v in by_regime.items()}
worst = min(by_regime, key=by_regime.get)
print(json.dumps({"score": _float(_mean(rates)), "mean_hit_rate": _float(_mean(rates)),
                  "per_regime": by_regime,
                  "diagnostic": diag + "; per-regime hit rate " + json.dumps(by_regime)
                                + f"; weakest regime: {worst}"}))
'''

def belady(trace: list[int], capacity: int) -> float:
    """Offline optimal hit rate: evict whatever is used farthest in the future. Not reachable online, but it
    is the honest ceiling, so the agent can see how much room is actually left."""
    nxt: dict[int, int] = {}
    following = [0] * len(trace)
    for i in range(len(trace) - 1, -1, -1):
        following[i] = nxt.get(trace[i], len(trace) + 1)
        nxt[trace[i]] = i
    cache: dict[int, int] = {}
    hits = 0
    for i, key in enumerate(trace):
        if key in cache:
            hits += 1
        elif len(cache) >= capacity:
            del cache[max(cache, key=lambda k: cache[k])]
        cache[key] = following[i]
    return hits / len(trace)


LRU = "def priority(now, last_used, freq, inserted):\n    return last_used\n"
LFU = "def priority(now, last_used, freq, inserted):\n    return freq\n"
FIFO = "def priority(now, last_used, freq, inserted):\n    return inserted\n"


class CacheTask:
    name = "cache_evict_zipf_drift_scan"
    signature = "priority(now: float, last_used: np.ndarray, freq: np.ndarray, inserted: np.ndarray) -> np.ndarray"
    metric_label = "best score so far: mean cache hit rate"
    regression_delta = 0.05  # a drop this far below the seed earns an explicit diagnostic for the agent
    train_spec = TRAIN_SPEC
    heldout_spec = HELDOUT_SPEC
    # Headroom above the seed is ~0.04 hit rate and a rung is worth 0.003-0.01, so attempts must cost far
    # less than on bin packing or the policy stops before it has climbed anything.
    default_beta1 = 0.0002
    default_beta2 = 0.0004

    def __init__(self):
        self.train = build_traces(TRAIN_SPEC)
        self.heldout = build_traces(HELDOUT_SPEC)
        self._tmp = Path(tempfile.mkdtemp(prefix="cache_"))
        atexit.register(shutil.rmtree, self._tmp, True)
        self.train_json = self._tmp / "train.json"
        self.heldout_json = self._tmp / "heldout.json"
        self.train_json.write_text(json.dumps(self.train))
        self.heldout_json.write_text(json.dumps(self.heldout))
        self.runner = self._tmp / "runner.py"
        self.runner.write_text(EVAL_RUNNER)
        self.baselines = {"lru": LRU, "lfu": LFU, "fifo": FIFO}
        self.baseline_scores = {k: self.evaluate(v)["score"] for k, v in self.baselines.items()}
        self.optimal_score = sum(belady(t["requests"], t["capacity"]) for t in self.train) / len(self.train)
        self.seed_name = "lru"  # what almost every real cache ships with
        self.seed_code = self.baselines[self.seed_name]
        self.seed_score = self.baseline_scores[self.seed_name]

    def description(self) -> str:
        return TASK_DESCRIPTION.format(
            ntraces=len(self.train), nreg=len(TRAIN_SPEC), regimes=", ".join(TRAIN_SPEC),
            lru=self.baseline_scores["lru"], lfu=self.baseline_scores["lfu"],
            fifo=self.baseline_scores["fifo"], opt=self.optimal_score)

    def evaluate(self, src: str, heldout: bool = False, timeout: float = 120.0) -> dict:
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
