"""Both tasks: reference scores, the sandbox guards, and the cache task's improvement ladder. No LLM."""
from dream_rsi.sandbox import static_check
from dream_rsi.tasks import make_task

assert static_check("def priority(a, b):\n    return a\n") is None
assert "no function named" in static_check("def other(a):\n    return a\n")
assert "disallowed import" in static_check("import os\ndef priority(a, b):\n    return a\n")
assert "disallowed call" in static_check("def priority(a, b):\n    return open('/etc/passwd')\n")
assert "SyntaxError" in static_check("def priority(:\n")
print("static_check OK")

bp = make_task("binpack")
print("binpack:", {k: round(v, 3) for k, v in bp.baseline_scores.items()})
assert abs(bp.baseline_scores["best_fit"] - -2.175) < 1e-6
assert abs(bp.seed_score - -2.775) < 1e-6
assert bp.baseline_scores["best_fit"] > bp.seed_score, "seed must be beatable"
bad = bp.evaluate("def priority(item, bins):\n    return 1.0\n")
assert not bad["ok"] and "shape" in bad["error"], bad
# the scorer must not be forgeable by monkeypatching numpy after the candidate loads
forge = bp.evaluate("def priority(item, bins):\n    np.mean = lambda *a, **k: 0.0\n    return -(bins - item)\n")
assert not forge["ok"] or abs(forge["score"] - bp.baseline_scores["best_fit"]) < 1e-9, forge
print("binpack guards OK")

ca = make_task("cache")
print("cache  :", {k: round(v, 4) for k, v in ca.baseline_scores.items()})
assert ca.seed_name == "lru" and ca.seed_score > ca.baseline_scores["fifo"] > ca.baseline_scores["lfu"]
bad = ca.evaluate("def priority(now, last_used, freq, inserted):\n    return 0.0\n")
assert not bad["ok"] and "shape" in bad["error"], bad
nan = ca.evaluate("def priority(now, last_used, freq, inserted):\n    return last_used * float('nan')\n")
assert not nan["ok"] and "NaN" in nan["error"], nan

# `seg3`, whose threshold is a hard-coded constant, LOSES to the seed once several regimes are scored
# together. That is the mixture doing its job, and why the assertions below are rank-based.
LADDER = {
    "lru + 20*f": "def priority(now, last_used, freq, inserted):\n    return last_used + 20.0*freq\n",
    "lru + 200*f": "def priority(now, last_used, freq, inserted):\n    return last_used + 200.0*freq\n",
    "f/(age+1)": "def priority(now, last_used, freq, inserted):\n    return freq/(now-last_used+1.0)\n",
    "seg3 (constant)": "def priority(now, last_used, freq, inserted):\n    return last_used + 1e6*(freq >= 3)\n",
    "seg2+decay": ("import numpy as np\ndef priority(now, last_used, freq, inserted):\n"
                   "    return last_used + 1e6*((freq >= 2) & (now-last_used < 3000))\n"),
    "seg2+adaptive": ("import numpy as np\ndef priority(now, last_used, freq, inserted):\n"
                      "    age = now - last_used\n"
                      "    return last_used + 1e6*((freq >= 2) & (age < 2.0*np.median(age)))\n"),
}
rows = [("lru (SEED)", ca.seed_score, ca.evaluate(ca.seed_code, heldout=True)["score"])]
for n, c in LADDER.items():
    a, b = ca.evaluate(c), ca.evaluate(c, heldout=True)
    assert a["ok"] and b["ok"], (n, a, b)
    rows.append((n, a["score"], b["score"]))
rows.sort(key=lambda r: r[1])
seed_a = ca.seed_score
seed_b = [b for n, _, b in rows if "SEED" in n][0]
for n, a, b in rows:
    print(f"  {n:16s} train {a:.4f}  held-out {b:.4f}   vs seed {a - seed_a:+.4f} / {b - seed_b:+.4f}")

above = [(n, a, b) for n, a, b in rows if a > seed_a]
assert len(above) >= 3, f"only {len(above)} rungs above the seed"
for n, a, b in above:
    assert b > seed_b, f"{n} beats the seed on the search set but not held-out ({b:.4f} vs {seed_b:.4f})"
head_a, head_b = above[-1][1] - seed_a, above[-1][2] - seed_b
assert head_a > 0.015, f"headroom above the seed is only {head_a:.4f}"
assert ca.optimal_score - above[-1][1] > 0.05, "offline optimum is too close: no room left to search"
print(f"cache ladder OK: {len(above)} rungs above the seed, headroom {head_a:+.4f} train / {head_b:+.4f} "
      f"held-out, offline optimum {ca.optimal_score:.4f} ({ca.optimal_score - above[-1][1]:+.4f} still open)")
print("OK")
