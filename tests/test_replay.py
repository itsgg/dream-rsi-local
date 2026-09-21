"""Sanity test of the replay simulator + baseline policy on a synthetic tree (no LLM)."""
import tempfile
from pathlib import Path
from dream_rsi.tree import Tree
from dream_rsi.replay import replay, evaluate_policy
from dream_rsi.policy_api import PolicyRunner
from dream_rsi.online import BASELINE_POLICY

t = Tree({"name": "synthetic"})
# 3 branches, depths 3/2/3; branch 1 fails at depth 2
b0 = t.add(0, 0, ok=True, score=-3.0, code="", mechanism="a")
b1 = t.add(0, 0, ok=True, score=-2.9, code="", mechanism="b")
b2 = t.add(0, 0, ok=False, error="boom", code="", mechanism="c")
n = t.add(b0.id, 1, ok=True, score=-2.5, mechanism="a2"); t.add(n.id, 2, ok=True, score=-2.4, mechanism="a3")
t.add(b1.id, 1, ok=False, error="shape", mechanism="b2")
n = t.add(b2.id, 1, ok=True, score=-2.7, mechanism="c2"); t.add(n.id, 2, ok=True, score=-2.2, mechanism="c3")
cfg = dict(W=3, K2=10, max_depth=6, max_branches=8, baseline_score=-2.85, beta1=0.02, beta2=0.04)
r = replay(t, PolicyRunner(BASELINE_POLICY), **cfg)
print({k: v for k, v in r.items() if k != "trajectory"})
for s in r["trajectory"]: print(" ", s["round"], s["batch"], [(x["id"], x["score"]) for x in s["revealed"]])
assert r["best"] == -2.2 and r["N"] == 8 and r["rounds"] == 3, r
# a greedy policy written the way the LLM would write one
TMP = Path(tempfile.mkdtemp())
greedy = TMP / "greedy_policy.py"; greedy.write_text('''
from dream_rsi.policy_api import View
class Policy:
    NAME = "greedy_test"
    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        br = view.branches()
        if not br:
            return [0] * min(view.W, view.root_slots())
        cands = [tr[-1] for tr in br.values() if tr[-1].id in legal]
        cands.sort(key=lambda o: -(o.score if o.ok and o.score is not None else -1e9))
        return [o.id for o in cands[:view.W]]
''')
ev = evaluate_policy(greedy, [t], cfg)
print("greedy mean V", ev["mean_V"], ev["errors"])
bad = TMP / "bad_policy.py"; bad.write_text("class Policy:\n    def select_batch(self, view):\n        return [999, 0, 0, 0, 0]\n")
ev = evaluate_policy(bad, [t], cfg); print("bad policy:", ev["mean_V"], ev["errors"], ev["per_tree"][0]["violations"], ev["per_tree"][0]["N"])
crash = TMP / "crash_policy.py"; crash.write_text("class Policy:\n    def select_batch(self, view):\n        raise RuntimeError('x')\n")
ev = evaluate_policy(crash, [t], cfg); print("crash policy:", ev["mean_V"], ev["errors"])

# --- plan_grid, per-episode state, and the beta sweep ---------------------------------------------
from dream_rsi.policy_api import GridPlanningContext
from dream_rsi.replay import eq1

ctx = GridPlanningContext(round=2, W=3, max_branches_cap=8, max_depth_cap=6, baseline_score=-2.85,
                          history=[{"attempts": 24, "rounds": 6, "best": -2.2}])

# a policy without plan_grid just gets the caps, which is the old fixed-grid behaviour
r = PolicyRunner(BASELINE_POLICY)
plan = r.plan_grid(ctx)
assert (plan.max_branches, plan.max_depth) == (8, 6), plan
r.close()

planner = TMP / "planner_policy.py"; planner.write_text('''
from dream_rsi.policy_api import GridPlan, View
class Policy:
    NAME = "planner"
    def plan_grid(self, ctx):
        # spend less once history says progress has stalled
        return GridPlan(max_branches=2 if ctx.history else 4, max_depth=99)
    def select_batch(self, view):
        legal = [i for i in view.legal_actions() if i != 0]
        return legal[:view.W] if legal else [0] * min(view.W, view.root_slots())
''')
r = PolicyRunner(planner)
plan = r.plan_grid(ctx)
assert (plan.max_branches, plan.max_depth) == (2, 6), f"depth must be clamped to the cap: {plan}"
assert r.violations == 1, "planning above the cap should count as a violation"
plan0 = PolicyRunner(planner).plan_grid(
    GridPlanningContext(1, 3, 8, 6, -2.85, []))
assert plan0.max_branches == 4, plan0
r.close()
print("plan_grid OK (clamped, history-sensitive, optional)")

# a policy may carry state across rounds of one episode, and it must not leak into the next
stateful = TMP / "stateful_policy.py"; stateful.write_text('''
import json, os
from dream_rsi.policy_api import View
class Policy:
    NAME = "stateful"
    def select_batch(self, view):
        seen = json.load(open("state.json")) if os.path.exists("state.json") else []
        seen.append(view.round)
        json.dump(seen, open("state.json", "w"))
        assert seen == list(range(len(seen))), f"state did not persist within the episode: {seen}"
        legal = [i for i in view.legal_actions() if i != 0]
        return legal[:view.W] if legal else [0] * min(view.W, view.root_slots())
''')
ev = evaluate_policy(stateful, [t, t], cfg)
assert not ev["errors"], ev["errors"]
print("per-episode state OK (persists across rounds, reset between replays)")

# the beta sweep must change the ranking, not just the number
swept = dict(cfg, beta_sweep=[0.5, 1.0, 2.0])
one = evaluate_policy(greedy, [t], dict(cfg, beta_sweep=[1.0]))["mean_V"]
many = evaluate_policy(greedy, [t], swept)["mean_V"]
p = evaluate_policy(greedy, [t], swept)["per_tree"][0]
assert abs(many - sum(eq1(p["best"], p["N"], p["rounds"], cfg["beta1"] * m, cfg["beta2"] * m)
                      for m in [0.5, 1.0, 2.0]) / 3) < 1e-12
assert one != many, "sweeping beta should move the score"
print(f"beta sweep OK (single point {one:.4f}, swept {many:.4f})")
print("OK")
