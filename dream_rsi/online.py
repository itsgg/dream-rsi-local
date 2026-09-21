"""Online rollout: the current policy picks batches; the discovery agent expands the tree for real."""
from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agent import run_attempt
from .llm import LLM, Embedder
from .policy_api import GridPlan, GridPlanningContext, PolicyError, PolicyRunner, make_view
from .task_binpack import BinPackTask
from .tree import Tree

BASELINE_POLICY = Path(__file__).resolve().parent / "policies" / "baseline.py"


def online_legal(tree: Tree, max_depth: int, max_branches: int) -> tuple[list[int], int]:
    slots = max_branches - len(tree.children[0])
    legal = [0] if slots > 0 else []
    legal += [l for l in tree.leaves() if tree.nodes[l].depth < max_depth]
    return legal, slots


def rollout(task: BinPackTask, policy_path: Path, history: list[Tree], llm: LLM, cfg: dict,
            round_idx: int, log, embedder: Embedder | None = None) -> Tree:
    tree = Tree({"name": f"round_{round_idx:02d}", "round": round_idx, "policy": str(policy_path), "rounds": 0,
                 "task": task.name, "W": cfg["W"], "max_depth": cfg["max_depth"], "max_branches": cfg["max_branches"]})
    runner = PolicyRunner(policy_path, timeout=cfg.get("policy_timeout", 30))
    fallback = None
    seeded = cfg.get("sampling_seed") is not None
    t0 = time.time()
    ctx = GridPlanningContext(round_idx, cfg["W"], cfg["max_branches"], cfg["max_depth"],
                              cfg["baseline_score"], [t.brief() for t in history])
    try:
        plan = runner.plan_grid(ctx)
    except PolicyError as e:
        log(f"  plan_grid failed ({e}); using the caps")
        plan = GridPlan(cfg["max_branches"], cfg["max_depth"])
    tree.meta["grid"] = [plan.max_branches, plan.max_depth]
    if [plan.max_branches, plan.max_depth] != [cfg["max_branches"], cfg["max_depth"]]:
        log(f"  policy planned a grid of {plan.max_branches} branches x {plan.max_depth} depth "
            f"(caps {cfg['max_branches']} x {cfg['max_depth']})")
    for k in range(cfg["K1"]):
        legal, slots = online_legal(tree, plan.max_depth, plan.max_branches)
        if not legal:
            log(f"  round {k}: no legal actions left"); break
        order = sorted(tree.nodes)   # online, creation order is already reveal order
        view = make_view(tree, order, {i: tree.nodes[i].round for i in order}, legal, cfg["W"], k,
                         cfg["K1"], plan.max_depth, plan.max_branches, cfg["baseline_score"], slots)
        try:
            batch = (fallback or runner).select_batch(view)
        except PolicyError as e:
            log(f"  round {k}: policy failed online ({e}); falling back to baseline policy")
            tree.meta["policy_fallback_round"] = k
            tree.meta["policy"] = f"{policy_path} (fell back to {BASELINE_POLICY.name} at round {k}: {e})"
            fallback = PolicyRunner(BASELINE_POLICY)
            batch = fallback.select_batch(view)
        if not batch:
            log(f"  round {k}: policy chose to stop"); break
        log(f"  round {k}: batch={batch} (legal={legal})")
        # Each attempt's sampling seed and example-sampling rng come from its (outer round, decision round,
        # slot) coordinates, not from arrival order, so --seed survives running W attempts in parallel.
        def attempt(job):
            j, parent = job
            base = ((round_idx * 1000 + k) * 100 + j) * 16
            return run_attempt(task, tree, parent, history, llm, round_idx,
                               repair_rounds=cfg.get("repair_rounds", 2), embedder=embedder,
                               novelty_threshold=cfg.get("novelty_threshold", 0.95),
                               rng=random.Random(base), seed_base=base if seeded else None)

        with ThreadPoolExecutor(max_workers=cfg["W"]) as ex:
            results = list(ex.map(attempt, list(enumerate(batch))))
        for parent, res in zip(batch, results):
            n = tree.add(parent, k, **res)
            s = f"{n.score:.3f}" if n.ok else f"FAIL: {(n.error or '')[:80]}"
            flags = "".join([f" repaired={n.metrics['repaired']}x{n.metrics['repairs']}" if n.metrics.get("repairs") else "",
                             f" dup={n.metrics['near_duplicate_of']}" if n.metrics.get("near_duplicate_of") else "",
                             " novelty-retry" if n.metrics.get("novelty_rejections") else ""])
            log(f"    node {n.id} (branch {n.branch}, depth {n.depth}) {s}{flags} | {n.mechanism[:80]}")
        tree.meta["rounds"] = k + 1
    tree.meta["seconds"] = round(time.time() - t0, 1)
    tree.meta["policy_violations"] = runner.violations + (fallback.violations if fallback else 0)
    return tree
