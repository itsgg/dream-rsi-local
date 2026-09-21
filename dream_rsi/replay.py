"""Replay simulator: run a policy over a frozen recorded tree, revealing only stored outcomes, and score
it with the paper's Eq. (1):  V = max revealed score - beta1 * N + beta2 * N / max(1, rounds)."""
from __future__ import annotations

from pathlib import Path

from .policy_api import GridPlanningContext, PolicyError, PolicyRunner, make_view
from .tree import Tree


def eq1(best: float, N: int, rounds: int, beta1: float, beta2: float) -> float:
    """The paper's Eq. (1): discovery quality, minus an execution cost, plus a parallelism bonus."""
    return best - beta1 * N + beta2 * N / max(1, rounds)


def replay_legal(tree: Tree, revealed: set[int], max_depth: int, max_branches: int) -> tuple[list[int], int]:
    """The same action set the policy would see online.

    The paper defines Child(v) as the unobserved recorded children of v and says it is empty when no recorded
    continuation remains, so selecting an off-tree node reveals nothing. Filtering those nodes out of the menu
    instead would tell the policy it is being replayed.
    """
    slots = max_branches - len([c for c in tree.root_children() if c in revealed])
    legal = [0] if slots > 0 else []
    legal += [l for l in tree.leaves(revealed) if tree.nodes[l].depth < max_depth]
    return legal, slots


def replay(tree: Tree, runner: PolicyRunner, W: int, K2: int, max_depth: int, max_branches: int,
           baseline_score: float, beta1: float, beta2: float, round_idx: int = 1,
           history: list | None = None) -> dict:
    ctx = GridPlanningContext(round_idx, W, max_branches, max_depth, baseline_score, history or [])
    try:
        plan = runner.plan_grid(ctx)
    except PolicyError as e:
        return {"tree": tree.meta.get("name"), "V": float("-inf"), "best": None, "N": 0, "rounds": 0,
                "revealed_all": False, "error": f"plan_grid: {e}", "violations": runner.violations,
                "grid": None, "trajectory": []}
    max_branches, max_depth = plan.max_branches, plan.max_depth
    revealed = {0}
    order = [0]                 # reveal order, so the policy sees ids in the sequence it uncovered them
    reveal_round = {0: 0}
    rounds = 0
    traj = []
    error = None
    total = len(tree.nodes)
    while rounds < K2 and len(revealed) < total:
        legal, root_slots = replay_legal(tree, revealed, max_depth, max_branches)
        if not legal:
            break
        view = make_view(tree, order, reveal_round, legal, W, rounds, K2, max_depth, max_branches,
                         baseline_score, root_slots)
        try:
            batch = runner.select_batch(view)
        except PolicyError as e:
            error = str(e)
            break
        if not batch:
            break
        newly = []
        n_roots = batch.count(0)
        newly += [c for c in tree.root_children() if c not in revealed][:n_roots]
        for b in batch:
            if b != 0:
                kids = [c for c in tree.children[b] if c not in revealed]
                if kids:
                    newly.append(kids[0])
        if not newly:
            break   # Child(v) was empty for every selection: the batch went off-tree, so replay ends here
        revealed.update(newly)
        order += newly
        for i in newly:
            reveal_round[i] = rounds
        rounds += 1
        traj.append({"round": rounds - 1, "batch": batch,
                     "revealed": [{"id": i, "branch": tree.nodes[i].branch, "depth": tree.nodes[i].depth,
                                   "ok": tree.nodes[i].ok, "score": tree.nodes[i].score} for i in newly]})
    ok_scores = [tree.nodes[i].score for i in revealed if i != 0 and tree.nodes[i].ok and tree.nodes[i].score is not None]
    all_ok = [n.score for n in tree.nodes.values() if n.ok and n.score is not None and n.id != 0]
    floor = min(all_ok + [baseline_score]) - 1.0
    best = max(ok_scores) if ok_scores else floor
    N = len(revealed) - 1
    V = eq1(best, N, rounds, beta1, beta2)
    return {"tree": tree.meta.get("name"), "V": V, "best": best, "N": N, "rounds": rounds,
            "revealed_all": len(revealed) == total, "error": error, "violations": runner.violations,
            "grid": [max_branches, max_depth], "trajectory": traj}


def evaluate_policy(policy_path: Path, trees: list[Tree], cfg: dict) -> dict:
    """Mean replay score of one policy over all historical trees (the paper's V^m).

    The paper's evaluator sweeps the beta knob when ranking policies rather than scoring at one point, so a
    policy has to be good across the cost/quality trade-off and not just at one weighting. The sweep is free:
    the policy never sees beta, so the trajectory is identical at every setting and only Eq. (1) is recomputed
    from the recorded (best, N, rounds).
    """
    per = []
    for i, t in enumerate(trees):
        runner = PolicyRunner(policy_path, timeout=cfg.get("policy_timeout", 30))
        try:
            per.append(replay(t, runner, cfg["W"], cfg["K2"], cfg["max_depth"], cfg["max_branches"],
                              cfg["baseline_score"], cfg["beta1"], cfg["beta2"],
                              round_idx=i + 1, history=[x.brief() for x in trees[:i]]))
        finally:
            runner.close()
    errors = [p["error"] for p in per if p["error"]]
    mults = cfg.get("beta_sweep") or [1.0]
    swept = [eq1(x["best"], x["N"], x["rounds"], cfg["beta1"] * m, cfg["beta2"] * m)
             for x in per for m in mults]
    mean_V = float("-inf") if errors else sum(swept) / len(swept)
    return {"policy": str(policy_path), "mean_V": mean_V, "beta_sweep": list(mults),
            "per_tree": per, "errors": errors}
