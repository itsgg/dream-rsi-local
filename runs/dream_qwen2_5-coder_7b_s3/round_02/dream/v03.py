"""Parallel refinement with dynamic batch sizing and exploration.

Improvement over parallel_refine: dynamically adjust batch size based on progress and exploration needs. Prioritize
exploitation and justified recovery while ensuring sufficient exploration. Adjust thresholds and batch composition
based on the current state of the revealed portfolio and remaining rounds."""
from dream_rsi.policy_api import View


class Policy:
    NAME = "dynamic_parallel_refine"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        if not branches:
            return [view.root] * min(view.W, view.root_slots())

        # Determine exploration vs exploitation ratio
        if view.round < view.max_rounds // 3:
            explore_ratio = 0.7
        elif view.round < 2 * view.max_rounds // 3:
            explore_ratio = 0.5
        else:
            explore_ratio = 0.3

        # Prioritize recovery of failed frontiers
        recoverable = [fr.id for fr in view.frontier(b).values() if fr.id in legal and fr.error and 'recoverable' in fr.error]
        if recoverable:
            return recoverable[: min(view.W, len(recoverable))]

        # Prioritize under-explored branches
        frontier_scores = {b: (fr.score, fr.depth) for b, fr in branches.items() if fr.id in legal and not fr.error}
        under_explored = sorted(frontier_scores, key=lambda b: (frontier_scores[b][1], -frontier_scores[b][0]), reverse=True)
        under_explored = [fr.id for fr in view.frontier(b).values() if fr.id in legal and fr.id not in under_explored[: view.W // 3]]

        # Select batch
        batch = under_explored[: int(explore_ratio * view.W)]
        batch += [view.root] * min(int((1 - explore_ratio) * view.W), view.root_slots())
        batch += [fr.id for fr in view.frontier(b).values() if fr.id in legal and fr.id not in batch]

        return batch[: view.W]
