"""
Improve exploration by prioritizing deepening under-explored branches, repairing recoverable failures,
and opening new branches. Balance exploitation, exploration, and recovery in each batch.
"""

from dream_rsi.policy_api import View


class Policy:
    NAME = "balanced_exploration"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        frontier_scores = {b: view.frontier(b).score for b in branches if view.frontier(b)}
        active_branches = [b for b in branches if frontier_scores[b] is not None]

        # Prioritize deepening under-explored branches
        under_explored = sorted(active_branches, key=lambda b: (frontier_scores[b], len(branches[b])), reverse=True)
        deepening = [b for b in under_explored if len(branches[b]) < view.max_depth]
        deepening = deepening[:view.W]

        # Repair recoverable failures
        recoverable = [b for b in branches if frontier_scores[b] is not None and b not in deepening]
        recoverable = recoverable[:view.W]

        # Open new branches
        new_branches = [view.root] * min(view.W, view.root_slots())
        new_branches = new_branches[:view.W]

        # Balance exploitation, exploration, and recovery
        batch = []
        if deepening:
            batch.extend(deepening)
        if recoverable:
            batch.extend(recoverable)
        if len(batch) < view.W and new_branches:
            batch.extend(new_branches)

        return batch
