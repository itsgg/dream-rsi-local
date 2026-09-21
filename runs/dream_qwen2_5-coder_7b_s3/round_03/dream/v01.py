"""
Improved exploration policy that balances exploitation and exploration more aggressively,
with a focus on repairing under-explored branches and exploring new frontiers.
"""

from dream_rsi.policy_api import View
from statistics import mean

class Policy:
    NAME = "dynamic_portfolio"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        if not branches:
            return [view.root] * min(view.W, view.root_slots())

        # Sort branches by depth and score
        sorted_branches = sorted(branches.items(), key=lambda x: (x[1][-1].depth, -x[1][-1].score))

        batch = []
        repairable_failures = []

        # Explore new roots or under-explored branches
        for branch_id, trajectory in sorted_branches:
            if branch_id in legal and len(trajectory) < view.max_depth:
                batch.append(branch_id)
                if len(batch) >= view.W:
                    break

        # Repair under-explored branches
        for branch_id, trajectory in sorted_branches:
            if branch_id in legal and len(trajectory) >= view.max_depth:
                if trajectory[-1].score is None:
                    repairable_failures.append(branch_id)

        # Ensure we have at least one slot for exploration
        if len(batch) < view.W and repairable_failures:
            batch.append(repairable_failures.pop(0))

        # Fill remaining slots with more exploration or refinement
        while len(batch) < view.W and legal:
            batch.append(legal.pop(0))

        return batch
