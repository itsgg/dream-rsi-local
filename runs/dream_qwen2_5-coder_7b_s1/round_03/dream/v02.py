"""Enhanced dynamic batch selection policy that prioritizes high-potential branches, handles recoverable failures,
and expands under-explored areas more effectively. Improved by dynamically adjusting batch composition based on
current performance and remaining depth."""

from dream_rsi.policy_api import View

class Policy:
    NAME = "enhanced_dynamic_batch_v2"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        if not legal:
            return []

        branches = view.branches()
        if not branches:
            return [view.root] * min(view.W, view.root_slots())

        prioritized_branches = []
        for branch_id, traj in branches.items():
            score = traj[-1].score
            depth = traj[-1].depth
            if score is not None:
                prioritized_branches.append((branch_id, score, depth))
        
        prioritized_branches = sorted(prioritized_branches, key=lambda x: (-x[1], x[2]))

        batch = []
        repairable_failures = [branch_id for branch_id, traj in branches.items() if traj[-1].score is None and traj[-2].score is not None]

        # Prioritize repairable failures
        for branch_id in repairable_failures:
            if branch_id in legal and len(batch) < view.W:
                batch.append(branch_id)

        # Prioritize high-potential branches
        for branch_id, _, _ in prioritized_branches:
            if branch_id in legal and len(batch) < view.W:
                batch.append(branch_id)

        # Fill remaining slots with under-explored branches
        underexplored_branches = [branch_id for branch_id in legal if branch_id not in batch]
        for branch_id in underexplored_branches:
            if len(batch) < view.W:
                batch.append(branch_id)

        return batch[:view.W]
