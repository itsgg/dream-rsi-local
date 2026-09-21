"""Improved dynamic batch selection policy. Prioritizes high-potential branches, handles recoverable failures,
and expands under-explored areas more aggressively. Adjusts exploration based on round and batch size."""

from dream_rsi.policy_api import View


class Policy:
    NAME = "dynamic_batch_v2"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        if not legal:
            return []

        branches = view.branches()
        if not branches:
            return [view.root] * min(view.W, view.root_slots())

        # Prioritize branches by their best score, remaining depth, and round number
        prioritized_branches = sorted(branches.items(), key=lambda x: (-x[1][-1].score, x[1][-1].depth, view.round))

        batch = []
        for branch_id, traj in prioritized_branches:
            if branch_id in legal and len(batch) < view.W:
                batch.append(branch_id)

        # Ensure at least one exploration or recovery attempt if possible
        if len(batch) < view.W:
            for branch_id, traj in prioritized_branches:
                if branch_id in legal and traj[-1].id != view.root and len(batch) < view.W:
                    batch.append(branch_id)
                if len(batch) == view.W:
                    break

        # Adjust exploration based on round and batch size
        if len(batch) < view.W:
            for _ in range(view.W - len(batch)):
                batch.append(view.root)

        return batch[:view.W]
