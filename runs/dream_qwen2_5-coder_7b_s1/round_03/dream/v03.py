"""Improved dynamic batch selection policy that balances exploitation, exploration, and recovery more effectively.
Prioritizes high-potential branches, handles recoverable failures, and expands under-explored areas more effectively."""

from dream_rsi.policy_api import View

class Policy:
    NAME = "improved_enhanced_dynamic_batch"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        if not legal:
            return []

        branches = view.branches()
        if not branches:
            return [view.root] * min(view.W, view.root_slots())

        prioritized_branches = []
        for branch_id, traj in branches.items():
            if traj[-1].score is not None:
                prioritized_branches.append((branch_id, traj[-1].score, traj[-1].depth))
        
        prioritized_branches = sorted(prioritized_branches, key=lambda x: (-x[1], x[2]))

        batch = []
        for branch_id, _, _ in prioritized_branches:
            if branch_id in legal and len(batch) < view.W:
                batch.append(branch_id)

        if len(batch) < view.W:
            for branch_id, traj in branches.items():
                if traj[-1].id != view.root and traj[-1].score is None and len(batch) < view.W:
                    if self.is_repairable_failure(traj):
                        batch.append(branch_id)
                        if len(batch) == view.W:
                            break
                    else:
                        batch.append(branch_id)
                        if len(batch) == view.W:
                            break
                if len(batch) == view.W:
                    break

        return batch[:view.W]

    def is_repairable_failure(self, traj: list[Obs]) -> bool:
        latest_attempt = traj[-1]
        if latest_attempt.error is not None:
            return True
        if latest_attempt.score is not None and latest_attempt.score < 0:
            return True
        return False
