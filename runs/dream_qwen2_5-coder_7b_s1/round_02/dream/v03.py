"""
Enhanced dynamic batch selection policy that refines the exploration strategy by incorporating more nuanced
rankings of nodes and branches. Prioritizes recovery attempts for recently failed frontiers and ensures
a balance between exploitation, exploration, and recovery.
"""

from dream_rsi.policy_api import View

class Policy:
    NAME = "enhanced_dynamic_batch"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        if not legal:
            return []

        branches = view.branches()
        if not branches:
            return [view.root] * min(view.W, view.root_slots())

        # Prioritize branches by their best score, recent failure recovery potential, and remaining depth
        prioritized_branches = sorted(
            branches.items(),
            key=lambda x: (
                -x[1][-1].score,
                x[1][-2].score if len(x[1]) > 1 else -float('inf'),  # Recent score if available
                x[1][-1].depth
            )
        )

        batch = []
        recovery_added = False

        for branch_id, traj in prioritized_branches:
            if branch_id in legal and len(batch) < view.W:
                if not recovery_added and traj[-1].id != view.root and traj[-1].error is not None:
                    batch.append(branch_id)  # Add recovery attempt for recently failed frontier
                    recovery_added = True
                elif traj[-1].id != view.root:
                    batch.append(branch_id)  # Add exploration or strong refinement attempt

        # Ensure at least one exploration or recovery attempt if possible
        if len(batch) < view.W:
            for branch_id, traj in prioritized_branches:
                if branch_id in legal and traj[-1].id != view.root and len(batch) < view.W:
                    batch.append(branch_id)
                if len(batch) == view.W:
                    break

        return batch[:view.W]
