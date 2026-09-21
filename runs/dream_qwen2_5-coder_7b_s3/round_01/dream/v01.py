"""Parallel exploration with selective refinement: open W independent workspaces in the first round,
then explore by opening new branches and refine promising leaves until the depth cap. Balance exploration
and exploitation by prioritizing under-explored branches and justified repairs."""

from dream_rsi.policy_api import View
import statistics


class Policy:
    NAME = "parallel_explore"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        frontier_scores = {b: [n.score for n in traj if n.score is not None] for b, traj in branches.items()}

        # Prioritize under-explored branches and branches with promising scores
        sorted_branches = sorted(
            branches.items(),
            key=lambda b: (
                len(b[1]),  # Branch depth
                -max(frontier_scores[b[0]], default=0),  # Negative of the best score
                -sum(frontier_scores[b[0]], default=0) / max(len(frontier_scores[b[0]]), 1),  # Average score
            ),
            reverse=True,
        )

        batch = []
        for b, traj in sorted_branches:
            fr = traj[-1]
            if fr.id in legal:
                batch.append(fr.id)
                if len(batch) >= view.W:
                    break

        # If no promising branches, explore by opening new branches
        if not batch and view.root_slots() > 0:
            batch = [view.root] * min(view.W, view.root_slots())

        return batch
