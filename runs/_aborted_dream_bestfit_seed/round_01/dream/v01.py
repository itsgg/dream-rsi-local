"""Parallel refinement with diverse exploration: open W independent workspaces in the first round,
then balance refining existing branches and opening new ones based on their performance relative to the best score."""

from dream_rsi.policy_api import View

class Policy:
    NAME = "balanced_refine"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        best_score = view.best_score() or -float('inf')
        batch = []
        refine_count = 0
        open_count = 0

        if not branches:
            return [view.root] * min(view.W, view.root_slots())

        for b, traj in sorted(branches.items()):
            fr = traj[-1]
            if fr.id in legal:
                if len(traj) < view.max_depth and fr.score < best_score:
                    batch.append(fr.id)
                    refine_count += 1
                elif open_count < view.root_slots():
                    batch.append(view.root)
                    open_count += 1

        if refine_count == 0 and open_count < view.root_slots():
            batch.append(view.root)
            open_count += 1

        return batch[: view.W]
