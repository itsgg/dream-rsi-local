"""
Improved exploration policy that balances refining existing branches with opening new ones. 
Prioritizes promising branches and uses larger batches to explore multiple promising directions concurrently.
"""

from dream_rsi.policy_api import View

class Policy:
    NAME = "balanced_exploration"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        best_score = view.best_score() or -float('inf')
        batch = []
        refine_count = 0
        open_count = 0
        batch_size = min(view.W, view.root_slots())

        if not branches:
            return [view.root] * batch_size

        for b, traj in sorted(branches.items(), key=lambda x: -x[1][-1].score):
            fr = traj[-1]
            if fr.id in legal:
                if len(traj) < view.max_depth and fr.score < best_score:
                    batch.append(fr.id)
                    refine_count += 1
                elif open_count < view.root_slots():
                    batch.append(view.root)
                    open_count += 1
                if len(batch) == batch_size:
                    break

        if refine_count == 0 and open_count < view.root_slots():
            batch.append(view.root)
            open_count += 1

        return batch[: batch_size]
