"""
Improved exploration policy that balances opening new branches and refining existing ones based
on their performance relative to the best seen score.
"""

from dream_rsi.policy_api import View


class Policy:
    NAME = "balanced_refine"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        best_score = view.best_score() or view.baseline_score
        batch = []

        # Close branches with repeated failures or long plateaus
        for b, traj in branches.items():
            fr = traj[-1]
            if fr.error or (fr.score and fr.score <= best_score - 1.5):
                batch.append(fr.id)

        # Refine branches that are still improving or close to the best score
        for b, traj in branches.items():
            fr = traj[-1]
            if fr.score and fr.score < best_score:
                batch.append(fr.id)

        # Open new branches if there are few or the open ones are stagnating
        if len(branches) < view.max_branches and len(batch) < view.W:
            open_slots = view.root_slots()
            open_actions = [view.root] * open_slots + list(legal)
            batch += open_actions[: view.W - len(batch)]

        return batch[: view.W]
