"""
Improved exploration policy that prioritizes refining branches with potential for improvement,
opening new branches when the open ones stagnate, and stopping when nothing promising remains.
Balances the exploration of independent promising probes in the same round.
"""

from dream_rsi.policy_api import View


class Policy:
    NAME = "prioritize_improvement"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        best_score = view.best_score() or view.baseline_score
        batch = []

        # Prioritize refining branches with potential for improvement
        for b, traj in branches.items():
            fr = traj[-1]
            if fr.score and fr.score < best_score:
                batch.append(fr.id)

        # Open new branches if there are few or the open ones are stagnating
        if len(branches) < view.max_branches and len(batch) < view.W:
            open_slots = view.root_slots()
            open_actions = [view.root] * open_slots + list(legal)
            batch += open_actions[: view.W - len(batch)]

        # Fill the batch with independent promising probes if available
        if len(batch) < view.W:
            batch += [a for a in legal if a not in batch][: view.W - len(batch)]

        return batch[: view.W]
