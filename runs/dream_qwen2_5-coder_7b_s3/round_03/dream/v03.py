"""
Improve the exploration policy by incorporating a more nuanced approach to batch selection. The new policy
prioritizes exploration over exploitation and recovery, while also considering the trajectory of each branch
and the recoverability of failed attempts.
"""

from dream_rsi.policy_api import View


class Policy:
    NAME = "nuanced_exploration"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        underexplored = []
        recoverable = []
        successful = []
        failed = []

        for b, traj in branches.items():
            fr = traj[-1]
            if fr.id in legal:
                if fr.score is None:
                    underexplored.append((b, traj))
                elif fr.error:
                    if self.is_repairable(fr.error):
                        recoverable.append((b, traj))
                    else:
                        failed.append((b, traj))
                else:
                    successful.append((b, traj))

        batch = []
        if recoverable:
            batch.extend([id for id, _ in recoverable[:1]])
        if underexplored:
            batch.extend([id for id, _ in underexplored[:view.W - len(batch) - 1]])
        if successful:
            batch.extend([id for id, _ in successful[:view.W - len(batch)]])

        return batch[: view.W]
