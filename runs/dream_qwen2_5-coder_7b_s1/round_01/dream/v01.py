"""
Enhanced exploration policy: Prioritize exploration of new branches and under-explored paths, while also
exploiting the best available refinements. Balance exploration and exploitation with recovery of recently
failed branches.
"""

from dream_rsi.policy_api import View


class Policy:
    NAME = "enhanced_exploration"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        best_score = view.best_score() or float('-inf')
        beta1 = 0.005
        beta2 = 0.01
        N = view.n_calls()

        # Calculate the value V for each branch
        branch_values = {}
        for b, traj in branches.items():
            fr = traj[-1]
            if fr.ok:
                value = best_score - beta1 * N + beta2 * N / max(1, view.round)
            else:
                value = best_score - beta1 * N + beta2 * N / max(1, view.round) - 1
            branch_values[b] = value

        # Prioritize exploration of new branches and under-explored paths
        batch = []
        for b in sorted(legal, key=lambda x: (x == view.root, branch_values.get(x, float('-inf')), -view.round)):
            if x == view.root and len(batch) < view.root_slots():
                batch.append(x)
            elif x != view.root and len(batch) < view.W:
                batch.append(x)
            if len(batch) == view.W:
                break

        return batch
