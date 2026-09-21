"""Parallel exploration with selective refinement: open new branches and refine existing ones in parallel,
favoring under-explored branches and justified recoveries. Exploitation and exploration balance."""

from dream_rsi.policy_api import View


class Policy:
    NAME = "parallel_explore"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        if not branches:
            return [view.root] * min(view.W, view.root_slots())

        batch = []
        max_depth = view.max_depth
        beta1, beta2 = 0.005, 0.01

        # Select under-explored branches
        for b, traj in branches.items():
            fr = traj[-1]
            if fr.depth < max_depth and len(batch) < view.W:
                batch.append(fr.id)

        # Select new branches if slots available
        if len(batch) < view.W and len(branches) < view.max_branches and view.root_slots() > 0:
            batch.extend([view.root] * min(view.W - len(batch), view.root_slots()))

        # Select justified recoveries
        for b, traj in branches.items():
            fr = traj[-1]
            if fr.depth == max_depth and fr.error:
                batch.append(fr.id)
                if len(batch) == view.W:
                    break

        return batch
