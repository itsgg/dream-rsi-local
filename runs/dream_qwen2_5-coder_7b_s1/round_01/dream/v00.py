"""Parallel refining (the paper's initial policy): open W independent workspaces in the first round,
then refine every open branch in parallel until the depth cap; never open more branches."""
from dream_rsi.policy_api import View


class Policy:
    NAME = "parallel_refine"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        if not branches:
            return [view.root] * min(view.W, view.root_slots())
        batch = []
        for b, traj in sorted(branches.items()):
            fr = traj[-1]
            if fr.id in legal:
                batch.append(fr.id)
        return batch[: view.W]
