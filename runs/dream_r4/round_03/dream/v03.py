"""Parallel refinement with adaptive branching: open W independent workspaces in the first round,
then refine every open branch in parallel until the depth cap. Open new branches when the open ones
stagnate, have limited progress, or when few branches exist. Stop when nothing promising remains.
Prefer refining branches near the best score or still improving, but also consider opening new branches
for diverse exploration. Balance exploration and exploitation by adjusting the threshold for opening new branches."""

from dream_rsi.policy_api import View


class Policy:
    NAME = "parallel_refine_adaptive"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        if not branches:
            return [view.root] * min(view.W, view.root_slots())
        
        best_score = view.best_score() or view.baseline_score
        beta1 = 0.005
        beta2 = 0.01
        V = (best_score - beta1 * len(legal) + beta2 * len(legal) / max(1, view.round)) if best_score else None

        batch = []
        for b, traj in sorted(branches.items()):
            fr = traj[-1]
            if fr.id in legal:
                if fr.score is not None and fr.score < best_score and len(traj) < view.max_depth:
                    batch.append(fr.id)
                elif len(traj) == view.max_depth:
                    batch.append(b)
        
        # Consider opening new branches if the open ones are stagnant or limited in progress
        if len(batch) < view.W:
            promising_ids = [obs.id for obs in view.legal_actions() if obs.id != view.root]
            for promising_id in promising_ids:
                if promising_id not in batch and len(batch) < view.W:
                    batch.append(promising_id)
        
        return batch[: view.W]
