"""Parallel refinement with selective branching and diverse exploration: open W independent workspaces in the first round,
then refine every open branch in parallel until the depth cap. Open new branches when the open ones stagnate, few branches exist,
or when diverse promising nodes are available. Stop when nothing promising remains. Prefer refining branches near the best score
or still improving. Diversify exploration by probing promising nodes independently when possible."""

from dream_rsi.policy_api import View


class Policy:
    NAME = "parallel_refine_explore"

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
        promising_ids = [obs.id for obs in legal if obs.id != view.root]
        for b, traj in sorted(branches.items()):
            fr = traj[-1]
            if fr.id in legal:
                if fr.score is not None and fr.score < best_score and len(traj) < view.max_depth:
                    batch.append(fr.id)
                elif len(traj) == view.max_depth:
                    batch.append(b)
        
        # Add diverse promising nodes if batch is not full
        while len(batch) < view.W and promising_ids:
            batch.append(promising_ids.pop(0))
        
        return batch[: view.W]
