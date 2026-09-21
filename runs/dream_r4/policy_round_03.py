"""Parallel refinement with selective branching: open W independent workspaces in the first round,
then refine every open branch in parallel until the depth cap. Open new branches when the open ones
stagnate or when few branches exist. Stop when nothing promising remains. Prefer refining branches
near the best score or still improving."""

from dream_rsi.policy_api import View


class Policy:
    NAME = "parallel_refine_selective"

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
        return batch[: view.W]
