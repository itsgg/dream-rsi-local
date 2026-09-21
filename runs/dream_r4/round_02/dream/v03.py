"""Improve exploration by prioritizing branches based on score improvement and novelty, 
while maintaining a balanced batch of new and refined probes. Open new branches when open ones stagnate 
or when few branches exist, and stop when nothing promising remains."""

from dream_rsi.policy_api import View


class Policy:
    NAME = "improved_parallel_refine"

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
                score_diff = best_score - fr.score if fr.score is not None else 0
                novelty = len(traj) < view.max_depth
                if score_diff > 0 or novelty:
                    batch.append(fr.id)
                elif len(traj) == view.max_depth:
                    batch.append(b)
        return batch[: view.W]
