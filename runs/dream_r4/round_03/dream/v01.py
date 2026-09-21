"""Parallel refinement with selective branching and adaptive batch size: open W independent workspaces in the first round,
then refine every open branch in parallel until the depth cap. Adjust batch size based on progress and available slots.
Open new branches when the open ones stagnate or when few branches exist. Stop when nothing promising remains.
Prefer refining branches near the best score or still improving."""

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

        # Determine the number of branches to refine this round
        n_refine = min(view.W, len(branches))
        if view.round > 0:
            avg_score = sum(b[-1].score for b in branches.values() if b[-1].score is not None) / len(branches)
            if avg_score >= best_score:
                n_refine = min(n_refine, len(branches) * 2)

        batch = []
        for b, traj in sorted(branches.items()):
            fr = traj[-1]
            if fr.id in legal:
                if fr.score is not None and fr.score < best_score and len(traj) < view.max_depth:
                    batch.append(fr.id)
                elif len(traj) == view.max_depth:
                    batch.append(b)
            if len(batch) >= n_refine:
                break
        return batch[: view.W]
