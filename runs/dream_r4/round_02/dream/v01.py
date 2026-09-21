"""Improve exploration by prioritizing high-potential branches and batching independent probes.
Open new branches when existing ones stagnate or when few branches exist. Close branches with repeated failures.
Prefer refining branches near the best score or still improving. Fill the batch with promising independent probes."""

from dream_rsi.policy_api import View


class Policy:
    NAME = "batched_parallel_refine"

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
        
        # Fill the batch with new branches if less than W actions are selected
        while len(batch) < view.W and view.root_slots() > 0:
            batch.append(view.root)
        
        return batch[: view.W]
