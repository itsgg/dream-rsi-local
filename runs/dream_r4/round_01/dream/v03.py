"""
Improve exploration by focusing on refining the most promising branches and opening new branches when existing ones stagnate. Prioritize branches with scores close to the best seen and those with ongoing improvement. Adjust the batch size dynamically based on the number of open branches.
"""

from dream_rsi.policy_api import View


class Policy:
    NAME = "improved_refine_selective"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        if not branches:
            return [view.root] * min(view.W, view.root_slots())
        
        best_score = view.best_score() or view.baseline_score
        beta1 = 0.005
        beta2 = 0.01
        V = (best_score - beta1 * len(legal) + beta2 * len(legal) / max(1, view.round)) if best_score else None

        promising_branches = []
        stagnated_branches = []

        for b, traj in sorted(branches.items()):
            fr = traj[-1]
            if fr.id in legal:
                if fr.score is not None and fr.score < best_score and len(traj) < view.max_depth:
                    promising_branches.append((b, fr.id))
                elif len(traj) == view.max_depth:
                    stagnated_branches.append((b, fr.id))

        batch = [b for b, _ in promising_branches[: view.W - len(stagnated_branches)]]
        batch.extend(stagnated_branches[: view.W - len(batch)])
        return batch[: view.W]
