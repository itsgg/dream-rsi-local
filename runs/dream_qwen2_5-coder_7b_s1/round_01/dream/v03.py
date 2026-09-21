"""
Dynamic portfolio policy that balances exploration and exploitation, prioritizes recoveries, and
adapts to branch success and failure.
"""

from dream_rsi.policy_api import View


class Policy:
    NAME = "dynamic_portfolio"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        frontier_scores = {b: max([obs.score for obs in traj]) for b, traj in branches.items() if traj[-1].score is not None}
        max_score = max(frontier_scores.values()) if frontier_scores else view.baseline_score

        batch = []
        for _ in range(view.W):
            if not batch:
                # Explore new branches
                if len(batch) < view.root_slots():
                    batch.append(view.root)
                else:
                    break
            else:
                # Exploit existing branches
                if frontier_scores:
                    b, score = max(frontier_scores.items(), key=lambda x: (x[1] - max_score, -x[1]))
                    batch.append(frontier_scores[b])
                    del frontier_scores[b]

        return batch
