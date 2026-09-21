from dream_rsi.policy_api import View, Obs
"""Adaptive exploration and exploitation: Open new branches and refine promising frontiers in each round.
Balance exploration and exploitation based on trajectory scores and failure recovery potential."""

class Policy:
    NAME = "adaptive_explore_exploit"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        best_score = view.best_score()
        beta1, beta2 = 0.005, 0.01
        N = view.n_calls()

        # Select new branches
        if len(branches) < view.root_slots():
            batch = [view.root] * min(view.W, view.root_slots())
        else:
            batch = []

        # Refine promising frontiers
        frontier_scores = {b: (fr.score, fr.depth) for b, fr in branches.items() if fr.id in legal}
        for b, (score, depth) in sorted(frontier_scores.items(), key=lambda x: (-x[1][0], x[1][1])):
            if score > best_score - beta1 * N + beta2 * N / max(1, view.round):
                batch.append(b)
                if len(batch) == view.W:
                    break

        return batch
