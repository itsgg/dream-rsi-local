from dream_rsi.policy_api import View, Obs
"""Dynamic Portfolio Exploration: Balance exploration and exploitation by dynamically adjusting the batch composition based on the current state of revealed branches and their potential for improvement. Prioritize under-explored branches and recovery of repairable failures, while also selecting strong refinements to maximize exploitation."""

from collections import defaultdict
from statistics import mean


class Policy:
    NAME = "dynamic_portfolio"

    def select_batch(self, view: View) -> list[int]:
        legal_actions = view.legal_actions()
        branches = view.branches()
        frontier_scores = {b: [obs.score for obs in traj if obs.score is not None] for b, traj in branches.items()}
        best_scores = {b: max(scores) for b, scores in frontier_scores.items()}
        avg_scores = {b: mean(scores) for b, scores in frontier_scores.items()}
        avg_improvement = {b: max(0, best_scores[b] - avg_scores[b]) for b in frontier_scores}
        repairable_failures = {b: any('repaired' in obs.error for obs in traj) for b, traj in branches.items()}

        exploration_batch = []
        recovery_batch = []
        exploitation_batch = []

        for b in legal_actions:
            if b == view.root and len(exploration_batch) < view.root_slots():
                exploration_batch.append(b)
            elif b in frontier_scores:
                if repairable_failures[b] and len(recovery_batch) < 1:
                    recovery_batch.append(b)
                elif avg_improvement[b] > 0 and len(exploitation_batch) < view.W - 1:
                    exploitation_batch.append(b)

        return exploration_batch + recovery_batch + exploitation_batch
