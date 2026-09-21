"""Parallel refinement with targeted exploration and recovery.

Explores new branches and refines promising frontiers while selectively recovering
from repairable failures. Prioritizes under-explored branches and recovery over
individual refinements when resources are limited."""
from dream_rsi.policy_api import View


class Policy:
    NAME = "parallel_refine_with_recovery"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        frontier_scores = {b: [obs.score for obs in traj if obs.ok] for b, traj in branches.items()}

        # Select new roots for under-explored branches
        new_roots = [view.root] * min(view.W, view.root_slots())

        # Select refinements of promising frontiers
        refinements = []
        for b, traj in branches.items():
            fr = traj[-1]
            if fr.id in legal and len(traj) < view.max_depth:
                refinements.append(fr.id)

        # Select recovery attempts for repairable failures
        recoveries = []
        for b, traj in branches.items():
            if traj[-1].error and 'repaired' in traj[-1].error:
                recoveries.append(traj[-1].id)

        # Combine and limit the batch size
        batch = new_roots + refinements + recoveries
        return batch[:view.W]
