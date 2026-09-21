"""Dynamic portfolio policy: Prioritize exploration of under-explored branches, repairable failures, and
exploitation of strong refinements. Balance exploration and exploitation based on trajectory trends and
remaining depth. Maintain a diverse portfolio to maximize the discovery space."""

from dream_rsi.policy_api import View


class Policy:
    NAME = "dynamic_portfolio"

    def select_batch(self, view: View) -> list[int]:
        legal = view.legal_actions()
        branches = view.branches()
        if not branches:
            return [view.root] * min(view.W, view.root_slots())

        # Prioritize under-explored branches
        underexplored = sorted(
            branches.items(),
            key=lambda x: (x[1][-1].depth, -x[1][0].score),
        )

        # Identify repairable failures
        repairable_failures = []
        for b, traj in branches.items():
            if traj[-1].score is None:
                continue
            if traj[-1].error in ('shape mismatch', 'NaN/inf', 'NameError', 'exception'):
                repairable_failures.append((b, traj))

        # Prioritize recovery of repairable failures
        recovery = [b for b, _ in repairable_failures]

        # Prioritize strong refinements
        strong_refinements = [
            b for b, traj in underexplored if traj[-1].score > max(traj[:-1], key=lambda x: x.score).score
        ]

        # Select actions based on the portfolio
        batch = []
        for b in recovery:
            if len(batch) < view.W:
                batch.append(b)
        for b in strong_refinements:
            if len(batch) < view.W:
                batch.append(b)
        for b, traj in underexplored:
            if len(batch) < view.W:
                batch.append(b)
        for _ in range(view.W - len(batch)):
            if view.root in legal and len(batch) < view.W:
                batch.append(view.root)

        return batch
