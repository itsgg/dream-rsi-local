from dream_rsi.policy_api import View, Obs
class Policy:
    NAME = "dynamic_portfolio"

    def select_batch(self, view: View) -> list[int]:
        """Select a dynamic portfolio of up to view.W actions to maximize exploration and exploitation."""
        legal_actions = view.legal_actions()
        if not legal_actions:
            return []

        # Prioritize recovery of failed frontiers
        recoverable = []
        underexplored = []
        exploitable = []
        for node in legal_actions:
            if node != view.root:
                obs = view.nodes()[node]
                if obs.error:
                    recoverable.append(node)
                elif obs.depth < view.max_depth:
                    underexplored.append(node)
                else:
                    exploitable.append(node)

        # Add one recovery attempt if possible
        batch = recoverable[:1]

        # Add underexplored branches if available
        batch += underexplored[:view.W - len(batch)]

        # Fill the rest with exploitable actions
        batch += exploitable[:view.W - len(batch)]

        return batch[:view.W]
