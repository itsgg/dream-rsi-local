from dream_rsi.policy_api import View, Obs
class Policy:
    NAME = "dynamic_portfolio"

    def select_batch(self, view: View) -> list[int]:
        """Dynamic portfolio selection: balance exploitation, exploration, and recovery based on
        trajectory analysis and remaining depth."""
        legal = view.legal_actions()
        branches = view.branches()
        frontier_scores = {b: (fr.score, fr.depth) for b, fr in branches.items() if fr.id in legal}
        
        # Sort by potential improvement, remaining depth, and recovery potential
        sorted_branches = sorted(frontier_scores.items(), key=lambda x: (x[1][0] - view.baseline_score, -x[1][1], x[0]))
        
        batch = []
        for branch, (score, depth) in sorted_branches:
            if branch in legal:
                batch.append(branch)
            if len(batch) == view.W:
                break
        
        # If under-explored branches, add a new root
        if len(batch) < view.W and view.root_slots() > 0:
            batch.append(view.root)
        
        return batch
