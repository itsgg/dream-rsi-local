"""Shared decision interface used identically in online rollout and offline replay.

The policy sees only the revealed prefix of a tree (a View of Obs records) and returns a batch of
node ids to continue from: the root (open a new branch) and/or leaves (refine a branch). A policy
is a pure function of the View; it is re-instantiated every decision round.
"""
from __future__ import annotations

import atexit
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

from .tree import Tree


@dataclass(frozen=True)
class Obs:
    id: int
    parent: int | None
    branch: int          # -1 for root
    depth: int           # root = 0, root children = 1
    round: int           # decision round the node was created in
    ok: bool             # program ran and produced a valid score
    score: float | None  # search score (higher is better); None if not ok
    error: str | None    # short failure diagnostic when not ok
    mechanism: str       # one-line description the discovery agent gave its attempt


@dataclass(frozen=True)
class GridPlan:
    max_branches: int    # how many root children this episode may open
    max_depth: int       # how many refinements deep a branch may go


@dataclass(frozen=True)
class GridPlanningContext:
    """What a policy knows before an episode starts, when it chooses its budget shape."""
    round: int                  # outer recursive round, 1-based
    W: int                      # max batch size
    max_branches_cap: int       # hard ceiling; a plan above this is clamped
    max_depth_cap: int
    baseline_score: float
    history: list               # one {"attempts", "rounds", "best"} summary per earlier tree


class View:
    """What a policy is allowed to see at one decision round."""

    def __init__(self, nodes: list[Obs], legal: list[int], W: int, round: int, max_rounds: int,
                 max_depth: int, max_branches: int, baseline_score: float, root_slots: int = 0):
        self._nodes = {o.id: o for o in nodes}
        self._legal = list(legal)
        self._tid: list[int] = []   # view id -> real tree id; set by make_view, never sent to the policy
        self._root_slots = root_slots
        self.W = W                        # max batch size (parallel workers)
        self.round = round                # 0-based index of the current decision round
        self.max_rounds = max_rounds      # rollout ends after this many rounds
        self.max_depth = max_depth        # deepest allowed node (number of refinements + 1)
        self.max_branches = max_branches  # max number of root children
        self.baseline_score = baseline_score
        self.root = 0

    def nodes(self) -> dict[int, Obs]:
        return dict(self._nodes)

    def legal_actions(self) -> list[int]:
        """Root (if a new branch may be opened) + leaves that may still be refined."""
        return list(self._legal)

    def root_slots(self) -> int:
        """How many NEW branches may still be opened this round (root may appear this many times in a batch)."""
        return self._root_slots

    def branches(self) -> dict[int, list[Obs]]:
        """branch id -> its nodes ordered by depth (the trajectory of that workspace)."""
        out: dict[int, list[Obs]] = {}
        for o in self._nodes.values():
            if o.id != 0:
                out.setdefault(o.branch, []).append(o)
        for b in out:
            out[b].sort(key=lambda o: o.depth)
        return out

    def frontier(self, branch: int) -> Obs | None:
        b = self.branches().get(branch)
        return b[-1] if b else None

    def best_score(self) -> float | None:
        s = [o.score for o in self._nodes.values() if o.ok and o.score is not None and o.id != 0]
        return max(s) if s else None

    def n_calls(self) -> int:
        return len(self._nodes) - 1

    # serialization for the subprocess boundary
    def to_json(self) -> dict:
        return {"nodes": [asdict(o) for o in self._nodes.values()], "legal": self._legal,
                "W": self.W, "round": self.round, "max_rounds": self.max_rounds,
                "max_depth": self.max_depth, "max_branches": self.max_branches,
                "baseline_score": self.baseline_score, "root_slots": self._root_slots}

    @classmethod
    def from_json(cls, d: dict) -> "View":
        return cls([Obs(**o) for o in d["nodes"]], d["legal"], d["W"], d["round"], d["max_rounds"],
                   d["max_depth"], d["max_branches"], d["baseline_score"], d.get("root_slots", 0))


def make_view(tree: Tree, order: list[int], reveal_round: dict[int, int], legal: list[int], W: int,
              round: int, max_rounds: int, max_depth: int, max_branches: int, baseline_score: float,
              root_slots: int) -> View:
    """Build the View the policy sees. `order` is the revealed node ids in the order they were revealed,
    root first.

    Ids and rounds are renumbered to that order. Online they already match, but a recorded tree replayed in a
    different order would otherwise hand the policy out-of-sequence ids and creation rounds, which is a plain
    tell that it is being replayed.
    """
    vid = {t: k for k, t in enumerate(order)}
    obs = []
    for t in order:
        n = tree.nodes[t]
        obs.append(Obs(vid[t], None if n.parent is None else vid.get(n.parent), n.branch, n.depth,
                       reveal_round[t], n.ok, n.score,
                       (n.error or "")[:160] if n.error else None, n.mechanism[:160]))
    view = View(obs, [vid[l] for l in legal], W, round, max_rounds, max_depth, max_branches,
                baseline_score, root_slots)
    view._tid = list(order)
    return view


POLICY_API_DOC = '''
class Obs:            # one revealed node (frozen dataclass)
    id: int; parent: int|None; branch: int; depth: int; round: int
    ok: bool; score: float|None; error: str|None; mechanism: str

class View:           # everything the policy may look at this round
    W: int                    # max batch size
    round: int                # current decision round (0-based)
    max_rounds: int           # rollout stops after this many rounds
    max_depth: int            # a node at this depth cannot be refined further
    max_branches: int         # root cannot be selected once this many branches exist
    baseline_score: float     # score of the seed program every branch starts from
    root: int = 0
    nodes() -> dict[int, Obs]           # revealed prefix, includes root (id 0, score None)
    legal_actions() -> list[int]        # root (if a branch can still be opened) + refinable leaves
    root_slots() -> int                 # how many new branches may be opened now (root may repeat that many times)
    branches() -> dict[int, list[Obs]]  # branch -> trajectory ordered by depth
    frontier(branch) -> Obs|None        # deepest node of a branch
    best_score() -> float|None
    n_calls() -> int                    # non-root nodes revealed so far (= generations spent)

class GridPlanningContext:   # what you know before an episode starts
    round: int                  # outer recursive round, 1-based
    W: int
    max_branches_cap: int       # a plan above this is clamped
    max_depth_cap: int
    baseline_score: float
    history: list               # one {"attempts", "rounds", "best"} summary per earlier episode

class Policy:
    NAME = "..."

    def plan_grid(self, ctx: GridPlanningContext) -> GridPlan:
        """OPTIONAL. The budget shape for the whole episode, chosen before it starts. Return
        GridPlan(max_branches=..., max_depth=...), or a (branches, depth) tuple. Both are clamped to the
        caps in ctx. Omit the method to just take the caps. This is how you spend LESS when the history
        says progress has stalled, and MORE when it has not: stopping early is not the only lever."""

    def select_batch(self, view: View) -> list[int]:
        """Return <= view.W ids from view.legal_actions(). Empty list = stop the rollout.
        Each occurrence of root (0) opens ONE new branch (a fresh attempt from scratch), so root may
        appear up to view.root_slots() times; every other id must be a distinct leaf and refines that
        branch by one attempt.

        Called fresh every round. If you need to carry something between rounds of the SAME episode, read
        and write a file in the working directory: it is a private temp dir that lives for one episode and
        is discarded afterwards. Never assume it exists on the first call."""
'''


class PolicyError(Exception):
    pass


def _runner_src() -> str:
    return r'''
import json, sys, importlib.util
sys.path.insert(0, sys.argv[3])
from dream_rsi.policy_api import GridPlanningContext, View
spec = importlib.util.spec_from_file_location("candidate_policy", sys.argv[1])
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
payload = json.load(open(sys.argv[2]))
policy = mod.Policy()
if sys.argv[4] == "grid":
    ctx = GridPlanningContext(**payload)
    if not hasattr(policy, "plan_grid"):
        print(json.dumps({"grid": None}))
    else:
        plan = policy.plan_grid(ctx)
        b = getattr(plan, "max_branches", None) if not isinstance(plan, (tuple, list, dict)) else None
        d = getattr(plan, "max_depth", None) if not isinstance(plan, (tuple, list, dict)) else None
        if isinstance(plan, dict):
            b, d = plan.get("max_branches"), plan.get("max_depth")
        elif isinstance(plan, (tuple, list)) and len(plan) == 2:
            b, d = plan
        print(json.dumps({"grid": [int(b), int(d)]}))
else:
    batch = policy.select_batch(View.from_json(payload))
    print(json.dumps({"batch": [int(x) for x in batch if isinstance(x, int) and not isinstance(x, bool)]}))
'''


class PolicyRunner:
    """Executes an (LLM-written) policy file in a subprocess with a timeout, sanitizing its batch."""

    def __init__(self, policy_path: Path, timeout: float = 30.0):
        self.policy_path = Path(policy_path)
        self.timeout = timeout
        self.violations = 0
        root = Path(__file__).resolve().parent
        self._pkg_root = str(root.parent)
        # Runs from a private temp dir so __file__ and cwd do not point at runs/<name>/, where the full
        # tree.json sits. Closes the accidental path to it, not the deliberate one: prefix-only is still
        # mainly enforced by the prompt.
        self._tmp = Path(tempfile.mkdtemp(prefix="policy_"))
        self._policy_copy = self._tmp / "candidate_policy.py"
        shutil.copy(self.policy_path, self._policy_copy)
        self._runner = self._tmp / "_policy_runner.py"
        self._runner.write_text(_runner_src())
        self._cleanup = lambda: shutil.rmtree(self._tmp, ignore_errors=True)
        atexit.register(self._cleanup)

    def close(self):
        """One runner is built per (policy, tree) pair during a dreaming phase, so relying on atexit alone
        leaves hundreds of temp dirs around for the life of the process."""
        atexit.unregister(self._cleanup)
        self._cleanup()

    def _run(self, payload: dict, mode: str) -> dict:
        path = self._tmp / "payload.json"
        path.write_text(json.dumps(payload))
        try:
            r = subprocess.run([sys.executable, str(self._runner), str(self._policy_copy), str(path),
                                self._pkg_root, mode],
                               capture_output=True, text=True, timeout=self.timeout, cwd=self._tmp)
        except subprocess.TimeoutExpired:
            raise PolicyError(f"policy timed out after {self.timeout}s")
        if r.returncode != 0:
            tail = " | ".join(r.stderr.strip().splitlines()[-3:])
            raise PolicyError(f"policy crashed: {tail[:500]}")
        try:
            return json.loads(r.stdout.strip().splitlines()[-1])
        except Exception as e:  # noqa: BLE001
            raise PolicyError(f"policy produced unparsable output: {e}")

    def plan_grid(self, ctx: GridPlanningContext) -> GridPlan:
        """The episode's budget shape, chosen by the policy and clamped to the caps. A policy without a
        plan_grid method just gets the caps, which is the old fixed-grid behaviour."""
        grid = self._run(asdict(ctx), "grid").get("grid")
        if not grid:
            return GridPlan(ctx.max_branches_cap, ctx.max_depth_cap)
        b, d = grid
        if not (1 <= b <= ctx.max_branches_cap and 1 <= d <= ctx.max_depth_cap):
            self.violations += 1
        return GridPlan(max(1, min(int(b), ctx.max_branches_cap)), max(1, min(int(d), ctx.max_depth_cap)))

    def select_batch(self, view: View) -> list[int]:
        batch = self._run(view.to_json(), "batch")["batch"]
        legal = set(view.legal_actions())
        clean, seen, roots = [], set(), 0
        for b in batch:
            if b == 0 and 0 in legal and roots < view.root_slots():
                clean.append(0); roots += 1
            elif b != 0 and b in legal and b not in seen:
                clean.append(b); seen.add(b)
            else:
                self.violations += 1
        if len(clean) > view.W:
            self.violations += len(clean) - view.W
            clean = clean[:view.W]
        return [view._tid[b] for b in clean] if view._tid else clean
