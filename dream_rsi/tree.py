"""Discovery tree: root = initial workspace; every non-root node has exactly one parent and records
the artifact (program), its score and diagnostics. Leaves + root are the selectable actions."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Node:
    id: int
    parent: int | None
    branch: int            # index of the root child this node descends from (root: -1)
    depth: int             # root = 0
    round: int             # decision round in which it was created
    created: int           # creation order (for replay of root children)
    ok: bool = False
    score: float | None = None
    error: str | None = None
    mechanism: str = ""
    code: str = ""
    metrics: dict = field(default_factory=dict)


class Tree:
    def __init__(self, meta: dict | None = None):
        self.meta = meta or {}
        self.nodes: dict[int, Node] = {0: Node(0, None, -1, 0, 0, 0, ok=True, mechanism="<root>")}
        self.children: dict[int, list[int]] = {0: []}

    def add(self, parent: int, round_idx: int, **kw) -> Node:
        p = self.nodes[parent]
        nid = len(self.nodes)
        branch = len(self.children[0]) if parent == 0 else p.branch
        n = Node(nid, parent, branch, p.depth + 1, round_idx, nid, **kw)
        self.nodes[nid] = n
        self.children[nid] = []
        self.children[parent].append(nid)
        return n

    def root_children(self) -> list[int]:
        return sorted(self.children[0], key=lambda i: self.nodes[i].created)

    def leaves(self, subset: set[int] | None = None) -> list[int]:
        ids = subset if subset is not None else set(self.nodes)
        return sorted(i for i in ids if i != 0 and not [c for c in self.children[i] if c in ids])

    def best(self) -> Node | None:
        ok = [n for n in self.nodes.values() if n.ok and n.score is not None and n.id != 0]
        return max(ok, key=lambda n: n.score) if ok else None

    def n_calls(self) -> int:
        return len(self.nodes) - 1

    def brief(self) -> dict:
        """One-line summary of a finished tree, for a policy planning its next episode."""
        b = self.best()
        return {"attempts": self.n_calls(), "rounds": self.meta.get("rounds", 0),
                "best": b.score if b else None}

    def to_json(self) -> dict:
        return {"meta": self.meta, "nodes": [asdict(n) for n in self.nodes.values()]}

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: Path) -> "Tree":
        d = json.loads(Path(path).read_text())
        t = cls(d["meta"])
        t.nodes = {}
        t.children = {}
        for nd in d["nodes"]:
            n = Node(**nd)
            t.nodes[n.id] = n
            t.children.setdefault(n.id, [])
            if n.parent is not None:
                t.children.setdefault(n.parent, []).append(n.id)
        return t

    def branch_table(self) -> dict[int, list[Node]]:
        out: dict[int, list[Node]] = {}
        for n in self.nodes.values():
            if n.id == 0:
                continue
            out.setdefault(n.branch, []).append(n)
        for b in out:
            out[b].sort(key=lambda n: n.depth)
        return out
