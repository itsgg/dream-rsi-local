"""Discovery agent: one LLM call resumes a node's workspace (its branch trajectory + global history)
and produces one new candidate program, which the fixed evaluator scores."""
from __future__ import annotations

import random
import re

from .llm import LLM, Embedder, extract_code
from .task_binpack import BinPackTask
from .tree import Tree

SYSTEM_TEMPLATE = """You are an expert algorithm engineer competing in an automated discovery run.
You write ONE complete Python program per reply, inside a single ```python fenced block.
The program's first line must be a comment of the form `# MECHANISM: <one-line description of the idea>`.
Then define `{signature}` (numpy is already imported as np; you may
`import math`). No other imports, no printing, no test code. Never claim results you have not measured."""


def system_prompt(task) -> str:
    return SYSTEM_TEMPLATE.format(signature=task.signature)

GUIDANCE = """\
## How to choose the next attempt
1. Read the whole history above before proposing. Trust measured scores over what a mechanism claims.
2. Learn from failures: if an attempt failed on a bug (shape error, NaN, exception), a targeted fix is worth
   one retry; a mechanism that ran fine but scored badly should not be repeated or renamed.
3. Do not converge on a local optimum: if the branches cluster around small variations of one idea with
   flattening returns, try a different kind of change rather than another small tweak of the same idea.
4. Keep what measurably works. A change that dropped the score far below the baseline (see the diagnostics in
   brackets) must be reverted, not refined. Never resubmit an identical or near-identical program.
"""


REPAIR_SYSTEM = """You fix a broken Python program. Reply with the complete corrected program in one ```python block.
Change only what is needed to fix the error; keep the mechanism and the `# MECHANISM:` line. No explanations."""


def _mechanism(code: str, fallback: str = "unspecified") -> str:
    m = re.search(r"#\s*MECHANISM:\s*(.+)", code)
    return m.group(1).strip()[:200] if m else fallback


def fmt_score(n) -> str:
    if n.ok and n.score is not None:
        d = n.metrics.get("diagnostic")
        return f"score={n.score:.3f}" + (f" [{d}]" if d else "")
    return f"FAILED ({(n.error or 'unknown')[:120]})"


def global_history(history: list[Tree], max_full: int = 2, max_lines: int = 8, rng: random.Random | None = None) -> str:
    """Compact rendering of all previous discovery trees (the paper's H_{t-1}).

    Full code is shown for `max_full` programs sampled from the top 5 distinct programs with probability
    proportional to rank (FunSearch-style best-shot sampling), so different attempts see different examples."""
    if not history:
        return "## Global history from previous rounds\n(none: this is the first round)\n"
    nodes = [n for t in history for n in t.nodes.values() if n.id != 0]
    ok = sorted([n for n in nodes if n.ok and n.score is not None], key=lambda n: -n.score)
    distinct, seen = [], set()
    for n in ok:
        key = Embedder.normalize(n.code)
        if key not in seen:
            seen.add(key); distinct.append(n)
    fails = [n for n in nodes if not n.ok]
    out = [f"## Global history from previous rounds ({len(history)} rounds, {len(nodes)} attempts, {len(fails)} failed)"]
    out.append("Best distinct programs so far (highest score first):")
    for n in distinct[:max_lines]:
        out.append(f"- score={n.score:.3f}: {n.mechanism}")
    pool = distinct[:5]
    rng = rng or random.Random(0)
    weights = [len(pool) - i for i in range(len(pool))]
    shown = []
    while pool and len(shown) < min(max_full, len(pool)):
        pick = rng.choices(range(len(pool)), weights=weights)[0]
        shown.append(pool.pop(pick)); weights.pop(pick)
    for n in shown:
        out.append(f"\nFull code of a top program (score={n.score:.3f}):\n```python\n{n.code.strip()}\n```")
    if fails:
        seen = set(); lines = []
        for n in fails:
            key = (n.error or "")[:60]
            if key in seen:
                continue
            seen.add(key); lines.append(f"- {n.mechanism} -> {(n.error or '')[:100]}")
            if len(lines) >= 5:
                break
        out.append("\nRepresentative failures:\n" + "\n".join(lines))
    return "\n".join(out) + "\n"


def workspace_context(tree: Tree, parent_id: int, baseline_code: str = "", baseline_score: float = 0.0) -> str:
    """The trajectory of the branch being resumed plus a glance at sibling branches in this run."""
    parts = []
    if parent_id == 0:
        parts.append("## This attempt\nYou are opening a NEW branch from the initial workspace, which contains the "
                     f"current baseline program (score={baseline_score:.3f}):\n```python\n{baseline_code.strip()}\n```\n"
                     "Start FROM this program and beat it with one clear, well-motivated change. Different branches "
                     "should try different kinds of change; do not repeat what the other branches did.")
    else:
        chain = []
        cur = parent_id
        while cur != 0:
            chain.append(tree.nodes[cur]); cur = tree.nodes[cur].parent
        chain.reverse()
        parts.append(f"## This attempt\nYou are refining branch {chain[-1].branch}, which has {len(chain)} attempt(s) so far:")
        for n in chain:
            parts.append(f"- attempt depth {n.depth}: {fmt_score(n)} | {n.mechanism}")
        last = chain[-1]
        parts.append(f"\nCurrent program of this branch ({fmt_score(last)}):\n```python\n{last.code.strip()}\n```")
        if not last.ok and len(chain) > 1 and chain[-2].ok:
            parts.append(f"Previous working version of this branch ({fmt_score(chain[-2])}):\n```python\n{chain[-2].code.strip()}\n```")
    sib = tree.branch_table()
    others = {b: traj for b, traj in sib.items() if parent_id == 0 or b != tree.nodes[parent_id].branch}
    if others:
        parts.append("\n## Other branches already open in this run (do not duplicate them)")
        for b, traj in sorted(others.items()):
            best = max([n.score for n in traj if n.ok and n.score is not None], default=None)
            bs = f"best={best:.3f}" if best is not None else "no valid score yet"
            parts.append(f"- branch {b} ({len(traj)} attempts, {bs}): " + "; ".join(n.mechanism[:80] for n in traj[-2:]))
    return "\n".join(parts) + "\n"


def _existing_codes(tree: Tree, history: list[Tree]) -> list[tuple[str, str]]:
    out = []
    for t in [*history, tree]:
        for n in t.nodes.values():
            if n.id != 0 and n.code:
                out.append((f"{t.meta.get('name')}#{n.id}", n.code))
    return out


def run_attempt(task: BinPackTask, tree: Tree, parent_id: int, history: list[Tree], llm: LLM,
                round_idx: int, repair_rounds: int = 2, embedder: Embedder | None = None,
                novelty_threshold: float = 0.95, rng: random.Random | None = None,
                seed_base: int | None = None) -> dict:
    """Generate and evaluate one child of `parent_id`; returns kwargs for Tree.add.

    A near-duplicate program (identical modulo whitespace, or within `novelty_threshold` cosine) costs one
    regeneration request, then is kept and tagged. A failed evaluation gets up to `repair_rounds`
    traceback-only calls. Both count towards `metrics["llm_calls"]`."""
    if parent_id == 0:
        ask = ("Write a new program that starts from the baseline and tries a kind of change not yet tried by the "
               "other open branches or saturated in the global history.")
    else:
        last = tree.nodes[parent_id]
        ask = ("Fix the specific bug shown above and keep the mechanism." if not last.ok else
               "Improve this branch's program. Keep what measurably works; change what does not. If the branch has "
               "plateaued, make a structurally different change rather than a tiny tweak.")
    user = "\n".join([task.description(), global_history(history, rng=rng),
                      workspace_context(tree, parent_id, task.seed_code, task.seed_score), GUIDANCE,
                      f"## Your task now\n{ask}\nReply with the complete program in one ```python block."])
    metrics: dict = {}
    llm_calls = 0

    def _key() -> int | None:
        return None if seed_base is None else seed_base + llm_calls

    def generate(extra: str, tag: str) -> str:
        nonlocal llm_calls
        text = llm.chat(system_prompt(task), user + extra, tag=f"{tag} r{round_idx} parent={parent_id}", seed_key=_key())
        llm_calls += 1
        c = extract_code(text)
        if c is None:
            text = llm.chat(system_prompt(task), user + extra + "\n\nYour previous reply had no ```python block. Reply with ONLY the code block.",
                            tag=f"{tag}-retry r{round_idx} parent={parent_id}", seed_key=_key())
            llm_calls += 1
            c = extract_code(text) or ""
        return c if "import numpy" in c else "import numpy as np\n" + c

    def near_duplicate(c: str) -> str | None:
        existing = _existing_codes(tree, history)
        key = Embedder.normalize(c)
        for name, other in existing:
            if Embedder.normalize(other) == key:
                return name
        if embedder is not None and novelty_threshold > 0 and existing:
            try:
                sim, idx = embedder.max_similarity(c, [o for _, o in existing])
            except Exception as e:  # noqa: BLE001  (embedding service down: skip the check)
                metrics["novelty_error"] = str(e)[:100]
                return None
            metrics["max_similarity"] = round(sim, 3)
            if sim >= novelty_threshold:
                return existing[idx][0]
        return None

    code = generate("", "discover")
    dup = near_duplicate(code)
    if dup:
        metrics["novelty_rejections"] = 1
        code = generate(f"\n\nYour previous program was a near-duplicate of an existing attempt ({dup}). "
                        "Propose a program that differs in mechanism, not just in constants or naming.", "discover-novel")
        dup2 = near_duplicate(code)
        if dup2:
            metrics["near_duplicate_of"] = dup2
    ev = task.evaluate(code)
    repairs = 0
    while not ev["ok"] and repairs < repair_rounds:
        repairs += 1
        fix_user = (f"{task.description()}\n## Program\n```python\n{code.strip()}\n```\n## Error\n{ev['error']}\n"
                    "Return the corrected complete program.")
        text = llm.chat(REPAIR_SYSTEM, fix_user, temperature=0.2,
                        tag=f"repair{repairs} r{round_idx} parent={parent_id}", seed_key=_key())
        llm_calls += 1
        fixed = extract_code(text)
        if not fixed:
            break
        if "import numpy" not in fixed:
            fixed = "import numpy as np\n" + fixed
        code = fixed
        ev = task.evaluate(code)
    if repairs:
        metrics["repairs"] = repairs
        metrics["repaired"] = ev["ok"]
    metrics.update({k: v for k, v in ev.items() if k not in {"ok", "score", "error", "bins"}})
    if ev["ok"] and ev["score"] < task.seed_score - getattr(task, "regression_delta", 1.0):
        # a large regression is worth an explicit diagnostic to the agent, like an evaluator error report
        metrics["diagnostic"] = f"{ev.get('diagnostic', '')}; {ev['score'] - task.seed_score:+.2f} vs baseline".strip("; ")
    if ev["ok"]:
        held = task.evaluate(code, heldout=True)
        metrics["heldout_score"] = held.get("score")
    metrics["llm_calls"] = llm_calls
    return {"ok": ev["ok"], "score": ev["score"], "error": ev["error"], "code": code,
            "mechanism": _mechanism(code), "metrics": metrics}
