"""Dreaming-based policy improvement: an LLM revises the exploration-policy code; every version is scored by
replay over all historical trees; the best of {incumbent, revisions} is deployed (so it never regresses)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .llm import LLM, extract_code
from .policy_api import POLICY_API_DOC
from .replay import evaluate_policy
from .tree import Tree

SYSTEM = """You are improving the EXPLORATION POLICY of an automated discovery system. You do not solve the
discovery task itself. You write ONE complete Python module in a single ```python block that defines
`class Policy` with `NAME` and `select_batch(self, view)`. Only the standard library (math, statistics,
collections) plus the given View/Obs API may be used. Code must be deterministic and must not raise."""

REPAIR_SYSTEM = """You fix a broken exploration-policy module. Reply with the complete corrected module in one
```python block: `class Policy` with `NAME` and `select_batch(self, view)`. Change only what is needed to stop
the error, and keep the strategy the module was going for. No explanations."""


def tree_summary(t: Tree) -> str:
    rows = []
    for b, traj in sorted(t.branch_table().items()):
        cells = " ".join(f"{n.score:.2f}" if n.ok and n.score is not None else "X" for n in traj)
        rows.append(f"  branch {b}: {cells}")
    best = t.best()
    bs = f"{best.score:.3f}" if best else "none"
    return f"tree {t.meta.get('name')} ({t.n_calls()} attempts, {t.meta.get('rounds')} rounds, best {bs}); scores by depth:\n" + "\n".join(rows)


def feedback_text(ev: dict) -> str:
    if ev["errors"]:
        return "REPLAY FAILED: " + "; ".join(ev["errors"])[:600]
    lines = [f"mean V = {ev['mean_V']:.4f}"]
    for p in ev["per_tree"]:
        lines.append(f"  {p['tree']}: V={p['V']:.4f} best={p['best']:.3f} N={p['N']} rounds={p['rounds']} "
                     f"attempts/round={p['N']/max(1,p['rounds']):.2f} revealed_all={p['revealed_all']}")
        for step in p["trajectory"][:8]:
            rv = ", ".join(f"b{r['branch']}d{r['depth']}={r['score']:.2f}" if r["ok"] else f"b{r['branch']}d{r['depth']}=X" for r in step["revealed"])
            lines.append(f"     round {step['round']}: batch={step['batch']} -> {rv}")
        if len(p["trajectory"]) > 8:
            lines.append(f"     ... {len(p['trajectory']) - 8} more rounds")
    return "\n".join(lines)


def build_prompt(cfg: dict, trees: list[Tree], best_code: str, best_eval: dict, versions: list[dict]) -> str:
    obj = (f"## Objective (maximise mean over trees)\n"
           f"V = best_revealed_score - beta1*N + beta2*N/max(1, rounds), with beta1={cfg['beta1']}, beta2={cfg['beta2']}.\n"
           f"N = attempts revealed (generations spent), rounds = decision rounds used. best_revealed_score is the best "
           f"score among the nodes your policy revealed (a policy that reveals nothing scores far below baseline).\n"
           f"So: reach the high-scoring nodes with FEW attempts, and batch independent promising probes in the same "
           f"round (up to W={cfg['W']}) rather than probing one at a time. A batch may open several new branches "
           f"(root repeated up to view.root_slots() times) and/or refine one frontier leaf per branch.\n"
           f"Policies are ranked by mean V over all trees AND over a sweep of beta "
           f"({', '.join(str(x) + 'x' for x in cfg.get('beta_sweep', [1.0]))} the values above), so a policy "
           f"tuned to one cost weighting does not win.\n")
    hist = "## Recorded discovery trees (replay worlds). X = failed attempt. Baseline (seed program) score = " + \
           f"{cfg['baseline_score']:.3f}\n" + "\n".join(tree_summary(t) for t in trees) + "\n"
    table = "## Policy versions evaluated so far this dreaming phase\n" + "\n".join(
        f"- {v['name']}: mean V = {v['mean_V']:.4f}" + (f" (FAILED: {v['errors'][0][:100]})" if v["errors"] else "")
        for v in versions) + "\n"
    rules = """## Rules for the policy (from the Dream-RSI paper's policy-improvement prompt, Appendix B.2)
- Prefix-only: decide from view.nodes()/branches()/legal_actions() only. Never hardcode node ids, branch ids,
  score targets, or anything read off the trees above; the policy will run on NEW trees online. Never sample
  randomly.
- Every id returned must be in view.legal_actions(); root (0) may repeat up to view.root_slots() times; at most
  view.W ids total. Return [] to stop.
- Reconstruct each branch's trajectory, not just its latest score: best successful score so far (its
  "anchor"), trend, regressions, failure/repair sequence, and remaining depth.
- Failure interpretation. Before closing a failed frontier, classify it: (a) repairable implementation failure
  (shape mismatch, NaN/inf, NameError, exception): normally worth one recovery attempt; (b) weak but
  under-explored; (c) repeatedly unpromising after enough valid evidence; (d) hard-unrecoverable. A single
  failure is a signal, not a closure. A later success reopens a branch. A repairable latest failure must not
  erase the branch's earlier successful anchor or starve it permanently.
- Rank legal roots and frontiers only from prefix-derived signals: anchor, parent-to-child gain, whole
  trajectory, success vs failure evidence, recoverability, remaining depth, and cross-branch comparison.
  Shallow weak scores alone are not enough to discard a branch: deeper attempts can recover.
- Build ONE dynamic portfolio batch per round, up to view.W: exploitation (strong refinements), exploration
  (new roots or under-explored branches), and at most ONE recovery (refining a frontier whose last attempt was
  a repairable failure). Give exploration and justified recovery a slot before filling the rest by priority;
  recovery must not displace successful refinements or leave workers idle. Do not default to a singleton
  batch, and do not use a fixed widen-all / deepen-all schedule.
- Stop only after considering the whole revealed portfolio: active, under-explored, recoverable, and unopened
  candidates. Do NOT stop while an eligible recovery or under-explored frontier remains; every remaining
  action needs an evidence-based decision to continue, reserve, or close.
- Keep thresholds relative (e.g. to the best score seen), never absolute score cutoffs.
- You may also define plan_grid(ctx) to choose the episode's branch and depth budget up front, from
  ctx.history. Spending less on a plateau by planning a smaller grid is usually better than opening a wide
  grid and then stopping early, because the objective charges you per attempt revealed, not per branch
  opened. Never plan a grid of 1x1 to game the cost term: revealing nothing scores far below baseline.
"""
    return "\n".join([
        "## View / Obs API (read carefully)\n```python" + POLICY_API_DOC + "```\n", obj, hist, table,
        f"## Current best policy (mean V = {best_eval['mean_V']:.4f})\n```python\n{best_code.strip()}\n```\n",
        "## Replay feedback for the current best policy\n" + feedback_text(best_eval) + "\n", rules,
        "## Your task\nWrite an improved complete policy module (class Policy with NAME and select_batch). "
        "Explain the change in a module docstring (2-4 lines). Reply with one ```python block only."])


def normalise(code: str) -> str:
    """The policy runs in a bare subprocess, so it needs the import even when the model forgets it."""
    if "from dream_rsi.policy_api import" not in code:
        code = "from dream_rsi.policy_api import View, Obs\n" + code
    return code


def repair_prompt(code: str, error: str) -> str:
    return "\n".join([
        "## View / Obs API\n```python" + POLICY_API_DOC + "```\n",
        f"## Module that failed\n```python\n{code.strip()}\n```\n",
        f"## Error it raised during replay\n{error}\n",
        "Common causes: returning an id that is not in view.legal_actions(), indexing an empty branch or an "
        "empty list, assuming view.nodes() contains a particular id, dividing by zero when no node has a "
        "score yet, or returning a non-int. Return the corrected complete module."])


NO_BLOCK = ("\n\nYour previous reply had no ```python block defining `class Policy`. "
            "Reply with ONLY that code block.")


def ask(llm: LLM, system: str, prompt: str, tag: str, temperature: float) -> str | None:
    """One request, retried once if the reply had no usable module. Mirrors what the discovery agent does."""
    for attempt in range(2):
        text = llm.chat(system, prompt + (NO_BLOCK if attempt else ""), temperature=temperature,
                        num_predict=2500, tag=tag if not attempt else f"{tag}-retry")
        code = extract_code(text)
        if code and "class Policy" in code:
            return normalise(code)
    return None


def dream(run_dir: Path, round_idx: int, trees: list[Tree], current_policy: Path, llm: LLM, cfg: dict, log) -> Path:
    ddir = run_dir / f"round_{round_idx:02d}" / "dream"
    ddir.mkdir(parents=True, exist_ok=True)
    v0 = ddir / "v00.py"
    shutil.copy(current_policy, v0)
    versions = []
    ev0 = evaluate_policy(v0, trees, cfg)
    versions.append({"name": "v00 (incumbent)", "path": str(v0), "mean_V": ev0["mean_V"], "errors": ev0["errors"], "eval": ev0})
    log(f"  dream v00 (incumbent): mean V = {ev0['mean_V']:.4f}")
    best = versions[0]
    for m in range(1, cfg["M"]):
        prompt = build_prompt(cfg, trees, Path(best["path"]).read_text(), best["eval"], versions)
        code = ask(llm, SYSTEM, prompt, f"dream r{round_idx} v{m:02d}", cfg.get("policy_temperature", 0.7))
        vp = ddir / f"v{m:02d}.py"
        if code is None:
            vp.write_text("# no ```python block with class Policy after two requests\n")
            ev = {"mean_V": float("-inf"), "errors": ["no valid ```python block with class Policy"], "per_tree": []}
        else:
            vp.write_text(code)
            ev = evaluate_policy(vp, trees, cfg)
        # A crashed policy is discarded, so a high crash rate turns the whole dreaming phase into a no-op.
        # Two traceback-only calls recovered 7 of the 11 crashes in the recorded matrix.
        repairs = 0
        while ev["errors"] and code is not None and repairs < cfg.get("policy_repair_rounds", 2):
            repairs += 1
            fixed = ask(llm, REPAIR_SYSTEM, repair_prompt(code, ev["errors"][0]),
                        f"dream-repair r{round_idx} v{m:02d}.{repairs}", 0.2)
            if fixed is None:
                break
            code = fixed
            vp.write_text(code)
            ev = evaluate_policy(vp, trees, cfg)
        versions.append({"name": f"v{m:02d}", "path": str(vp), "mean_V": ev["mean_V"], "errors": ev["errors"],
                         "repairs": repairs, "eval": ev})
        rep = f" (repaired after {repairs})" if repairs and not ev["errors"] else (f" (repair failed x{repairs})" if repairs else "")
        log(f"  dream v{m:02d}: mean V = {ev['mean_V']:.4f}{rep}" + (f"  FAILED: {ev['errors'][0][:120]}" if ev["errors"] else ""))
        if ev["mean_V"] > best["mean_V"]:
            best = versions[-1]
    def _j(x):  # -inf is not valid JSON
        return None if isinstance(x, float) and x == float("-inf") else x
    (ddir / "versions.json").write_text(json.dumps(
        [{k: _j(v) for k, v in ver.items() if k != "eval"} | {"per_tree": [{kk: vv for kk, vv in p.items() if kk != "trajectory"} for p in ver["eval"].get("per_tree", [])]}
         for ver in versions], indent=1))
    selected = ddir / "selected.py"
    shutil.copy(best["path"], selected)
    log(f"  selected {best['name']} (mean V = {best['mean_V']:.4f}) for the next round")
    return selected
