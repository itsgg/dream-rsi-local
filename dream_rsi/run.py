"""Main loop:  online rollout -> (dream: replay-based policy improvement) -> redeploy.  --mode fixed keeps the
initial policy (the paper's Recursive Fixed Exploration baseline)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from .dream import dream
from . import llm as llm_mod
from .llm import LLM, Embedder
from .online import BASELINE_POLICY, rollout
from .tasks import TASKS, make_task
from .tree import Tree

ROOT = Path(__file__).resolve().parent.parent

# Set DREAM_RSI_MODEL once instead of passing --model to every command.
DEFAULT_MODEL = os.environ.get("DREAM_RSI_MODEL", "qwen2.5-coder:7b")
DEFAULT_EMBED_MODEL = os.environ.get("DREAM_RSI_EMBED_MODEL", "bge-m3")


def _git_commit() -> str | None:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dream", "fixed"], default="dream")
    ap.add_argument("--task", choices=sorted(TASKS), default="cache",
                    help="cache: eviction policy, a long improvement ladder. binpack: the FunSearch task, "
                         "kept for the earlier results but exhausted within one round by a small model")
    ap.add_argument("--rounds", type=int, default=4, help="outer recursive rounds T")
    ap.add_argument("--W", type=int, default=4, help="parallel workers / max batch size")
    ap.add_argument("--max-branches", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=6, help="root children have depth 1")
    ap.add_argument("--K1", type=int, default=10, help="max decision rounds per online rollout")
    ap.add_argument("--K2", type=int, default=None, help="max decision rounds per replay (default: same as K1, so the view's max_rounds does not reveal replay vs online)")
    ap.add_argument("--M", type=int, default=4, help="policy versions per dreaming phase (incl. incumbent)")
    ap.add_argument("--beta1", type=float, default=None,
                    help="replay cost per revealed attempt (Eq. 1); default is the task's own scale")
    ap.add_argument("--beta2", type=float, default=None,
                    help="replay parallelism bonus per attempt/round (Eq. 1); default is the task's own scale")
    ap.add_argument("--beta-sweep", default="0.5,1,2",
                    help="multipliers on beta1/beta2 that the replay evaluator sweeps when ranking policies; "
                         "\"1\" scores at the nominal betas only")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Ollama chat model (default {DEFAULT_MODEL}; set DREAM_RSI_MODEL to change it)")
    ap.add_argument("--policy-model", default=os.environ.get("DREAM_RSI_POLICY_MODEL") or None,
                    help="model for the dreaming phase; defaults to --model. Only M-1 calls per round, so a "
                         "frontier model here is cheap and is the single highest-value upgrade")
    ap.add_argument("--base-url", default=llm_mod.BASE_URL,
                    help="OpenAI-compatible endpoint for the discovery agent, e.g. https://api.openai.com/v1. "
                         "Unset means local Ollama. API keys come from the environment, never the command "
                         "line: DREAM_RSI_API_KEY, DREAM_RSI_POLICY_API_KEY, DREAM_RSI_EMBED_API_KEY")
    ap.add_argument("--policy-base-url", default=os.environ.get("DREAM_RSI_POLICY_BASE_URL") or None,
                    help="OpenAI-compatible endpoint for the policy writer; defaults to --base-url")
    ap.add_argument("--embed-base-url", default=llm_mod.EMBED_BASE_URL,
                    help="OpenAI-compatible endpoint for embeddings; defaults to local Ollama")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=None, help="Ollama sampling seed (seed + call index per call); default unseeded")
    ap.add_argument("--temperature", type=float, default=0.8, help="discovery-agent sampling temperature")
    ap.add_argument("--policy-temperature", type=float, default=0.7, help="policy-development sampling temperature")
    ap.add_argument("--policy-timeout", type=float, default=30.0, help="seconds a policy may take per decision")
    ap.add_argument("--repair-rounds", type=int, default=2, help="traceback-only repair calls per failed attempt (0 disables)")
    ap.add_argument("--policy-repair-rounds", type=int, default=2, help="traceback-only repair calls per crashed policy version (0 disables)")
    ap.add_argument("--novelty-threshold", type=float, default=0.95, help="cosine-similarity threshold for near-duplicate rejection (0 disables)")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                    help=f"Ollama embedding model for novelty rejection (default {DEFAULT_EMBED_MODEL}; "
                         "set DREAM_RSI_EMBED_MODEL to change it)")
    ap.add_argument("--smoke", action="store_true", help="tiny config to validate the pipeline")
    a = ap.parse_args()
    if a.smoke:
        a.rounds, a.W, a.max_branches, a.max_depth, a.K1, a.M = 2, 2, 3, 2, 3, 2
    if a.K2 is None:
        a.K2 = a.K1

    name = a.name or f"{a.mode}_{a.task}_{datetime.now():%Y%m%d_%H%M}" + ("_smoke" if a.smoke else "")
    run_dir = ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / "run.log", "a")

    def log(msg: str):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    task = make_task(a.task)
    # Score scales differ between tasks, so the Eq. (1) costs do too. An explicit --beta1/--beta2 wins.
    if a.beta1 is None:
        a.beta1 = task.default_beta1
    if a.beta2 is None:
        a.beta2 = task.default_beta2
    beta_sweep = [float(x) for x in a.beta_sweep.split(",") if x.strip()] or [1.0]
    cfg = {"W": a.W, "max_branches": a.max_branches, "max_depth": a.max_depth, "K1": a.K1, "K2": a.K2, "M": a.M,
           "beta1": a.beta1, "beta2": a.beta2, "beta_sweep": beta_sweep, "baseline_score": task.seed_score, "seed_program": task.seed_name,
           "mode": a.mode, "model": a.model, "policy_model": a.policy_model or a.model, "rounds": a.rounds,
           # endpoints are recorded so a run is reproducible; API keys never are
           "base_url": a.base_url, "policy_base_url": a.policy_base_url or a.base_url,
           "embed_base_url": a.embed_base_url,
           "sampling_seed": a.seed, "temperature": a.temperature, "policy_temperature": a.policy_temperature,
           "policy_timeout": a.policy_timeout, "num_ctx": 8192, "num_predict": 1200, "policy_num_predict": 2500,
           "task": task.name, "task_key": a.task, "metric_label": task.metric_label,
           "train_spec": task.train_spec, "heldout_spec": task.heldout_spec,
           "reference_scores": task.baseline_scores, "code_commit": _git_commit(),
           "repair_rounds": a.repair_rounds, "policy_repair_rounds": a.policy_repair_rounds,
           "novelty_threshold": a.novelty_threshold, "embed_model": a.embed_model}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    log(f"run {name}: {json.dumps(cfg)}")
    log(f"baseline scores: {task.baseline_scores}")

    log_path = str(run_dir / "llm_calls.jsonl")
    policy_key = os.environ.get("DREAM_RSI_POLICY_API_KEY") or llm_mod.API_KEY
    llm = LLM(a.model, temperature=a.temperature, log_path=log_path, seed=a.seed,
              base_url=a.base_url, api_key=llm_mod.API_KEY)
    same = cfg["policy_model"] == a.model and cfg["policy_base_url"] == a.base_url
    pllm = llm if same else LLM(cfg["policy_model"], temperature=a.policy_temperature, log_path=log_path,
                                seed=a.seed, base_url=cfg["policy_base_url"], api_key=policy_key)
    log(f"discovery agent: {llm.describe()}")
    if pllm is not llm:
        log(f"policy writer:   {pllm.describe()}")
    if a.seed is not None and "openai" in (llm.provider + pllm.provider):
        log("note: --seed is best-effort at most on an OpenAI-compatible endpoint; this run is not reproducible")

    embedder = (Embedder(a.embed_model, base_url=a.embed_base_url, api_key=llm_mod.EMBED_API_KEY)
                if a.novelty_threshold > 0 else None)
    policy = run_dir / "policy_round_01.py"
    shutil.copy(BASELINE_POLICY, policy)
    history: list[Tree] = []
    summary = {"name": name, "config": cfg, "rounds": []}
    cumulative = 0
    for t in range(1, a.rounds + 1):
        rdir = run_dir / f"round_{t:02d}"
        rdir.mkdir(exist_ok=True)
        log(f"=== round {t}/{a.rounds}: online rollout with {policy.name} ===")
        t0 = time.time()
        tree = rollout(task, policy, history, llm, cfg, t, log, embedder=embedder)
        tree.save(rdir / "tree.json")
        history.append(tree)
        cumulative += tree.n_calls()
        best = tree.best()
        hist_best = max((n for tr in history for n in tr.nodes.values() if n.ok and n.score is not None and n.id != 0),
                        key=lambda n: n.score, default=None)
        agent_llm_calls = sum(n.metrics.get("llm_calls", 1) for n in tree.nodes.values() if n.id)
        rec = {"round": t, "policy": policy.name, "calls": tree.n_calls(), "cumulative_calls": cumulative,
               "agent_llm_calls": agent_llm_calls,
               "repaired": sum(1 for n in tree.nodes.values() if n.id and n.metrics.get("repaired")),
               "near_duplicates": sum(1 for n in tree.nodes.values() if n.id and n.metrics.get("near_duplicate_of")),
               "decision_rounds": tree.meta.get("rounds"), "failed": sum(1 for n in tree.nodes.values() if n.id and not n.ok),
               "round_best": best.score if best else None,
               "round_best_heldout": best.metrics.get("heldout_score") if best else None,
               "best_so_far": hist_best.score if hist_best else None,
               "best_so_far_heldout": hist_best.metrics.get("heldout_score") if hist_best else None,
               "best_mechanism": hist_best.mechanism if hist_best else None,
               "rollout_seconds": tree.meta.get("seconds"), "policy_fallback_round": tree.meta.get("policy_fallback_round")}
        log(f"round {t} done: {tree.n_calls()} calls in {tree.meta.get('seconds')}s, round best "
            f"{rec['round_best']}, best so far {rec['best_so_far']} (held-out {rec['best_so_far_heldout']}), cumulative {cumulative}")
        if hist_best:
            (rdir / "best_program.py").write_text(hist_best.code)
        if a.mode == "dream" and t < a.rounds:
            log(f"--- dreaming over {len(history)} replay world(s) ---")
            t1 = time.time()
            selected = dream(run_dir, t, history, policy, pllm, cfg, log)
            policy = run_dir / f"policy_round_{t+1:02d}.py"
            shutil.copy(selected, policy)
            rec["dream_seconds"] = round(time.time() - t1, 1)
            rec["next_policy_changed"] = (run_dir / f"policy_round_{t:02d}.py").read_text() != policy.read_text()
        elif t < a.rounds:
            nxt = run_dir / f"policy_round_{t+1:02d}.py"
            shutil.copy(policy, nxt); policy = nxt
        rec["llm"] = llm.stats() | ({"policy_llm": pllm.stats()} if pllm is not llm else {})
        summary["rounds"].append(rec)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"finished. total discovery-agent calls {cumulative}; llm stats {llm.stats()}")


if __name__ == "__main__":
    main()
