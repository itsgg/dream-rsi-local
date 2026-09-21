"""Check that this machine can actually run an experiment, before spending an hour finding out it cannot.

    uv run python -m tools.preflight                          # checks the default model
    uv run python -m tools.preflight --model qwen3-coder:30b

Verifies: Ollama reachable, the chat model and the embedding model present, one real call to each, both
tasks build and reproduce their reference scores, and the policy subprocess runs. Exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

from dream_rsi import llm as llm_mod
from dream_rsi.llm import OLLAMA_URL, Embedder, LLM
from dream_rsi.run import DEFAULT_EMBED_MODEL, DEFAULT_MODEL

OK, BAD = "  ok  ", " FAIL "
failures: list[str] = []


def check(label: str, fn):
    t0 = time.time()
    try:
        detail = fn() or ""
        print(f"[{OK}] {label}: {detail} ({time.time() - t0:.1f}s)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[{BAD}] {label}: {type(e).__name__}: {e}", flush=True)
        failures.append(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="chat model the run will use")
    ap.add_argument("--policy-model", default=None, help="only if you plan to pass --policy-model")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--base-url", default=llm_mod.BASE_URL)
    ap.add_argument("--policy-base-url", default=os.environ.get("DREAM_RSI_POLICY_BASE_URL") or None)
    ap.add_argument("--embed-base-url", default=llm_mod.EMBED_BASE_URL)
    a = ap.parse_args()

    policy_key = os.environ.get("DREAM_RSI_POLICY_API_KEY") or llm_mod.API_KEY
    endpoints = [("discovery agent", a.model, a.base_url, llm_mod.API_KEY)]
    if a.policy_model or a.policy_base_url:
        endpoints.append(("policy writer", a.policy_model or a.model,
                          a.policy_base_url or a.base_url, policy_key))
    for role, model, base, key in endpoints:
        print(f"{role:16s} {model} via {'openai-compatible at ' + base if base else 'ollama at ' + OLLAMA_URL}")
    print(f"{'embeddings':16s} {a.embed_model} via "
          f"{'openai-compatible at ' + a.embed_base_url if a.embed_base_url else 'ollama at ' + OLLAMA_URL}\n")

    # Local models have to be pulled first; a remote endpoint just has to answer, which the live call checks.
    local = [m for _, m, base, _ in endpoints if not base] + ([a.embed_model] if not a.embed_base_url else [])
    installed: list[str] = []
    if local:
        def tags():
            with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=10) as r:
                data = json.loads(r.read())
            installed.extend(m["name"] for m in data.get("models", []))
            return f"{len(installed)} model(s) installed"

        check("Ollama reachable", tags)
        if failures:
            print("\nStart it with `ollama serve`, or set OLLAMA_URL if it listens elsewhere.")
            raise SystemExit(1)

        def present(name):
            def fn():
                # Ollama reports "qwen3-coder:30b"; a bare "bge-m3" is stored as "bge-m3:latest".
                if name in installed or f"{name}:latest" in installed:
                    return "installed"
                raise RuntimeError(f"not installed, run `ollama pull {name}`")
            return fn

        for m in dict.fromkeys(local):
            check(f"model {m}", present(m))

    if not failures:
        for role, model, base, key in endpoints:
            def chat(model=model, base=base, key=key):
                if base and not key:
                    raise RuntimeError("no API key: set DREAM_RSI_API_KEY (or DREAM_RSI_POLICY_API_KEY)")
                out = LLM(model, num_predict=24, base_url=base, api_key=key).chat(
                    "Reply with exactly one word.", "Say: ready", temperature=0.0)
                return f"replied {out.strip()[:30]!r}"
            check(f"chat call, {role} ({model})", chat)

        def embed():
            v = Embedder(a.embed_model, base_url=a.embed_base_url, api_key=llm_mod.EMBED_API_KEY).embed(
                "def priority(now, last_used, freq, inserted):\n    return last_used\n")
            return f"{len(v)}-dim vector"

        check(f"embedding call to {a.embed_model}", embed)

    from dream_rsi.tasks import TASKS, make_task

    for key in sorted(TASKS):
        def build(key=key):
            # The seed may be the strongest reference (cache) or the weakest (binpack), so neither ordering
            # is checkable here. Headroom is asserted in tests/test_tasks.py.
            t = make_task(key)
            spread = max(t.baseline_scores.values()) - min(t.baseline_scores.values())
            assert spread > 1e-6, "all reference policies score the same: evaluator is not discriminating"
            assert abs(t.evaluate(t.seed_code)["score"] - t.seed_score) < 1e-9, "seed score not reproducible"
            assert not t.evaluate("def priority(*a):\n    return 0.0\n")["ok"], "evaluator accepts junk"
            return (f"seed {t.seed_name}={t.seed_score:.4f}, reference spread {spread:.4f} over "
                    f"{len(t.baseline_scores)} policies")
        check(f"task {key} builds and scores", build)

    def policy():
        from pathlib import Path
        from dream_rsi.online import BASELINE_POLICY
        from dream_rsi.policy_api import GridPlanningContext, PolicyRunner, make_view
        from dream_rsi.tree import Tree
        t = Tree({"name": "preflight"})
        t.add(0, 0, ok=True, score=-1.0, mechanism="a")
        order = sorted(t.nodes)
        runner = PolicyRunner(Path(BASELINE_POLICY))
        try:
            plan = runner.plan_grid(GridPlanningContext(1, 4, 8, 6, -2.0, []))
            view = make_view(t, order, {i: t.nodes[i].round for i in order}, [0, 1], 4, 0, 10,
                             plan.max_depth, plan.max_branches, -2.0, 4)
            batch = runner.select_batch(view)
        finally:
            runner.close()
        assert batch, "baseline policy returned an empty batch"
        return f"grid {plan.max_branches}x{plan.max_depth}, batch {batch}"

    check("policy subprocess", policy)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        raise SystemExit(1)
    print(f"All checks passed. Suggested first run:\n"
          f"  uv run python -m dream_rsi.run --smoke --model {a.model}")


if __name__ == "__main__":
    main()
