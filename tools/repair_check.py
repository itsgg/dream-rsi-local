"""Replay the policy versions that crashed in a recorded matrix through the repair step in dream.py.

Measures what the fix claims: how many crashed policies become usable within two traceback-only calls, and
whether any of the repaired ones would have beaten their incumbent and been deployed.

    uv run python -m tools.repair_check [run-glob]

Default glob is the recorded bin-packing matrix. Needs Ollama running; costs one or two calls per crash.
"""
import json
import pathlib
import shutil
import sys
import tempfile

from dream_rsi.dream import REPAIR_SYSTEM, normalise, repair_prompt
from dream_rsi.llm import LLM, extract_code
from dream_rsi.replay import evaluate_policy
from dream_rsi.tree import Tree


def main():

    ROOT = pathlib.Path(__file__).resolve().parent.parent
    GLOB = sys.argv[1] if len(sys.argv) > 1 else "dream_qwen2_5-coder_7b_s*"
    TMP = pathlib.Path(tempfile.mkdtemp(prefix="repaircheck_"))

    cases = []
    model = None
    for vj in sorted(ROOT.glob(f"runs/{GLOB}/round_*/dream/versions.json")):
        run_dir = vj.parents[2]
        cfg_all = json.loads((run_dir / "config.json").read_text())
        model = model or cfg_all.get("policy_model") or cfg_all.get("model")
        rnd = int(vj.parents[1].name.split("_")[1])
        trees = [Tree.load(run_dir / f"round_{r:02d}" / "tree.json") for r in range(1, rnd + 1)]
        cfg = {k: cfg_all[k] for k in ("W", "K2", "max_depth", "max_branches", "baseline_score", "beta1", "beta2")}
        cfg["policy_timeout"] = cfg_all.get("policy_timeout", 30)
        versions = json.loads(vj.read_text())
        incumbent = versions[0]["mean_V"]
        for v in versions:
            if v["name"].startswith("v00") or not v["errors"]:
                continue
            src = pathlib.Path(v["path"])
            if not src.exists():
                continue
            cases.append((f"{run_dir.name}/r{rnd}/{v['name']}", src.read_text(), v["errors"][0], trees, cfg, incumbent))

    if not cases:
        raise SystemExit(f"no crashed policy versions under runs/{GLOB}")
    llm = LLM(model, temperature=0.2, log_path=str(TMP / "calls.jsonl"), seed=7)
    print(f"crashed policy versions found: {len(cases)} (repairing with {model})\n")
    fixed = beat = 0
    for name, code, err, trees, cfg, incumbent in cases:
        if "no valid" in err:
            print(f"{name:42s} SKIP (never produced a code block)")
            continue
        cur, ev, rounds = normalise(code), {"errors": [err], "mean_V": float("-inf")}, 0
        while ev["errors"] and rounds < 2:
            rounds += 1
            out = extract_code(llm.chat(REPAIR_SYSTEM, repair_prompt(cur, ev["errors"][0]),
                                        temperature=0.2, num_predict=2500, tag=f"{name}.{rounds}"))
            if not out or "class Policy" not in out:
                break
            cur = normalise(out)
            vp = TMP / "cand.py"
            vp.write_text(cur)
            ev = evaluate_policy(vp, trees, cfg)
        if ev["errors"]:
            print(f"{name:42s} still broken after {rounds}: {ev['errors'][0][:70]}")
        else:
            fixed += 1
            better = incumbent is None or ev["mean_V"] > incumbent
            beat += better
            mark = "  BEATS INCUMBENT" if better else ""
            inc = f"{incumbent:.4f}" if incumbent is not None else "-inf"
            print(f"{name:42s} FIXED in {rounds}: V={ev['mean_V']:.4f} (incumbent {inc}){mark}")

    n = sum(1 for c in cases if "no valid" not in c[2])
    print(f"\nrepaired {fixed}/{n} crashed policies; {beat} of them would have been deployed")
    print("llm:", llm.stats())
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
