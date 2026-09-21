"""Summarise runs: markdown table + plot of best-so-far score vs cumulative discovery-agent calls."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(run_names: list[str]):
    runs = []
    for n in run_names:
        p = ROOT / "runs" / n / "summary.json"
        if p.exists():
            runs.append(json.loads(p.read_text()))
    if not runs:
        print("no runs found"); return
    # Scores from different tasks are not comparable (bin packing is around -2, cache hit rate around 0.5),
    # so never put them in one table or on one axis. With no arguments this globs every run, which makes
    # mixing the default rather than the exception.
    by_task: dict[str, list[dict]] = {}
    for r in runs:
        by_task.setdefault(r["config"].get("task", "?"), []).append(r)   # task name, present in every run
    for i, (task, group) in enumerate(sorted(by_task.items())):
        if len(by_task) > 1:
            print(f"{'' if i == 0 else chr(10)}# task: {task}\n")
        report(task, group, [r["name"] for r in group])


def report(task: str, runs: list[dict], run_names: list[str]):
    print("| run | round | policy | attempts | cumulative | LLM calls | failed | repaired | round best | best so far | held-out | best mechanism |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        for x in r["rounds"]:
            f = lambda v: "-" if v is None else f"{v:.3f}"
            print(f"| {r['name']} | {x['round']} | {x['policy']} | {x['calls']} | {x['cumulative_calls']} | {x.get('agent_llm_calls', '-')} | {x['failed']} | "
                  f"{x.get('repaired', '-')} | {f(x['round_best'])} | {f(x['best_so_far'])} | {f(x['best_so_far_heldout'])} | {(x['best_mechanism'] or '')[:70]} |")
    by_mode: dict[str, list[dict]] = {}
    for r in runs:
        by_mode.setdefault(r["config"]["mode"], []).append(r)
    if any(len(v) > 1 for v in by_mode.values()):
        print("\n### Aggregate over runs (mean [min, max])")
        print("| mode | runs | total attempts | total LLM calls | final best | final held-out |")
        print("|---|---|---|---|---|---|")
        for mode, rs in by_mode.items():
            def agg(vals):
                vals = [v for v in vals if v is not None]
                return f"{sum(vals)/len(vals):.3f} [{min(vals):.3f}, {max(vals):.3f}]" if vals else "-"
            print(f"| {mode} | {len(rs)} | {agg([r['rounds'][-1]['cumulative_calls'] for r in rs])} | "
                  f"{agg([sum(x.get('agent_llm_calls', x['calls']) for x in r['rounds']) for r in rs])} | "
                  f"{agg([r['rounds'][-1]['best_so_far'] for r in rs])} | {agg([r['rounds'][-1]['best_so_far_heldout'] for r in rs])} |")
    for r in runs:
        rd = ROOT / "runs" / r["name"]
        print(f"\n### {r['name']}: policy evolution")
        for x in r["rounds"]:
            t = x["round"]
            pol = rd / f"policy_round_{t:02d}.py"
            doc = ""
            if pol.exists():
                src = pol.read_text()
                m = re.search(r'"""(.*?)"""', src, re.S)
                nm = re.search(r'NAME\s*=\s*"([^"]+)"', src)
                doc = (nm.group(1) if nm else "?") + ": " + " ".join((m.group(1) if m else "").split())[:220]
            print(f"- round {t} deployed `{pol.name}` -> {doc}")
            vj = rd / f"round_{t:02d}" / "dream" / "versions.json"
            if vj.exists():
                vs = json.loads(vj.read_text())
                def fv(v):  # mean_V is null in versions.json when the policy failed replay (-inf)
                    return f"{v['mean_V']:.4f}" if v["mean_V"] is not None else "-inf"
                print("    dreaming: " + ", ".join(f"{v['name']} V={fv(v)}" + (" (failed)" if v["errors"] else "") for v in vs))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})
    for r in runs:
        xs = [0] + [x["cumulative_calls"] for x in r["rounds"]]
        ys = [r["config"]["baseline_score"]] + [x["best_so_far"] if x["best_so_far"] is not None else float("nan") for x in r["rounds"]]
        ax.plot(xs, ys, marker="o", label=r["name"])
        for x in r["rounds"]:
            if x["best_so_far"] is not None:
                ax.annotate(str(x["round"]), (x["cumulative_calls"], x["best_so_far"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(runs[0]["config"]["baseline_score"], ls="--", c="gray", lw=0.8, label="seed program")
    ax.set_xlabel("cumulative discovery-agent calls")
    ax.set_ylabel(runs[0]["config"].get("metric_label", "best score so far"))
    ax.legend(); ax.grid(alpha=0.3)
    width = 0.8 / max(1, len(runs))
    for i, r in enumerate(runs):
        xs = [x["round"] + (i - (len(runs) - 1) / 2) * width for x in r["rounds"]]
        ax2.bar(xs, [x["calls"] for x in r["rounds"]], width=width, label=r["name"])
    ax2.set_xlabel("recursive round"); ax2.set_ylabel("discovery-agent calls in round"); ax2.set_xticks([x["round"] for x in runs[0]["rounds"]])
    ax2.legend(); ax2.grid(alpha=0.3, axis="y"); fig.tight_layout()
    # Name the plot after what the runs have in common, not all of them concatenated and truncated.
    tags = sorted({re.sub(r"_s\d+$", "", re.sub(r"^(dream|fixed)_", "", n)) for n in run_names})
    stem = tags[0] if len(tags) == 1 else task
    if len(stem) > 60:
        stem = stem[:50] + "_" + hashlib.sha1("_".join(run_names).encode()).hexdigest()[:8]
    out = ROOT / "runs" / f"report_{stem}.png"
    fig.savefig(out, dpi=130)
    print(f"\nplot: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(p.name for p in (ROOT / "runs").iterdir() if (p / "summary.json").exists()))
