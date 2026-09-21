"""Task registry. A task carries `name`, `signature`, `metric_label`, `regression_delta`, `train_spec` and
`heldout_spec`, `default_beta1` / `default_beta2`, `baselines` / `baseline_scores`, `seed_name` / `seed_code`
/ `seed_score`, plus `description()` and `evaluate(src, heldout=False)`.
"""
from __future__ import annotations

from .task_binpack import BinPackTask
from .task_cache import CacheTask

TASKS = {"binpack": BinPackTask, "cache": CacheTask}


def make_task(name: str):
    if name not in TASKS:
        raise SystemExit(f"unknown task {name!r}; choose one of {', '.join(TASKS)}")
    return TASKS[name]()
