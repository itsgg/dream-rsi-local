# Dream-RSI Local

A laptop-sized reimplementation of Dream-RSI ([arXiv 2609.14858](https://arxiv.org/abs/2609.14858)), which
improves a coding agent's *exploration policy* by replaying recorded discovery trees instead of paying for
fresh agent runs. The authors' code is still unreleased, so this is written from Section 3 and Appendix B.
Everything runs against a local Ollama model.

The loop: run the agent under the current policy; freeze the resulting tree of attempts into a replay
simulator; have an LLM write revised policies; score each by replay over every recorded tree; deploy the
best. The incumbent is always a candidate, so the deployed policy never scores worse on the trees it was
picked against.

## Status

The mechanism works. A three-seed matrix on bin packing came out null ([RESULTS.md](RESULTS.md)) for two
measured reasons, both now fixed: the 7B policy writer crashed on 41% of the policies it wrote, and the task
was exhausted in round 1. Crashed policies now get repaired, and the default task is cache eviction, which
was checked for headroom before being adopted.

The rerun on the new task has not happened. It needs more memory than the 16 GB machine this was built on,
and it is the open question.

## Setup

Python 3.11+, [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com), on macOS or Linux.

```bash
OLLAMA_NUM_PARALLEL=4 ollama serve   # drop this under 32 GB; without it W=4 is only nominal
ollama pull qwen3-coder:30b          # or qwen2.5-coder:7b on 16 GB
ollama pull bge-m3                   # embeddings, for duplicate rejection

uv sync
export DREAM_RSI_MODEL=qwen3-coder:30b
uv run python -m tools.preflight     # checks Ollama, models, both tasks, the policy subprocess
```

On sizing: the 7B is 4.7 GB and is what the recorded runs used, at the limit of a 16 GB machine.
`qwen3-coder:30b` is the real step up at 19 GB, mixture of experts, so it generates at roughly 3B speed.
`qwen2.5-coder:14b` is the trap: it loads on 16 GB but runs four times slower than the 7B under memory
pressure. `OLLAMA_URL` and `DREAM_RSI_EMBED_MODEL` work if you need them.

### Frontier models

Any OpenAI-compatible endpoint works. The highest-value place to spend one is the policy writer: it is the
job a small model actually fails at, crashing on 41% of the policies it wrote, and it costs only M-1 calls
per round against dozens of agent calls.

```bash
export DREAM_RSI_POLICY_BASE_URL=https://api.openai.com/v1
export DREAM_RSI_POLICY_API_KEY=...
export DREAM_RSI_POLICY_MODEL=<a frontier model>
./run_matrix.sh                       # agent stays local, dreaming goes remote
```

`DREAM_RSI_BASE_URL` and `DREAM_RSI_API_KEY` do the same for the discovery agent, `DREAM_RSI_EMBED_BASE_URL`
for embeddings, which otherwise stay on Ollama because they are cheap and called constantly. Equivalent
`--base-url`, `--policy-base-url` and `--embed-base-url` flags exist; keys are environment-only, never
command-line arguments, and are never written to `config.json` or `llm_calls.jsonl`.

Two things to know. `--seed` is best-effort at most over an API, so seeded reproducibility is a local-only
property and a run that mixes the two logs a warning. And if an endpoint rejects `temperature`, `seed` or
`max_tokens`, as reasoning models tend to, the client drops or renames the parameter and retries rather than
making you configure it.

Whether this helps is an open question worth being clear about. It reliably fixes the policy writer. It may
well shrink the measured dream-versus-fixed gap, because a stronger discovery agent exhausts the task sooner
and leaves an exploration policy less to find, which is exactly how bin packing failed. Expect to need a
harder task alongside a better model.

## Run

```bash
uv run python -m tests.test_replay      # no LLM, 1 s
uv run python -m tests.test_tasks       # no LLM, 10 s
uv run python -m tests.test_llm         # no LLM, mock endpoint, 1 s
uv run python -m dream_rsi.run --smoke  # end to end, ~2 min

./run_matrix.sh                         # the experiment: 3 seeds x {dream, fixed}, then a report
```

`run_matrix.sh [model] [rounds] [seeds...]` also honours `TASK=binpack`, resumes cells it already finished,
and writes `runs/report_<tag>.md` with a plot. Budget roughly 3 hours on a 7B without parallelism.

For one-off runs use `dream_rsi.run --mode {dream,fixed} --rounds 4 --name X`, then `dream_rsi.report X Y`.
`--help` covers the rest. Defaults: W=4, caps of 8 branches and depth 6 (a policy may plan a smaller grid
per episode), K1=10 rounds, M=4 policy versions, per-task beta1/beta2 because the two tasks' scores differ
by two orders of magnitude, and `--beta-sweep 0.5,1,2` so ranking is not tuned to one cost weighting.
`--seed N` makes a run reproducible, including across attempts that run in parallel.

## The task

Write a cache eviction rule and beat LRU.

```python
def priority(now, last_used, freq, inserted) -> np.ndarray   # lowest score is evicted
```

LRU is `return last_used`, LFU is `return freq`, FIFO is `return inserted`. The score is the mean hit rate
over ten traces from five workload regimes scored *together*. They differ in how fast the popular set
drifts, how much traffic is scans of keys never requested again, how skewed popularity is, and how big the
cache is, so a rule with hard-coded thresholds wins one regime and loses another: `last_used + 1e6*(freq>=3)`
beats LRU comfortably on a single regime and lands below it over the mixture.

| | search set | held-out |
|---|---|---|
| LFU | 0.4393 | 0.4077 |
| FIFO | 0.5084 | 0.4955 |
| LRU, the seed every branch starts from | 0.5422 | 0.5263 |
| best rule I hand-wrote | 0.5628 | 0.5467 |
| Belady's MIN, offline optimum | 0.6662 | |

`tests/test_tasks.py` asserts the ladder is real: several rungs above the seed, every one of them still
ahead on held-out, and the optimum well clear of the top.

`--task binpack` is the original FunSearch bin-packing task, kept because RESULTS.md reports a matrix on it.
A 7B reaches its practical ceiling there in 24 attempts, which is why it was replaced.

## Layout

| file | |
|---|---|
| `run.py` | the outer loop |
| `online.py` | rollout: the policy picks batches, the agent expands the tree |
| `agent.py` | one Ollama call per attempt, plus traceback repair and duplicate rejection |
| `replay.py` | the replay simulator and Eq. (1) |
| `dream.py` | LLM writes revised policies, replay scores them, the best is deployed |
| `policy_api.py` | the `View`/`Obs` a policy sees, and the subprocess that runs it |
| `llm.py` | Ollama's native API, or any OpenAI-compatible endpoint |
| `task_cache.py`, `task_binpack.py` | evaluators; shared guards in `sandbox.py` |
| `tools/` | `preflight.py` checks the machine, `repair_check.py` measures the repair rate |

The algorithm follows Section 3, checked line by line against it: the action set and batch rules, the replay
reveal rules including `Child(v) = ∅` off-tree, Eq. (1), `argmax` over M candidates with the incumbent as
candidate 0, the policy's own `plan_grid()` before each episode, per-episode policy state, and the beta sweep
when ranking. What differs is everything around it. The paper's agent is Gemini through a CLI with filesystem
and execution access, ours is one chat call with no tools. The paper runs a Lasso solver, circle packing and
GPU kernels at W=10 or 32; we run bin packing and cache eviction at W=4. The Appendix B prompts are
condensed.

Each run writes `runs/<name>/` holding `config.json` (every parameter and the git commit),
`round_XX/tree.json` (the replay worlds), `round_XX/dream/` (every proposed policy with its score),
`summary.json` (what the report reads) and `run.log`. Prompts and replies go to `llm_calls.jsonl`, which is
gitignored.

## Known limits

Generated code runs as you. The evaluator pre-binds what the scorer needs so a candidate cannot forge its
score, restricts builtins and imports, and sets rlimits, but that is a hurdle rather than a sandbox.

Replay only covers where history actually went. A policy that wants eight branches earns no credit for them
if the recorded tree has four, and the paper's only mitigation is that support widens as trees accumulate.
The interface itself no longer gives the game away: `legal_actions()` and `root_slots()` report the same
thing in both modes, ids and rounds are renumbered to reveal order, and an off-tree pick reveals nothing
instead of being hidden from the menu. What a policy can still infer is indirect, from selections that come
back empty.

Monotonicity holds only against the trees you have at the time you choose. A newly added tree can make the
incumbent crash retroactively, which happened in seed 1.

The policy writer is still the weak link, crashing about 15% of the time after repair. Pointing
`--policy-model` at a bigger model is the cheapest remaining improvement: three policy calls per round
against dozens of agent calls. [NEXT_STEPS.md](NEXT_STEPS.md) ranks the rest.

## Sources

- Paper: [arXiv 2609.14858](https://arxiv.org/abs/2609.14858), <https://www.dream-rsi.com/>, unreleased code
  at [zhengkid/Dream-RSI](https://github.com/zhengkid/Dream-RSI)
- [FunSearch](https://github.com/google-deepmind/funsearch) for the bin-packing protocol, and
  [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/binpackinfo.html) for its instances
- [Ollama concurrency settings](https://docs.ollama.com/faq)
