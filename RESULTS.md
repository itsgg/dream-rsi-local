# Results

A three-seed matrix on bin packing, which came out null, and the measurements behind the two fixes made in
response. The rerun on the cache task has not happened.

## The bin-packing matrix

`./run_matrix.sh qwen2.5-coder:7b 4 1 2 3`, run 2026-09-19/20, about 3.3 h across the six cells.
`qwen2.5-coder:7b` on an M1 Pro (16 GB) as both the discovery agent and the policy writer. Seed program is
FunSearch's OR-Library heuristic at -2.775; best-fit is -2.175 on the search set and -4.20 held out. W=4, 8
branches, depth 6, M=4, beta1=0.005, beta2=0.01, four rounds. Round 1 is the same procedure in both arms.

| arm | attempts (s1/s2/s3) | mean | LLM calls | final best | held-out |
|---|---|---|---|---|---|
| Dream-RSI | 92 / 60 / 96 | 82.7 | 165 | -2.150 / -2.175 / -2.150 | -4.10 / -4.20 / -4.20 |
| Fixed | 96 / 96 / 96 | 96.0 | 188 | -2.150 / -2.150 / -2.175 | -4.20 / -4.15 / -4.20 |

Mean final best is -2.158 in both arms. The earlier unseeded pair (`runs/dream_r4` vs `runs/fixed_r4`, 58 vs
96 calls at the same score) looked like a 1.66x saving. With three seeds it is 1.16x, it comes almost
entirely from one seed, and that seed ended at the worse score.

Per seed:

- s1 spent 92 attempts and matched fixed at -2.150, with the best held-out score in the matrix. Not a smooth
  run: the round-2 policy crashed mid-rollout in round 3 and the baseline took over, so round 3 cost 29
  attempts against fixed's 24.
- s2 stopped early and often (24/13/12/11) for 60 attempts total, but finished at -2.175 where fixed reached
  -2.150. That is the entire saving in the table, and it was paid for in score.
- s3 had all nine policy revisions crash in replay, so the incumbent was never replaced and the dream arm ran
  the initial policy for four rounds. The identical 96 attempts confirm it.

Where the learned policy saved calls it gave up score; where it matched the score it saved nothing. The
stopping behaviour of Figure 6 shows up in s2. The efficiency claim of Figure 3(b) does not hold here.

### Why

The policy writer crashed on 11 of 27 versions, 3/9, 1/9 and 7/9 by seed, and a crashed version is discarded,
so a high rate collapses the dreaming phase into a no-op. Writing a correct module against the `View`/`Obs`
API, with no state and no exceptions, is harder for a 7B than writing a numpy one-liner.

The discovery agent hit the task ceiling in round 1 of every run: best-fit at -2.175, sometimes plus a small
threshold term. Held out, the discovered programs tie plain best-fit in four of the six runs, so the 0.025
gain on the search set does not generalise. With nothing left to find, both arms necessarily end level.

### What did work

Replay scoring is well defined and pinned by `tests/test_replay.py`. Selection is monotone, since the
incumbent is always a candidate. The policy code genuinely changes: s1 and s2 both deployed policies
structurally different from the baseline. And the cost term behaves as designed, which is why s2's policy
stops once nothing beats the best: at beta1=0.005 the extra attempts cost more than the 0.025 they might buy.

One caveat. Monotonicity holds only against the trees available at the time you choose. In s1 the policy
deployed in round 2 scored -inf in round 3, because the tree added in round 2 made it crash.

### Earlier runs, kept for reference

`runs/dream_r4` and `runs/fixed_r4` are the first unseeded pair (K2=12, no `--seed`), 58 vs 96 calls at
-2.150 both, superseded by the matrix. `runs/_aborted_*` are two configurations on OR-Library data, where the
FunSearch seed heuristic is near-optimal: the agent never beat the seed in 24 attempts, every tree had a flat
ceiling, and the dreaming phase correctly learned to stop after round 0. That is why the uniform distribution
is used instead.

All of these predate two rounds of fixes: a review pass (evaluator score forgery, the policy subprocess'
path to the tree, the replay/online `max_rounds` mismatch, thread safety) and then a fidelity pass against
Section 3 that added `plan_grid()`, per-episode policy state, the beta sweep, `Child(v) = ∅` in replay, and
view ids renumbered to reveal order. The prompt logs were checked at the time and none of the holes were
exploited, so the numbers stand, but they were not produced by the current code. Each run's `config.json`
records its parameters and commit.

## Policy repair

A crashed version now gets up to two traceback-only repair calls, the same treatment failed candidate
programs already had. Replaying the 11 recorded crashes through it, same 7B model:

| | |
|---|---|
| crashed versions | 11 of 27 (41%) |
| repaired within two calls | 7 |
| of those, good enough to deploy | 3 |
| still broken | 4, so 15% effective |
| cost | 17 LLM calls |

The four that stayed broken are all from s3. Three of its seven crashes recover, one of them beating the
incumbent, so that run would no longer have been a no-op. Reproduce with `uv run python -m tools.repair_check`,
which is seeded and gives the same numbers every time.

## The cache task

I swept the workload until three things held, all of them now asserted by `tests/test_tasks.py`: a graded
ladder above the seed rather than a single step (five hand-written policies beat LRU, +0.005 to +0.021 hit
rate), gains that transfer to held-out, and room left over (Belady's MIN at 0.6662 against 0.5628 for the
best rule I wrote).

The property bin packing lacked is that the five regimes are scored together and their individual winners
disagree. `last_used + 1e6*(freq >= 3)` beats LRU comfortably on one regime and scores 0.5383 over the
mixture, below the LRU seed at 0.5422. A random search over a six-parameter family put the cost of using one
rule everywhere at 0.013 hit rate, which is ladder only an adaptive rule can claim. On bin packing the total
headroom was 0.6 of a bin, taken inside the first 24 attempts, with no held-out gain at all.

A 19-attempt probe (2 rounds, W=4, seed 1, before the regimes were mixed in) moved from LRU's 0.5225 to
0.5583, roughly 28% of the way to the offline optimum, and held up on held-out at 0.5312 against LRU's
0.5117. The repair step fired and worked in its dreaming phase. One run and no control arm, so this is a
smoke test rather than a result.

## What would make this a real test

```bash
OLLAMA_NUM_PARALLEL=4 ollama serve
./run_matrix.sh qwen3-coder:30b 4 1 2 3
```

That is the open question. After it: point `--policy-model` at something larger still, since repair only
takes the crash rate to 15% and policy calls are a small share of the total. Three seeds is the floor, not
the target, given that the per-seed spread on bin packing (60 to 96 attempts) was wider than the effect being
measured. And sweep beta1, the one thing s2 did demonstrate, remembering that the cache task defaults to
0.0002 rather than bin packing's 0.005 because its scores live on a 0..1 scale.

## Files

`runs/report_qwen2_5-coder_7b.md` and the matching `.png` hold the matrix table and the best-so-far plot.
Each run has `round_XX/tree.json` (the replay worlds) and `round_XX/dream/` (every policy version with its
replay trajectories). `llm_calls.jsonl` holds every prompt and reply; it is written on every run but not
committed, since the six recorded runs produced 9 MB of it.
