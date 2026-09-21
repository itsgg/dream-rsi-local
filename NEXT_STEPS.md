# Next steps

Ranked, with sources. Nothing here has been run. The two things that made the bin-packing matrix a null
result are already fixed and listed at the bottom; most of what is left needs more memory than the 16 GB
machine this was built on.

## What the paper and its readers say

The official repo still says code is being prepared for release (<https://github.com/zhengkid/Dream-RSI>,
checked 2026-09-20). The paper has no limitations section and never states its beta1/beta2
(<https://arxiv.org/html/2609.14858v1>).

Off-tree actions are not simulated: when no recorded child exists, replay just ends. The authors' only
mitigation is evaluating every policy on all accumulated trees so support widens over rounds. The project
page puts it as "a policy can only be dreamt where history actually went" (<https://dream-rsi.com/>).

Appendix B.2 hard-codes anti-degeneracy rules into the policy-writer prompt: never stop while a repairable or
under-explored node remains, at most one recovery per batch, classify failures as hard, repairable, or
weak-but-underexplored. Section 5.1 finds that semantic hints about where to search consistently underperform
unguided search.

A Hacker News thread asked how the policy avoids overfitting to already-discovered branches and got no answer
(<https://news.ycombinator.com/item?id=49726955>). The first author's AutoTTS uses the same offline-replay
idea and counters overfitting with many pre-collected trajectories, random-subset evaluation, one beta knob,
and held-out problems for selection (<https://arxiv.org/html/2605.08083v2>).

## Highest value

1. Run the matrix on `--task cache` with a 30B-class agent. Nothing about the new task has been tested beyond
   a 19-attempt probe and a smoke run.
2. Separate the policy model from the agent model, and make the policy one frontier. `--policy-model` and
   `--policy-base-url` now reach any OpenAI-compatible endpoint, and it is three calls per round against
   24-plus agent calls, so it is nearly free. Repair only takes the crash rate from 41% to 15%; a model that
   can write a correct module against a documented interface should take it to roughly zero. Watch for what that
   exposes: a capable policy writer scored on three or four recorded trees will start
   overfitting them, which is item 10.
3. Move the agent to `qwen3-coder:30b`: 30B total, 3.3B active, 19 GB at Q4_K_M, so it generates at roughly
   dense-3B speed (<https://ollama.com/library/qwen3-coder>). Needs 32 GB. `qwen2.5-coder:14b` was tried on
   the 16 GB M1 Pro and ran at 6 tok/s, 54 s per attempt, four times slower than the 7B and no better on a
   five-attempt probe. Also worth a look: `gemma4:12b` for output-format reliability, `gpt-oss:20b` if you
   have the room (<https://insiderllm.com/guides/best-local-coding-models-2026/>). Ollama's `think: false`
   works for Qwen3; gpt-oss only accepts low/medium/high (<https://docs.ollama.com/capabilities/thinking>).
4. Turn on real parallelism. `OLLAMA_NUM_PARALLEL` defaults to 1, so W=4 is currently nominal. Setting it to 4
   cuts wall clock roughly fourfold and makes the Eq. (1) parallelism bonus mean something. Memory scales as
   `OLLAMA_NUM_PARALLEL * context` (<https://docs.ollama.com/faq>).
5. A third task, once cache is exhausted too. The shape that works is now known: a short scoring function, a
   fast deterministic evaluator, several regimes scored together so tuned constants cannot win, and a
   computable upper bound.

## Cheap

6. Keep a running failure summary in the agent prompt, as SimpleTES does with "recurring failures such as
   compilation errors, verifier failures, timeouts" (<https://arxiv.org/html/2604.19341>). The prompt shows
   representative failures today but no counts per error type.
7. Sweep beta1, three seeds per cell, reporting calls-to-best and best score. s2 showed the stopping
   behaviour tracking the cost term; this would measure it. The cache default is 0.0002, not bin packing's
   0.005, because its scores live on a 0..1 scale.
8. Show the per-regime hit rates to the policy writer as well. The discovery agent already sees which regime
   it is weakest on; the exploration policy does not, and "this branch is only good at one regime" is exactly
   the signal that should shape where attempts go.

## Medium

9. Bootstrapped seeds and a small MAP-Elites archive: start branches from best-fit, first-fit, worst-fit and
   a couple of hand-written variants so the agent has structurally different parents. LEVI's ablation says
   removing bootstrapped seeds hurts more than anything else (<https://arxiv.org/html/2605.09764>).
10. Weighted parent sampling instead of hill-climbing: fitness-sigmoid times novelty (1/(1+visits)), two to
    four islands, migrate every ten generations (ShinkaEvolve, <https://arxiv.org/html/2509.19349>).
11. Fix the objective for narrow replay support: an explicit off-tree penalty rather than silent termination,
    evaluation on random subsets of trees with the newest held out for selection (AutoTTS), and sampling two
    or three children per frontier online so replay has siblings to choose among.
12. Alternate full-rewrite and diff-edit prompts (ShinkaEvolve used roughly 45/45/10 rewrite/diff/crossover).

## Large

13. Hybrid meta-evaluation: replay to prune bad policies, short online rollouts to rank the survivors, with
    bandit-style budget allocation as in AdaEvolve, EvoX or SkyDiscover (<https://arxiv.org/abs/2602.20133>,
    <https://arxiv.org/pdf/2602.23413>, <https://github.com/skydiscover-ai/skydiscover>). Buys protection
    from replay overfitting at the cost of some online calls.
14. Real statistics. Three seeds is the floor; the per-seed spread on bin packing was wider than the effect.

## Already done

Traceback repair for crashed policy versions (`--policy-repair-rounds`), which recovers 7 of the 11 recorded
crashes. Traceback self-repair for failed candidate programs (`--repair-rounds`): an 8B repairs NameErrors
about 77% of the time from the traceback alone but logic errors only 45%, so those are better routed to a new
branch (<https://arxiv.org/html/2604.10508>). Novelty rejection by embedding cosine similarity
(`--novelty-threshold`), which ShinkaEvolve's ablation ranks as its strongest component. Two rotating
in-prompt examples sampled by rank from the top five distinct programs, with the mechanism shopping-list
removed, following FunSearch (<https://www.nature.com/articles/s41586-023-06924-6>). The Appendix B.2 guards
in the policy-writer prompt. A cheap probe before full scoring, which is OpenEvolve's cascade evaluation at
stage 1 (<https://algorithmicsuperintelligence.ai/blog/openevolve-overview/index.html>). And `--seed` plus
`run_matrix.sh` for repeated seeded runs, with the seed derived from each attempt's round and batch slot so
it survives parallel execution.

Section 3 is now followed rather than approximated: the policy chooses its own grid through `plan_grid()`
before each episode, may carry state across rounds of one episode, and is ranked over a beta sweep rather
than at one fixed cost weighting. Replay also matches `Child(v) = ∅`, and view ids and rounds are renumbered
to reveal order, so the decision interface is identical online and in replay.

The cache task itself belongs on this list: five hand-written rungs above the LRU seed, gains that transfer
to held-out, five regimes scored together, and Belady's MIN as a ceiling. `tests/test_tasks.py` asserts all
of it without an LLM.
