| run | round | policy | attempts | cumulative | LLM calls | failed | repaired | round best | best so far | held-out (500 items) | best mechanism |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dream_r4 | 1 | policy_round_01.py | 24 | 24 | - | 20 | - | -2.150 | -2.150 | -4.050 | Introduce a tie-break term based on best-fit to encourage spread acros |
| dream_r4 | 2 | policy_round_02.py | 14 | 38 | - | 1 | - | -2.175 | -2.150 | -4.050 | Introduce a tie-break term based on best-fit to encourage spread acros |
| dream_r4 | 3 | policy_round_03.py | 16 | 54 | - | 1 | - | -2.175 | -2.150 | -4.050 | Introduce a tie-break term based on best-fit to encourage spread acros |
| dream_r4 | 4 | policy_round_04.py | 4 | 58 | - | 1 | - | -2.150 | -2.150 | -4.050 | Introduce a tie-break term based on best-fit to encourage spread acros |
| fixed_r4 | 1 | policy_round_01.py | 24 | 24 | - | 5 | - | -2.150 | -2.150 | -4.250 | Add a preference for already-used bins over fresh ones to reduce fragm |
| fixed_r4 | 2 | policy_round_02.py | 24 | 48 | - | 0 | - | -2.150 | -2.150 | -4.250 | Add a preference for already-used bins over fresh ones to reduce fragm |
| fixed_r4 | 3 | policy_round_03.py | 24 | 72 | - | 0 | - | -2.150 | -2.150 | -4.250 | Add a preference for already-used bins over fresh ones to reduce fragm |
| fixed_r4 | 4 | policy_round_04.py | 24 | 96 | - | 0 | - | -2.150 | -2.150 | -4.250 | Add a preference for already-used bins over fresh ones to reduce fragm |

### dream_r4: policy evolution
- round 1 deployed `policy_round_01.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
    dreaming: v00 (incumbent) V=-2.2300, v01 V=-2.1750, v02 V=-2.1750, v03 V=-2.1750
- round 2 deployed `policy_round_02.py` -> parallel_refine_selective: Parallel refinement with selective branching: open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap. Open new branches when the open ones stagnate or when few bra
    dreaming: v00 (incumbent) V=-2.1983, v01 V=-2.1983, v02 V=-2.1983, v03 V=-2.2258
- round 3 deployed `policy_round_03.py` -> parallel_refine_selective: Parallel refinement with selective branching: open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap. Open new branches when the open ones stagnate or when few bra
    dreaming: v00 (incumbent) V=-2.2083, v01 V=-2.2083, v02 V=-inf (failed), v03 V=-inf (failed)
- round 4 deployed `policy_round_04.py` -> parallel_refine_selective: Parallel refinement with selective branching: open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap. Open new branches when the open ones stagnate or when few bra

### fixed_r4: policy evolution
- round 1 deployed `policy_round_01.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
- round 2 deployed `policy_round_02.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
- round 3 deployed `policy_round_03.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
- round 4 deployed `policy_round_04.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.

plot: runs/report_r4.png
