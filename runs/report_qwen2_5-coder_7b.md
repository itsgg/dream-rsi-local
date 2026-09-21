| run | round | policy | attempts | cumulative | LLM calls | failed | repaired | round best | best so far | held-out (500 items) | best mechanism |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dream_qwen2_5-coder_7b_s1 | 1 | policy_round_01.py | 24 | 24 | 53 | 6 | 1 | -2.175 | -2.175 | -4.200 | Use a linear decay function for preference scores based on remaining c |
| dream_qwen2_5-coder_7b_s1 | 2 | policy_round_02.py | 24 | 48 | 43 | 0 | 0 | -2.150 | -2.150 | -4.100 | Use a modified piecewise linear function with a focus on encouraging t |
| dream_qwen2_5-coder_7b_s1 | 3 | policy_round_03.py | 29 | 77 | 65 | 5 | 0 | -2.150 | -2.150 | -4.100 | Use a modified piecewise linear function with a focus on encouraging t |
| dream_qwen2_5-coder_7b_s1 | 4 | policy_round_04.py | 15 | 92 | 29 | 0 | 1 | -2.175 | -2.150 | -4.100 | Use a modified piecewise linear function with a focus on encouraging t |
| fixed_qwen2_5-coder_7b_s1 | 1 | policy_round_01.py | 24 | 24 | 49 | 4 | 0 | -2.175 | -2.175 | -4.200 | Use a linear decay function for preference scores based on remaining c |
| fixed_qwen2_5-coder_7b_s1 | 2 | policy_round_02.py | 24 | 48 | 42 | 0 | 0 | -2.175 | -2.175 | -4.200 | Use a linear decay function for preference scores based on remaining c |
| fixed_qwen2_5-coder_7b_s1 | 3 | policy_round_03.py | 24 | 72 | 48 | 0 | 2 | -2.150 | -2.150 | -4.200 | Use a piecewise linear function with different slopes for different re |
| fixed_qwen2_5-coder_7b_s1 | 4 | policy_round_04.py | 24 | 96 | 47 | 0 | 0 | -2.150 | -2.150 | -4.200 | Use a piecewise linear function with different slopes for different re |
| dream_qwen2_5-coder_7b_s2 | 1 | policy_round_01.py | 24 | 24 | 37 | 0 | 0 | -2.175 | -2.175 | -4.200 | unspecified |
| dream_qwen2_5-coder_7b_s2 | 2 | policy_round_02.py | 13 | 37 | 33 | 4 | 0 | -2.175 | -2.175 | -4.200 | unspecified |
| dream_qwen2_5-coder_7b_s2 | 3 | policy_round_03.py | 12 | 49 | 24 | 0 | 0 | -2.175 | -2.175 | -4.200 | unspecified |
| dream_qwen2_5-coder_7b_s2 | 4 | policy_round_04.py | 11 | 60 | 22 | 0 | 0 | -2.175 | -2.175 | -4.200 | unspecified |
| fixed_qwen2_5-coder_7b_s2 | 1 | policy_round_01.py | 24 | 24 | 40 | 1 | 0 | -2.175 | -2.175 | -4.200 | unspecified |
| fixed_qwen2_5-coder_7b_s2 | 2 | policy_round_02.py | 24 | 48 | 49 | 1 | 0 | -2.175 | -2.175 | -4.200 | unspecified |
| fixed_qwen2_5-coder_7b_s2 | 3 | policy_round_03.py | 24 | 72 | 46 | 0 | 0 | -2.150 | -2.150 | -4.150 | Use a combination of logarithmic function for remaining capacity and e |
| fixed_qwen2_5-coder_7b_s2 | 4 | policy_round_04.py | 24 | 96 | 51 | 0 | 3 | -2.150 | -2.150 | -4.150 | Use a combination of logarithmic function for remaining capacity and e |
| dream_qwen2_5-coder_7b_s3 | 1 | policy_round_01.py | 24 | 24 | 43 | 0 | 1 | -2.175 | -2.175 | -4.200 | Use a logarithmic scale to prioritize bins with larger remaining capac |
| dream_qwen2_5-coder_7b_s3 | 2 | policy_round_02.py | 24 | 48 | 49 | 1 | 2 | -2.175 | -2.175 | -4.200 | Use a logarithmic scale to prioritize bins with larger remaining capac |
| dream_qwen2_5-coder_7b_s3 | 3 | policy_round_03.py | 24 | 72 | 48 | 2 | 1 | -2.150 | -2.150 | -4.200 | Use a piecewise linear function to prioritize bins based on remaining  |
| dream_qwen2_5-coder_7b_s3 | 4 | policy_round_04.py | 24 | 96 | 50 | 1 | 1 | -2.175 | -2.150 | -4.200 | Use a piecewise linear function to prioritize bins based on remaining  |
| fixed_qwen2_5-coder_7b_s3 | 1 | policy_round_01.py | 24 | 24 | 41 | 0 | 1 | -2.175 | -2.175 | -4.200 | Use a logarithmic scale to prioritize bins with larger remaining capac |
| fixed_qwen2_5-coder_7b_s3 | 2 | policy_round_02.py | 24 | 48 | 49 | 1 | 0 | -2.175 | -2.175 | -4.200 | Use a logarithmic scale to prioritize bins with larger remaining capac |
| fixed_qwen2_5-coder_7b_s3 | 3 | policy_round_03.py | 24 | 72 | 51 | 2 | 0 | -2.175 | -2.175 | -4.200 | Use a logarithmic scale to prioritize bins with larger remaining capac |
| fixed_qwen2_5-coder_7b_s3 | 4 | policy_round_04.py | 24 | 96 | 51 | 1 | 1 | -2.175 | -2.175 | -4.200 | Use a logarithmic scale to prioritize bins with larger remaining capac |

### Aggregate over runs (mean [min, max])
| mode | runs | total attempts | total LLM calls | final best | final held-out |
|---|---|---|---|---|---|
| dream | 3 | 82.667 [60.000, 96.000] | 165.333 [116.000, 190.000] | -2.158 [-2.175, -2.150] | -4.167 [-4.200, -4.100] |
| fixed | 3 | 96.000 [96.000, 96.000] | 188.000 [186.000, 192.000] | -2.158 [-2.175, -2.150] | -4.183 [-4.200, -4.150] |

### dream_qwen2_5-coder_7b_s1: policy evolution
- round 1 deployed `policy_round_01.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
    dreaming: v00 (incumbent) V=-2.2550, v01 V=-inf (failed), v02 V=-2.2550, v03 V=-2.2850
- round 2 deployed `policy_round_02.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
    dreaming: v00 (incumbent) V=-2.2425, v01 V=-2.1625, v02 V=-2.1625, v03 V=-2.1625
- round 3 deployed `policy_round_03.py` -> dynamic_batch: Dynamic batch selection policy that balances exploitation, exploration, and recovery. Prioritizes high-potential branches, handles recoverable failures, and expands under-explored areas.
    dreaming: v00 (incumbent) V=-inf (failed), v01 V=-2.1717, v02 V=-inf (failed), v03 V=-inf (failed)
- round 4 deployed `policy_round_04.py` -> enhanced_dynamic_batch: Enhanced dynamic batch selection policy that balances exploitation, exploration, and recovery. Prioritizes high-potential branches, handles recoverable failures, and expands under-explored areas more effectively.

### fixed_qwen2_5-coder_7b_s1: policy evolution
- round 1 deployed `policy_round_01.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
- round 2 deployed `policy_round_02.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
- round 3 deployed `policy_round_03.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
- round 4 deployed `policy_round_04.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.

### dream_qwen2_5-coder_7b_s2: policy evolution
- round 1 deployed `policy_round_01.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
    dreaming: v00 (incumbent) V=-2.2550, v01 V=-133.2250, v02 V=-2.1750, v03 V=-2.1750
- round 2 deployed `policy_round_02.py` -> improved_exploration: Improve exploration by prioritizing deepening under-explored branches and repairing recoverable failures before opening new branches.
    dreaming: v00 (incumbent) V=-2.1912, v01 V=-2.1900, v02 V=-2.1900, v03 V=-2.1900
- round 3 deployed `policy_round_03.py` -> improved_exploration_v2: Improve exploration by prioritizing deepening under-explored branches and repairing recoverable failures before opening new branches. Adjusted to balance exploitation, exploration, and recovery more effectively.
    dreaming: v00 (incumbent) V=-2.1970, v01 V=-2.1952, v02 V=-2.1952, v03 V=-inf (failed)
- round 4 deployed `policy_round_04.py` -> improved_exploration_v3: Improve exploration by prioritizing deepening under-explored branches and repairing recoverable failures before opening new branches. Adjusted to balance exploitation, exploration, and recovery more effectively. Reduced 

### fixed_qwen2_5-coder_7b_s2: policy evolution
- round 1 deployed `policy_round_01.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
- round 2 deployed `policy_round_02.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
- round 3 deployed `policy_round_03.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
- round 4 deployed `policy_round_04.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.

### dream_qwen2_5-coder_7b_s3: policy evolution
- round 1 deployed `policy_round_01.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
    dreaming: v00 (incumbent) V=-2.2550, v01 V=-inf (failed), v02 V=-inf (failed), v03 V=-2.2550
- round 2 deployed `policy_round_02.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
    dreaming: v00 (incumbent) V=-2.2550, v01 V=-inf (failed), v02 V=-inf (failed), v03 V=-inf (failed)
- round 3 deployed `policy_round_03.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
    dreaming: v00 (incumbent) V=-2.2467, v01 V=-inf (failed), v02 V=-inf (failed), v03 V=-133.2250
- round 4 deployed `policy_round_04.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.

### fixed_qwen2_5-coder_7b_s3: policy evolution
- round 1 deployed `policy_round_01.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
- round 2 deployed `policy_round_02.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
- round 3 deployed `policy_round_03.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.
- round 4 deployed `policy_round_04.py` -> parallel_refine: Parallel refining (the paper's initial policy): open W independent workspaces in the first round, then refine every open branch in parallel until the depth cap; never open more branches.

plot: runs/report_qwen2_5-coder_7b.png
