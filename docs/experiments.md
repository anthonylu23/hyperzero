# Experiment Plan

## Experimental Goal

Measure how AlphaZero-style self-play performs as Connect-K scales across dimensions, rules, and model architectures.

Current status:

- 3D 4x4x4 Connect-4 is stable enough to use as the promoted higher-dimensional
  baseline. The guarded line-ResNet checkpoint reached high final win rates
  against random, tactical, heuristic, and MCTS-32 baselines.
- 4D 4x4x4x4 Connect-4 is feasible but tactically immature. Completed 4D runs
  show stable loss/resource behavior and strong random/MCTS performance, but
  weak tactical/heuristic performance.
- The 4D tactical-weighted multi-seed follow-up completed. Seed 2 is the current
  specialist 4D baseline, but results remain seed/checkpoint sensitive.
- The active research phase is a universal agent: one shared checkpoint that can
  play selected 2D, 3D, and 4D Connect-K variants.

The experiments should distinguish between:

- game complexity
- search budget
- neural architecture
- training curriculum
- symmetry augmentation

## Core Metrics

Game performance:

- win rate against random agent
- win rate against heuristic agent
- win rate against pure MCTS
- win rate against previous checkpoints
- draw rate
- average game length

Learning behavior:

- policy loss
- value loss
- policy entropy
- value calibration error
- games required to reach target win rate
- wall-clock time required to reach target win rate

Search behavior:

- MCTS simulations per move
- average search depth
- visit distribution entropy
- move time
- node expansion count

Scaling behavior:

- number of legal actions
- number of winning lines
- self-play games per hour
- memory usage
- training throughput

## Baseline Matchups

Every major trained checkpoint should be evaluated against:

```text
random
one-ply tactical
heuristic line scorer
pure MCTS with small budget
pure MCTS with matched compute budget
previous best neural checkpoint
```

## Experiment E1: 2D Validation

Variant:

```text
shape=(6, 7)
K=4
gravity_axis=0
```

Purpose:

Validate the full system on a familiar game.

Expected result:

The agent should learn to beat random and simple heuristics. This experiment is primarily a correctness and infrastructure check.

## Experiment E2: 3D Main Target

Variant:

```text
shape=(4, 4, 4)
K=4
gravity_axis=0
```

Purpose:

Demonstrate learning in the first serious higher-dimensional game.

Expected result:

The AlphaZero-style agent should beat random, heuristic, and pure-MCTS baselines under fixed evaluation settings.

## Experiment E3: Search Budget Sweep

Variant:

```text
shape=(4, 4, 4)
K=4
gravity_axis=0
```

Compare MCTS budgets:

```text
25, 50, 100, 200, 400 simulations per move
```

Purpose:

Determine how much search is needed for useful play and whether the learned policy reduces search requirements.

## Experiment E4: Architecture Comparison

Variant:

```text
shape=(4, 4, 4)
K=4
gravity_axis=0
```

Compare:

```text
residual MLP
cell transformer
optional graph model
```

Purpose:

Measure whether spatial or relational inductive bias improves sample efficiency and final strength.

## Experiment E5: Curriculum Learning

Compare direct training versus curriculum:

```text
direct:      train on 3D 4x4x4 K=4 from scratch
curriculum: 2D or smaller 3D variants -> 3D 4x4x4 K=4
```

Purpose:

Test whether simpler variants produce reusable representations or policies.

## Experiment E6: Symmetry Augmentation

Compare training with and without board symmetries.

Purpose:

Measure whether exploiting geometric symmetries improves sample efficiency.

Important note:

Symmetry handling must preserve action labels correctly. Bad augmentation can silently corrupt policy targets.

## Experiment E7: 4D Feasibility

Variant:

```text
shape=(4, 4, 4, 4)
K=4
gravity_axis=0
```

Purpose:

Stress test the system and identify bottlenecks.

Report:

- number of actions
- number of winning lines
- MCTS time per move
- self-play games per hour
- memory usage
- whether training shows improvement

## Experiment E8: Universal 2D/3D/4D Agent

Variants:

```text
2D validation: shape=(6, 7), K=4, gravity_axis=0
3D target:     shape=(4, 4, 4), K=4, gravity_axis=0
4D stretch:    shape=(4, 4, 4, 4), K=4, gravity_axis=0
```

Purpose:

Train one shared policy-value agent that can play across dimensionalities. This
tests whether the system can learn reusable Connect-K concepts such as line
completion, blocking, forks, and gravity-aware move timing instead of only
learning specialist policies for one board shape.

Required measurements:

- per-variant win rate against random, tactical, heuristic, and MCTS baselines
- per-variant loss curves or held-out value/policy calibration
- per-variant self-play throughput
- transfer comparison against specialist checkpoints
- aggregate score with per-dimension floors

Minimum success criteria:

- A single checkpoint can load once and legally play all selected 2D, 3D, and
  4D variants.
- It beats random and tactical baselines in every dimension.
- It does not regress below the current specialist 3D baseline by more than a
  documented tolerance.
- 4D tactical/heuristic performance is at least competitive with the current
  specialist 4D continuation before scaling budget.

## Suggested First Results Table

```text
Variant | Agent | Eval Budget | Win % vs Random | Win % vs Heuristic | Win % vs MCTS | Avg Game Length
```

## Phase 3 3D Target Gate - 2026-05-19

Artifacts:

```text
runs/phase3_3d_target_20260519-121124/
```

Configs:

```text
5x5x5 K=4 line_resnet + random symmetry, 120 iterations, 48 games/iter, 48 sims
5x5x5 K=4 resnet control + random symmetry, 140 iterations, 48 games/iter, 48 sims
4x4x4 K=4 line_resnet filler + random symmetry, 80 iterations, 48 games/iter, 48 sims
```

Final checkpoint evals used 200 games per opponent for the 5x5x5 target/control and 160 games per opponent for the 4x4x4 filler. Trace evals reran 100 games versus the heuristic opponent and exported eight loss traces per config.

```text
Config                         Random  Tactical  Heuristic  MCTS-32  Trace heuristic
5x5x5 line_resnet target       100.0%    95.5%      76.5%   100.0%            74.0%
5x5x5 resnet control           100.0%    93.5%      61.0%   100.0%            61.0%
4x4x4 line_resnet filler       100.0%    81.9%      21.3%   100.0%            22.0%
```

Key readout:

- The 5x5x5 line-aware target cleared the initial 3D gate and materially outperformed the plain ResNet control against the heuristic baseline.
- Both 5x5x5 models beat random, tactical, and MCTS-32 strongly under the current eval protocol.
- The 4x4x4 filler exposed a sharp heuristic weakness despite high random/MCTS scores, so heuristic-specific traces remain required before promoting 3D agents.
- In all trace exports, sampled losses included moves that allowed opponent immediate wins; tactical threat defense is still the main failure mode to improve before larger 3D/4D scaling.

## 3D Stability and 4D Initial Runs - 2026-05-19

Code changes:

- PUCT root action selection now applies a tactical guard: immediate wins are
  forced, and one-ply losing moves are masked when at least one safe legal move
  exists. This affects self-play targets and checkpoint evaluation.
- GPU experiment runs now write `gpu-monitor.csv` with 15-second GPU telemetry.
- The training CLI and GPU runner can pass custom eval score weights, allowing
  4D follow-ups to select best checkpoints mostly on tactical/heuristic
  performance instead of random-opponent wins.

3D stability run:

```text
runs/phase4_3d_stability_guard_20260519-144210/
4x4x4 K=4 line_resnet + guard, 120 iterations, 48 games/iteration,
64 PUCT simulations
```

Train-time readout:

```text
Config                              Iteration  Random  Tactical  Heuristic
4x4x4 K=4 line_resnet + guard              40  100.0%     95.8%      87.5%
4x4x4 K=4 line_resnet + guard              60  100.0%     91.7%     100.0%
4x4x4 K=4 line_resnet + guard              80  100.0%    100.0%     100.0%
4x4x4 K=4 line_resnet + guard             115  100.0%    100.0%     100.0%
4x4x4 K=4 line_resnet + guard             120  100.0%    100.0%      95.8%
```

Final checkpoint evals used 160 games per opponent:

```text
Random  Tactical  Heuristic  MCTS-32
100.0%     97.5%      94.4%    99.4%
```

The 3D stability gate is cleared. Loss stayed finite and improved from `3.44`
to `2.43`, while train-time eval recovered from early heuristic dips to two
perfect aggregate evals. Resource use was stable on the RTX 3060 Ti: max sampled
GPU memory `3626 MiB`, max sampled utilization `63%`, average utilization
`19.8%`, max sampled temperature `62 C`.

4D smoke:

```text
runs/phase6_4d_smoke_20260519-150123/
4x4x4x4 K=4 line_mlp, 4 iterations, 8 games/iteration, 8 PUCT simulations
```

Final checkpoint evals used 8 games per opponent:

```text
Random  Tactical  Heuristic  MCTS-32
100.0%     25.0%      25.0%   100.0%
```

4D scaled initial training:

```text
runs/phase6_4d_training_20260519-151442/
4x4x4x4 K=4 line_mlp, 12 iterations, 12 games/iteration, 16 PUCT simulations
```

Final checkpoint evals used 24 games per opponent:

```text
Random  Tactical  Heuristic  MCTS-32
100.0%     50.0%       0.0%   100.0%
```

Best checkpoint evals also used 24 games per opponent:

```text
Random  Tactical  Heuristic  MCTS-32
100.0%     29.2%      12.5%   100.0%
```

4D heuristic continuation:

```text
runs/phase6_4d_heuristic_followup_20260519-155628/
4x4x4x4 K=4 line_mlp, resumed from the prior 4D best checkpoint,
24 iterations, 16 games/iteration, 24 PUCT simulations, learning rate 5e-4
```

Train-time best checkpoint was iteration 21:

```text
Iteration  Random  Tactical  Heuristic  Eval score
       21  100.0%     41.7%      33.3%       49.0%
       24  100.0%     41.7%      16.7%       41.7%
```

Final checkpoint evals used 32 games per opponent:

```text
Random  Tactical  Heuristic  MCTS-32
100.0%     43.8%      25.0%    96.9%
```

Trace readout:

- Heuristic loss traces: 8 losses, 95 neural moves, 8 selected moves gave the
  opponent immediate wins, and no immediate wins were missed.
- Tactical loss traces: 8 losses, 226 neural moves, 8 selected moves gave the
  opponent immediate wins, and no immediate wins were missed.
- Many selected actions in the sampled losses had low raw policy probability
  before search (`56` heuristic-loss moves and `61` tactical-loss moves below
  `5%`), so the next run should test whether stronger search plus
  tactical/heuristic-weighted checkpoint selection improves promotion.

Key readout:

- 4D training is technically stable: losses stayed finite and total loss
  improved from `4.76` at the initial 4D run to `3.57` at the end of the
  continuation.
- 4D resource use was stable while co-scheduled with the 3D run: the
  continuation peaked at `3619 MiB` GPU memory, `65%` sampled utilization, and
  `54 C`.
- 4D throughput is the current bottleneck. The scaled run averaged roughly
  `8-10` seconds per self-play game at `16` PUCT simulations.
- 4D policy/value learning did not yet translate to heuristic strength. The
  next 4D run should target tactical/heuristic failures directly instead of only
  increasing the same self-play budget.
- Heuristic loss traces from the best 4D checkpoint show all sampled losses
  reached positions where no safe one-ply reply remained. The current root
  tactical guard prevents immediate blunders but does not prevent earlier fork
  creation, so follow-up 4D experiments should emphasize earlier tactical
  threat/fork avoidance.
- Next run prepared: `configs/phase6_4d_tactical_weighted_20260519.json`
  resumes from the 4D heuristic continuation best checkpoint, raises self-play
  and eval search to `32` simulations, lowers learning rate to `2.5e-4`, and
  weights best-checkpoint selection `50%` heuristic / `35%` tactical / `15%`
  random. Because training resume uses absolute iteration numbers, the config
  ends at iteration `39`, giving 18 additional iterations after the resumed
  iteration-21 checkpoint.

## 4D Tactical-Weighted Fresh Seeds and Universal Early Run - 2026-05-20

4D tactical-weighted fresh seeds:

```text
runs/phase6_4d_tactical_fresh_seed0_20260520-0938/
runs/phase6_4d_tactical_fresh_seed1_20260520-0941/
runs/phase6_4d_tactical_fresh_seed2_20260520-1015/
runs/phase6_4d_tactical_fresh_seed3_20260520-1015/
4x4x4x4 K=4 line_mlp, 39 iterations, 12 games/iteration,
32 PUCT simulations, learning rate 2.5e-4, best checkpoint selected
with 50% heuristic / 35% tactical / 15% random train-time eval score
```

Final evals used 40 games per opponent:

```text
Seed  Checkpoint  Random  Tactical  Heuristic  MCTS-32
   0  best         100.0%     30.0%      20.0%    97.5%
   0  latest       100.0%     40.0%      22.5%   100.0%
   1  best         100.0%     47.5%      52.5%   100.0%
   1  latest       100.0%     52.5%      12.5%   100.0%
   2  best         100.0%     55.0%      55.0%   100.0%
   2  latest       100.0%     57.5%      40.0%   100.0%
   3  best         100.0%     17.5%      27.5%   100.0%
   3  latest        97.5%     32.5%      25.0%    95.0%
```

Resource readout on the remote GPU workstation:

- The first serial run showed low GPU utilization because 4D PUCT/search is
  CPU-bound and uses CUDA in short batched-inference bursts.
- Running four independent seeds in parallel raised the machine to roughly four
  active CPU workers while keeping VRAM under about `1 GiB` and temperature near
  the low `40s C`.
- The seed-2 best checkpoint is the current specialist 4D baseline.

Interpretation:

- The tactical-weighted 32-simulation recipe improved 4D specialist strength
  over the prior continuation (`43.8%` tactical / `25.0%` heuristic).
- The result is still seed/checkpoint sensitive. Best-checkpoint selection is
  important because latest checkpoints can preserve tactical strength while
  regressing heuristic performance.
- Additional specialist 4D runs should be targeted, not open-ended. The better
  next step is to continue the universal-agent phase or add curriculum/eval
  pressure for heuristic consistency.

Universal early run:

```text
runs/universal_early_20260520-1150/
2D 6x7 K=4, 2D 4x4 K=3, 3D 4x4x4 K=4, 4D 4x4x4x4 K=4
UniversalPolicyValueTransformer, 36 iterations, 16 PUCT simulations,
58 mixed self-play games/iteration, eval every 3 iterations
```

Best train-time eval was iteration 33:

```text
Variant         Random  Tactical  Heuristic
2d_4x4_k3       100.0%     75.0%      50.0%
2d_6x7_k4        87.5%     50.0%      50.0%
3d_4x4x4_k4     100.0%     75.0%      25.0%
4d_4x4x4x4_k4   100.0%    100.0%      37.5%
```

Key readout:

- The universal checkpoint legally plays all configured 2D, 3D, and 4D
  variants, and the shared model can learn nontrivial tactical behavior across
  variants.
- Aggregate score improved from `0.4512` at iteration 3 to `0.6055` at
  iteration 33.
- Heuristic consistency remains weak and noisy, especially on 2D 6x7 and 4D.
  The next universal run should either extend the same checkpoint with stronger
  heuristic/tactical eval weighting or adjust the curriculum to prevent easy
  random/tactical gains from hiding heuristic failures.

Universal curriculum v2 continuation:

```text
configs/universal_curriculum_v2_20260520.json
runs/universal_curriculum_v2_20260520-1356/
2D 6x7 K=4:        32 games/iteration
2D 4x4 K=3:         8 games/iteration
3D 4x4x4 K=4:      16 games/iteration
4D 4x4x4x4 K=4:    12 games/iteration
resumed from the universal early best checkpoint at iteration 33
continued through iteration 40 with 16 PUCT simulations
weighted eval: 45% heuristic / 35% tactical / 20% random
```

Resume validation now permits curriculum-count changes while keeping variant ids,
order, shape, connect length, gravity axis, and model shape strict.

Best v2 eval was the final iteration 40:

```text
Variant         Random  Tactical  Heuristic
2d_4x4_k3        87.5%     87.5%      50.0%
2d_6x7_k4       100.0%     62.5%       0.0%
3d_4x4x4_k4     100.0%     50.0%      50.0%
4d_4x4x4x4_k4   100.0%     50.0%      12.5%
```

Key readout:

- V2 did not improve over the universal early best. Its best score was `0.4789`
  versus the prior `0.6055`, and no checkpoint passed the configured floors.
- The run lowered training loss from `2.6457` at iteration 34 to `2.5812` at
  iteration 40, but lower self-play loss did not translate into stronger
  external baseline performance.
- The curriculum recovered some 3D heuristic behavior, but 2D 6x7 heuristic and
  4D heuristic remained the major blockers.
- Eval floors worked as intended: they prevented these weaker checkpoints from
  replacing the prior best checkpoint.

4D stronger-search probe:

```text
configs/phase6_4d_stronger_search_probe_20260520.json
runs/phase6_4d_stronger_search_probe_20260520-1416/
resumed seed-2 specialist best checkpoint at iteration 6
48 PUCT simulations, 8 games/iteration, learning rate 1.25e-4
```

Train-time evals:

```text
Iteration  Random  Tactical  Heuristic  Eval score
        7  100.0%     37.5%      75.0%       65.6%
        8  100.0%     37.5%      37.5%       46.9%
```

Final eval of iteration-7 best used 16 games per opponent:

```text
Random     100.0%
Tactical    43.8%
Heuristic   56.2%
```

Interpretation:

- Raising search to 48 simulations can improve 4D heuristic performance, but it
  did not preserve tactical strength in this short continuation.
- The original seed-2 specialist remains the current promoted 4D baseline because
  it balanced tactical and heuristic better over larger 40-game evals.
- The next 4D specialist experiment should target the tactical/heuristic tradeoff
  directly rather than simply increasing search again.

Universal curriculum v3 reduced 2D 6x7 exposure and shifted that budget toward
3D:

```text
configs/universal_curriculum_v3_20260520.json
runs/universal_curriculum_v3_20260520-1428/
2D 6x7 K=4:        16 games/iteration
2D 4x4 K=3:         8 games/iteration
3D 4x4x4 K=4:      24 games/iteration
4D 4x4x4x4 K=4:    12 games/iteration
```

Best v3 eval was iteration 34:

```text
Variant         Random  Tactical  Heuristic
2d_4x4_k3        87.5%     75.0%      50.0%
2d_6x7_k4        87.5%     50.0%      50.0%
3d_4x4x4_k4      75.0%     62.5%      37.5%
4d_4x4x4x4_k4   100.0%     87.5%      12.5%
```

Key readout:

- V3 improved the best continuation score to `0.5609`, better than v2's `0.4789`
  but still below the universal early best `0.6055`.
- No v3 checkpoint passed floors. The main failures were still 4D heuristic and
  random/heuristic instability on the smaller variants.
- Reweighting alone helped 2D 6x7 heuristic at the best checkpoint, but did not
  solve cross-variant heuristic consistency. The next universal experiment should
  change search budget or add explicit hard-position/anti-fork pressure instead
  of only changing game counts.

## Reproducibility Requirements

Each experiment should record:

- git commit
- config file
- random seed
- model architecture
- training duration
- number of self-play games
- MCTS simulations per move
- hardware notes
