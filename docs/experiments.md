# Experiment Plan

## Experimental Goal

Measure how AlphaZero-style self-play performs as Connect-K scales across dimensions, rules, and model architectures.

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

## Suggested First Results Table

```text
Variant | Agent | Eval Budget | Win % vs Random | Win % vs Heuristic | Win % vs MCTS | Avg Game Length
```

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
