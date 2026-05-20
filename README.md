# HyperZero

AlphaZero-style self-play for N-dimensional Connect-K.

HyperZero is a reinforcement learning research project focused on a configurable family of abstract strategy games: Connect-K played on N-dimensional boards with gravity. The project starts with a rigorous game engine and baseline agents, then builds toward neural-guided Monte Carlo Tree Search, self-play training, and experiments on how dimensionality affects learning and search.

## Core Goal

Build a general N-dimensional Connect-K platform and use it to study whether AlphaZero-style agents can learn strong play as the game scales from 2D to 3D and 4D.

The first serious target is:

> Train an AlphaZero-style agent that beats pure MCTS and heuristic baselines on 3D 4x4x4 Connect-4.

The stretch target is:

> Extend the same system to 4D 4x4x4x4 Connect-4 and analyze what breaks.

## Repository Structure

```text
hyperzero/
  game/                Core N-dimensional Connect-K engine
  agents/              Random, tactical, heuristic, and pure-MCTS baselines
  search/              Reusable Monte Carlo Tree Search
  eval/                Head-to-head matchup evaluation
benchmarks/
  benchmark_engine.py  Engine throughput smoke benchmark
scripts/
  evaluate_baselines.py  Baseline matchup runner
docs/
  status.md            Current project status and active run notes
  proposal.md          Research proposal and project framing
  roadmap.md           Milestones and implementation phases
  architecture.md      Planned system architecture
  experiments.md       Experimental design and evaluation metrics
  game-spec.md         N-dimensional Connect-K rules and definitions
  engine.md            Core engine API notes
tests/
  test_*.py            Engine unit tests
```

## Project Status

See [docs/status.md](docs/status.md) for the full current snapshot.

Core game engine, baseline agents, pure MCTS search, head-to-head evaluation,
PUCT search, neural agents, self-play training, checkpoint evaluation, and
terminal play are implemented with tests. The training stack now supports
batched self-play leaf inference, line-aware models, GPU experiment orchestration,
loss traces, GPU telemetry, and custom eval score weights for best-checkpoint
selection.

Current validation:

- Local lint and unit tests pass: `71 passed`.
- Remote validation on `anthonypc` passes in the `torch` conda environment.
- CUDA is available on `anthonypc` with an NVIDIA GeForce RTX 3060 Ti.
- 3D 4x4x4 Connect-4 is promoted as stable. The guarded line-ResNet run reached
  final evals of `100.0%` vs random, `97.5%` vs tactical, `94.4%` vs heuristic,
  and `99.4%` vs MCTS-32 over 160 games per opponent.
- 4D 4x4x4x4 Connect-4 is feasible but not solved. Smoke, initial training, and
  a heuristic continuation completed with stable loss/resource behavior. The
  current best readout is strong against random/MCTS but still weak against
  tactical and heuristic opponents.
- The active 4D tactical-weighted follow-up was paused on `anthonypc` so the
  machine can be rebooted into Windows. It can be resumed from the stopped
  process set or restarted from its latest checkpoint artifacts.

Next experiments:

- Resume or restart the 4D tactical-weighted follow-up and evaluate whether
  deeper search plus heuristic/tactical-weighted checkpoint selection improves
  4D threat defense.
- Start the new universal-agent phase: train a single dimension-conditioned
  agent that can play 2D, 3D, and 4D Connect-K variants with one shared policy
  and value model.

## Working Name

HyperZero: AlphaZero-Style Self-Play for N-Dimensional Connect-K
