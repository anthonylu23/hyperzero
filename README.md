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

Core game engine, baseline agents, pure MCTS search, and head-to-head
evaluation are implemented with tests. Models, self-play, and training are
still upcoming.

## Working Name

HyperZero: AlphaZero-Style Self-Play for N-Dimensional Connect-K
