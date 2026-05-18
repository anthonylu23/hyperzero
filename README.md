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
evaluation are implemented with tests. The first v1 neural stack is also
available: a small residual MLP, non-batched PUCT agent, self-play generation,
batched self-play leaf inference, replay buffer, and basic training loop. This
v1 loop is intended for correctness smoke tests and early debugging before
larger training and evaluation runs. See [docs/training-v1.md](docs/training-v1.md)
for train/eval commands.

Current validation:

- Local lint and unit tests pass: `60 passed`.
- Remote validation on `anthonypc` passes in the `torch` conda environment.
- CUDA is available on `anthonypc` with an NVIDIA GeForce RTX 3060 Ti.
- A remote CUDA smoke run completed with per-iteration losses, baseline evals,
  checkpoints, `metrics.jsonl`, and checkpoint-series eval output.
- A fuller 3x3 Connect-3 run completed for 50 iterations. It learned a stable
  draw/win policy, reached `84%` vs random, `80%` vs tactical, `50%` wins plus
  `50%` draws vs heuristic, and `10%` wins plus `90%` draws vs matched MCTS.
- A larger 4x4 Connect-3 validation run completed for 30 iterations. The best
  checkpoint reached `76%` vs heuristic and the final checkpoint reached `76%`
  vs matched 32-simulation MCTS in checkpoint-series eval.
- Batched self-play/inference is implemented as an optional training path.
  Remote GPU comparison on 4x4 Connect-3 improved a small two-iteration run
  from `6.13s` to `4.08s`; a small 3D smoke run completed successfully.

Next experiment:

Move to an initial 3D smoke experiment on 4x4x4 Connect-4 with conservative
budgets. The v1 loop is now verified on small 2D games, but serious 3D training
will likely need better exploration controls and higher self-play throughput.

## Working Name

HyperZero: AlphaZero-Style Self-Play for N-Dimensional Connect-K
