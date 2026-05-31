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
  universal/           Dimension-conditioned board/action encoding
  eval/                Head-to-head matchup evaluation
  server/              Demo-session and universal-agent service logic
services/api/          FastAPI demo API
apps/web/              Vite/React/Three.js playable demo
benchmarks/
  benchmark_engine.py  Engine throughput smoke benchmark
scripts/
  evaluate_baselines.py  Baseline matchup runner
  train_universal.py     Universal mixed-game training runner
  smoke_deployed_demo.py Deployed API/web smoke test
docs/
  status.md            Current project status and active run notes
  deployment.md        Render/Vercel deployment notes
  local-web-demo.md    Local playable demo setup
  proposal.md          Research proposal and project framing
  roadmap.md           Milestones and implementation phases
  architecture.md      Planned system architecture
  experiments.md       Experimental design and evaluation metrics
  universal-agent.md   Universal-agent scaffold and run notes
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

- Local lint, unit tests, and web build pass: `ruff`, `126 passed`, and
  `npm run build --prefix apps/web`.
- The public demo is deployed at `https://hyperzero-web-demo.vercel.app`, backed
  by `https://hyperzero-api.onrender.com`.
- Remote GPU validation has passed in the `torch` conda environment on an
  NVIDIA GeForce RTX 3060 Ti workstation.
- 3D 4x4x4 Connect-4 is promoted as stable. The guarded line-ResNet run reached
  final evals of `100.0%` vs random, `97.5%` vs tactical, `94.4%` vs heuristic,
  and `99.4%` vs MCTS-32 over 160 games per opponent.
- 4D 4x4x4x4 Connect-4 is feasible but not solved. The tactical-weighted
  multi-seed follow-up completed, with seed 2 now the current specialist 4D
  baseline at `100.0%` vs random, `55.0%` vs tactical, `55.0%` vs heuristic,
  and `100.0%` vs MCTS-32 over 40 games per opponent.
- The universal-agent path is implemented and deployed. The promoted
  residual-recovery checkpoint is iteration 36 with eval score `0.8328`, and it
  passed eval floors across selected 2D, 3D, and 4D variants.

Next experiments:

- Run larger robust evals for the promoted universal residual-recovery
  checkpoint and continue tuning with stronger search or hard-position
  curriculum pressure.
- Run targeted 4D specialist follow-ups only when they directly address the
  tactical/heuristic tradeoff seen in the stronger-search probe.

## Working Name

HyperZero: AlphaZero-Style Self-Play for N-Dimensional Connect-K

## License

MIT. See [LICENSE](LICENSE).
