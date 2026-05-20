# Roadmap

## Phase 0: Repository and Project Framing

Status: implemented

Deliverables:

- Project proposal
- Game specification
- Initial architecture plan
- Experiment plan
- Implementation roadmap

## Phase 1: Game Engine

Goal: Build a reliable N-dimensional Connect-K environment.

Status: implemented and covered by unit tests.

Deliverables:

- Board representation using NumPy arrays or compact tensors
- Configurable board shape, connect length, and gravity axis
- Legal move generation
- Move application and player switching
- Fast terminal-state detection
- Programmatic winning-line generation
- Canonical state representation from the current player's perspective
- Unit tests for lines, moves, wins, draws, and invalid actions

Exit criteria:

- 2D Connect Four behavior is correct.
- 3D 4x4x4 Connect-4 behavior is correct.
- Win detection works for axes, diagonals, and hyperdiagonals.

## Phase 2: Baseline Agents

Goal: Establish non-neural opponents and debugging baselines.

Status: implemented and covered by unit tests.

Deliverables:

- Random agent
- One-ply tactical agent
- Heuristic line-scoring agent
- Pure MCTS agent
- Evaluation script for head-to-head matches

Exit criteria:

- Heuristic agent beats random reliably.
- Pure MCTS beats random and is competitive with the heuristic agent.
- Evaluation outputs win rate, draw rate, average game length, and runtime.

## Phase 3: Minimal AlphaZero Loop

Goal: Train the first neural-guided self-play agent.

Status: v1 infrastructure implemented and smoke-tested locally and on the
remote GPU machine.

Current implementation note: the policy-value model and training loop are v1
infrastructure. They are intentionally small and non-batched at the search layer
so correctness is easy to inspect before moving to larger self-play throughput,
arena promotion, and 3D experiments.

Deliverables:

- Policy-value network: v1 residual MLP implemented
- PUCT-based MCTS: v1 non-batched neural-guided search implemented
- Self-play data generation: v1 implemented
- Replay buffer: v1 in-memory buffer implemented
- Training loop: v1 supervised policy/value updates and optional baseline evals implemented
- Checkpointing: v1 per-iteration checkpoints implemented
- Arena evaluation against previous checkpoints and baselines: v1 scripts implemented

Exit criteria:

- Training runs end to end on a small game: complete.
- Model improves against random and heuristic baselines: complete on 4x4
  Connect-3 validation.
- Training metrics are logged consistently: complete for v1 JSONL metrics.

Phase 3 validation runs:

- 3x3 Connect-3, 50 iterations, `32` games per iteration, `32` PUCT
  simulations: final checkpoint reached `84%` vs random, `80%` vs tactical,
  `50%` wins plus `50%` draws vs heuristic, and `10%` wins plus `90%` draws vs
  matched MCTS.
- 4x4 Connect-3, 30 iterations, `32` games per iteration, `32` PUCT
  simulations: best checkpoint reached `76%` vs heuristic, and final checkpoint
  reached `76%` vs matched 32-simulation MCTS.
- Batched self-play/inference: optional v1 path implemented. Remote GPU
  comparison on 4x4 Connect-3 improved a two-iteration smoke run from `6.13s`
  to `4.08s`; a small 3D batched smoke completed successfully.

Deferred Phase 3 improvements:

- Add root Dirichlet noise and temperature scheduling if exploration is too weak.
- Add best-checkpoint selection/promotion if repeated long runs need automatic
  checkpoint management.

Status for moving forward: complete. The v1 loop has now supported completed
2D validation, promoted 3D training, and 4D feasibility runs.

## Phase 4: 3D Target Experiment

Goal: Demonstrate meaningful learning on 3D 4x4x4 Connect-4.

Status: complete for the 4x4x4 stability gate.

Deliverables:

- Configured 3D training run
- Baseline comparisons
- Elo or win-rate curves
- Analysis of MCTS simulation count versus playing strength
- Saved best checkpoint

Exit criteria:

- AlphaZero-style agent beats random, heuristic, and pure-MCTS baselines under a fixed evaluation budget: complete.

Current result:

- Guarded 4x4x4 line-ResNet, 120 iterations, 48 games/iteration, 64 PUCT
  simulations.
- Final evals over 160 games per opponent: `100.0%` vs random, `97.5%` vs
  tactical, `94.4%` vs heuristic, and `99.4%` vs MCTS-32.
- Loss improved from `3.44` to `2.43`; GPU telemetry stayed stable.

## Phase 5: Architecture and Training Experiments

Goal: Compare design choices.

Status: partially complete, with the most important finding already folded into
the promoted 3D run.

Candidate experiments:

- MLP versus transformer over board cells
- Direct 3D training versus curriculum learning
- Symmetry augmentation versus no augmentation
- Gravity-axis and board-shape variants
- Different MCTS simulation budgets

Exit criteria:

- At least three controlled experiments with plots and written analysis.

Current result:

- Line-aware models materially improved tactical/heuristic performance versus
  plain controls in 3D experiments.
- Root tactical guarding is now enabled in PUCT by default and fixed the
  immediate one-ply blunder mode that appeared in earlier traces.
- GPU experiment tooling now records telemetry and supports custom eval score
  weights for best-checkpoint selection.

## Phase 6: 4D Stretch

Goal: Scale the system to 4D 4x4x4x4 Connect-4.

Status: complete as a feasibility and bottleneck-finding phase; active research
continues because 4D strength is still weak against tactical/heuristic baselines.

Deliverables:

- 4D environment validation
- Feasibility benchmark for legal moves, win checks, MCTS rollout time, and self-play throughput
- Initial training run
- Analysis of bottlenecks

Exit criteria:

- A clear result about whether the existing system scales directly or which components become limiting: complete.

Current result:

- 4D smoke, initial training, and heuristic continuation all completed with
  finite losses, checkpoint evals, loss traces, and GPU telemetry.
- The best completed continuation reached `100.0%` vs random, `43.8%` vs
  tactical, `25.0%` vs heuristic, and `96.9%` vs MCTS-32 in final evals.
- Interpretation: the system runs in 4D and learns general play, but current
  self-play/search still fails to consistently prevent forks and tactical
  threats.
- Active follow-up: `phase6_4d_tactical_weighted_20260519` raises search to
  32 PUCT simulations and weights best-checkpoint selection toward heuristic
  and tactical opponents. This run is currently paused on `anthonypc` for a
  Windows reboot.

## Phase 7: Universal Multi-Dimensional Agent

Goal: Train one agent that can play 2D, 3D, and 4D Connect-K variants with a
single shared model.

Status: planned.

Motivation:

- The current checkpoints are specialized by board shape and action space.
- A universal agent would test whether learned Connect-K concepts transfer
  across dimensionality instead of being memorized per geometry.
- A shared model may improve 4D sample efficiency by reusing tactical patterns
  learned from cheaper 2D and 3D games.

Required design changes:

- Shape-conditioned observations: encode board dimensionality, board extents,
  connect length, gravity axis, and legal-action mask.
- Variable action spaces: support policy heads that can score legal moves for
  different board/action shapes without retraining a separate output layer per
  variant.
- Dimension-aware architecture: prefer token/coordinate or line-incidence
  models over fixed flattened MLP heads. Candidate first model is a
  coordinate-conditioned transformer or line-token model.
- Mixed-game replay: store game config with every replay example and sample
  balanced batches across 2D, 3D, and 4D curricula.
- Mixed-game self-play scheduler: interleave cheap 2D/3D games with smaller
  4D batches, with explicit promotion gates per dimension.
- Evaluation harness: report per-variant win rates and an aggregate score, but
  prevent strong 2D performance from hiding weak 4D play.

Initial curriculum:

```text
2D: 6x7 K=4 and small 4x4 K=3 validation games
3D: 4x4x4 K=4 promoted stability target
4D: 4x4x4x4 K=4 tactical/heuristic stress target
```

Exit criteria:

- A single checkpoint loads once and can legally play all selected 2D, 3D, and
  4D variants.
- The shared agent beats random and tactical baselines in each dimension under
  fixed per-dimension search budgets.
- The shared agent is compared against specialist checkpoints to quantify
  transfer benefit or cost.
- Per-dimension evals, loss curves, and resource telemetry are documented.

## Phase 8: Interface and Final Report

Goal: Make the project understandable and demonstrable.

Deliverables:

- Human-play interface against trained agents
- Game replay viewer
- Training plots
- Final research report
- Short demo video or notebook

Exit criteria:

- A reader can understand the game, reproduce core experiments, and inspect trained-agent behavior.
