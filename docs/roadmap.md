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

Remaining Phase 3 improvements before serious long 3D runs:

- Add root Dirichlet noise and temperature scheduling if exploration is too weak.
- Add best-checkpoint selection/promotion if repeated long runs need automatic
  checkpoint management.

Status for moving forward: complete enough for the first 3D smoke experiment.

## Phase 4: 3D Target Experiment

Goal: Demonstrate meaningful learning on 3D 4x4x4 Connect-4.

Deliverables:

- Configured 3D training run
- Baseline comparisons
- Elo or win-rate curves
- Analysis of MCTS simulation count versus playing strength
- Saved best checkpoint

Exit criteria:

- AlphaZero-style agent beats random, heuristic, and pure-MCTS baselines under a fixed evaluation budget.

## Phase 5: Architecture and Training Experiments

Goal: Compare design choices.

Candidate experiments:

- MLP versus transformer over board cells
- Direct 3D training versus curriculum learning
- Symmetry augmentation versus no augmentation
- Gravity-axis and board-shape variants
- Different MCTS simulation budgets

Exit criteria:

- At least three controlled experiments with plots and written analysis.

## Phase 6: 4D Stretch

Goal: Scale the system to 4D 4x4x4x4 Connect-4.

Deliverables:

- 4D environment validation
- Feasibility benchmark for legal moves, win checks, MCTS rollout time, and self-play throughput
- Initial training run
- Analysis of bottlenecks

Exit criteria:

- A clear result about whether the existing system scales directly or which components become limiting.

## Phase 7: Interface and Final Report

Goal: Make the project understandable and demonstrable.

Deliverables:

- Human-play interface against trained agents
- Game replay viewer
- Training plots
- Final research report
- Short demo video or notebook

Exit criteria:

- A reader can understand the game, reproduce core experiments, and inspect trained-agent behavior.
