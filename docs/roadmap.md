# Roadmap

## Phase 0: Repository and Project Framing

Status: in progress

Deliverables:

- Project proposal
- Game specification
- Initial architecture plan
- Experiment plan
- Implementation roadmap

## Phase 1: Game Engine

Goal: Build a reliable N-dimensional Connect-K environment.

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

Deliverables:

- Policy-value network
- PUCT-based MCTS
- Self-play data generation
- Replay buffer
- Training loop
- Checkpointing
- Arena evaluation against previous checkpoints and baselines

Exit criteria:

- Training runs end to end on a small game.
- Model improves against random and heuristic baselines.
- Training metrics are logged consistently.

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
