# Architecture Plan

## System Overview

HyperZero should be built as a layered system:

```text
Game engine
Baseline agents
MCTS
Neural policy-value model
Self-play data generation
Training loop
Evaluation and analysis
Visualization/interface
```

Each layer should be independently testable. The project should avoid coupling the game engine to neural training logic.

## Proposed Python Package Layout

```text
hyperzero/
  game/
    config.py
    state.py
    lines.py
    env.py
  agents/
    random_agent.py
    tactical_agent.py
    heuristic_agent.py
    mcts_agent.py
    alphazero_agent.py
  search/
    mcts.py
    node.py
  models/
    mlp.py
    transformer.py
  training/
    self_play.py
    replay_buffer.py
    train.py
    arena.py
  eval/
    tournament.py
    elo.py
  viz/
    render.py
tests/
```

This layout is a target, not a requirement for the first commit. The first implementation should be small enough to remain easy to debug.

## Game Engine

Responsibilities:

- Store board state.
- Generate legal moves.
- Apply moves.
- Detect wins and draws.
- Expose canonical observations.
- Precompute winning lines.

The engine should be deterministic and free of model dependencies.

## MCTS

The search implementation should support both pure MCTS and neural-guided MCTS.

Pure MCTS can use uniform priors and rollout or value heuristics. AlphaZero-style MCTS should use PUCT:

```text
score(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
```

Each node should track:

```text
prior probability
visit count
total value
mean value
children
terminal status
```

## Neural Model

The first model is a v1 residual MLP over a flattened board. It is intentionally
minimal: useful for verifying data flow, policy/value losses, checkpointing, and
PUCT integration before introducing larger architecture experiments.

Inputs:

```text
canonical board tensor
optional side features such as move count or legal-action mask
```

Outputs:

```text
policy logits over the full action space
value scalar in [-1, 1]
```

The second serious architecture should be a transformer over board cells:

- one token per cell
- learned or coordinate-based positional encodings
- current-player canonical cell value as token feature
- policy head maps cell tokens to move logits
- value head pools cell representations

Graph neural networks are a stretch architecture if time permits.

### Universal Multi-Dimensional Model

The next architecture phase is a single shared agent for 2D, 3D, and 4D
Connect-K. The current specialist checkpoints assume a fixed board shape and
fixed action count, so they cannot directly share one policy head across
dimensions. A universal model should treat board cells and legal actions as
sets of tokens with geometry metadata.

Recommended representation:

```text
cell token:
  canonical occupancy
  normalized coordinate padded to max supported dimension
  valid-coordinate mask
  gravity-axis indicator
  connect-k embedding
  board-shape embedding

action token:
  column/action coordinate
  legal-action bit
  resulting cell coordinate if played
```

Recommended model shape:

- shared cell/action encoder for all dimensions
- coordinate-conditioned transformer or line-incidence attention block
- value head pools cell tokens
- policy head scores action tokens independently, then masks illegal actions
- optional line-feature side channel reusing the line-aware 3D/4D findings

This avoids a dimension-specific final linear layer and lets one checkpoint
score a 7-action 2D Connect Four board, a 16-action 3D board, and a 64-action
4D board. The eval API should load the checkpoint once, pass the requested
`GameConfig`, and require the model to produce logits matching that config's
action count.

## Training Loop

The v1 training loop implements the smallest useful AlphaZero-style cycle:

1. Generate self-play games with current model plus MCTS.
2. Store `(state, MCTS policy, outcome)` examples.
3. Train policy-value network on replay buffer.
4. Evaluate against configured baselines when requested.
5. Save a checkpoint and append per-iteration metrics to JSONL.

Arena evaluation and promotion are deliberately deferred until the basic loop is
stable. Standalone checkpoint and checkpoint-series evaluation scripts provide
the current promotion signal. The default search API remains simple and
single-state, but v1 training can optionally use batched self-play: many active
games select PUCT leaves independently, then their leaf states are evaluated in
one batched model call.

For the universal-agent phase, replay entries must include their `GameConfig`.
Training batches should be mixed by curriculum weights rather than raw sample
availability; otherwise cheap 2D games will dominate the replay distribution.
Metrics should be reported both globally and per variant.

Training losses:

```text
policy_loss = cross_entropy(MCTS_visit_distribution, predicted_policy)
value_loss = mean_squared_error(game_outcome, predicted_value)
total_loss = policy_loss + value_weight * value_loss + l2_regularization
```

## Evaluation

Evaluation must be isolated from training. Agents should play with fixed budgets so comparisons are fair.

Important outputs:

- win rate
- draw rate
- average game length
- average move time
- Elo rating across checkpoints
- matchup matrix

Universal-agent evaluation adds per-dimension gates:

- 2D validation win rates
- 3D promoted-target win rates
- 4D tactical/heuristic stress win rates
- aggregate score with per-dimension floors, so 2D/3D strength cannot mask 4D
  weakness

## Experiment Tracking

Use TensorBoard or Weights & Biases. Track at minimum:

- training losses
- self-play game length
- policy entropy
- value error
- evaluation win rates
- MCTS simulations per move
- self-play games per hour

## Engineering Priorities

1. Correct game engine.
2. Reliable evaluation harness.
3. Minimal AlphaZero loop.
4. 3D training result.
5. Architecture experiments.
6. 4D stretch.
7. Universal 2D/3D/4D agent.
