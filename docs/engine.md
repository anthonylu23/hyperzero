# Game Engine API

The core engine is dependency-light and deterministic. It stores absolute board
state internally and exposes canonical current-player observations for agents.

## Configuration

```python
from hyperzero.game import GameConfig

config = GameConfig(shape=(4, 4, 4), connect_k=4, gravity_axis=0)

print(config.num_actions)     # 16
print(config.action_shape)    # (4, 4)
print(len(config.winning_lines))
```

`GameConfig` owns all precomputed lookup tables:

- flat action ids to non-gravity column coordinates
- gravity column cells
- winning lines
- lines touching each cell
- deterministic Zobrist hash keys

## State

```python
from hyperzero.game import GameState

state = GameState.new(config)
move = state.make_move(5)

print(move.cell_coord)
print(state.legal_mask())
print(state.canonical_board())
```

The internal board uses absolute player ids:

```text
 0 empty
 1 player one
-1 player two
```

Agent observations use current-player perspective:

```python
canonical = state.canonical_board()
```

## Search-Oriented State Updates

For MCTS, use in-place transitions:

```python
state.make_move(action)
state.undo_move()
```

For simpler callers, use copy-on-apply:

```python
next_state = state.apply(action)
```

Search code can enable incremental line-count win detection:

```python
state = GameState.new(config, use_line_counts=True)
```

The default win check scans only precomputed lines that touch the last move.
Line-count mode updates per-player line occupancy counts on move and undo.

## Hashing

Each state maintains a Zobrist hash that includes board occupancy and side to
move:

```python
key = int(state.zobrist_hash)
```

Use `state.recompute_hash()` as a defensive consistency check in tests.

## Policy Utilities

Flat action ids are the policy/search interface:

```python
from hyperzero.game import logits_to_policy, policy_to_action_tensor

policy = logits_to_policy(logits, state.legal_mask())
policy_grid = policy_to_action_tensor(policy, config)
```

## Symmetries

Symmetries preserve the gravity axis and transform both boards and action
policies:

```python
from hyperzero.game import gravity_preserving_symmetries

for symmetry in gravity_preserving_symmetries(config):
    augmented_board = symmetry.transform_board(state.canonical_board())
    augmented_policy = symmetry.transform_policy(policy)
```

## Replays and GUI Snapshots

Use snapshots for UI/debug views:

```python
snapshot = state.to_snapshot()
```

Use replays for serializable game records:

```python
from hyperzero.game import GameReplay

replay = GameReplay.from_state(state)
payload = replay.to_dict()
restored = GameReplay.from_dict(payload).playback()
```

## RL-Style Environment

The environment wrapper is intentionally thin:

```python
from hyperzero.game import ConnectKEnv

env = ConnectKEnv(config)
obs = env.reset()
obs, reward, terminated, info = env.step(action)
```

Rewards are from the acting player's perspective for the just-applied move.

## Benchmarks

Run:

```bash
python3 benchmarks/benchmark_engine.py
```

The benchmark reports legal-mask throughput, move/undo throughput, and random
playout throughput for the initial 2D, 3D, and 4D target variants.
