# Game Specification: N-Dimensional Connect-K

## Parameters

An N-dimensional Connect-K game is defined by:

```text
N: number of dimensions
shape: board size along each dimension
K: number of connected pieces required to win
gravity: whether moves fall along a gravity axis
gravity_axis: axis pieces fall along when gravity is enabled
players: two-player zero-sum game
```

Examples:

```text
2D classic-style: shape=(6, 7), K=4, gravity=true
3D target:        shape=(4, 4, 4), K=4, gravity=false or true
4D stretch:       shape=(4, 4, 4, 4), K=4, gravity=false
```

## Board State

The board is an N-dimensional tensor.

```text
0  = empty cell
1  = current player
-1 = opponent
```

Internally, states should be canonicalized from the current player's perspective before being passed to agents or neural networks. This keeps the value target consistent:

```text
+1 = win for current player
 0 = draw or unknown during search
-1 = loss for current player
```

## Actions

There are two action modes.

### Free Placement

The player chooses any empty coordinate in the N-dimensional board.

For shape `(4, 4, 4)`, the action space has up to 64 legal actions.

### Gravity Placement

The player chooses a coordinate across all non-gravity dimensions. The piece occupies the lowest available cell along the gravity axis.

For shape `(4, 4, 4)` with gravity along axis 0, the action space has up to 16 columns.

Gravity mode reduces branching factor but adds rule-specific geometry. Free placement is easier to generalize across dimensions and should be implemented first.

## Winning Lines

A player wins by occupying K consecutive cells along any valid direction.

A direction is an N-dimensional vector where each component is one of:

```text
-1, 0, 1
```

The all-zero vector is excluded. Opposite directions are duplicates, so only one of each pair should be retained.

Examples in 3D:

```text
(1, 0, 0)  axis-aligned
(0, 1, 0)  axis-aligned
(0, 0, 1)  axis-aligned
(1, 1, 0)  plane diagonal
(1, 0, 1)  plane diagonal
(0, 1, 1)  plane diagonal
(1, 1, 1)  space diagonal
(1, 1, -1) space diagonal
```

For each direction, the engine should generate all length-K segments that fit within the board. A terminal win occurs when every cell in one segment is occupied by the same player.

## Draws

A draw occurs when:

- no legal moves remain, and
- neither player has a winning line.

## Recommended Initial Variants

### Validation Variant

```text
shape=(6, 7)
K=4
gravity=true
```

Purpose: Validate against known Connect Four intuition.

### Main Target Variant

```text
shape=(4, 4, 4)
K=4
gravity=false
```

Purpose: First serious higher-dimensional learning task.

### Stretch Variant

```text
shape=(4, 4, 4, 4)
K=4
gravity=false
```

Purpose: Stress test search, memory, architecture, and training throughput.

## Implementation Notes

The engine should separate:

- immutable game configuration
- mutable board state
- move generation
- win-line precomputation
- terminal evaluation
- agent-facing canonicalization

Winning lines can be precomputed once per game configuration and reused for every state. This is likely simpler and less error-prone than scanning in every direction after each move during the first implementation.

