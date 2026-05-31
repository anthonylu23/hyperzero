# Universal Agent Scaffold

Phase 7 adds a parallel universal training path for one shared checkpoint across
multiple Connect-K variants. The specialist v1 loop remains intact. The current
promoted universal checkpoint is the residual-recovery iteration-36 checkpoint
used by the web/API demo.

## Design

The universal path keeps PUCT unchanged. Variable board shapes are handled by a
state encoder and model adapter:

- `hyperzero/universal/encoding.py` converts each position into cell tokens and
  action tokens with board values, normalized coordinates, rank, shape, gravity
  axis, connect length, ply, and column fill features.
- `UniversalPolicyValueTransformer` runs one transformer over a global token,
  cell tokens, and action tokens.
- Policy logits are produced by scoring action tokens, so the same model can
  return four logits for 2D 4x4 K=3, sixteen logits for 3D 4x4x4 K=4, and
  sixty-four logits for 4D 4x4x4x4 K=4.
- `UniversalEvaluator` trims padded logits back to each state's legal action
  space and satisfies the existing `PolicyValueEvaluator` protocol consumed by
  PUCT.

## Training

Use `scripts/train_universal.py` for mixed-game training. Replay examples store
their `config_id` and `GameConfig`, and minibatches are sampled in a balanced
round-robin style across variants present in replay.

Smoke:

```bash
python3 scripts/train_universal.py \
  --variants-json configs/universal_smoke_20260520.json \
  --iterations 1 \
  --simulations 1 \
  --training-steps 1 \
  --batch-size 4 \
  --hidden-size 16 \
  --checkpoint-dir runs/universal_smoke
```

Starter GPU run for the RTX 3060 Ti box:

```bash
python3 scripts/train_universal.py \
  --variants-json configs/universal_gpu_starter_20260520.json \
  --iterations 80 \
  --simulations 32 \
  --training-steps 64 \
  --batch-size 128 \
  --hidden-size 128 \
  --residual-blocks 2 \
  --heads 4 \
  --device cuda \
  --checkpoint-dir runs/universal_gpu_starter \
  --checkpoint-keep-last 3 \
  --eval-games 16 \
  --eval-opponents random tactical heuristic mcts \
  --eval-simulations 32 \
  --eval-mcts-simulations 32 \
  --batched-self-play \
  --max-active-self-play-games 8
```

Resume:

```bash
python3 scripts/train_universal.py \
  --variants-json configs/universal_gpu_starter_20260520.json \
  --iterations 120 \
  --simulations 32 \
  --training-steps 64 \
  --batch-size 128 \
  --hidden-size 128 \
  --residual-blocks 2 \
  --heads 4 \
  --device cuda \
  --checkpoint-dir runs/universal_gpu_starter \
  --resume-from-checkpoint runs/universal_gpu_starter/iteration_0080.pt
```

Resume commands must preserve the checkpoint's model-shape options: hidden
size, residual blocks, heads, max rank, and max board extent.

Curriculum continuation:

```bash
python3 scripts/train_universal.py \
  --variants-json configs/universal_curriculum_v2_20260520.json \
  --iterations 80 \
  --simulations 24 \
  --training-steps 48 \
  --batch-size 128 \
  --hidden-size 128 \
  --residual-blocks 2 \
  --heads 4 \
  --device cuda \
  --checkpoint-dir runs/universal_curriculum_v2 \
  --resume-from-checkpoint runs/universal_early_20260520-1150/checkpoints/best_by_eval_score.pt \
  --checkpoint-keep-last 3 \
  --eval-games 16 \
  --eval-opponents random tactical heuristic mcts \
  --eval-simulations 24 \
  --eval-mcts-simulations 32 \
  --eval-score-weights '{"heuristic": 0.45, "tactical": 0.35, "mcts": 0.10, "random": 0.10}' \
  --eval-score-floors '{"default": {"random": 0.90, "tactical": 0.50}, "3d_4x4x4_k4": {"heuristic": 0.80}, "4d_4x4x4x4_k4": {"heuristic": 0.375}}' \
  --batched-self-play \
  --max-active-self-play-games 8
```

Curriculum continuations may change `self_play_games_per_iteration` for an
existing variant set. Variant ids, order, board shape, connect length, gravity
axis, hidden size, residual blocks, heads, max rank, and max board extent must
still match the resumed checkpoint.

Eval floors gate best-checkpoint promotion. The raw `eval_score` is still logged,
but a checkpoint is not copied to `best_by_eval_score.pt` when any configured
floor fails or a required opponent result is missing.

Result so far:

```text
runs/universal_curriculum_v2_20260520-1356/
iterations 34-40
best v2 score: 0.4789 at iteration 40
prior universal early best: 0.6055 at iteration 33
```

The v2 continuation did not promote a new best checkpoint. It reduced training
loss, but baseline evals remained weaker than the prior universal best and failed
the configured floors, mainly on 2D 6x7 heuristic and 4D heuristic.

Curriculum v3 reduced 2D 6x7 from 32 to 16 games/iteration and increased 3D
from 16 to 24 games/iteration:

```text
runs/universal_curriculum_v3_20260520-1428/
best v3 score: 0.5609 at iteration 34
prior universal early best: 0.6055 at iteration 33
```

V3 improved over v2, but it also failed floors and did not replace the initial
universal best. The next universal pass should change search budget or add
hard-position pressure rather than only changing curriculum counts.

Promoted residual-recovery checkpoint:

```text
runs/universal_residual_followup_20260528/residual_recovery_lr2e5_seed6603/checkpoints/best_by_eval_score.pt
iteration: 36
eval score: 0.8328
eval simulations: 24
eval games: 16 per variant/opponent
floor status: passed
```

Train-time eval win rates:

```text
Variant         Random  Tactical  Heuristic
2d_6x7_k4       100.0%    100.0%     100.0%
2d_4x4_k3       100.0%     68.8%      81.2%
3d_4x4x4_k4     100.0%    100.0%      68.8%
4d_4x4x4x4_k4   100.0%    100.0%      81.2%
```

This checkpoint is loaded by `hyperzero.server.agent_service.DEFAULT_CHECKPOINT`
and is included in the Docker image for the public demo. The next useful
research step is to run larger robust evals before treating the 16-game
train-time eval table as a final result.

## Checkpoints

Universal checkpoints use `checkpoint_version=2`, include `game_specs`,
`universal_model_config`, optimizer state, replay state, RNG state, and metrics.
Use `load_universal_training_checkpoint` or `build_universal_checkpoint_agent`
to load them.

## Metrics

Each iteration logs:

- self-play games/examples by config
- replay size by config
- policy/value/total losses
- per-variant, per-opponent evals
- an eval score that combines mean per-variant score and worst per-variant score
- minimum tactical win rate when tactical evals are enabled

This prevents strong 2D performance from fully hiding weak 4D behavior.
