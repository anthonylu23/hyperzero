# Training V1

This document covers the first HyperZero neural training stack. The model,
search, and training loop are v1 infrastructure: they are meant to establish a
correct train/evaluate/checkpoint cycle before optimizing throughput or running
large 3D experiments.

## Local Correctness

Run the full local suite:

```bash
python3 -m ruff check .
python3 -m pytest -q
```

Current known-good result:

```text
60 passed
```

## Current Progress

Implemented:

- residual MLP policy-value model
- single-state and batched neural evaluator API
- non-batched neural PUCT search
- stepwise PUCT search sessions for batched coordination
- `AlphaZeroAgent`
- self-play generation with MCTS visit-policy targets
- batched self-play leaf inference across active games
- replay buffer
- v1 training loop
- checkpoint save/load helpers
- standalone checkpoint evaluation
- checkpoint-series evaluation
- JSONL metrics
- train-time evals against baselines
- CPU/CUDA device validation

Validation so far:

- Local lint and tests pass.
- `anthonypc` remote lint and tests pass in the `torch` conda environment.
- CUDA was verified on `anthonypc`: NVIDIA GeForce RTX 3060 Ti.
- A 3-iteration CUDA smoke run completed with checkpoints, losses, baseline
  evals, `metrics.jsonl`, and checkpoint-series JSONL output.
- A 50-iteration 3x3 Connect-3 run completed at `/tmp/hyperzero-v1-full`.
  Final checkpoint summary from 50-game checkpoint-series eval:
  - random: `84%` wins, `16%` draws
  - tactical: `80%` wins, `20%` draws
  - heuristic: `50%` wins, `50%` draws, no losses
  - matched MCTS-32: `10%` wins, `90%` draws, no losses
- A 30-iteration 4x4 Connect-3 run completed at `/tmp/hyperzero-v1-4x4-k3`.
  Best/final checkpoint-series eval:
  - best vs heuristic: iteration 21, `76%` wins
  - final vs heuristic: `66%` wins
  - final vs matched MCTS-32: `76%` wins
  - final vs random: `92%` wins
  - final vs tactical: `78%` wins
- Batched self-play comparison on `anthonypc`:
  - 4x4 Connect-3, 2 iterations, same budget: single path `6.13s`, batched path
    `4.08s`
  - 4x4x4 Connect-4 smoke, 1 iteration, 8 games: single path `3.96s`, batched
    path `3.62s`
- GPT Pro reviewed the value perspective, backup, self-play target, loss
  masking, eval, and batching logic. The actionable hardening item was applied:
  PUCT backup now uses the copied simulation state's actual `player_to_move`,
  and training validates policy targets.

## Training Smoke Run

Run a small CPU smoke test from the repository root:

```bash
python3 scripts/train_v1.py \
  --iterations 2 \
  --games 4 \
  --simulations 16 \
  --training-steps 8 \
  --batch-size 32 \
  --hidden-size 32 \
  --residual-blocks 1 \
  --eval-games 4 \
  --eval-opponents random tactical heuristic \
  --eval-simulations 8 \
  --checkpoint-dir /tmp/hyperzero-v1-checkpoints
```

The loop prints a compact per-iteration summary and writes a richer JSONL row.
Each `metrics.jsonl` row logs:

- policy loss mean/min/max
- value loss mean/min/max
- total loss mean/min/max
- training steps, batch size, and PUCT simulations
- self-play game length
- replay size
- self-play mode and maximum active games for batched self-play
- eval game count, eval opponents, eval PUCT simulations, and MCTS baseline
  simulations
- per-opponent eval win/draw/loss rates when eval is enabled
- checkpoint path

When `--checkpoint-dir` is set, metrics are also appended to
`metrics.jsonl` in that directory unless `--metrics-path` overrides it. If
evals are not enabled with both `--eval-games` and `--eval-opponents`,
`evaluations` is logged as `{}` while the eval configuration fields are still
present.

## Checkpoint Evaluation

Evaluate one checkpoint:

```bash
python3 scripts/evaluate_checkpoint.py \
  --checkpoint /tmp/hyperzero-v1-checkpoints/iteration_0002.pt \
  --opponent heuristic \
  --games 50 \
  --simulations 32
```

Supported opponents:

- `random`
- `tactical`
- `heuristic`
- `mcts`
- `untrained`
- `checkpoint` with `--opponent-checkpoint`

Evaluate a checkpoint series:

```bash
python3 scripts/evaluate_checkpoint_series.py \
  --checkpoint-dir /tmp/hyperzero-v1-checkpoints \
  --opponents random tactical heuristic \
  --games 50 \
  --simulations 32 \
  --jsonl-output /tmp/hyperzero-v1-checkpoints/eval.jsonl
```

The v1 promotion check is intentionally simple: a checkpoint passes if its
win rate against the configured promotion opponent exceeds the configured
threshold. The default threshold is `0.55` against `heuristic`.

## Remote GPU Machine

Use `anthonypc` for fuller runs:

```bash
ssh anthonylu@anthonypc
cd /tmp/hyperzero-codex
conda run -n torch python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Example GPU run:

```bash
conda run -n torch python scripts/train_v1.py \
  --iterations 20 \
  --games 16 \
  --simulations 32 \
  --training-steps 32 \
  --batch-size 64 \
  --hidden-size 64 \
  --residual-blocks 2 \
  --device cuda \
  --batched-self-play \
  --max-active-self-play-games 16 \
  --eval-games 20 \
  --eval-opponents random tactical heuristic \
  --eval-simulations 16 \
  --checkpoint-dir /tmp/hyperzero-v1-checkpoints
```

## Next Full V1 Run

The 2D v1 validation runs above are complete. The next run should be an initial
3D smoke experiment with conservative budgets, not a full target experiment.

For reference, the completed fuller 2D training command was:

Training:

```bash
ssh anthonylu@anthonypc
cd /tmp/hyperzero-codex
rm -rf /tmp/hyperzero-v1-full
conda run -n torch python scripts/train_v1.py \
  --iterations 50 \
  --games 32 \
  --simulations 32 \
  --training-steps 64 \
  --batch-size 128 \
  --hidden-size 64 \
  --residual-blocks 2 \
  --device cuda \
  --eval-games 20 \
  --eval-opponents random tactical heuristic \
  --eval-simulations 16 \
  --checkpoint-dir /tmp/hyperzero-v1-full
```

Post-run checkpoint series evaluation:

```bash
conda run -n torch python scripts/evaluate_checkpoint_series.py \
  --checkpoint-dir /tmp/hyperzero-v1-full \
  --opponents random tactical heuristic mcts \
  --games 50 \
  --simulations 32 \
  --mcts-simulations 32 \
  --device cuda \
  --jsonl-output /tmp/hyperzero-v1-full/eval-series.jsonl
```

Primary questions:

- Does win rate improve over checkpoint index? Yes on 4x4 Connect-3.
- Does the trained agent reliably beat random? Yes.
- Does it become competitive with tactical or heuristic? Yes on 4x4 Connect-3.
- Does pure MCTS still dominate at matched or small budgets? No on 4x4
  Connect-3; final v1 checkpoint beat MCTS-32 in the checkpoint-series eval.
- Are value loss, policy loss, and game length stable enough to justify scaling?
  Yes for a 3D smoke experiment.

## Suggested 3D Smoke Run

This is the next recommended command after the completed 2D validation:

```bash
ssh anthonylu@anthonypc
cd /tmp/hyperzero-codex
rm -rf /tmp/hyperzero-v1-3d-smoke
conda run -n torch python scripts/train_v1.py \
  --shape 4 4 4 \
  --connect-k 4 \
  --iterations 10 \
  --games 16 \
  --simulations 16 \
  --training-steps 32 \
  --batch-size 128 \
  --hidden-size 128 \
  --residual-blocks 2 \
  --device cuda \
  --eval-games 8 \
  --eval-opponents random tactical heuristic \
  --eval-simulations 8 \
  --checkpoint-dir /tmp/hyperzero-v1-3d-smoke
```

Treat this as a throughput and correctness smoke test. A serious 3D target run
should follow only after inspecting game lengths, runtime, replay diversity, and
baseline evals from this smoke run.

If `--device cuda` is requested on a machine without CUDA, the training code
fails early instead of silently falling back to CPU.

## Known V1 Limits

- Neural inference inside one PUCT search is still non-batched.
- Batched self-play coordinates leaf evaluation across active games, but it does
  not yet reuse trees across moves or run multiple workers.
- Arena promotion is reported by eval scripts but not wired into automatic
  checkpoint replacement.
- The v1 model is a small MLP over flattened boards, not the final architecture
  for 3D or 4D experiments.
