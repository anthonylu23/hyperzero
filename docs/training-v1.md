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
64 passed
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
- model factory support for `mlp`, `line_mlp`, `cnn`, `resnet`, and
  `transformer`
- gravity-preserving random symmetry augmentation
- weighted eval score tracking and `best_by_eval_score.pt`
- optional checkpoint retention with `--checkpoint-keep-last`
- GPU experiment runner with benchmark, train, eval-series, final eval, GPU
  snapshots, and explicit co-scheduling via `--allow-existing-compute`

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

## May 19 GPU Experiment Block

The larger 2D/3D experiment block ran on `anthonypc` in `/tmp` run roots named
`/tmp/hyperzero-gpu-runs-20260519-*`. The runs used batched self-play,
symmetry augmentation, train-time baseline evals every 5 iterations, JSONL loss
and timing logs, eval-series, and final evals against random, tactical,
heuristic, and MCTS-32. Intermediate checkpoints were pruned for later runs
with `--checkpoint-keep-last 3` after `/tmp` quota pressure was observed.

Final neural-agent win rates:

| Config | Model | Rows | Random | Tactical | Heuristic | MCTS-32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 6x7 K=4 | ResNet | 90 | 99.5% | 84.0% | 100.0% | 91.0% |
| 6x7 K=4 | line-MLP | 90 | 99.5% | 81.5% | 50.0% | 91.0% |
| 6x7 K=4 | Transformer | 60 | 100.0% | 82.5% | 100.0% | 92.5% |
| 8x8 K=4 | line-MLP | 71 | 100.0% | 70.6% | 77.5% | 91.9% |
| 8x8 K=4 | ResNet | 70 | 100.0% | 76.9% | 100.0% | 98.1% |
| 8x8 K=4 | Transformer | 70 | 95.6% | 44.4% | 0.0% | 65.6% |
| 10x10 K=5 | line-MLP, 40 rows | 40 | 96.7% | 16.7% | 0.0% | 65.8% |
| 10x10 K=5 | ResNet, 40 rows | 40 | 99.2% | 48.3% | 39.2% | 90.8% |
| 10x10 K=5 | ResNet, 80 rows | 80 | 98.8% | 49.4% | 0.0% | 90.6% |
| 10x10 K=5 | line-MLP, 100 rows | 100 | 98.8% | 60.0% | 77.5% | 83.8% |
| 10x10 K=5 | ResNet, 120 rows | 120 | 98.8% | 70.0% | 0.0% | 88.7% |
| 10x10 K=5 | Transformer, 80 rows | 80 | 98.1% | 28.7% | 0.0% | 78.1% |
| 4x4x4 K=4 | ResNet | 80 | 100.0% | 52.5% | 9.4% | 98.8% |
| 4x4x4 K=4 | line-MLP | 80 | 100.0% | 63.1% | 56.2% | 100.0% |
| 5x5x5 K=4 | ResNet, 50 rows | 50 | 100.0% | 75.0% | 0.0% | 100.0% |
| 5x5x5 K=4 | line-MLP, 50 rows | 50 | 98.3% | 25.0% | 0.0% | 96.7% |
| 5x5x5 K=4 | ResNet, 100 rows | 100 | 100.0% | 95.0% | 63.1% | 100.0% |

Main takeaways:

- ResNet is the best default architecture so far for 2D 6x7/8x8 and 3D 5x5x5
  once given enough training.
- Line-aware MLP features helped the 4x4x4 heuristic matchup and the long
  10x10 K=5 run, but did not solve 5x5x5 by themselves.
- Transformer was strong on 6x7, but degraded on 8x8 and 10x10 in these v1
  budgets.
- The heuristic baseline exposes strategic blind spots that random, tactical,
  and MCTS-32 do not always catch. For 10x10 K=5, the long line-MLP was the
  first run to beat heuristic decisively.
- Longer training fixed the 5x5x5 ResNet heuristic gap: 50 rows scored 0%
  against heuristic, while 100 rows reached 63.1%.

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
- iteration, cumulative training, self-play, optimizer-step, eval, checkpoint,
  and per-step/per-game/per-example wall-clock timings
- model forward-pass inference time, batches, states, and per-state inference
  time for self-play, eval, and their combined total
- eval game count, eval opponents, eval PUCT simulations, and MCTS baseline
  simulations
- per-opponent eval win/draw/loss rates when eval is enabled
- checkpoint path
- best eval score, best checkpoint path, and whether this row produced a new
  best checkpoint

When `--checkpoint-dir` is set, metrics are also appended to
`metrics.jsonl` in that directory unless `--metrics-path` overrides it. If
evals are not enabled with both `--eval-games` and `--eval-opponents`,
`evaluations` is logged as `{}` while the eval configuration fields are still
present.

Use `--checkpoint-keep-last N` for large runs on quota-constrained disks. This
keeps the latest `N` `iteration_*.pt` files plus `best_by_eval_score.pt`,
without dropping `metrics.jsonl`.

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

The first larger 2D and 3D v1 experiment block is complete. Next work should
focus on improving robustness rather than adding another blind sweep:

- Add a first-class experiment manifest for these GPU runs instead of relying
  on ad hoc `/tmp` JSON shards.
- Add resumable training from checkpoints, including replay state if we want
  exact continuation.
- Add checkpoint-series downsampling controls so eval-series cost is explicit
  instead of controlled indirectly by checkpoint pruning.
- Investigate why heuristic matchups are much harder on 10x10 K=5 and larger
  3D boards; prioritize line-aware conv/resnet features or stronger tactical
  curriculum data before moving to higher dimensions.
- Promote ResNet as the default v1 architecture for serious 3D runs and keep
  line-MLP as a diagnostic/comparison model.

The older 2D validation command remains below for reference.

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

This older command is kept as a quick correctness smoke. It is no longer the
next strategic experiment after the May 19 block:

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
- The v1 model implementations are initial baselines. ResNet is the strongest
  default so far, but the architecture search is not final.
- Training is not resumable from optimizer/replay state yet.
