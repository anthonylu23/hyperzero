# Project Status

Last updated: 2026-05-19

## Summary

HyperZero has moved past infrastructure smoke tests. The engine, baseline
agents, PUCT search, neural agents, self-play training, checkpoint evaluation,
GPU experiment runner, telemetry logging, and terminal demo are implemented and
validated.

The main research result so far:

```text
3D 4x4x4 Connect-4: stable/promoted
4D 4x4x4x4 Connect-4: feasible, stable, but tactically immature
```

## Validation

Current local validation:

```text
python3 -m ruff check .
python3 -m pytest -q
71 passed
```

Remote validation has passed on `anthonypc` in the `torch` conda environment.
The GPU box has an NVIDIA GeForce RTX 3060 Ti and has run all major 3D/4D
experiments with stable memory and temperature.

## Completed Milestones

- N-dimensional Connect-K game engine with configurable shape, connect length,
  and gravity axis.
- Random, tactical, heuristic, pure-MCTS, and AlphaZero-style neural agents.
- PUCT root tactical guard that forces immediate wins and masks one-ply losing
  moves when a safe alternative exists.
- Self-play training with replay, checkpointing, resume support, train-time
  evals, checkpoint-series evals, loss traces, and GPU telemetry.
- Terminal demo supports playing 2D games against the best neural checkpoint.
- 3D stability gate completed.
- 4D smoke, initial training, and heuristic continuation completed.

## 3D Result

Run:

```text
runs/phase4_3d_stability_guard_20260519-144210/
```

Configuration:

```text
4x4x4 K=4
line_resnet
120 iterations
48 games/iteration
64 PUCT simulations
```

Final evals over 160 games per opponent:

```text
Random     100.0%
Tactical    97.5%
Heuristic   94.4%
MCTS-32     99.4%
```

Interpretation: 3D is stable enough to treat as the promoted baseline. The
guarded search policy fixed the earlier immediate-threat failure mode, and
loss/eval/resource curves were stable.

## 4D Result

Completed runs:

```text
runs/phase6_4d_smoke_20260519-150123/
runs/phase6_4d_training_20260519-151442/
runs/phase6_4d_heuristic_followup_20260519-155628/
```

Best completed continuation final evals over 32 games per opponent:

```text
Random     100.0%
Tactical    43.8%
Heuristic   25.0%
MCTS-32     96.9%
```

Interpretation: 4D training is technically healthy. Losses stayed finite,
training improved, and GPU resources were stable. The limiting factor is
strategic: the specialist 4D agent still fails to prevent tactical and
heuristic threat/fork structures.

## Active/Pending Work

The 4D tactical-weighted continuation was paused on `anthonypc` for a Windows
reboot request.

Run root:

```text
/tmp/hyperzero-gpu-runs-phase6-4d-tactical-weighted-20260519-173543
```

Historical stopped process set at pause time:

```text
1039575  conda run wrapper
1039590  bash wrapper
1039595  run_gpu_experiments.py
1039610  gpu telemetry sampler
1043812  train_v1.py
```

The run had passed the resume point and reached iteration 23 with finite loss.
If the same Linux session is still alive, the process set can be resumed with
`SIGCONT`. After a reboot, treat the PIDs as invalid and restart from a verified
checkpoint path. The GPU runner now fails before waiting for GPU availability if
`resume_from_checkpoint` or `previous_best_checkpoint` points at a missing file,
so copy any required `/tmp` artifacts to persistent storage before restart.

## Next Research Phase

The next major phase is a universal multi-dimensional agent:

```text
one checkpoint
one shared policy-value model
plays selected 2D, 3D, and 4D Connect-K variants
```

This phase should test whether learned Connect-K concepts transfer across board
dimensionality. The likely architecture is a coordinate-conditioned token model
or line-incidence model with action-token scoring, mixed-game replay, and
per-dimension evaluation floors.
