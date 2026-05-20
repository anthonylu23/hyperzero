# Project Status

Last updated: 2026-05-20

## Summary

HyperZero has moved past infrastructure smoke tests. The engine, baseline
agents, PUCT search, neural agents, self-play training, checkpoint evaluation,
GPU experiment runner, telemetry logging, and terminal demo are implemented and
validated.

The main research result so far:

```text
3D 4x4x4 Connect-4: stable/promoted
4D 4x4x4x4 Connect-4: feasible, stable, improved but still noisy
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
- 4D smoke, initial training, heuristic continuation, and tactical-weighted
  multi-seed follow-up completed.

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
runs/phase6_4d_tactical_fresh_seed0_20260520-0938/
runs/phase6_4d_tactical_fresh_seed1_20260520-0941/
runs/phase6_4d_tactical_fresh_seed2_20260520-1015/
runs/phase6_4d_tactical_fresh_seed3_20260520-1015/
```

Previous best completed continuation final evals over 32 games per opponent:

```text
Random     100.0%
Tactical    43.8%
Heuristic   25.0%
MCTS-32     96.9%
```

Best tactical-weighted follow-up final evals over 40 games per opponent:

```text
Seed  Random  Tactical  Heuristic  MCTS-32
   1  100.0%     47.5%      52.5%   100.0%
   2  100.0%     55.0%      55.0%   100.0%
```

Interpretation: 4D training is technically healthy and the tactical-weighted
32-simulation follow-up improved the specialist baseline. Results are still
noisy across seeds and checkpoints, but the best seed-2 checkpoint is now the
current 4D specialist baseline. Late checkpoints can regress heuristic strength,
so promotion should use `best_by_eval_score.pt`, not the latest checkpoint.

Targeted stronger-search probe:

```text
runs/phase6_4d_stronger_search_probe_20260520-1416/
resumed seed-2 best checkpoint at iteration 6
48 PUCT simulations, 8 games/iteration, 2 continuation iterations
```

Best checkpoint was iteration 7. Final evals over 16 games per opponent:

```text
Random     100.0%
Tactical    43.8%
Heuristic   56.2%
```

Interpretation: stronger search can lift heuristic performance, but tactical
performance regressed versus the seed-2 specialist baseline. This supports
targeted follow-up, not an open-ended 4D specialist continuation.

## Universal Agent Scaffold

Phase 7 scaffolding is implemented as a parallel path:

```text
hyperzero/universal/encoding.py
hyperzero/models/universal_transformer.py
hyperzero/models/universal_evaluator.py
hyperzero/training/train_universal.py
scripts/train_universal.py
```

The same checkpoint can evaluate and play selected 2D, 3D, and 4D variants
through the existing PUCT interface. See `docs/universal-agent.md` for the smoke
command and RTX 3060 Ti starter command.

Initial universal run:

```text
runs/universal_early_20260520-1150/
2D 6x7 K=4, 2D 4x4 K=3, 3D 4x4x4 K=4, 4D 4x4x4x4 K=4
36 iterations, 16 PUCT simulations, mixed-game replay
```

Best train-time eval was iteration 33, score `0.6055`. It legally played every
configured variant and reached 4D evals of `100.0%` vs random, `100.0%` vs
tactical, and `37.5%` vs heuristic over 8 games. The main weakness is still
heuristic consistency, especially on 2D 6x7 and 4D.

Universal curriculum v2 continuation:

```text
configs/universal_curriculum_v2_20260520.json
runs/universal_curriculum_v2_20260520-1356/
resumed from runs/universal_early_20260520-1150/checkpoints/best_by_eval_score.pt
2D 6x7 K=4:        32 games/iteration
2D 4x4 K=3:         8 games/iteration
3D 4x4x4 K=4:      16 games/iteration
4D 4x4x4x4 K=4:    12 games/iteration
iterations 34-40, 16 PUCT simulations, eval every 2 iterations
```

Resume now allows per-variant games/iteration to change as curriculum, while
still requiring the resumed checkpoint to match variant ids, shapes, connect
lengths, gravity axis, and model shape.

The v2 run did not beat the initial universal best. Iteration 40 scored
`0.4789` versus the previous best `0.6055` and failed eval floors on 2D 4x4
random, 3D heuristic, and 4D heuristic. It did recover 3D heuristic to `50.0%`
and held minimum tactical win rate at `50.0%`, but 2D 6x7 and 4D heuristic
remained weak.

Universal curriculum v3 reduced 2D 6x7 exposure and shifted games toward 3D:

```text
configs/universal_curriculum_v3_20260520.json
runs/universal_curriculum_v3_20260520-1428/
2D 6x7 K=4:        16 games/iteration
2D 4x4 K=3:         8 games/iteration
3D 4x4x4 K=4:      24 games/iteration
4D 4x4x4x4 K=4:    12 games/iteration
```

The best v3 eval was iteration 34, score `0.5609`, still below the initial
universal best and still floor-failing. V3 improved over v2 on early aggregate
score and restored 2D 6x7 heuristic to `50.0%` at that checkpoint, but it did
not fix 4D heuristic and later iterations regressed.

## Next Research Phase

The next useful universal experiment should increase search budget or add
explicit anti-heuristic/fork pressure rather than only reweighting the curriculum.
The stronger-search 4D probe showed heuristic upside but tactical regression, so
specialist 4D work should remain diagnostic rather than the main path.
