# HyperZero — Consolidated Findings for the Technical Report / Paper

Last compiled: 2026-05-31

This document is a planning artifact: it consolidates what we actually ran, where
every significant result lives on disk, the techniques worth writing up, and the
gaps that still need closing before any result becomes a final research claim.

**Sourcing convention.** Older specialist and sweep win-rate tables are
reproduced from the project docs (`docs/experiments.md`, `docs/status.md`,
`docs/universal-agent.md`, `docs/roadmap.md`, `docs/training-v1.md`) after
cross-checking against local summaries where available. The final universal
promotion evidence is taken from live remote artifacts on `anthonypc`, especially
`runs/universal_validation_block_20260531/summary.md` and
`robust_s24_g64x3_seed7100.json`. Sweep structure, hyperparameter grids,
floor-gating behaviour, file inventory, and hardware were verified directly
against raw run artifacts. **All concrete paths in Section 5 were verified to
exist** (local unless tagged `[REMOTE]`).

---

## 1. Compute & environment

- **GPU box:** single NVIDIA GeForce RTX 3060 Ti (8 GiB), remote
  (`ssh anthonylu@anthonypc`), Fedora Linux (kernel `7.0.9-205.fc44`), conda env
  `torch`. Working trees on the box:
  `~/hyperzero-universal-next-20260525` (7.3 GB runs) and
  `~/hyperzero-universal-next-20260528` (1.6 GB runs — the authoritative store for
  the residual-recovery and 2026-05-31 validation blocks).
- **Local artifact store:** `runs/` ≈ 7.8 GB — 212 checkpoints (`.pt`), 288 JSON /
  JSONL summaries and eval records, 55 `metrics.jsonl`, 146 train logs, 58
  telemetry CSVs (paired `gpu-monitor.csv` / `cpu-monitor.csv`).
- **Resource envelope (3D, from telemetry):** peak GPU mem 3626 MiB, util avg
  19.8 % / peak 63 %, temp ≤ 62 °C. **4D:** < 1 GiB VRAM, CPU-bound, ≈ 8–10 s
  per self-play game at 16 sims; 4-seed parallelism used to fill the machine.

---

## 2. Experimental substrate (tunable complexity)

N-dimensional Connect-K with gravity. Variants actually trained:

| Variant id | Shape | K | Actions | Role |
|---|---|---|---:|---|
| `2d_4x4_k3` | (4,4) | 3 | 4 | easy validation |
| `2d_6x7_k4` | (6,7) | 4 | 7 | Connect Four benchmark |
| (2D arch block) | (8,8),(10,10) | 4,5 | 8,10 | v1 architecture study |
| `3d_4x4x4_k4` | (4,4,4) | 4 | 16 | **main target** |
| (3D scaling) | (5,5,5) | 4 | 25 | scaling probe |
| `4d_4x4x4x4_k4` | (4,4,4,4) | 4 | 64 | **stretch target** |

Engine: `hyperzero/game/` — precomputed winning lines + per-cell line incidence,
Zobrist hashing, canonical current-player view, gravity-preserving symmetry
group, incremental line-count win detection, replay serialization.

---

## 3. System components (all present in source tree)

- **Baselines:** `agents/{random,tactical,heuristic,mcts}_agent.py`.
- **Search:** `search/{puct,node,mcts}.py` — PUCT with **root tactical guard**
  (force immediate wins, mask one-ply losses when a safe move exists), batched
  leaf inference, stepwise search sessions.
- **Models:** `models/factory.py` exposes `mlp`, `line_mlp`, `cnn`, `resnet`,
  `line_resnet`, and `transformer`; universal path adds
  `UniversalPolicyValueTransformer` (`models/universal_transformer.py`) — token
  transformer over a global token + cell tokens + action tokens, scoring action
  tokens directly so one checkpoint emits 4 / 7 / 16 / 64 logits.
- **Training:** specialist `training/train.py`; universal
  `training/train_universal.py` + `universal_self_play.py` + `universal_replay.py`;
  resumable checkpoints (`training/checkpoint.py`, `checkpoint_version=2`).
- **Encoding:** `universal/encoding.py` — cell/action tokens with normalized
  coords, rank, shape, gravity axis, connect length, ply, column-fill features.
- **Orchestration scripts:** `scripts/{train_v1,train_universal,run_gpu_experiments,
  run_universal_hparam_sweep,run_universal_targeted_sweep,run_universal_scaling_block,
  run_universal_staged_block,distill_universal_heuristic,evaluate_checkpoint,
  evaluate_checkpoint_series,evaluate_universal_checkpoints}.py`.

---

## 4. Technique inventory (the methods worth a paper)

1. AlphaZero self-play (MCTS visit-count policy targets, outcome value targets).
2. Gravity-preserving **symmetry augmentation** (action-label-safe).
3. **Line-aware feature planes** (`line_*` models) — most repeated arch win.
4. **Weighted eval-score + per-variant eval floors** gating promotion. Universal
   runs persist three checkpoint heads per run:
   `best_by_eval_score.pt`, `best_current_run_floor_passing.pt`,
   `best_current_run_raw.pt`.
5. **best-checkpoint selection** vs latest (latest regresses heuristic in 4D).
6. **Batched self-play** with capped active games.
7. **Mixed-variant balanced replay sampling** (config-aware).
8. **Curriculum reweighting** of per-variant games/iteration across continuations.
9. **Teacher-replay anchored "residual recovery"** continual learning
   (anti-forgetting) — the method behind the promoted universal checkpoint
   (`teacher010` = teacher batch fraction 0.10, lr 2e-5).
10. **Line-feature distillation** (bounded distillation run).
11. **Heuristic-opponent injection** into self-play (`heuristic_pressure`).
12. Hyperparameter sweep (lr / value_weight / weight_decay), **capacity scaling**,
    replay/training-step/sims-budget sweeps.
13. Paired GPU+CPU **telemetry** and **loss-trace failure analysis**.

---

## 5. Experiment ledger — significant blocks with verified artifact paths

Artifact-type glossary: *loss curve* = `train/metrics.jsonl` or
`checkpoints/metrics.jsonl` (per-iteration policy/value/total loss + eval rows);
*eval-over-checkpoints* = `eval-series.jsonl`; *final eval* = per-opponent JSON
under `final-evals/`; *robust eval* = `robust_evals/*.json[l]`; *telemetry* =
`gpu-monitor.csv` / `cpu-monitor.csv`. `[REMOTE]` = exists only on
`~/hyperzero-universal-next-20260528/` (not synced locally).

### 5.1 — 3D target gate (5×5×5 line_resnet vs ResNet control + 4×4×4 filler)
Dir: `runs/phase3_3d_target_20260519-121124/`
- `line-resnet/target_3d_5x5x5_line_resnet_sym/{summary.json, eval-series.jsonl, heuristic-loss-traces.json, heuristic-trace-eval.json}`
- `resnet-control/target_3d_5x5x5_resnet_sym_longer/{summary.json, eval-series.jsonl, heuristic-loss-traces.json, heuristic-trace-eval.json}`
- `filler-4x4x4/filler_3d_4x4x4_line_resnet_sym/{summary.json, eval-series.jsonl, heuristic-loss-traces.json, heuristic-trace-eval.json}`
- Config: `configs/phase3_3d_target_ready.json`
- Result: 5×5×5 line_resnet R100 / T95.5 / H76.5 / MCTS100, beats plain ResNet
  control on heuristic (H61.0); 4×4×4 filler exposed a sharp heuristic hole (H21.3).
- ⚠️ Per-iteration checkpoints were pruned; these sub-runs keep summaries + eval
  series + loss traces only.

### 5.2 — 3D stability gate (PROMOTED 3D baseline)
Dir: `runs/phase4_3d_stability_guard_20260519-144210/stable_3d_4x4x4_line_resnet_guard/`
- Loss curve: `train/metrics.jsonl` (benchmark warmup at `benchmark/metrics.jsonl`)
- Eval-over-checkpoints: `eval-series.jsonl`
- Final eval (160 games/opp), per opponent: `final-evals/random.json`,
  `final-evals/tactical.json`, `final-evals/heuristic.json`, `final-evals/mcts.json`
- Run summary: `summary.json`; run config snapshot: `config.json`;
  roll-up at `runs/phase4_3d_stability_guard_20260519-144210/all-summaries.json`
- Telemetry: `gpu-monitor.csv` (+ `gpu-before.txt`/`gpu-after.txt`)
- Source config: `configs/phase4_3d_stability_guard_20260519.json`
- Result: **R100 / T97.5 / H94.4 / MCTS-32 99.4**; loss 3.44→2.43.
- ⚠️ Trained checkpoints pruned locally; re-train from config or pull from the
  remote worktree if weights are needed.

### 5.3 — 4D smoke / initial / heuristic continuation
- Smoke: `runs/phase6_4d_smoke_20260519-150123/smoke_4d_4x4x4x4_line_mlp/`
  (`train/metrics.jsonl`, `eval-series.jsonl`, `summary.json`, `gpu-monitor.csv`)
  — config `configs/phase6_4d_smoke_20260519.json`
- Initial: `runs/phase6_4d_training_20260519-151442/target_4d_4x4x4x4_line_mlp/`
  (`train/metrics.jsonl`, `eval-series.jsonl`, `summary.json`, `gpu-monitor.csv`,
  **fork failure traces** at `best-evals/heuristic-loss-traces.json`) — config
  `configs/phase6_4d_training_20260519.json`
- Heuristic continuation:
  `runs/phase6_4d_heuristic_followup_20260519-155628/followup_4d_4x4x4x4_line_mlp_heuristic/`
  (`train/metrics.jsonl`, `eval-series.jsonl`, `summary.json`, `gpu-monitor.csv`)
  — config `configs/phase6_4d_heuristic_followup_20260519.json`

### 5.4 — 4D tactical-weighted fresh seeds (specialist 4D baseline = seed 2)
Seeds: `runs/phase6_4d_tactical_fresh_seed{0,1,2,3}_20260520-*/`.
Seed-2 (promoted specialist):
`runs/phase6_4d_tactical_fresh_seed2_20260520-1015/fresh_4d_4x4x4x4_line_mlp_tactical_weighted_seed2/`
- Loss curve: `train/metrics.jsonl`; eval-over-checkpoints: `eval-series.jsonl`;
  run summary: `summary.json`; telemetry: `gpu-monitor.csv`
- Checkpoints: `train/best_by_eval_score.pt`, `train/iteration_0039.pt` (35–39 kept)
- Configs: `configs/phase6_4d_tactical_weighted_fresh_seed{1,2,3}_20260520.json`
  (+ base `configs/phase6_4d_tactical_weighted_fresh_20260520.json`)
- Result (seed-2 best, 40 games/opp): R100 / T55 / H55 / MCTS100.

### 5.5 — 4D stronger-search probe (48 sims)
Dir: `runs/phase6_4d_stronger_search_probe_20260520-1416/`
- Loss curve: `train/metrics.jsonl`; checkpoints: `train/best_by_eval_score.pt`,
  `train/iteration_0007.pt`, `train/iteration_0008.pt`
- Final eval of iter-7 best, per opponent:
  `final-evals/best/{random,tactical,heuristic}.json`
- Config: `configs/phase6_4d_stronger_search_probe_20260520.json`
- Result: H↑ to 56.2 but T↓ to 43.8 → seed-2 stays promoted.

### 5.6 — Universal early run (first multi-dim checkpoint)
Dir: `runs/universal_early_20260520-1150/`
- Loss curve: `checkpoints/metrics.jsonl`
- Checkpoints: `checkpoints/best_by_eval_score.pt` (+ iter 34–36)
- Best score 0.6055 @ iter 33. Baseline config: `configs/universal_gpu_starter_20260520.json`.

### 5.7 — Universal curriculum v2 / v3 (negative results)
- v2: `runs/universal_curriculum_v2_20260520-1356/checkpoints/metrics.jsonl`
  (+ iter 38–40) — config `configs/universal_curriculum_v2_20260520.json`
- v3: `runs/universal_curriculum_v3_20260520-1428/checkpoints/metrics.jsonl`
  (+ iter 38–40) — config `configs/universal_curriculum_v3_20260520.json`
- Best scores 0.4789 (v2) / 0.5609 (v3), both below early-best 0.6055; **no floors
  passed → no `best_by_eval_score.pt` was emitted** for either run.

### 5.8 — Universal hyperparameter sweep (7 trials A–G)
Dir: `runs/universal_hparam_parallel_queued8_20260521/`
- Sweep records: `sweep_config.json`, `sweep_summary.json`
- Per trial `<TRIAL>/`: loss curve `checkpoints/metrics.jsonl`; best raw weights
  `checkpoints/best_current_run_raw.pt`; `summary.json`, `command.json`,
  `trial.json`, `gpu-monitor.csv`, `cpu-monitor.csv`. Trials:
  `A_lr3e4_vw1_wd1e4_active24` … `G_lr3e4_vw075_wd1e4_heuristic_pressure`.
- Trial configs: `configs/universal_sprint_active24_20260521.json`,
  `configs/universal_heuristic_pressure_20260521.json`
- Result: grid lr {1e-4,3e-4,5e-4} × vw {0.5,0.75,1.0} × wd {1e-4,3e-4}; top raw
  ≈ 0.48 (trial A), heuristic-pressure G ≈ 0.43; **none passed floors**.

### 5.9 — Universal targeted "repair" sweep (8 trials → teacher source)
Dir: `runs/universal_targeted_sweep_promoted_20260522/`
- Sweep records: `sweep_config.json`, `sweep_summary.json`
- Per trial `<TRIAL>/`: `checkpoints/metrics.jsonl`,
  `checkpoints/best_current_run_raw.pt`, `summary.json`, `command.json`,
  `trial.json`, `gpu-monitor.csv`, `cpu-monitor.csv`. Key trials:
  `promoted_repair2d6x7_sims32_seed2108` (top raw ≈ 0.48, basis for teacher),
  `promoted_repair4d_sims32_seed2107`, `promoted_train128_seed2105`.
- ⚠️ `promoted_replay50k_seed2104` **failed to run** (no `checkpoints/`; only
  `summary.json`/`command.json`/telemetry, return code 1).
- Trial configs: `configs/universal_repair_2d6x7_20260522.json`,
  `configs/universal_repair_4d_20260522.json`,
  `configs/universal_repair_balanced_20260522.json`
- Result: per-variant repair curricula + 32 sims narrowed failures to mostly the
  2D-6×7 heuristic floor; **none passed floors**, but set up the recovery run.

### 5.10 — Universal capacity scaling (negative result)
Dir: `runs/universal_scaling_block_20260523/` (`scale_config.json`, `scale_summary.json`)
- Medium (192×3/6h ≈ 1.84 M):
  `scale_medium_192x3_seed5102/{checkpoints/metrics.jsonl, checkpoints/best_current_run_raw.pt, summary.json}`
- Large (256×4/8h ≈ 4.21 M):
  `scale_large_256x4_seed5103/{checkpoints/metrics.jsonl, checkpoints/best_current_run_raw.pt}`
  (no `summary.json` — run did not finish a clean summary)
- Result: medium ≥ large at fixed budget → capacity was not the bottleneck.

### 5.11 — Line-feature distillation / line-rank architecture runs
- Bounded line-distill: `runs/universal_4d_line_distill_bounded_20260523/`
  - weights `distill_iter24_bounded_s800.pt`; metrics `distill_iter24_bounded_s800_metrics.json`
  - robust evals: `robust_evals/4d_heuristic_distill_vs_repair_s24_g32x2.jsonl`,
    `robust_evals/all_variants_distill_vs_arch_s24_g16.jsonl`,
    `robust_evals/final_all_variants_distill_s24_g32x2.jsonl`
  - driver `scripts/distill_universal_heuristic.py`
- Line-rank arch:
  `runs/universal_arch_line_rank_medium_b384_20260523/checkpoints/{metrics.jsonl, best_current_run_raw.pt}` (+ iter 13–18)
- 4D-repair arch:
  `runs/universal_arch_4drepair_b384_20260523/checkpoints/{metrics.jsonl, best_current_run_raw.pt}` (+ iter 23–28),
  robust eval `robust_evals/4d_heuristic_arch_repair_s24_g32x2.jsonl`
- Result: competitive but did not surpass teacher-replay anchored recovery.

### 5.12 — Robust-eval comparison harnesses (candidate selection)
Each dir holds a per-matchup `records.jsonl` + roll-up `summary.json` (+ `eval.log`):
- `runs/universal_candidate_robust_eval_20260522/{records.jsonl, summary.json}`
- `runs/universal_robust_full72_vs_train128_20260522/{records.jsonl, summary.json}`
- `runs/universal_robust_full72_vs_train128_sims48_20260522/`
- `runs/universal_final_robust_full72_train128_seed2_20260523/{records.jsonl, summary.json}`

### 5.13 — Universal residual-recovery training + older two-seed robust eval
Run root: `runs/universal_residual_followup_20260528/`.
**Locally only the promoted checkpoint is synced; the loss curves, robust evals,
run metadata, and the entire no-teacher control are `[REMOTE]`** on
`~/hyperzero-universal-next-20260528/`.

- **Teacher-replay candidate (teacher batch fraction 0.10, lr 2e-5, seed 6604):**
  `residual_recovery_teacher010_lr2e5_seed6604/`
  - **`checkpoints/best_by_eval_score.pt`** (iter 36; the demo checkpoint) — *local*
  - `[REMOTE]` `checkpoints/{best_current_run_floor_passing.pt, best_current_run_raw.pt, iteration_0035..40.pt}`
  - `[REMOTE]` loss curve `checkpoints/metrics.jsonl`
  - `[REMOTE]` robust eval `robust_evals/residual_recovery_teacher010_best_s24_g32x2_seed6700.{json,jsonl}` → **score 0.8214**, passed all floors
  - `[REMOTE]` run metadata `run_meta.json`
- **No-teacher control (matched recipe, seed 6603) — `[REMOTE]` in full:**
  `residual_recovery_lr2e5_seed6603/`
  - `checkpoints/{best_by_eval_score.pt, best_current_run_floor_passing.pt, best_current_run_raw.pt, iteration_0035..40.pt, metrics.jsonl}`
  - `robust_evals/residual_recovery_best_s24_g32x2_seed6700.{json,jsonl}`, `run_meta.json`
  - Older two-seed robust eval: **score 0.8249**, passed all floors. This looked
    slightly better than the teacher-replay candidate under the older protocol,
    so do **not** cite this pair alone as proof of a clean teacher-replay ablation.
- `[REMOTE]` launcher: `runs/universal_residual_followup_20260528/launch_followup.sh`
- Variant curriculum config (local): `configs/universal_repair_recovery_20260525.json`
  (2d6x7=28, 2d4x4=8, 3d=24, 4d=24 games/iter).

Interpretation: both residual-recovery branches were viable under the older
`s24_g32x2_seed6700` evaluation. The teacher-replay branch becomes the promoted
candidate only after the stricter 2026-05-31 validation block in §5.14.

### 5.14 — Promotion-grade universal validation block (current headline evidence)
Run root: `[REMOTE]`
`/home/anthonylu/hyperzero-universal-next-20260528/runs/universal_validation_block_20260531/`
- Inputs: `checkpoints.json`
- Protocol metadata: `run_meta.json`
- Primary eval: `robust_s24_g64x3_seed7100.{json,jsonl}`; summary:
  `summary.md`
- Launcher: `launch_robust_eval_s24_g64x3.sh`
- Telemetry: `gpu-monitor.csv`
- Protocol: 5 checkpoints × 4 variants × 4 opponents × 3 seeds × 64 games = 240
  records. Agent simulations = 24; MCTS opponent simulations = 32; workers = 6;
  device = CUDA.
- Scored weights: heuristic 0.55, tactical 0.35, random 0.10, MCTS 0.0. MCTS-32
  is reported as a diagnostic column, not a scored promotion term.
- Aggregate floors: default random ≥ 0.90 and tactical ≥ 0.75; 2D-6×7
  heuristic ≥ 0.75; 3D heuristic ≥ 0.65; 4D heuristic ≥ 0.65.
- Resource note: 391 GPU samples; mean util 93.3 %, max util 99 %, max memory
  1458 MiB, max temp 63 °C.

Result:

| Rank | Checkpoint | Score | Floor result |
|---:|---|---:|---|
| 1 | `residual_recovery_teacher010_best` | 0.8221 | PASS |
| 2 | `residual_explore_best` | 0.8165 | `3d_4x4x4_k4:heuristic=0.635<0.650` |
| 3 | `residual_recovery_best` | 0.8017 | `2d_4x4_k3:tactical=0.740<0.750` |
| 4 | `teacher_anchor_best` | 0.7571 | `4d_4x4x4x4_k4:heuristic=0.620<0.650` |
| 5 | `distill_iter24` | 0.7560 | `4d_4x4x4x4_k4:heuristic=0.641<0.650` |

This is the cleanest promotion evidence: `residual_recovery_teacher010_best`
is the top scalar checkpoint and the only candidate in this validation block
that passes every aggregate floor.

---

## 6. Headline results

**Promoted universal (teacher-replay, iter 36; 2026-05-31 validation score
0.8221):**

| Variant | Random | Tactical | Heuristic | MCTS-32 |
|---|---:|---:|---:|---:|
| 2D 4×4 K3 | 96.4 (185/192) | 79.7 (153/192) | 72.9 (140/192) | 57.8 (111/192) |
| 2D 6×7 K4 | 100 (192/192) | 96.9 (186/192) | 100 (192/192) | 98.4 (189/192) |
| 3D 4×4×4 K4 | 100 (192/192) | 96.9 (186/192) | 71.4 (137/192) | 100 (192/192) |
| 4D 4×4×4×4 K4 | 100 (192/192) | 96.9 (186/192) | 79.2 (152/192) | 100 (192/192) |

**Promoted 3D specialist:** R100 / T97.5 / H94.4 / MCTS99.4 (160 games/opp).
**Best 4D specialist (seed 2):** R100 / T55 / H55 / MCTS100 (40 games/opp).

**Teacher-replay signal.** The teacher-replay run (seed 6604, teacher batch
fraction 0.10) and no-teacher recovery run (seed 6603, teacher batch fraction 0)
are matched on the main recipe and learning rate but not on seed. The older
two-seed eval slightly favored no-teacher, while the stricter 2026-05-31
validation favored teacher-replay recovery and failed the no-teacher checkpoint
on the 2D-4×4 tactical floor. Treat this as evidence that teacher replay improved
promotion stability, not as a fully isolated ablation until a multi-seed matched
study is run.

**Selection lineage:** hparam sweep (raw ≈ 0.48) → targeted repair sweep
(`repair2d6x7_sims32`, raw ≈ 0.48) → residual-explore / teacher-anchor branches
→ residual-recovery follow-up → teacher-replay recovery iter 36 (validation
**0.8221**, only candidate in the 2026-05-31 block to clear every aggregate
floor).

---

## 7. Honest limitations

- Modest promotion-eval sample sizes (≈ 64 games/seed × 3 seeds, sims 24).
- Confidence intervals are not yet reported. Several headline cells are still
  close enough to need uncertainty bars, especially 2D-4×4 vs MCTS-32 (111/192).
- 4D heuristic/fork robustness only partial; 4D specialists are seed-sensitive.
- The root guard prevents one-ply blunders but not earlier fork creation (see
  `runs/phase6_4d_training_20260519-151442/target_4d_4x4x4x4_line_mlp/best-evals/heuristic-loss-traces.json`).
- Heuristic-opponent injection gave only marginal lift.
- Recurring: **lower self-play loss ≠ stronger external play** (v2/v3, train128).
- Operational debt: one sweep trial (`promoted_replay50k_seed2104`) failed to run;
  the promoted residual-recovery run is local-only as a single checkpoint — its
  metrics/robust JSON and the entire no-teacher control live on the remote box.
- Reproducibility debt: `python3 -m pip install -e . --dry-run` currently fails
  because setuptools discovers multiple top-level packages in the flat layout
  (`apps`, `runs`, `configs`, `services`, `hyperzero`). Tests pass via
  `pytest`'s `pythonpath = ["."]`, but package installation should be fixed
  before a paper artifact release.

---

## 8. Gaps to close before the paper (with where the work would land)

1. **Search-budget strength curve (planned E3, never run cleanly).** Only ad-hoc
   16/24/32/48 sims exist. Need a controlled 25/50/100/200/400 sims-vs-strength
   curve on a fixed checkpoint → new figure. Use `scripts/evaluate_checkpoint.py`
   over the promoted universal checkpoint (§5.13) and the 3D specialist (§5.2 must
   be re-trained — its weights were pruned).
2. **Symmetry on/off ablation (planned E6).** Not isolated as a matched pair.
3. **Curriculum vs direct (planned E5).** No clean direct-from-scratch 3D control
   against a curriculum arm.
4. **Universal-vs-specialist transfer table.** Central thesis; only partially done.
   Run the promoted universal and the 3D/4D specialists head-to-head under one
   fixed budget and tabulate per-variant deltas.
5. **Teacher-replay ablation.** Current evidence compares seed 6603 (no teacher) with
   seed 6604 (teacher fraction 0.10), plus stricter validation over several
   candidate families. For a clean method claim, run matched multi-seed no-teacher
   vs teacher continuations or phrase the paper claim as a stability observation.
6. **Confidence intervals / uncertainty bars.** Add Wilson or bootstrap intervals
   for every headline rate and score; at minimum annotate the 192-game validation
   cells and 40-game 4D specialist rows.
7. **Value calibration & policy-entropy analysis.** Logged in every `metrics.jsonl`
   but never extracted/plotted.
8. **Elo / cross-checkpoint matchup matrix.** `eval/tournament.py` exists but the
   matrix was never computed across the checkpoint family.
9. **4D throughput / scaling numbers.** games/hr, sims/move time, memory-vs-N only
   partially collected; the per-run `gpu-monitor.csv` / `cpu-monitor.csv` allow
   retrospective extraction.
10. **Larger-N / stronger-budget robust evals** before making claims beyond the
    current board sizes, opponent set, and sims-24 validation budget.
11. **Package-install reproducibility.** Fix `pyproject.toml` package discovery so
    `pip install -e .` succeeds without relying on pytest's local `pythonpath`.
12. **Sync/restore the residual-recovery and validation blocks** from
   `~/hyperzero-universal-next-20260528/runs/universal_residual_followup_20260528/`
   and `~/hyperzero-universal-next-20260528/runs/universal_validation_block_20260531/`
   so the teacher comparison and promotion evidence are reproducible locally, not
   just on the remote box.

---

## 9. Reproducibility

Per-run provenance is captured as follows:
- **Specialist phase runs:** `summary.json` + `config.json` + `train/metrics.jsonl`
  + per-opponent `final-evals/*.json` + telemetry CSVs.
- **Sweeps:** `sweep_config.json` / `scale_config.json` at the root, and per trial
  `command.json` (full CLI), `trial.json` (hparam overrides), `summary.json`,
  `checkpoints/metrics.jsonl`, paired `gpu-monitor.csv` / `cpu-monitor.csv`.
- **Residual-recovery block:** `run_meta.json` + `checkpoints/metrics.jsonl` +
  `robust_evals/*.json` + `launch_followup.sh` (all `[REMOTE]`).
- **Promotion validation block:** `[REMOTE]`
  `runs/universal_validation_block_20260531/{run_meta.json,checkpoints.json,summary.md,robust_s24_g64x3_seed7100.json,robust_s24_g64x3_seed7100.jsonl,gpu-monitor.csv}`.

Promoted checkpoint (loaded by `hyperzero.server.agent_service.DEFAULT_CHECKPOINT`
and shipped in the demo image):
`runs/universal_residual_followup_20260528/residual_recovery_teacher010_lr2e5_seed6604/checkpoints/best_by_eval_score.pt`.
