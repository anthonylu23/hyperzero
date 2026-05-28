#!/usr/bin/env python3
"""Run staged full self-play experiments for the universal agent."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_universal_hparam_sweep import (  # noqa: E402
    _start_cpu_monitor,
    now,
    summarize_trial,
)

PYTHON = ["conda", "run", "-n", "torch", "python", "-u"]

COMMON: dict[str, str | bool] = {
    "simulations": "32",
    "training_steps": "96",
    "batch_size": "256",
    "replay_capacity": "50000",
    "hidden_size": "256",
    "residual_blocks": "4",
    "heads": "8",
    "line_features": True,
    "input_layer_norm": True,
    "rank_adapters": True,
    "adapter_size": "64",
    "lr": "3e-4",
    "value_weight": "1.0",
    "weight_decay": "1e-4",
    "eval_games": "16",
    "eval_simulations": "24",
    "eval_mcts_simulations": "32",
    "eval_interval": "4",
    "max_active": "24",
    "checkpoint_keep_last": "8",
    "eval_workers": "8",
    "self_play_workers": "8",
    "central_batched_self_play": True,
    "line_policy_residual": False,
    "root_dirichlet_alpha": "0.0",
    "root_exploration_fraction": "0.0",
    "root_temperature": "0.0",
    "root_temperature_move_cutoff": "",
    "teacher_batch_fraction": "0.0",
}

EVAL_WEIGHTS = {"heuristic": 0.55, "tactical": 0.35, "random": 0.10}
EVAL_FLOORS = {
    "default": {"random": 0.85, "tactical": 0.50},
    "2d_6x7_k4": {"heuristic": 0.125},
    "4d_4x4x4x4_k4": {"heuristic": 0.375},
}

STAGES: list[dict[str, str]] = [
    {
        "name": "balanced_warmup",
        "config": "configs/universal_repair_balanced_20260522.json",
        "iterations": "48",
    },
    {
        "name": "4d_heavy_pressure",
        "config": "configs/universal_repair_4d_heavy_20260523.json",
        "iterations": "48",
    },
    {
        "name": "balanced_recovery",
        "config": "configs/universal_repair_recovery_20260525.json",
        "iterations": "32",
    },
]

ADAPTIVE_PULSES: dict[str, dict[str, str]] = {
    "4d_4x4x4x4_k4": {
        "name": "adaptive_4d_pressure",
        "config": "configs/universal_repair_4d_heavy_20260523.json",
        "iterations": "16",
    },
    "2d_6x7_k4": {
        "name": "adaptive_2d6x7_pressure",
        "config": "configs/universal_repair_2d6x7_20260522.json",
        "iterations": "16",
    },
    "default": {
        "name": "adaptive_balanced_recovery",
        "config": "configs/universal_repair_recovery_20260525.json",
        "iterations": "16",
    },
}

TRIALS: list[dict[str, str | bool]] = [
    {
        "name": "line_rank_large_staged_seed6201",
        "seed": "6201",
    },
    {
        "name": "line_rank_large_staged_lr2e4_seed6202",
        "seed": "6202",
        "lr": "2e-4",
    },
    {
        "name": "line_residual_explore_seed7001",
        "seed": "7001",
        "line_policy_residual": True,
        "root_dirichlet_alpha": "0.25",
        "root_exploration_fraction": "0.20",
        "root_temperature": "1.0",
        "root_temperature_move_cutoff": "24",
    },
    {
        "name": "teacher_anchor_residual_seed7002",
        "seed": "7002",
        "line_policy_residual": True,
        "root_dirichlet_alpha": "0.20",
        "root_exploration_fraction": "0.15",
        "root_temperature": "1.0",
        "root_temperature_move_cutoff": "24",
        "teacher_replay_path": (
            "runs/teacher_replays/universal_4d_heuristic_bounded_20260528.pt"
        ),
        "teacher_batch_fraction": "0.25",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only trials whose names contain this substring.",
    )
    parser.add_argument(
        "--start-after",
        default=None,
        help="Skip trials through and including the named trial.",
    )
    parser.add_argument(
        "--skip-existing-summaries",
        action="store_true",
        help="Skip trials that already have a summary.json in the run root.",
    )
    parser.add_argument(
        "--adaptive-pulses",
        type=int,
        default=1,
        help="Run this many 16-iteration pulses after fixed stages if floors fail.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one tiny stage with eval disabled for CUDA/memory validation.",
    )
    return parser.parse_args()


def merged_config(
    trial: dict[str, str | bool],
    stage: dict[str, str],
    *,
    end_iteration: int,
    smoke: bool = False,
) -> dict[str, str | bool | int]:
    config: dict[str, str | bool | int] = dict(COMMON)
    config.update(trial)
    config.update(stage)
    config["end_iteration"] = end_iteration
    if smoke:
        config.update(
            {
                "iterations": "1",
                "end_iteration": 1,
                "config": "configs/universal_smoke_20260525.json",
                "simulations": "2",
                "training_steps": "1",
                "batch_size": "64",
                "replay_capacity": "256",
                "eval_games": "0",
                "eval_interval": "99",
                "checkpoint_keep_last": "2",
                "max_active": "4",
                "eval_workers": "1",
                "self_play_workers": "2",
            }
        )
    return config


def stage_plan(
    *,
    smoke: bool = False,
) -> list[dict[str, str]]:
    if smoke:
        return [
            {
                "name": "smoke_balanced",
                "config": "configs/universal_smoke_20260525.json",
                "iterations": "1",
            }
        ]
    return [dict(stage) for stage in STAGES]


def stage_end_iterations(stages: list[dict[str, str]]) -> list[int]:
    total = 0
    ends = []
    for stage in stages:
        total += int(stage["iterations"])
        ends.append(total)
    return ends


def train_command(
    trial: dict[str, str | bool],
    stage: dict[str, str],
    checkpoint_dir: Path,
    *,
    end_iteration: int,
    resume_from_checkpoint: Path | None = None,
    smoke: bool = False,
) -> list[str]:
    config = merged_config(trial, stage, end_iteration=end_iteration, smoke=smoke)
    command = PYTHON + [
        "scripts/train_universal.py",
        "--variants-json",
        str(config["config"]),
        "--iterations",
        str(config["end_iteration"]),
        "--simulations",
        str(config["simulations"]),
        "--training-steps",
        str(config["training_steps"]),
        "--batch-size",
        str(config["batch_size"]),
        "--replay-capacity",
        str(config["replay_capacity"]),
        "--hidden-size",
        str(config["hidden_size"]),
        "--residual-blocks",
        str(config["residual_blocks"]),
        "--heads",
        str(config["heads"]),
        "--learning-rate",
        str(config["lr"]),
        "--value-weight",
        str(config["value_weight"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--seed",
        str(config["seed"]),
        "--device",
        "cuda",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-keep-last",
        str(config["checkpoint_keep_last"]),
        "--eval-games",
        str(config["eval_games"]),
    ]
    if bool(config["line_features"]):
        command.append("--line-features")
    if bool(config["input_layer_norm"]):
        command.append("--input-layer-norm")
    if bool(config["rank_adapters"]):
        command.extend(
            ["--rank-adapters", "--adapter-size", str(config["adapter_size"])]
        )
    if bool(config.get("line_policy_residual", False)):
        command.append("--line-policy-residual")
    if float(str(config.get("root_dirichlet_alpha", "0.0"))) > 0.0:
        command.extend(["--root-dirichlet-alpha", str(config["root_dirichlet_alpha"])])
    if float(str(config.get("root_exploration_fraction", "0.0"))) > 0.0:
        command.extend(
            [
                "--root-exploration-fraction",
                str(config["root_exploration_fraction"]),
            ]
        )
    if float(str(config.get("root_temperature", "0.0"))) > 0.0:
        command.extend(["--root-temperature", str(config["root_temperature"])])
    if str(config.get("root_temperature_move_cutoff", "")):
        command.extend(
            [
                "--root-temperature-move-cutoff",
                str(config["root_temperature_move_cutoff"]),
            ]
        )
    teacher_replay_path = str(config.get("teacher_replay_path", ""))
    if teacher_replay_path:
        command.extend(["--teacher-replay-path", teacher_replay_path])
    if float(str(config.get("teacher_batch_fraction", "0.0"))) > 0.0:
        command.extend(
            ["--teacher-batch-fraction", str(config["teacher_batch_fraction"])]
        )
    if resume_from_checkpoint is not None:
        command.extend(["--resume-from-checkpoint", str(resume_from_checkpoint)])
    if int(config["eval_games"]) > 0:
        command.extend(
            [
                "--eval-opponents",
                "random",
                "tactical",
                "heuristic",
                "--eval-simulations",
                str(config["eval_simulations"]),
                "--eval-mcts-simulations",
                str(config["eval_mcts_simulations"]),
                "--eval-interval",
                str(config["eval_interval"]),
                "--eval-score-weights",
                json.dumps(EVAL_WEIGHTS, sort_keys=True),
                "--eval-score-floors",
                json.dumps(EVAL_FLOORS, sort_keys=True),
            ]
        )
    command.extend(
        [
            "--batched-self-play",
            "--max-active-self-play-games",
            str(config["max_active"]),
            "--eval-workers",
            str(config["eval_workers"]),
            "--self-play-workers",
            str(config["self_play_workers"]),
        ]
    )
    if bool(config["central_batched_self_play"]):
        command.append("--central-batched-self-play")
    return command


def run_trial(
    trial: dict[str, str | bool],
    *,
    run_root: Path,
    poll_seconds: float,
    adaptive_pulses: int,
    smoke: bool,
) -> dict[str, Any]:
    trial_dir = run_root / trial["name"]
    checkpoint_dir = trial_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stages = stage_plan(smoke=smoke)
    end_iterations = stage_end_iterations(stages)
    (trial_dir / "trial.json").write_text(
        json.dumps(
            {
                "trial": trial,
                "common": COMMON,
                "stages": stages,
                "end_iterations": end_iterations,
                "eval_weights": EVAL_WEIGHTS,
                "eval_floors": EVAL_FLOORS,
                "smoke": smoke,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    monitor_log = (trial_dir / "gpu-monitor.csv").open("w", encoding="utf-8")
    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,memory.used,memory.free,"
            "utilization.gpu,utilization.memory,temperature.gpu,power.draw",
            "--format=csv,nounits",
            "--loop=15",
        ],
        stdout=monitor_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return_code = 0
    resume_from_checkpoint: Path | None = None
    stage_summaries = []
    try:
        for stage_index, (stage, end_iteration) in enumerate(
            zip(stages, end_iterations, strict=True),
            start=1,
        ):
            stage_summary = run_stage(
                trial,
                stage,
                trial_dir=trial_dir,
                checkpoint_dir=checkpoint_dir,
                stage_index=stage_index,
                end_iteration=end_iteration,
                resume_from_checkpoint=resume_from_checkpoint,
                poll_seconds=poll_seconds,
                smoke=smoke,
            )
            stage_summaries.append(stage_summary)
            return_code = int(stage_summary["return_code"])
            if return_code != 0:
                break
            resume_from_checkpoint = preferred_resume_checkpoint(
                checkpoint_dir,
                end_iteration,
            )
            if resume_from_checkpoint is None:
                return_code = 2
                break
        if return_code == 0 and not smoke:
            stage_index = len(stages)
            end_iteration = end_iterations[-1]
            for pulse_index in range(adaptive_pulses):
                rows = load_metrics(checkpoint_dir / "metrics.jsonl")
                if not rows or rows[-1].get("eval_floor_passed") is True:
                    break
                stage_index += 1
                pulse = adaptive_stage_for_metric(rows[-1], pulse_index=pulse_index)
                end_iteration += int(pulse["iterations"])
                stage_summary = run_stage(
                    trial,
                    pulse,
                    trial_dir=trial_dir,
                    checkpoint_dir=checkpoint_dir,
                    stage_index=stage_index,
                    end_iteration=end_iteration,
                    resume_from_checkpoint=resume_from_checkpoint,
                    poll_seconds=poll_seconds,
                    smoke=smoke,
                )
                stage_summaries.append(stage_summary)
                return_code = int(stage_summary["return_code"])
                if return_code != 0:
                    break
                resume_from_checkpoint = preferred_resume_checkpoint(
                    checkpoint_dir,
                    end_iteration,
                )
                if resume_from_checkpoint is None:
                    return_code = 2
                    break
    finally:
        monitor.terminate()
        try:
            monitor.wait(timeout=10)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()
        monitor_log.close()

    summary = summarize_trial(trial, trial_dir, return_code)
    summary["stage_summaries"] = stage_summaries
    (trial_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{now()}] END {trial['name']} rc={return_code} "
        f"best_pass={summary.get('best_floor_passing_score')} "
        f"best_raw={summary.get('best_raw_score')}",
        flush=True,
    )
    return summary


def run_stage(
    trial: dict[str, str | bool],
    stage: dict[str, str],
    *,
    trial_dir: Path,
    checkpoint_dir: Path,
    stage_index: int,
    end_iteration: int,
    resume_from_checkpoint: Path | None,
    poll_seconds: float,
    smoke: bool,
) -> dict[str, Any]:
    stage_dir = trial_dir / f"stage{stage_index:02d}_{stage['name']}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    cmd = train_command(
        trial,
        stage,
        checkpoint_dir,
        end_iteration=end_iteration,
        resume_from_checkpoint=resume_from_checkpoint,
        smoke=smoke,
    )
    (stage_dir / "command.json").write_text(
        json.dumps(cmd, indent=2) + "\n",
        encoding="utf-8",
    )
    (stage_dir / "stage.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "stage_index": stage_index,
                "end_iteration": end_iteration,
                "resume_from_checkpoint": (
                    None
                    if resume_from_checkpoint is None
                    else str(resume_from_checkpoint)
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[{now()}] START {trial['name']} stage={stage['name']} "
        f"end_iteration={end_iteration}",
        flush=True,
    )
    return_code = -999
    cpu_monitor: threading.Event | None = None
    try:
        with (stage_dir / "train.log").open("w", encoding="utf-8") as log:
            log.write("[{}] RUN {}\n".format(now(), " ".join(cmd)))
            log.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=train_environment(),
            )
            cpu_monitor = _start_cpu_monitor(stage_dir, proc.pid)
            last_seen_iteration = None
            while proc.poll() is None:
                time.sleep(poll_seconds)
                rows = load_metrics(checkpoint_dir / "metrics.jsonl")
                status = metric_status(rows)
                if status["iteration"] != last_seen_iteration:
                    last_seen_iteration = status["iteration"]
                    print_status(trial["name"], stage["name"], status)
            return_code = proc.wait()
    finally:
        if cpu_monitor is not None:
            cpu_monitor.set()
    rows = load_metrics(checkpoint_dir / "metrics.jsonl")
    stage_summary = {
        "stage": stage,
        "stage_index": stage_index,
        "end_iteration": end_iteration,
        "resume_from_checkpoint": (
            None if resume_from_checkpoint is None else str(resume_from_checkpoint)
        ),
        "return_code": return_code,
        "latest_metric": rows[-1] if rows else None,
    }
    (stage_dir / "summary.json").write_text(
        json.dumps(stage_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{now()}] END {trial['name']} stage={stage['name']} rc={return_code}",
        flush=True,
    )
    if return_code != 0:
        terminate_process_group(proc)
    return stage_summary


def train_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def terminate_process_group(
    proc: subprocess.Popen[str],
    *,
    terminate_timeout_seconds: float = 20.0,
) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=terminate_timeout_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        proc.kill()
    proc.wait()


def latest_checkpoint_for_iteration(
    checkpoint_dir: Path,
    iteration: int,
) -> Path | None:
    expected = checkpoint_dir / f"iteration_{iteration:04d}.pt"
    if expected.exists():
        return expected
    checkpoints = sorted(checkpoint_dir.glob("iteration_*.pt"))
    return checkpoints[-1] if checkpoints else None


def preferred_resume_checkpoint(
    checkpoint_dir: Path,
    iteration: int,
) -> Path | None:
    for filename in ("best_current_run_floor_passing.pt", "best_by_eval_score.pt"):
        candidate = checkpoint_dir / filename
        if candidate.exists():
            return candidate
    return latest_checkpoint_for_iteration(checkpoint_dir, iteration)


def adaptive_stage_for_metric(
    metric: dict[str, Any],
    *,
    pulse_index: int,
) -> dict[str, str]:
    failures = metric.get("eval_floor_failures") or []
    if any(str(failure).startswith("4d_4x4x4x4_k4:") for failure in failures):
        pulse = dict(ADAPTIVE_PULSES["4d_4x4x4x4_k4"])
    elif any(str(failure).startswith("2d_6x7_k4:") for failure in failures):
        pulse = dict(ADAPTIVE_PULSES["2d_6x7_k4"])
    else:
        pulse = dict(ADAPTIVE_PULSES["default"])
    pulse["name"] = f"{pulse['name']}_{pulse_index + 1}"
    return pulse


def load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def metric_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"iteration": None}
    latest = rows[-1]
    eval_rows = [row for row in rows if row.get("eval_score") is not None]
    best = max(eval_rows, key=lambda row: row["eval_score"], default=None)
    return {
        "iteration": latest.get("iteration"),
        "loss": latest.get("total_loss"),
        "policy_loss": latest.get("policy_loss"),
        "value_loss": latest.get("value_loss"),
        "self_play_seconds": latest.get("self_play_time_seconds"),
        "train_seconds": latest.get("training_step_time_seconds"),
        "eval_score": latest.get("eval_score"),
        "floor_passed": latest.get("eval_floor_passed"),
        "failures": latest.get("eval_floor_failures"),
        "best_eval": None if best is None else best.get("eval_score"),
        "best_iteration": None if best is None else best.get("iteration"),
        "eval_rows": len(eval_rows),
    }


def print_status(trial_name: str, stage_name: str, status: dict[str, Any]) -> None:
    iteration = status.get("iteration")
    if iteration is None:
        print(
            f"[{now()}] {trial_name}/{stage_name}: waiting for first metric",
            flush=True,
        )
        return
    failures = status.get("failures") or []
    print(
        f"[{now()}] {trial_name}/{stage_name}: iter={iteration} "
        f"loss={_fmt(status.get('loss'))} "
        f"policy={_fmt(status.get('policy_loss'))} "
        f"value={_fmt(status.get('value_loss'))} "
        f"eval={_fmt(status.get('eval_score'))} "
        f"best={_fmt(status.get('best_eval'))} "
        f"best_iter={status.get('best_iteration')} "
        f"eval_rows={status.get('eval_rows')} "
        f"self_play={_fmt(status.get('self_play_seconds'))}s "
        f"train={_fmt(status.get('train_seconds'))}s "
        f"floor={status.get('floor_passed')} failures={';'.join(failures)}",
        flush=True,
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def selected_trials(args: argparse.Namespace) -> list[dict[str, str | bool]]:
    trials = TRIALS
    for needle in args.only:
        trials = [trial for trial in trials if needle in trial["name"]]
    if args.start_after is not None:
        matching_indices = [
            index
            for index, trial in enumerate(trials)
            if args.start_after in trial["name"]
        ]
        if not matching_indices:
            raise SystemExit(
                f"--start-after did not match any trial: {args.start_after}"
            )
        trials = trials[matching_indices[-1] + 1 :]
    if args.max_trials is not None:
        trials = trials[: args.max_trials]
    return trials


def existing_trial_summaries(run_root: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    if not run_root.exists():
        return summaries
    for summary_path in sorted(run_root.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        trial = summary.get("trial", {})
        name = trial.get("name") if isinstance(trial, dict) else None
        summaries[str(name or summary_path.parent.name)] = summary
    return summaries


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    if args.adaptive_pulses < 0:
        raise SystemExit("--adaptive-pulses must be nonnegative")
    args.run_root.mkdir(parents=True, exist_ok=True)
    trials = selected_trials(args)
    existing_summaries = existing_trial_summaries(args.run_root)
    if args.skip_existing_summaries:
        trials = [trial for trial in trials if trial["name"] not in existing_summaries]
    stages = stage_plan(smoke=args.smoke)
    (args.run_root / "staged_config.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "common": COMMON,
                "eval_weights": EVAL_WEIGHTS,
                "eval_floors": EVAL_FLOORS,
                "stages": stages,
                "stage_end_iterations": stage_end_iterations(stages),
                "adaptive_pulses": args.adaptive_pulses,
                "adaptive_pulse_options": ADAPTIVE_PULSES,
                "smoke": args.smoke,
                "trials": trials,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summaries = []
    if args.skip_existing_summaries:
        summaries = [
            existing_summaries[trial["name"]]
            for trial in TRIALS
            if trial["name"] in existing_summaries
        ]
    for trial in trials:
        summaries.append(
            run_trial(
                trial,
                run_root=args.run_root,
                poll_seconds=args.poll_seconds,
                adaptive_pulses=args.adaptive_pulses,
                smoke=args.smoke,
            )
        )
        (args.run_root / "staged_summary.json").write_text(
            json.dumps(summaries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"[{now()}] STAGED BLOCK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
