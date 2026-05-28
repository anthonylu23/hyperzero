#!/usr/bin/env python3
"""Run controlled universal-transformer scaling experiments."""

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

from hyperzero.models import (  # noqa: E402
    UniversalModelConfig,
    UniversalPolicyValueTransformer,
    count_parameters,
)
from scripts.run_universal_hparam_sweep import (  # noqa: E402
    _start_cpu_monitor,
    now,
    summarize_trial,
)

PYTHON = ["conda", "run", "-n", "torch", "python", "-u"]

COMMON: dict[str, str | bool] = {
    "iterations": "64",
    "simulations": "32",
    "training_steps": "64",
    "batch_size": "512",
    "replay_capacity": "20000",
    "config": "configs/universal_repair_balanced_20260522.json",
    "lr": "3e-4",
    "value_weight": "1.0",
    "weight_decay": "1e-4",
    "eval_games": "16",
    "eval_simulations": "24",
    "eval_mcts_simulations": "32",
    "eval_interval": "4",
    "max_active": "24",
    "checkpoint_keep_last": "6",
    "eval_workers": "8",
    "self_play_workers": "8",
    "central_batched_self_play": True,
}

EVAL_WEIGHTS = {"heuristic": 0.55, "tactical": 0.35, "random": 0.10}
EVAL_FLOORS = {
    "default": {"random": 0.85, "tactical": 0.50},
    "2d_6x7_k4": {"heuristic": 0.125},
    "4d_4x4x4x4_k4": {"heuristic": 0.375},
}

TRIALS: list[dict[str, str]] = [
    {
        "name": "scale_small_control_128x2_seed5101",
        "hidden_size": "128",
        "residual_blocks": "2",
        "heads": "4",
        "seed": "5101",
    },
    {
        "name": "scale_medium_192x3_seed5102",
        "hidden_size": "192",
        "residual_blocks": "3",
        "heads": "6",
        "seed": "5102",
    },
    {
        "name": "scale_large_256x4_seed5103",
        "hidden_size": "256",
        "residual_blocks": "4",
        "heads": "8",
        "seed": "5103",
        "batch_size": "256",
        "training_steps": "128",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--prune-below", type=float, default=0.32)
    parser.add_argument("--min-prune-evals", type=int, default=2)
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
        "--smoke",
        action="store_true",
        help="Run a 2-iteration CUDA/memory probe with eval disabled.",
    )
    return parser.parse_args()


def merged_config(
    trial: dict[str, str],
    *,
    smoke: bool = False,
) -> dict[str, str | bool | int]:
    config: dict[str, str | bool | int] = dict(COMMON)
    config.update(trial)
    config["parameter_count"] = parameter_count(config)
    if smoke:
        config.update(
            {
                "iterations": "2",
                "training_steps": "2",
                "eval_games": "0",
                "eval_interval": "99",
                "checkpoint_keep_last": "2",
            }
        )
    return config


def parameter_count(config: dict[str, str | bool | int]) -> int:
    model = UniversalPolicyValueTransformer(
        UniversalModelConfig(
            hidden_size=int(config["hidden_size"]),
            residual_blocks=int(config["residual_blocks"]),
            heads=int(config["heads"]),
        )
    )
    return int(count_parameters(model))


def train_command(
    trial: dict[str, str],
    checkpoint_dir: Path,
    *,
    smoke: bool = False,
) -> list[str]:
    config = merged_config(trial, smoke=smoke)
    command = PYTHON + [
        "scripts/train_universal.py",
        "--variants-json",
        str(config["config"]),
        "--iterations",
        str(config["iterations"]),
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
    trial: dict[str, str],
    *,
    run_root: Path,
    poll_seconds: float,
    prune_below: float,
    min_prune_evals: int,
    smoke: bool,
) -> dict[str, Any]:
    trial_dir = run_root / trial["name"]
    checkpoint_dir = trial_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "trial.json").write_text(
        json.dumps(merged_config(trial, smoke=smoke), indent=2, sort_keys=True) + "\n",
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
    cmd = train_command(trial, checkpoint_dir, smoke=smoke)
    (trial_dir / "command.json").write_text(
        json.dumps(cmd, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"[{now()}] START {trial['name']} params="
        f"{merged_config(trial, smoke=smoke)['parameter_count']}",
        flush=True,
    )
    return_code = -999
    cpu_monitor: threading.Event | None = None
    try:
        with (trial_dir / "train.log").open("w", encoding="utf-8") as log:
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
            cpu_monitor = _start_cpu_monitor(trial_dir, proc.pid)
            last_seen_iteration = None
            prune_reason = None
            while proc.poll() is None:
                time.sleep(poll_seconds)
                rows = load_metrics(checkpoint_dir / "metrics.jsonl")
                status = metric_status(rows)
                if status["iteration"] != last_seen_iteration:
                    last_seen_iteration = status["iteration"]
                    print_status(trial["name"], status)
                if not smoke:
                    prune_reason = should_prune(
                        rows,
                        prune_below=prune_below,
                        min_prune_evals=min_prune_evals,
                    )
                    if prune_reason is not None:
                        print(
                            f"[{now()}] PRUNE {trial['name']} reason={prune_reason}",
                            flush=True,
                        )
                        terminate_process_group(proc)
                        break
            return_code = proc.wait()
            if prune_reason is not None:
                (trial_dir / "prune_reason.txt").write_text(
                    prune_reason + "\n",
                    encoding="utf-8",
                )
    finally:
        if cpu_monitor is not None:
            cpu_monitor.set()
        monitor.terminate()
        try:
            monitor.wait(timeout=10)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait()
        monitor_log.close()

    summary = summarize_trial(trial, trial_dir, return_code)
    print(
        f"[{now()}] END {trial['name']} rc={return_code} "
        f"best_pass={summary.get('best_floor_passing_score')} "
        f"best_raw={summary.get('best_raw_score')}",
        flush=True,
    )
    return summary


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


def print_status(trial_name: str, status: dict[str, Any]) -> None:
    iteration = status.get("iteration")
    if iteration is None:
        print(f"[{now()}] {trial_name}: waiting for first metric", flush=True)
        return
    failures = status.get("failures") or []
    print(
        f"[{now()}] {trial_name}: iter={iteration} "
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


def should_prune(
    rows: list[dict[str, Any]],
    *,
    prune_below: float,
    min_prune_evals: int,
) -> str | None:
    eval_rows = [row for row in rows if row.get("eval_score") is not None]
    if len(eval_rows) < min_prune_evals:
        return None
    scores = [float(row["eval_score"]) for row in eval_rows]
    if max(scores) < prune_below:
        return f"best_eval<{prune_below:.3f} after {len(eval_rows)} evals"
    recent = eval_rows[-min(2, len(eval_rows)) :]
    if all(_has_zero_2d6x7_heuristic(row) for row in recent) and len(eval_rows) >= 3:
        return "persistent_zero_2d6x7_heuristic"
    if all(_has_zero_4d_heuristic(row) for row in recent) and len(eval_rows) >= 3:
        return "persistent_zero_4d_heuristic"
    return None


def _has_zero_2d6x7_heuristic(row: dict[str, Any]) -> bool:
    stats = row.get("evaluations", {}).get("2d_6x7_k4", {}).get("heuristic", {})
    return stats.get("agent_a_win_rate") == 0.0


def _has_zero_4d_heuristic(row: dict[str, Any]) -> bool:
    stats = (
        row.get("evaluations", {})
        .get("4d_4x4x4x4_k4", {})
        .get("heuristic", {})
    )
    return stats.get("agent_a_win_rate") == 0.0


def selected_trials(args: argparse.Namespace) -> list[dict[str, str]]:
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
        name = trial.get("name") or summary_path.parent.name
        summaries[str(name)] = summary
    return summaries


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")
    args.run_root.mkdir(parents=True, exist_ok=True)
    trials = selected_trials(args)
    existing_summaries = existing_trial_summaries(args.run_root)
    if args.skip_existing_summaries:
        trials = [
            trial
            for trial in trials
            if trial["name"] not in existing_summaries
        ]
    (args.run_root / "scale_config.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "common": COMMON,
                "eval_weights": EVAL_WEIGHTS,
                "eval_floors": EVAL_FLOORS,
                "smoke": args.smoke,
                "trials": [
                    merged_config(trial, smoke=args.smoke) for trial in trials
                ],
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
                prune_below=args.prune_below,
                min_prune_evals=args.min_prune_evals,
                smoke=args.smoke,
            )
        )
        (args.run_root / "scale_summary.json").write_text(
            json.dumps(summaries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"[{now()}] SCALING BLOCK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
