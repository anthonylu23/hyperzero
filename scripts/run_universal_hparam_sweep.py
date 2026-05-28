#!/usr/bin/env python3
"""Run the current universal-agent hyperparameter sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PYTHON = ["conda", "run", "-n", "torch", "python", "-u"]

COMMON = {
    "iterations": "56",
    "simulations": "24",
    "training_steps": "64",
    "batch_size": "512",
    "hidden_size": "128",
    "residual_blocks": "2",
    "heads": "4",
    "eval_games": "8",
    "eval_simulations": "24",
    "eval_mcts_simulations": "32",
    "eval_interval": "4",
    "max_active": "24",
    "checkpoint_keep_last": "3",
    "eval_workers": "4",
    "self_play_workers": "8",
    "central_batched_self_play": True,
}

EVAL_WEIGHTS = {"heuristic": 0.55, "tactical": 0.35, "random": 0.10}
EVAL_FLOORS = {
    "default": {"random": 0.85, "tactical": 0.50},
    "2d_6x7_k4": {"heuristic": 0.125},
    "4d_4x4x4x4_k4": {"heuristic": 0.375},
}

TRIALS = [
    {
        "name": "A_lr3e4_vw1_wd1e4_active24",
        "lr": "3e-4",
        "value_weight": "1.0",
        "weight_decay": "1e-4",
        "config": "configs/universal_sprint_active24_20260521.json",
        "seed": "1101",
    },
    {
        "name": "B_lr1e4_vw1_wd1e4_active24",
        "lr": "1e-4",
        "value_weight": "1.0",
        "weight_decay": "1e-4",
        "config": "configs/universal_sprint_active24_20260521.json",
        "seed": "1102",
    },
    {
        "name": "C_lr5e4_vw1_wd1e4_active24",
        "lr": "5e-4",
        "value_weight": "1.0",
        "weight_decay": "1e-4",
        "config": "configs/universal_sprint_active24_20260521.json",
        "seed": "1103",
    },
    {
        "name": "D_lr3e4_vw075_wd1e4_active24",
        "lr": "3e-4",
        "value_weight": "0.75",
        "weight_decay": "1e-4",
        "config": "configs/universal_sprint_active24_20260521.json",
        "seed": "1104",
    },
    {
        "name": "E_lr3e4_vw05_wd1e4_active24",
        "lr": "3e-4",
        "value_weight": "0.5",
        "weight_decay": "1e-4",
        "config": "configs/universal_sprint_active24_20260521.json",
        "seed": "1105",
    },
    {
        "name": "F_lr3e4_vw075_wd3e4_active24",
        "lr": "3e-4",
        "value_weight": "0.75",
        "weight_decay": "3e-4",
        "config": "configs/universal_sprint_active24_20260521.json",
        "seed": "1106",
    },
    {
        "name": "G_lr3e4_vw075_wd1e4_heuristic_pressure",
        "lr": "3e-4",
        "value_weight": "0.75",
        "weight_decay": "1e-4",
        "config": "configs/universal_heuristic_pressure_20260521.json",
        "seed": "1107",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=Path("runs/universal_early_20260520-1150/checkpoints/best_by_eval_score.pt"),
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_trial(
    trial: dict[str, str],
    *,
    run_root: Path,
    resume_from_checkpoint: Path,
) -> dict[str, Any]:
    trial_dir = run_root / trial["name"]
    checkpoint_dir = trial_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "trial.json").write_text(
        json.dumps(trial, indent=2, sort_keys=True) + "\n",
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
    cmd = _train_command(trial, checkpoint_dir, resume_from_checkpoint)
    (trial_dir / "command.json").write_text(
        json.dumps(cmd, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[{now()}] START {trial['name']}", flush=True)
    try:
        with (trial_dir / "train.log").open("w", encoding="utf-8") as log:
            log.write("[{}] RUN {}\n".format(now(), " ".join(cmd)))
            log.flush()
            start = time.perf_counter()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            cpu_monitor = _start_cpu_monitor(trial_dir, proc.pid)
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return_code = proc.wait()
                elapsed = time.perf_counter() - start
                log.write(
                    f"[{now()}] EXIT rc={return_code} elapsed_seconds={elapsed:.3f}\n"
                )
            finally:
                cpu_monitor.set()
    finally:
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


def _start_cpu_monitor(trial_dir: Path, root_pid: int) -> threading.Event:
    stop = threading.Event()
    path = trial_dir / "cpu-monitor.csv"
    if not Path("/proc").exists():
        path.write_text("procfs unavailable on this platform\n", encoding="utf-8")
        stop.set()
        return stop

    def run() -> None:
        previous: dict[int, tuple[float, int]] = {}
        clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        start = time.perf_counter()
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "elapsed_seconds",
                    "pid",
                    "ppid",
                    "cpu_percent",
                    "rss_mib",
                    "cmdline",
                ]
            )
            while not stop.is_set():
                sample_time = time.perf_counter()
                for pid in _process_tree_pids(root_pid):
                    stat = _read_proc_stat(pid)
                    if stat is None:
                        continue
                    ppid, cpu_ticks = stat
                    previous_sample = previous.get(pid)
                    cpu_percent = 0.0
                    if previous_sample is not None:
                        previous_time, previous_ticks = previous_sample
                        elapsed = max(sample_time - previous_time, 1e-9)
                        cpu_percent = (
                            (cpu_ticks - previous_ticks) / clock_ticks / elapsed * 100.0
                        )
                    previous[pid] = (sample_time, cpu_ticks)
                    writer.writerow(
                        [
                            f"{sample_time - start:.3f}",
                            pid,
                            ppid,
                            f"{cpu_percent:.2f}",
                            f"{_read_rss_mib(pid):.1f}",
                            _read_cmdline(pid),
                        ]
                    )
                handle.flush()
                stop.wait(15.0)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return stop


def _process_tree_pids(root_pid: int) -> list[int]:
    children_by_parent: dict[int, list[int]] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        pid = int(stat_path.parent.name)
        stat = _read_proc_stat(pid)
        if stat is None:
            continue
        ppid, _ = stat
        children_by_parent.setdefault(ppid, []).append(pid)
    pids = []
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        pids.append(pid)
        stack.extend(children_by_parent.get(pid, ()))
    return pids


def _read_proc_stat(pid: int) -> tuple[int, int] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    close = text.rfind(")")
    if close == -1:
        return None
    fields = text[close + 2 :].split()
    if len(fields) < 13:
        return None
    return int(fields[1]), int(fields[11]) + int(fields[12])


def _read_rss_mib(pid: int) -> float:
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        return 0.0
    return 0.0


def _read_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _train_command(
    trial: dict[str, str],
    checkpoint_dir: Path,
    resume_from_checkpoint: Path,
) -> list[str]:
    command = PYTHON + [
        "scripts/train_universal.py",
        "--variants-json",
        trial["config"],
        "--iterations",
        COMMON["iterations"],
        "--simulations",
        COMMON["simulations"],
        "--training-steps",
        COMMON["training_steps"],
        "--batch-size",
        COMMON["batch_size"],
        "--hidden-size",
        COMMON["hidden_size"],
        "--residual-blocks",
        COMMON["residual_blocks"],
        "--heads",
        COMMON["heads"],
        "--learning-rate",
        trial["lr"],
        "--value-weight",
        trial["value_weight"],
        "--weight-decay",
        trial["weight_decay"],
        "--seed",
        trial["seed"],
        "--device",
        "cuda",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--resume-from-checkpoint",
        str(resume_from_checkpoint),
        "--checkpoint-keep-last",
        COMMON["checkpoint_keep_last"],
        "--eval-games",
        COMMON["eval_games"],
        "--eval-opponents",
        "random",
        "tactical",
        "heuristic",
        "--eval-simulations",
        COMMON["eval_simulations"],
        "--eval-mcts-simulations",
        COMMON["eval_mcts_simulations"],
        "--eval-interval",
        COMMON["eval_interval"],
        "--eval-score-weights",
        json.dumps(EVAL_WEIGHTS, sort_keys=True),
        "--eval-score-floors",
        json.dumps(EVAL_FLOORS, sort_keys=True),
        "--batched-self-play",
        "--max-active-self-play-games",
        COMMON["max_active"],
        "--eval-workers",
        COMMON["eval_workers"],
        "--self-play-workers",
        COMMON["self_play_workers"],
    ]
    if COMMON["central_batched_self_play"]:
        command.append("--central-batched-self-play")
    return command


def summarize_trial(
    trial: dict[str, str],
    trial_dir: Path,
    return_code: int,
) -> dict[str, Any]:
    metrics_path = trial_dir / "checkpoints" / "metrics.jsonl"
    rows = []
    if metrics_path.exists():
        rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    eval_rows = [row for row in rows if row.get("eval_score") is not None]
    best_raw = max(eval_rows, key=lambda row: row["eval_score"], default=None)
    passing = [row for row in eval_rows if row.get("eval_floor_passed") is True]
    best_passing = max(passing, key=lambda row: row["eval_score"], default=None)
    summary = {
        "trial": trial,
        "return_code": return_code,
        "run_dir": str(trial_dir),
        "metrics_rows": len(rows),
        "eval_rows": len(eval_rows),
        "last_iteration": rows[-1]["iteration"] if rows else None,
        "last_total_loss": rows[-1]["total_loss"] if rows else None,
        "min_total_loss": min((row["total_loss"] for row in rows), default=None),
        "best_raw_iteration": None if best_raw is None else best_raw["iteration"],
        "best_raw_score": None if best_raw is None else best_raw["eval_score"],
        "best_raw_floor_passed": (
            None if best_raw is None else best_raw.get("eval_floor_passed")
        ),
        "best_raw_failures": (
            None if best_raw is None else best_raw.get("eval_floor_failures")
        ),
        "best_floor_passing_iteration": (
            None if best_passing is None else best_passing["iteration"]
        ),
        "best_floor_passing_score": (
            None if best_passing is None else best_passing["eval_score"]
        ),
    }
    (trial_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "sweep_config.json").write_text(
        json.dumps(
            {
                "common": COMMON,
                "eval_weights": EVAL_WEIGHTS,
                "eval_floors": EVAL_FLOORS,
                "trials": TRIALS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summaries = []
    for trial in TRIALS:
        summaries.append(
            run_trial(
                trial,
                run_root=args.run_root,
                resume_from_checkpoint=args.resume_from_checkpoint,
            )
        )
        (args.run_root / "sweep_summary.json").write_text(
            json.dumps(summaries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"[{now()}] SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
