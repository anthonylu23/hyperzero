#!/usr/bin/env python3
"""Run a bounded sequence of GPU training/eval experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "2d_5x5_k4",
        "shape": [5, 5],
        "connect_k": 4,
        "iterations": 30,
        "games": 32,
        "simulations": 32,
        "steps": 64,
        "batch": 128,
        "hidden": 128,
        "blocks": 2,
        "eval_games": 16,
        "eval_simulations": 16,
        "series_games": 16,
        "final_games": 24,
        "estimate_minutes": 22,
    },
    {
        "name": "2d_6x7_k4",
        "shape": [6, 7],
        "connect_k": 4,
        "iterations": 30,
        "games": 32,
        "simulations": 32,
        "steps": 64,
        "batch": 128,
        "hidden": 128,
        "blocks": 2,
        "eval_games": 16,
        "eval_simulations": 16,
        "series_games": 16,
        "final_games": 24,
        "estimate_minutes": 28,
    },
    {
        "name": "2d_8x8_k4",
        "shape": [8, 8],
        "connect_k": 4,
        "iterations": 20,
        "games": 24,
        "simulations": 24,
        "steps": 48,
        "batch": 128,
        "hidden": 160,
        "blocks": 2,
        "eval_games": 12,
        "eval_simulations": 16,
        "series_games": 12,
        "final_games": 20,
        "estimate_minutes": 30,
    },
    {
        "name": "2d_10x10_k5",
        "shape": [10, 10],
        "connect_k": 5,
        "iterations": 12,
        "games": 20,
        "simulations": 20,
        "steps": 40,
        "batch": 128,
        "hidden": 192,
        "blocks": 2,
        "eval_games": 8,
        "eval_simulations": 12,
        "series_games": 10,
        "final_games": 16,
        "estimate_minutes": 26,
    },
    {
        "name": "3d_4x4x4_k4",
        "shape": [4, 4, 4],
        "connect_k": 4,
        "iterations": 20,
        "games": 24,
        "simulations": 24,
        "steps": 48,
        "batch": 128,
        "hidden": 160,
        "blocks": 2,
        "eval_games": 12,
        "eval_simulations": 16,
        "series_games": 12,
        "final_games": 20,
        "estimate_minutes": 30,
    },
    {
        "name": "3d_5x5x4_k4",
        "shape": [5, 5, 4],
        "connect_k": 4,
        "iterations": 12,
        "games": 20,
        "simulations": 20,
        "steps": 40,
        "batch": 128,
        "hidden": 192,
        "blocks": 2,
        "eval_games": 8,
        "eval_simulations": 12,
        "series_games": 10,
        "final_games": 16,
        "estimate_minutes": 26,
    },
    {
        "name": "3d_4x4x5_k4",
        "shape": [4, 4, 5],
        "connect_k": 4,
        "iterations": 10,
        "games": 16,
        "simulations": 16,
        "steps": 32,
        "batch": 128,
        "hidden": 160,
        "blocks": 2,
        "eval_games": 8,
        "eval_simulations": 12,
        "series_games": 10,
        "final_games": 16,
        "estimate_minutes": 18,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cutoff", required=True, help="Local time in HH:MM format")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-gpu-free-mb", type=int, default=6000)
    parser.add_argument("--max-gpu-utilization", type=int, default=20)
    parser.add_argument(
        "--allow-existing-compute",
        action="store_true",
        help="Allow launch when other GPU compute processes are already running.",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        help="Optional JSON file containing a list of experiment configs.",
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def cutoff_datetime(value: str) -> datetime:
    hour, minute = (int(part) for part in value.split(":", maxsplit=1))
    cutoff = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    if cutoff < datetime.now():
        cutoff += timedelta(days=1)
    return cutoff


def run_logged(cmd: list[str], log_path: Path, *, timeout: float | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{now()}] RUN {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[{now()}] RUN {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        start = time.perf_counter()
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            if timeout is not None and time.perf_counter() - start > timeout:
                proc.terminate()
                message = f"[{now()}] TIMEOUT after {timeout:.1f}s\n"
                print(message, end="", flush=True)
                log.write(message)
                break
        try:
            return_code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            return_code = proc.wait()
        elapsed = time.perf_counter() - start
        log.write(f"[{now()}] EXIT rc={return_code} elapsed_seconds={elapsed:.3f}\n")
        print(
            f"[{now()}] EXIT rc={return_code} elapsed_seconds={elapsed:.3f}",
            flush=True,
        )
        return return_code


def capture(cmd: list[str]) -> str:
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout


def snapshot(path: Path) -> None:
    text = [
        f"timestamp={now()}\n",
        capture(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                "utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        ),
        "PROCESSES\n",
        capture(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
        "LOAD\n",
        capture(["uptime"]),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(text), encoding="utf-8")


def gpu_free(
    min_free_mb: int,
    max_utilization: int,
    *,
    allow_existing_compute: bool,
) -> tuple[bool, str]:
    out = capture(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).strip()
    if not out:
        return False, "missing nvidia-smi output"
    free_mb, utilization = [int(part.strip()) for part in out.split(",")]
    processes = capture(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    ).strip()
    ok = (
        free_mb >= min_free_mb
        and utilization <= max_utilization
        and (allow_existing_compute or not processes)
    )
    return (
        ok,
        f"free_mb={free_mb} utilization={utilization} "
        f"compute_processes={processes or 'none'}",
    )


def wait_for_gpu(
    cutoff: datetime,
    min_free_mb: int,
    max_utilization: int,
    *,
    allow_existing_compute: bool,
) -> bool:
    while datetime.now() < cutoff:
        ok, status = gpu_free(
            min_free_mb,
            max_utilization,
            allow_existing_compute=allow_existing_compute,
        )
        print(f"[{now()}] GPU status: {status}", flush=True)
        if ok:
            return True
        time.sleep(60)
    return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def latest_checkpoint(path: Path) -> Path | None:
    checkpoints = sorted(path.glob("iteration_*.pt"))
    return checkpoints[-1] if checkpoints else None


def shape_args(shape: list[int]) -> list[str]:
    return [str(item) for item in shape]


def train_command(
    config: dict[str, Any],
    out_dir: Path,
    *,
    benchmark: bool,
    device: str,
) -> list[str]:
    if benchmark:
        iterations = 2
        games = min(8, config["games"])
        simulations = min(8, config["simulations"])
        steps = min(8, config["steps"])
        eval_games = min(4, config["eval_games"])
        eval_simulations = min(8, config["eval_simulations"])
    else:
        iterations = config["iterations"]
        games = config["games"]
        simulations = config["simulations"]
        steps = config["steps"]
        eval_games = config["eval_games"]
        eval_simulations = config["eval_simulations"]

    command = [
        sys.executable,
        "-u",
        "scripts/train_v1.py",
        "--shape",
        *shape_args(config["shape"]),
        "--connect-k",
        str(config["connect_k"]),
        "--iterations",
        str(iterations),
        "--games",
        str(games),
        "--simulations",
        str(simulations),
        "--training-steps",
        str(steps),
        "--batch-size",
        str(config["batch"]),
        "--replay-capacity",
        str(config.get("replay_capacity", 100_000)),
        "--hidden-size",
        str(config["hidden"]),
        "--residual-blocks",
        str(config["blocks"]),
        "--model-type",
        str(config.get("model_type", "mlp")),
        "--symmetry-augmentation",
        str(config.get("symmetry_augmentation", "none")),
        "--device",
        device,
        "--batched-self-play",
        "--max-active-self-play-games",
        str(config.get("max_active_games", min(games, 32))),
        "--eval-games",
        str(eval_games),
        "--eval-opponents",
        "random",
        "tactical",
        "heuristic",
        "--eval-simulations",
        str(eval_simulations),
        "--eval-interval",
        str(config.get("eval_interval", 1)),
        "--checkpoint-dir",
        str(out_dir),
    ]
    checkpoint_keep_last = config.get("checkpoint_keep_last")
    if checkpoint_keep_last is not None:
        command.extend(["--checkpoint-keep-last", str(checkpoint_keep_last)])
    return command


def summarize_config(
    config: dict[str, Any],
    config_dir: Path,
    status: str,
) -> dict[str, Any]:
    train_dir = config_dir / "train"
    metrics = load_jsonl(train_dir / "metrics.jsonl")
    benchmark_metrics = load_jsonl(config_dir / "benchmark" / "metrics.jsonl")
    eval_series = load_jsonl(config_dir / "eval-series.jsonl")
    final_evals: dict[str, Any] = {}
    for path in sorted((config_dir / "final-evals").glob("*.json")):
        final_evals[path.stem] = json.loads(path.read_text(encoding="utf-8"))

    best_by_opponent: dict[str, Any] = {}
    for row in eval_series:
        opponent = row.get("opponent")
        win_rate = row.get("stats", {}).get("agent_a_win_rate")
        if opponent is None or win_rate is None:
            continue
        current = best_by_opponent.get(opponent)
        if current is None or win_rate > current["agent_a_win_rate"]:
            best_by_opponent[opponent] = {
                "iteration": row.get("iteration"),
                "agent_a_win_rate": win_rate,
                "draw_rate": row.get("stats", {}).get("draw_rate"),
            }

    summary = {
        "name": config["name"],
        "status": status,
        "shape": config["shape"],
        "connect_k": config["connect_k"],
        "run_dir": str(config_dir),
        "benchmark_final_metric": benchmark_metrics[-1] if benchmark_metrics else None,
        "train_metrics_rows": len(metrics),
        "eval_series_rows": len(eval_series),
        "final_metric": metrics[-1] if metrics else None,
        "best_eval_series_by_opponent": best_by_opponent,
        "final_evals": final_evals,
    }
    (config_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_config(
    config: dict[str, Any],
    run_root: Path,
    cutoff: datetime,
    device: str,
    min_gpu_free_mb: int,
    max_gpu_utilization: int,
    allow_existing_compute: bool,
) -> dict[str, Any]:
    config_dir = run_root / config["name"]
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not wait_for_gpu(
        cutoff,
        min_gpu_free_mb,
        max_gpu_utilization,
        allow_existing_compute=allow_existing_compute,
    ):
        return {"name": config["name"], "status": "gpu_unavailable_before_cutoff"}

    snapshot(config_dir / "gpu-before.txt")
    status = "started"

    rc = run_logged(
        train_command(config, config_dir / "benchmark", benchmark=True, device=device),
        config_dir / "benchmark.log",
    )
    if rc != 0:
        status = f"benchmark_failed_rc_{rc}"
        snapshot(config_dir / "gpu-after.txt")
        return summarize_config(config, config_dir, status)

    rc = run_logged(
        train_command(config, config_dir / "train", benchmark=False, device=device),
        config_dir / "train.log",
    )
    if rc != 0:
        status = f"train_failed_rc_{rc}"
        snapshot(config_dir / "gpu-after.txt")
        return summarize_config(config, config_dir, status)

    final_checkpoint = latest_checkpoint(config_dir / "train")
    if final_checkpoint is None:
        status = "missing_checkpoint"
        snapshot(config_dir / "gpu-after.txt")
        return summarize_config(config, config_dir, status)

    if datetime.now() < cutoff - timedelta(minutes=3):
        rc = run_logged(
            [
                sys.executable,
                "-u",
                "scripts/evaluate_checkpoint_series.py",
                "--checkpoint-dir",
                str(config_dir / "train"),
                "--opponents",
                "random",
                "tactical",
                "heuristic",
                "--games",
                str(config["series_games"]),
                "--simulations",
                str(max(16, min(32, config["simulations"]))),
                "--mcts-simulations",
                "32",
                "--device",
                device,
                "--jsonl-output",
                str(config_dir / "eval-series.jsonl"),
            ],
            config_dir / "eval-series.log",
            timeout=max(60.0, (cutoff - datetime.now()).total_seconds() - 120.0),
        )
        if rc != 0:
            status = f"eval_series_failed_rc_{rc}"
    else:
        status = "skipped_eval_series_time_budget"

    final_eval_dir = config_dir / "final-evals"
    final_eval_dir.mkdir(exist_ok=True)
    for opponent in ("random", "tactical", "heuristic", "mcts"):
        if datetime.now() >= cutoff - timedelta(minutes=2):
            if status == "started":
                status = "partial_final_eval_time_budget"
            break
        rc = run_logged(
            [
                sys.executable,
                "-u",
                "scripts/evaluate_checkpoint.py",
                "--checkpoint",
                str(final_checkpoint),
                "--opponent",
                opponent,
                "--games",
                str(config["final_games"]),
                "--simulations",
                str(max(16, min(32, config["simulations"]))),
                "--mcts-simulations",
                "32",
                "--device",
                device,
                "--json-output",
                str(final_eval_dir / f"{opponent}.json"),
            ],
            config_dir / f"final-eval-{opponent}.log",
            timeout=max(60.0, (cutoff - datetime.now()).total_seconds() - 60.0),
        )
        if rc != 0:
            status = f"final_eval_{opponent}_failed_rc_{rc}"
            break

    if status == "started":
        status = "complete"
    snapshot(config_dir / "gpu-after.txt")
    return summarize_config(config, config_dir, status)


def main() -> None:
    args = parse_args()
    cutoff = cutoff_datetime(args.cutoff)
    args.run_root.mkdir(parents=True, exist_ok=True)
    configs = DEFAULT_CONFIGS
    if args.config_json is not None:
        configs = json.loads(args.config_json.read_text(encoding="utf-8"))
    snapshot(args.run_root / "gpu-start.txt")
    print(f"[{now()}] cutoff={cutoff:%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"[{now()}] run_root={args.run_root}", flush=True)

    summaries = []
    for config in configs:
        remaining_minutes = (cutoff - datetime.now()).total_seconds() / 60.0
        if remaining_minutes < config["estimate_minutes"] + 8:
            print(
                f"[{now()}] SKIP {config['name']}: remaining_minutes="
                f"{remaining_minutes:.1f} estimate={config['estimate_minutes']}",
                flush=True,
            )
            summaries.append(
                {
                    "name": config["name"],
                    "status": "skipped_time_budget",
                    "remaining_minutes": remaining_minutes,
                }
            )
            continue
        summary = run_config(
            config,
            args.run_root,
            cutoff,
            args.device,
            args.min_gpu_free_mb,
            args.max_gpu_utilization,
            args.allow_existing_compute,
        )
        summaries.append(summary)
        print(
            f"[{now()}] SUMMARY {config['name']}: status={summary['status']} "
            f"metrics_rows={summary.get('train_metrics_rows')} "
            f"eval_rows={summary.get('eval_series_rows')}",
            flush=True,
        )

    snapshot(args.run_root / "gpu-end.txt")
    output = args.run_root / "all-summaries.json"
    output.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(f"[{now()}] DONE summaries={output}", flush=True)


if __name__ == "__main__":
    main()
