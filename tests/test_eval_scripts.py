import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from hyperzero.game import GameConfig
from hyperzero.training import TrainingConfig, train_v1
from scripts.connect4_terminal import configure_agents, resolve_checkpoint_path
from scripts.evaluate_checkpoint_series import select_checkpoints
from scripts.run_gpu_experiments import (
    final_checkpoint_targets,
    run_config,
    summarize_config,
    train_command,
)


def test_select_checkpoints_supports_stride_latest_best_and_max(tmp_path) -> None:
    paths = []
    for iteration in range(1, 6):
        path = tmp_path / f"iteration_{iteration:04d}.pt"
        path.write_text("checkpoint", encoding="utf-8")
        paths.append(path)
    best = tmp_path / "best_by_eval_score.pt"
    best.write_text("best", encoding="utf-8")

    assert select_checkpoints(tmp_path, checkpoint_stride=2) == [
        paths[0],
        paths[2],
        paths[4],
    ]
    assert select_checkpoints(tmp_path, latest_only=True) == [paths[-1]]
    assert select_checkpoints(tmp_path, best_only=True) == [best]
    assert select_checkpoints(tmp_path, max_checkpoints=2) == paths[-2:]


def test_select_checkpoints_rejects_conflicting_modes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        select_checkpoints(tmp_path, latest_only=True, best_only=True)


def test_terminal_demo_resolves_checkpoint_directory_best(tmp_path: Path) -> None:
    best = tmp_path / "best_by_eval_score.pt"
    best.write_text("checkpoint", encoding="utf-8")

    assert resolve_checkpoint_path(None, tmp_path) == best
    assert resolve_checkpoint_path(best, None) == best

    with pytest.raises(ValueError, match="either --checkpoint or --checkpoint-dir"):
        resolve_checkpoint_path(best, tmp_path)


def test_terminal_demo_loads_2d_neural_opponent_from_best_checkpoint(
    tmp_path: Path,
) -> None:
    train_v1(
        TrainingConfig(
            game_config=GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0),
            iterations=1,
            self_play_games_per_iteration=1,
            puct_simulations=1,
            training_steps_per_iteration=1,
            batch_size=4,
            replay_capacity=16,
            hidden_size=8,
            residual_blocks=0,
            model_type="mlp",
            seed=0,
            checkpoint_dir=tmp_path,
            eval_games_per_iteration=1,
            eval_opponents=("random",),
            eval_simulations=1,
        )
    )

    args = argparse.Namespace(
        opponent="neural",
        human_player="X",
        x_agent="none",
        o_agent="none",
        seed=0,
        x_seed=None,
        o_seed=None,
        mcts_simulations=1,
        neural_simulations=1,
        c_puct=1.5,
        resolved_checkpoint=tmp_path / "best_by_eval_score.pt",
        device="cpu",
    )

    agents, config = configure_agents(args)

    assert config.shape == (3, 3)
    assert -1 in agents
    assert agents[-1].name.startswith("O-neural-")


def test_gpu_experiment_train_command_passes_eval_score_weights(
    tmp_path: Path,
) -> None:
    config = {
        "batch": 4,
        "blocks": 1,
        "connect_k": 3,
        "eval_games": 1,
        "eval_interval": 1,
        "eval_score_weights": {"heuristic": 0.6, "tactical": 0.4},
        "eval_simulations": 1,
        "games": 1,
        "hidden": 8,
        "iterations": 1,
        "max_active_games": 1,
        "name": "weighted",
        "shape": [3, 3],
        "simulations": 1,
        "steps": 1,
    }

    command = train_command(config, tmp_path, benchmark=False, device="cpu")

    index = command.index("--eval-score-weights")
    assert command[index + 1] == '{"heuristic": 0.6, "tactical": 0.4}'


def test_gpu_experiment_train_command_passes_seed(tmp_path: Path) -> None:
    config = {
        "batch": 4,
        "blocks": 1,
        "connect_k": 3,
        "eval_games": 1,
        "eval_interval": 1,
        "eval_simulations": 1,
        "games": 1,
        "hidden": 8,
        "iterations": 1,
        "max_active_games": 1,
        "name": "seeded",
        "seed": 123,
        "shape": [3, 3],
        "simulations": 1,
        "steps": 1,
    }

    command = train_command(config, tmp_path, benchmark=False, device="cpu")

    index = command.index("--seed")
    assert command[index + 1] == "123"


def test_final_checkpoint_targets_selects_latest_and_best(tmp_path: Path) -> None:
    first = tmp_path / "iteration_0001.pt"
    latest = tmp_path / "iteration_0002.pt"
    best = tmp_path / "best_by_eval_score.pt"
    first.write_text("first", encoding="utf-8")
    latest.write_text("latest", encoding="utf-8")
    best.write_text("best", encoding="utf-8")

    assert final_checkpoint_targets(tmp_path) == {
        "latest": latest,
        "best": best,
    }
    assert final_checkpoint_targets(tmp_path, selection="latest") == {
        "latest": latest,
    }
    assert final_checkpoint_targets(tmp_path, selection="best") == {
        "best": best,
    }


def test_summarize_config_reads_nested_final_evals_and_records_seed(
    tmp_path: Path,
) -> None:
    eval_path = tmp_path / "final-evals" / "best" / "heuristic.json"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text('{"ok": true}\n', encoding="utf-8")
    config = {
        "connect_k": 3,
        "name": "summary",
        "seed": 7,
        "shape": [3, 3],
    }

    summary = summarize_config(config, tmp_path, "complete")

    assert summary["seed"] == 7
    assert summary["final_evals"] == {"best/heuristic": {"ok": True}}
    assert "commit" in summary["git"]


def test_run_config_fails_before_gpu_wait_for_missing_resume_checkpoint(
    tmp_path: Path,
) -> None:
    config = {
        "connect_k": 3,
        "name": "missing-resume",
        "resume_from_checkpoint": str(tmp_path / "missing.pt"),
        "shape": [3, 3],
    }

    summary = run_config(
        config,
        tmp_path,
        datetime.now() + timedelta(minutes=1),
        "cpu",
        0,
        100,
        True,
    )

    assert summary["status"] == "missing_resume_checkpoint"
