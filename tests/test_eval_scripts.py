import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from hyperzero.game import GameConfig
from hyperzero.training import TrainingConfig, train_v1
from scripts.connect4_terminal import configure_agents, resolve_checkpoint_path
from scripts.evaluate_checkpoint_series import select_checkpoints
from scripts.evaluate_universal_checkpoints import (
    aggregate_records,
    parse_checkpoint_spec,
)
from scripts.run_gpu_experiments import (
    final_checkpoint_targets,
    run_config,
    summarize_config,
    train_command,
)
from scripts.run_universal_scaling_block import (
    selected_trials as selected_scaling_trials,
)
from scripts.run_universal_scaling_block import (
    train_command as universal_scaling_command,
)
from scripts.run_universal_staged_block import (
    adaptive_stage_for_metric,
    preferred_resume_checkpoint,
)
from scripts.run_universal_staged_block import (
    stage_end_iterations as staged_end_iterations,
)
from scripts.run_universal_staged_block import (
    stage_plan as universal_staged_plan,
)
from scripts.run_universal_staged_block import (
    train_command as universal_staged_command,
)
from scripts.run_universal_targeted_sweep import (
    existing_trial_summaries,
    selected_trials,
)
from scripts.run_universal_targeted_sweep import (
    train_command as universal_sweep_command,
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


def test_universal_checkpoint_arg_supports_explicit_label() -> None:
    spec = parse_checkpoint_spec("old_best=runs/old/best.pt")

    assert spec.label == "old_best"
    assert spec.path == Path("runs/old/best.pt")


def test_universal_eval_aggregate_scores_and_floors() -> None:
    from hyperzero.training import UniversalGameSpec

    variants = (
        UniversalGameSpec(
            config_id="2d",
            game_config=GameConfig(shape=(3, 3), connect_k=3),
            self_play_games_per_iteration=1,
        ),
    )
    records = [
        {
            "checkpoint": "candidate",
            "checkpoint_path": "candidate.pt",
            "iteration": 4,
            "variant": "2d",
            "opponent": "random",
            "stats": {
                "games": 4,
                "agent_a_wins": 4,
                "agent_b_wins": 0,
                "draws": 0,
            },
        },
        {
            "checkpoint": "candidate",
            "checkpoint_path": "candidate.pt",
            "iteration": 4,
            "variant": "2d",
            "opponent": "heuristic",
            "stats": {
                "games": 4,
                "agent_a_wins": 1,
                "agent_b_wins": 3,
                "draws": 0,
            },
        },
    ]

    summaries = aggregate_records(
        records,
        variants,
        {"random": 0.5, "heuristic": 0.5},
        {"default": {"random": 0.75}, "2d": {"heuristic": 0.5}},
    )

    assert summaries[0]["eval_score"] == pytest.approx(0.625)
    assert summaries[0]["floor_passed"] is False
    assert summaries[0]["floor_failures"] == ["2d:heuristic=0.250<0.500"]


def test_universal_targeted_sweep_command_allows_trial_overrides(
    tmp_path: Path,
) -> None:
    trial = {
        "name": "override",
        "lr": "3e-4",
        "value_weight": "1.0",
        "weight_decay": "1e-4",
        "config": "configs/universal_sprint_active24_20260521.json",
        "seed": "1",
        "simulations": "32",
        "replay_capacity": "50000",
    }

    command = universal_sweep_command(
        trial,
        tmp_path / "checkpoints",
        Path("runs/base.pt"),
    )

    assert command[command.index("--simulations") + 1] == "32"
    assert command[command.index("--replay-capacity") + 1] == "50000"


def test_universal_targeted_sweep_start_after_filters_remaining_trials() -> None:
    args = argparse.Namespace(
        only=[],
        start_after="promoted_sims32",
        max_trials=2,
    )

    trials = selected_trials(args)

    assert [trial["name"] for trial in trials] == [
        "promoted_sims48_active16_seed2103",
        "promoted_replay50k_seed2104",
    ]


def test_universal_targeted_sweep_loads_existing_summaries(tmp_path: Path) -> None:
    trial_dir = tmp_path / "promoted_control_active24_seed2101"
    trial_dir.mkdir()
    (trial_dir / "summary.json").write_text(
        (
            '{"trial": {"name": "promoted_control_active24_seed2101"}, '
            '"best_raw_score": 0.4}\n'
        ),
        encoding="utf-8",
    )

    summaries = existing_trial_summaries(tmp_path)

    assert summaries["promoted_control_active24_seed2101"]["best_raw_score"] == 0.4


def test_universal_scaling_command_is_fresh_architecture_run(tmp_path: Path) -> None:
    trial = {
        "name": "scale_medium_192x3_seed5102",
        "hidden_size": "192",
        "residual_blocks": "3",
        "heads": "6",
        "seed": "5102",
    }

    command = universal_scaling_command(trial, tmp_path / "checkpoints")

    assert "--resume-from-checkpoint" not in command
    assert command[command.index("--hidden-size") + 1] == "192"
    assert command[command.index("--residual-blocks") + 1] == "3"
    assert command[command.index("--heads") + 1] == "6"
    assert command[command.index("--batch-size") + 1] == "512"


def test_universal_scaling_smoke_disables_eval(tmp_path: Path) -> None:
    trial = {
        "name": "scale_large_256x4_seed5103",
        "hidden_size": "256",
        "residual_blocks": "4",
        "heads": "8",
        "seed": "5103",
    }

    command = universal_scaling_command(
        trial,
        tmp_path / "checkpoints",
        smoke=True,
    )

    assert command[command.index("--iterations") + 1] == "2"
    assert command[command.index("--training-steps") + 1] == "2"
    assert command[command.index("--eval-games") + 1] == "0"
    assert "--eval-opponents" not in command


def test_universal_scaling_large_uses_smaller_batch(tmp_path: Path) -> None:
    trial = {
        "name": "scale_large_256x4_seed5103",
        "hidden_size": "256",
        "residual_blocks": "4",
        "heads": "8",
        "seed": "5103",
        "batch_size": "256",
        "training_steps": "128",
    }

    command = universal_scaling_command(trial, tmp_path / "checkpoints")

    assert command[command.index("--batch-size") + 1] == "256"
    assert command[command.index("--training-steps") + 1] == "128"


def test_universal_scaling_selection_filters_trials() -> None:
    args = argparse.Namespace(
        only=["medium"],
        start_after=None,
        max_trials=None,
    )

    trials = selected_scaling_trials(args)

    assert [trial["name"] for trial in trials] == ["scale_medium_192x3_seed5102"]


def test_universal_staged_command_enables_line_rank_architecture(
    tmp_path: Path,
) -> None:
    trial = {"name": "line_rank_large_staged_seed6201", "seed": "6201"}
    stage = universal_staged_plan()[0]

    command = universal_staged_command(
        trial,
        stage,
        tmp_path / "checkpoints",
        end_iteration=48,
    )

    assert command[command.index("--hidden-size") + 1] == "256"
    assert command[command.index("--residual-blocks") + 1] == "4"
    assert command[command.index("--heads") + 1] == "8"
    assert "--line-features" in command
    assert "--input-layer-norm" in command
    assert "--rank-adapters" in command
    assert command[command.index("--adapter-size") + 1] == "64"
    assert "--resume-from-checkpoint" not in command
    assert command.count("cuda") == 1


def test_universal_staged_command_passes_exploration_teacher_and_residual(
    tmp_path: Path,
) -> None:
    trial = {
        "name": "teacher_anchor_residual_seed7002",
        "seed": "7002",
        "line_policy_residual": True,
        "root_dirichlet_alpha": "0.20",
        "root_exploration_fraction": "0.15",
        "root_temperature": "1.0",
        "root_temperature_move_cutoff": "24",
        "teacher_replay_path": "runs/teacher_replays/example.pt",
        "teacher_batch_fraction": "0.25",
    }
    stage = universal_staged_plan()[0]

    command = universal_staged_command(
        trial,
        stage,
        tmp_path / "checkpoints",
        end_iteration=48,
    )

    assert "--line-policy-residual" in command
    assert command[command.index("--root-dirichlet-alpha") + 1] == "0.20"
    assert command[command.index("--root-exploration-fraction") + 1] == "0.15"
    assert command[command.index("--root-temperature") + 1] == "1.0"
    assert command[command.index("--root-temperature-move-cutoff") + 1] == "24"
    assert command[command.index("--teacher-replay-path") + 1].endswith("example.pt")
    assert command[command.index("--teacher-batch-fraction") + 1] == "0.25"


def test_universal_staged_command_resumes_between_stages(tmp_path: Path) -> None:
    trial = {"name": "line_rank_large_staged_seed6201", "seed": "6201"}
    stage = universal_staged_plan()[1]
    checkpoint = tmp_path / "iteration_0048.pt"

    command = universal_staged_command(
        trial,
        stage,
        tmp_path / "checkpoints",
        end_iteration=96,
        resume_from_checkpoint=checkpoint,
    )

    assert command[command.index("--iterations") + 1] == "96"
    assert command[command.index("--resume-from-checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--variants-json") + 1].endswith(
        "universal_repair_4d_heavy_20260523.json"
    )


def test_universal_staged_plan_uses_cumulative_iterations() -> None:
    stages = universal_staged_plan()

    assert staged_end_iterations(stages) == [48, 96, 128]


def test_universal_staged_prefers_floor_passing_resume_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    latest = checkpoint_dir / "iteration_0048.pt"
    best = checkpoint_dir / "best_by_eval_score.pt"
    floor = checkpoint_dir / "best_current_run_floor_passing.pt"
    latest.write_text("latest", encoding="utf-8")
    best.write_text("best", encoding="utf-8")
    floor.write_text("floor", encoding="utf-8")

    assert preferred_resume_checkpoint(checkpoint_dir, 48) == floor
    floor.unlink()
    assert preferred_resume_checkpoint(checkpoint_dir, 48) == best
    best.unlink()
    assert preferred_resume_checkpoint(checkpoint_dir, 48) == latest


def test_universal_staged_smoke_uses_tiny_curriculum(tmp_path: Path) -> None:
    trial = {"name": "line_rank_large_staged_seed6201", "seed": "6201"}
    stage = universal_staged_plan(smoke=True)[0]

    command = universal_staged_command(
        trial,
        stage,
        tmp_path / "checkpoints",
        end_iteration=1,
        smoke=True,
    )

    assert command[command.index("--variants-json") + 1].endswith(
        "universal_smoke_20260525.json"
    )
    assert command[command.index("--simulations") + 1] == "2"
    assert command[command.index("--training-steps") + 1] == "1"
    assert command[command.index("--batch-size") + 1] == "64"
    assert command[command.index("--eval-games") + 1] == "0"


def test_universal_staged_adaptive_pulse_targets_failed_floor() -> None:
    pulse = adaptive_stage_for_metric(
        {
            "eval_floor_failures": [
                "4d_4x4x4x4_k4:heuristic=0.250<0.375",
                "default:tactical=0.375<0.500",
            ]
        },
        pulse_index=0,
    )

    assert pulse["name"] == "adaptive_4d_pressure_1"
    assert pulse["config"].endswith("universal_repair_4d_heavy_20260523.json")


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
