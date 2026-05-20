"""Training loop for one universal multi-dimensional Connect-K agent."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hyperzero.agents import (
    AlphaZeroAgent,
    HeuristicAgent,
    MCTSAgent,
    RandomAgent,
    TacticalAgent,
)
from hyperzero.eval import evaluate_matchup
from hyperzero.models import count_parameters
from hyperzero.models.universal_evaluator import UniversalEvaluator
from hyperzero.models.universal_transformer import (
    UniversalModelConfig,
    UniversalPolicyValueTransformer,
)
from hyperzero.search.puct import PUCTConfig
from hyperzero.training.checkpoint import resolve_device
from hyperzero.training.universal_replay import (
    UniversalReplayBuffer,
    UniversalSelfPlayExample,
)
from hyperzero.training.universal_self_play import (
    UniversalGameSpec,
    generate_universal_examples,
)
from hyperzero.universal.encoding import (
    UniversalBatch,
    UniversalEncoderConfig,
    collate_positions,
    encode_position,
)


@dataclass(frozen=True, slots=True)
class UniversalTrainingConfig:
    """Configuration for universal mixed-variant training."""

    game_specs: tuple[UniversalGameSpec, ...]
    iterations: int = 1
    puct_simulations: int = 8
    c_puct: float = 1.5
    replay_capacity: int = 20_000
    batch_size: int = 64
    training_steps_per_iteration: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    hidden_size: int = 128
    residual_blocks: int = 2
    heads: int = 4
    max_rank: int = 4
    max_board_extent: int = 8
    seed: int = 0
    device: str = "cpu"
    checkpoint_dir: str | Path | None = None
    checkpoint_keep_last: int | None = None
    resume_from_checkpoint: str | Path | None = None
    metrics_path: str | Path | None = None
    eval_games_per_variant: int = 0
    eval_opponents: tuple[str, ...] = ()
    eval_simulations: int = 8
    eval_mcts_simulations: int = 32
    eval_interval: int = 1
    eval_score_weights: dict[str, float] | None = None
    eval_score_floors: dict[str, Any] | None = None
    batched_self_play: bool = False
    max_active_self_play_games: int | None = None

    def __post_init__(self) -> None:
        if not self.game_specs:
            raise ValueError("game_specs must be nonempty")
        if len({spec.config_id for spec in self.game_specs}) != len(self.game_specs):
            raise ValueError("game_specs must have unique config_id values")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.puct_simulations <= 0:
            raise ValueError("puct_simulations must be positive")
        if self.c_puct < 0.0:
            raise ValueError("c_puct must be nonnegative")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.training_steps_per_iteration <= 0:
            raise ValueError("training_steps_per_iteration must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative")
        if self.value_weight < 0.0:
            raise ValueError("value_weight must be nonnegative")
        if self.checkpoint_keep_last is not None and self.checkpoint_keep_last <= 0:
            raise ValueError("checkpoint_keep_last must be positive when set")
        if self.eval_games_per_variant < 0:
            raise ValueError("eval_games_per_variant must be nonnegative")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.eval_simulations <= 0:
            raise ValueError("eval_simulations must be positive")
        if self.eval_mcts_simulations <= 0:
            raise ValueError("eval_mcts_simulations must be positive")
        if (
            self.max_active_self_play_games is not None
            and self.max_active_self_play_games <= 0
        ):
            raise ValueError("max_active_self_play_games must be positive when set")
        for opponent in self.eval_opponents:
            if opponent not in ("random", "tactical", "heuristic", "mcts"):
                raise ValueError(f"unknown eval opponent: {opponent}")


@dataclass(frozen=True, slots=True)
class UniversalTrainingMetrics:
    """Aggregate metrics for one universal training iteration."""

    iteration: int
    self_play_games: int
    self_play_examples: int
    self_play_games_by_config: dict[str, int]
    self_play_examples_by_config: dict[str, int]
    replay_size: int
    replay_size_by_config: dict[str, int]
    training_steps: int
    batch_size: int
    puct_simulations: int
    model_parameters: int
    eval_games_per_variant: int
    eval_opponents: tuple[str, ...]
    eval_simulations: int
    eval_mcts_simulations: int
    eval_interval: int
    eval_score: float | None
    best_eval_score: float | None
    is_best_checkpoint: bool
    best_checkpoint_path: str | None
    policy_loss: float
    policy_loss_min: float
    policy_loss_max: float
    value_loss: float
    value_loss_min: float
    value_loss_max: float
    total_loss: float
    total_loss_min: float
    total_loss_max: float
    average_game_length_by_config: dict[str, float]
    evaluations: dict[str, dict[str, dict[str, float | int]]]
    min_tactical_win_rate: float | None
    batched_self_play: bool
    max_active_self_play_games: int | None
    iteration_time_seconds: float
    total_training_time_seconds: float
    self_play_time_seconds: float
    training_step_time_seconds: float
    eval_time_seconds: float
    self_play_inference_time_seconds: float
    self_play_inference_batches: int
    self_play_inference_states: int
    eval_inference_time_seconds: float
    eval_inference_batches: int
    eval_inference_states: int
    checkpoint_time_seconds: float
    eval_floor_passed: bool | None = None
    eval_floor_failures: tuple[str, ...] = ()
    checkpoint_path: str | None = None


@dataclass(frozen=True, slots=True)
class UniversalTrainingResult:
    """Output from a universal training run."""

    model: nn.Module
    metrics: tuple[UniversalTrainingMetrics, ...]
    replay_size: int


def train_universal(
    config: UniversalTrainingConfig,
    *,
    model: nn.Module | None = None,
) -> UniversalTrainingResult:
    """Run universal mixed-game self-play and policy/value updates."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = resolve_device(config.device)
    model_config = _model_config(config)
    resume_payload: dict[str, Any] | None = None
    start_iteration = 1
    previous_metrics: list[UniversalTrainingMetrics] = []
    model = UniversalPolicyValueTransformer(model_config) if model is None else model
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    replay = UniversalReplayBuffer(config.replay_capacity, seed=config.seed)
    if config.resume_from_checkpoint is not None:
        resume_payload = _load_resume_payload(config.resume_from_checkpoint, device)
        _restore_training_state(
            resume_payload,
            config=config,
            model=model,
            optimizer=optimizer,
            replay=replay,
            rng=rng,
        )
        start_iteration = int(resume_payload["iteration"]) + 1
        previous_metrics = _deserialize_metrics(
            resume_payload.get("metrics", resume_payload.get("previous_metrics", []))
        )

    search_config = PUCTConfig(
        simulations=config.puct_simulations,
        c_puct=config.c_puct,
    )
    metrics: list[UniversalTrainingMetrics] = list(previous_metrics)
    model_parameters = count_parameters(model)
    best_eval_score: float | None = None
    best_checkpoint_path: str | None = None
    for previous_metric in metrics:
        if previous_metric.best_eval_score is not None:
            best_eval_score = previous_metric.best_eval_score
            best_checkpoint_path = previous_metric.best_checkpoint_path
    training_start = time.perf_counter()

    for iteration in range(start_iteration, config.iterations + 1):
        iteration_start = time.perf_counter()
        evaluator = UniversalEvaluator(model, model_config.encoder, device=device)
        self_play_start = time.perf_counter()
        examples, game_counts, average_lengths = _generate_iteration_examples(
            config,
            evaluator,
            search_config,
            rng,
        )
        self_play_time_seconds = time.perf_counter() - self_play_start
        replay.add_many(examples)

        losses = []
        training_step_time_seconds = 0.0
        for _ in range(config.training_steps_per_iteration):
            _synchronize_if_cuda(device)
            step_start = time.perf_counter()
            losses.append(
                _training_step(
                    model,
                    optimizer,
                    replay.sample(config.batch_size, balanced=True),
                    value_weight=config.value_weight,
                    model_config=model_config,
                    device=device,
                )
            )
            _synchronize_if_cuda(device)
            training_step_time_seconds += time.perf_counter() - step_start

        eval_start = time.perf_counter()
        evaluations, eval_time_stats = _evaluate_training_model(
            config,
            model,
            model_config=model_config,
            iteration=iteration,
            device=device,
        )
        eval_time_seconds = time.perf_counter() - eval_start
        loss_summary = _summarize_losses(losses)
        eval_score = _score_evaluations(config, evaluations)
        eval_floor_failures = _eval_floor_failures(config, evaluations)
        eval_floor_passed = None if eval_score is None else not eval_floor_failures
        min_tactical_win_rate = _min_tactical_win_rate(evaluations)
        checkpoint_path = _checkpoint_path(config, iteration)
        is_best_checkpoint = (
            checkpoint_path is not None
            and eval_score is not None
            and eval_floor_passed is not False
            and (best_eval_score is None or eval_score > best_eval_score)
        )
        if is_best_checkpoint:
            best_eval_score = eval_score
            best_checkpoint_path = str(
                Path(checkpoint_path).parent / "best_by_eval_score.pt"
            )

        examples_by_config = _count_examples(examples)
        metric = UniversalTrainingMetrics(
            iteration=iteration,
            self_play_games=sum(game_counts.values()),
            self_play_examples=len(examples),
            self_play_games_by_config=game_counts,
            self_play_examples_by_config=examples_by_config,
            replay_size=len(replay),
            replay_size_by_config=replay.counts_by_config(),
            training_steps=config.training_steps_per_iteration,
            batch_size=config.batch_size,
            puct_simulations=config.puct_simulations,
            model_parameters=model_parameters,
            eval_games_per_variant=config.eval_games_per_variant,
            eval_opponents=config.eval_opponents,
            eval_simulations=config.eval_simulations,
            eval_mcts_simulations=config.eval_mcts_simulations,
            eval_interval=config.eval_interval,
            eval_score=eval_score,
            best_eval_score=best_eval_score,
            is_best_checkpoint=is_best_checkpoint,
            best_checkpoint_path=best_checkpoint_path,
            policy_loss=loss_summary["policy_loss"]["mean"],
            policy_loss_min=loss_summary["policy_loss"]["min"],
            policy_loss_max=loss_summary["policy_loss"]["max"],
            value_loss=loss_summary["value_loss"]["mean"],
            value_loss_min=loss_summary["value_loss"]["min"],
            value_loss_max=loss_summary["value_loss"]["max"],
            total_loss=loss_summary["total_loss"]["mean"],
            total_loss_min=loss_summary["total_loss"]["min"],
            total_loss_max=loss_summary["total_loss"]["max"],
            average_game_length_by_config=average_lengths,
            evaluations=evaluations,
            min_tactical_win_rate=min_tactical_win_rate,
            batched_self_play=config.batched_self_play,
            max_active_self_play_games=config.max_active_self_play_games,
            iteration_time_seconds=time.perf_counter() - iteration_start,
            total_training_time_seconds=time.perf_counter() - training_start,
            self_play_time_seconds=self_play_time_seconds,
            training_step_time_seconds=training_step_time_seconds,
            eval_time_seconds=eval_time_seconds,
            self_play_inference_time_seconds=evaluator.inference_time_seconds,
            self_play_inference_batches=evaluator.inference_batches,
            self_play_inference_states=evaluator.inference_states,
            eval_inference_time_seconds=eval_time_stats[0],
            eval_inference_batches=eval_time_stats[1],
            eval_inference_states=eval_time_stats[2],
            checkpoint_time_seconds=0.0,
            eval_floor_passed=eval_floor_passed,
            eval_floor_failures=eval_floor_failures,
            checkpoint_path=checkpoint_path,
        )
        checkpoint_start = time.perf_counter()
        saved_checkpoint_path = _save_checkpoint(
            config,
            model_config,
            model,
            optimizer,
            replay,
            rng=rng,
            iteration=iteration,
            metrics=[*metrics, metric],
        )
        if saved_checkpoint_path is not None and is_best_checkpoint:
            best_checkpoint_path = _copy_best_checkpoint(saved_checkpoint_path)
        if saved_checkpoint_path is not None:
            _prune_checkpoints(
                Path(saved_checkpoint_path).parent,
                keep_last=config.checkpoint_keep_last,
            )
        checkpoint_time_seconds = time.perf_counter() - checkpoint_start
        metric = _replace_metric_checkpoint_time(metric, checkpoint_time_seconds)
        if saved_checkpoint_path is not None:
            _sync_checkpoint_metric(saved_checkpoint_path, metric)
            if is_best_checkpoint and best_checkpoint_path is not None:
                _sync_checkpoint_metric(best_checkpoint_path, metric)
        metrics.append(metric)
        _append_metrics_jsonl(config, metric)

    return UniversalTrainingResult(
        model=model,
        metrics=tuple(metrics[len(previous_metrics) :]),
        replay_size=len(replay),
    )


def _model_config(config: UniversalTrainingConfig) -> UniversalModelConfig:
    return UniversalModelConfig(
        hidden_size=config.hidden_size,
        residual_blocks=config.residual_blocks,
        heads=config.heads,
        encoder=UniversalEncoderConfig(
            max_rank=config.max_rank,
            max_board_extent=config.max_board_extent,
        ),
    )


def _generate_iteration_examples(
    config: UniversalTrainingConfig,
    evaluator: UniversalEvaluator,
    search_config: PUCTConfig,
    rng: np.random.Generator,
) -> tuple[list[UniversalSelfPlayExample], dict[str, int], dict[str, float]]:
    examples: list[UniversalSelfPlayExample] = []
    game_counts: dict[str, int] = {}
    average_lengths: dict[str, float] = {}
    for spec in config.game_specs:
        spec_rng = np.random.default_rng(int(rng.integers(2**63 - 1)))
        spec_examples, games, average_length = generate_universal_examples(
            spec,
            evaluator,
            search_config=search_config,
            rng=spec_rng,
            batched_self_play=config.batched_self_play,
            max_active_games=config.max_active_self_play_games,
        )
        examples.extend(spec_examples)
        game_counts[spec.config_id] = games
        average_lengths[spec.config_id] = average_length
    return examples, game_counts, average_lengths


def _training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    examples: list[UniversalSelfPlayExample],
    *,
    value_weight: float,
    model_config: UniversalModelConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    batch, policies, values, legal_masks = _batch_tensors(
        examples,
        model_config=model_config,
        device=device,
    )
    _validate_policy_targets(policies, legal_masks)
    optimizer.zero_grad(set_to_none=True)
    policy_logits, predicted_values = model(batch)
    masked_logits = policy_logits.masked_fill(~legal_masks, -1e9)
    log_policy = F.log_softmax(masked_logits, dim=1)
    policy_loss = -(policies * log_policy).sum(dim=1).mean()
    value_loss = F.mse_loss(predicted_values, values)
    total_loss = policy_loss + value_weight * value_loss
    total_loss.backward()
    optimizer.step()
    return (
        float(policy_loss.detach().cpu().item()),
        float(value_loss.detach().cpu().item()),
        float(total_loss.detach().cpu().item()),
    )


def _batch_tensors(
    examples: list[UniversalSelfPlayExample],
    *,
    model_config: UniversalModelConfig,
    device: torch.device,
) -> tuple[UniversalBatch, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = [
        encode_position(
            example.game_config,
            board=example.board,
            legal_mask=example.legal_mask,
            ply=example.ply,
            encoder_config=model_config.encoder,
        )
        for example in examples
    ]
    batch = collate_positions(positions, device=device)
    max_actions = batch.action_mask.shape[1]
    policies_np = np.zeros((len(examples), max_actions), dtype=np.float32)
    legal_np = np.zeros((len(examples), max_actions), dtype=bool)
    for index, example in enumerate(examples):
        action_count = example.game_config.num_actions
        policies_np[index, :action_count] = example.policy
        legal_np[index, :action_count] = example.legal_mask
    policies = torch.as_tensor(policies_np, dtype=torch.float32, device=device)
    values = torch.as_tensor(
        [example.value for example in examples],
        dtype=torch.float32,
        device=device,
    )
    legal_masks = torch.as_tensor(legal_np, dtype=torch.bool, device=device)
    return batch, policies, values, legal_masks


def _validate_policy_targets(
    policies: torch.Tensor,
    legal_masks: torch.Tensor,
) -> None:
    illegal_mass = policies.masked_select(~legal_masks)
    if illegal_mass.numel() > 0 and not torch.allclose(
        illegal_mass,
        torch.zeros_like(illegal_mass),
        atol=1e-6,
    ):
        raise ValueError("policy targets must assign zero mass to illegal actions")
    policy_sums = policies.sum(dim=1)
    if not torch.allclose(policy_sums, torch.ones_like(policy_sums), atol=1e-5):
        raise ValueError("policy targets must sum to 1 for each example")


def _summarize_losses(
    losses: list[tuple[float, float, float]],
) -> dict[str, dict[str, float]]:
    names = ("policy_loss", "value_loss", "total_loss")
    values_by_name = {
        name: np.asarray([loss[index] for loss in losses], dtype=np.float64)
        for index, name in enumerate(names)
    }
    return {
        name: {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        for name, values in values_by_name.items()
    }


def _evaluate_training_model(
    config: UniversalTrainingConfig,
    model: nn.Module,
    *,
    model_config: UniversalModelConfig,
    iteration: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, dict[str, float | int]]], tuple[float, int, int]]:
    if (
        config.eval_games_per_variant == 0
        or not config.eval_opponents
        or iteration % config.eval_interval != 0
    ):
        return {}, (0.0, 0, 0)

    results: dict[str, dict[str, dict[str, float | int]]] = {}
    inference_time_seconds = 0.0
    inference_batches = 0
    inference_states = 0
    for spec_index, spec in enumerate(config.game_specs):
        spec_results: dict[str, dict[str, float | int]] = {}
        for opponent_index, opponent_name in enumerate(config.eval_opponents):
            seed_offset = iteration * 1000 + spec_index * 100 + opponent_index
            evaluator = UniversalEvaluator(model, model_config.encoder, device=device)
            agent = AlphaZeroAgent(
                evaluator,
                simulations=config.eval_simulations,
                c_puct=config.c_puct,
                seed=config.seed + 30_000 + seed_offset,
                name=f"universal-iteration-{iteration:04d}",
            )
            opponent = _build_baseline_agent(
                opponent_name,
                seed=config.seed + 40_000 + seed_offset,
                mcts_simulations=config.eval_mcts_simulations,
            )
            stats = evaluate_matchup(
                spec.game_config,
                agent,
                opponent,
                games=config.eval_games_per_variant,
            )
            spec_results[opponent_name] = stats.to_dict()
            inference_time_seconds += evaluator.inference_time_seconds
            inference_batches += evaluator.inference_batches
            inference_states += evaluator.inference_states
        results[spec.config_id] = spec_results
    return results, (inference_time_seconds, inference_batches, inference_states)


def _score_evaluations(
    config: UniversalTrainingConfig,
    evaluations: dict[str, dict[str, dict[str, float | int]]],
) -> float | None:
    if not evaluations:
        return None
    weights = config.eval_score_weights or {
        "heuristic": 0.35,
        "tactical": 0.30,
        "mcts": 0.20,
        "random": 0.15,
    }
    per_variant_scores = []
    for spec in config.game_specs:
        spec_results = evaluations.get(spec.config_id, {})
        total_weight = 0.0
        score = 0.0
        for opponent, weight in weights.items():
            stats = spec_results.get(opponent)
            if stats is None:
                continue
            score += float(weight) * float(stats["agent_a_win_rate"])
            total_weight += float(weight)
        if total_weight > 0.0:
            per_variant_scores.append(float(score / total_weight))
    if not per_variant_scores:
        return None
    return float(0.5 * np.mean(per_variant_scores) + 0.5 * min(per_variant_scores))


def _min_tactical_win_rate(
    evaluations: dict[str, dict[str, dict[str, float | int]]],
) -> float | None:
    values = [
        float(results["tactical"]["agent_a_win_rate"])
        for results in evaluations.values()
        if "tactical" in results
    ]
    return None if not values else float(min(values))


def _eval_floor_failures(
    config: UniversalTrainingConfig,
    evaluations: dict[str, dict[str, dict[str, float | int]]],
) -> tuple[str, ...]:
    if not evaluations or not config.eval_score_floors:
        return ()
    failures: list[str] = []
    for spec in config.game_specs:
        thresholds = _floor_thresholds_for_variant(
            config.eval_score_floors,
            spec.config_id,
        )
        if not thresholds:
            continue
        variant_results = evaluations.get(spec.config_id, {})
        for opponent, threshold in thresholds.items():
            stats = variant_results.get(opponent)
            if stats is None:
                failures.append(f"{spec.config_id}:{opponent}=missing<{threshold:.3f}")
                continue
            win_rate = float(stats["agent_a_win_rate"])
            if win_rate < threshold:
                failures.append(
                    f"{spec.config_id}:{opponent}={win_rate:.3f}<{threshold:.3f}"
                )
    return tuple(failures)


def _floor_thresholds_for_variant(
    floors: dict[str, Any],
    config_id: str,
) -> dict[str, float]:
    default_thresholds: dict[str, float] = {}
    variant_thresholds: dict[str, float] = {}
    for key, value in floors.items():
        if isinstance(value, dict):
            if key == "default":
                default_thresholds.update(_coerce_floor_mapping(value, key))
            elif key == config_id:
                variant_thresholds.update(_coerce_floor_mapping(value, key))
        elif key in ("random", "tactical", "heuristic", "mcts"):
            default_thresholds[key] = float(value)
    return {**default_thresholds, **variant_thresholds}


def _coerce_floor_mapping(data: dict[str, Any], label: str) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for opponent, value in data.items():
        if opponent not in ("random", "tactical", "heuristic", "mcts"):
            raise ValueError(f"unknown eval floor opponent for {label}: {opponent}")
        threshold = float(value)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"eval floor for {label}:{opponent} must be in [0, 1]")
        thresholds[opponent] = threshold
    return thresholds


def _count_examples(examples: list[UniversalSelfPlayExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.config_id] = counts.get(example.config_id, 0) + 1
    return counts


def _checkpoint_path(config: UniversalTrainingConfig, iteration: int) -> str | None:
    if config.checkpoint_dir is None:
        return None
    return str(Path(config.checkpoint_dir) / f"iteration_{iteration:04d}.pt")


def _save_checkpoint(
    config: UniversalTrainingConfig,
    model_config: UniversalModelConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: UniversalReplayBuffer,
    *,
    rng: np.random.Generator,
    iteration: int,
    metrics: list[UniversalTrainingMetrics],
) -> str | None:
    path = _checkpoint_path(config, iteration)
    if path is None:
        return None
    checkpoint_dir = Path(path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": 2,
            "iteration": iteration,
            "global_step": iteration * config.training_steps_per_iteration,
            "encoding_version": "universal-coordinate-v1",
            "game_specs": [spec.to_dict() for spec in config.game_specs],
            "universal_model_config": model_config.to_dict(),
            "training_config": _serializable_training_config(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "replay_buffer": replay.state_dict(),
            "torch_rng_state": torch.random.get_rng_state(),
            "numpy_rng_state": rng.bit_generator.state,
            "metrics": [asdict(metric) for metric in metrics],
            "previous_metrics": [asdict(metric) for metric in metrics[:-1]],
        },
        path,
    )
    return str(path)


def _load_resume_payload(
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    return torch.load(Path(checkpoint_path), map_location=device, weights_only=False)


def _restore_training_state(
    payload: dict[str, Any],
    *,
    config: UniversalTrainingConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: UniversalReplayBuffer,
    rng: np.random.Generator,
) -> None:
    checkpoint_specs = [
        UniversalGameSpec.from_dict(spec) for spec in payload["game_specs"]
    ]
    _validate_resume_game_specs(checkpoint_specs, config.game_specs)
    checkpoint_training_config = dict(payload.get("training_config", {}))
    for key in (
        "hidden_size",
        "residual_blocks",
        "heads",
        "max_rank",
        "max_board_extent",
    ):
        expected = getattr(config, key)
        actual = checkpoint_training_config.get(key)
        if actual is not None and actual != expected:
            raise ValueError(
                f"resume checkpoint {key}={actual!r} does not match config "
                f"{expected!r}"
            )
    if int(payload["iteration"]) >= config.iterations:
        raise ValueError("resume checkpoint iteration is already at or past iterations")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    replay_payload = payload.get("replay_buffer")
    if replay_payload is None:
        raise ValueError("resume checkpoint does not include replay_buffer state")
    replay.load_state_dict(replay_payload)
    torch_rng_state = payload.get("torch_rng_state")
    if torch_rng_state is not None:
        torch.random.set_rng_state(torch_rng_state.cpu())
    numpy_rng_state = payload.get("numpy_rng_state")
    if numpy_rng_state is not None:
        rng.bit_generator.state = numpy_rng_state


def _validate_resume_game_specs(
    checkpoint_specs: list[UniversalGameSpec],
    config_specs: tuple[UniversalGameSpec, ...],
) -> None:
    """Validate resume compatibility while allowing curriculum count changes."""
    checkpoint_identity = [
        (spec.config_id, spec.game_config.to_dict()) for spec in checkpoint_specs
    ]
    config_identity = [
        (spec.config_id, spec.game_config.to_dict()) for spec in config_specs
    ]
    if checkpoint_identity != config_identity:
        raise ValueError(
            "resume checkpoint game variants do not match config; "
            "config_id, order, shape, connect_k, and gravity_axis must match"
        )


def _deserialize_metrics(rows: list[dict[str, Any]]) -> list[UniversalTrainingMetrics]:
    return [UniversalTrainingMetrics(**row) for row in rows]


def _copy_best_checkpoint(checkpoint_path: str) -> str:
    source = Path(checkpoint_path)
    target = source.parent / "best_by_eval_score.pt"
    shutil.copy2(source, target)
    return str(target)


def _sync_checkpoint_metric(
    checkpoint_path: str,
    metric: UniversalTrainingMetrics,
) -> None:
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metric_payload = asdict(metric)
    metrics = list(payload.get("metrics", []))
    if metrics:
        metrics[-1] = metric_payload
        payload["metrics"] = metrics
    torch.save(payload, path)


def _prune_checkpoints(checkpoint_dir: Path, *, keep_last: int | None) -> None:
    if keep_last is None:
        return
    checkpoints = sorted(checkpoint_dir.glob("iteration_*.pt"))
    for checkpoint_path in checkpoints[:-keep_last]:
        checkpoint_path.unlink(missing_ok=True)


def _replace_metric_checkpoint_time(
    metric: UniversalTrainingMetrics,
    checkpoint_time_seconds: float,
) -> UniversalTrainingMetrics:
    return replace(metric, checkpoint_time_seconds=checkpoint_time_seconds)


def _metrics_path(config: UniversalTrainingConfig) -> Path | None:
    if config.metrics_path is not None:
        return Path(config.metrics_path)
    if config.checkpoint_dir is not None:
        return Path(config.checkpoint_dir) / "metrics.jsonl"
    return None


def _append_metrics_jsonl(
    config: UniversalTrainingConfig,
    metric: UniversalTrainingMetrics,
) -> None:
    path = _metrics_path(config)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(metric), sort_keys=True) + "\n")


def _serializable_training_config(config: UniversalTrainingConfig) -> dict[str, object]:
    return {
        "game_specs": [spec.to_dict() for spec in config.game_specs],
        "iterations": config.iterations,
        "puct_simulations": config.puct_simulations,
        "c_puct": config.c_puct,
        "replay_capacity": config.replay_capacity,
        "batch_size": config.batch_size,
        "training_steps_per_iteration": config.training_steps_per_iteration,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "value_weight": config.value_weight,
        "hidden_size": config.hidden_size,
        "residual_blocks": config.residual_blocks,
        "heads": config.heads,
        "max_rank": config.max_rank,
        "max_board_extent": config.max_board_extent,
        "seed": config.seed,
        "device": config.device,
        "checkpoint_dir": (
            None if config.checkpoint_dir is None else str(config.checkpoint_dir)
        ),
        "checkpoint_keep_last": config.checkpoint_keep_last,
        "resume_from_checkpoint": (
            None
            if config.resume_from_checkpoint is None
            else str(config.resume_from_checkpoint)
        ),
        "metrics_path": (
            None if config.metrics_path is None else str(config.metrics_path)
        ),
        "eval_games_per_variant": config.eval_games_per_variant,
        "eval_opponents": config.eval_opponents,
        "eval_simulations": config.eval_simulations,
        "eval_mcts_simulations": config.eval_mcts_simulations,
        "eval_interval": config.eval_interval,
        "eval_score_weights": config.eval_score_weights,
        "eval_score_floors": config.eval_score_floors,
        "batched_self_play": config.batched_self_play,
        "max_active_self_play_games": config.max_active_self_play_games,
    }


def _build_baseline_agent(name: str, *, seed: int, mcts_simulations: int):
    if name == "random":
        return RandomAgent(seed=seed, name=f"random-{seed}")
    if name == "tactical":
        return TacticalAgent(seed=seed, name=f"tactical-{seed}")
    if name == "heuristic":
        return HeuristicAgent(seed=seed, name=f"heuristic-{seed}")
    if name == "mcts":
        return MCTSAgent(
            simulations=mcts_simulations,
            seed=seed,
            name=f"mcts-{mcts_simulations}-{seed}",
        )
    raise ValueError(f"unknown baseline agent: {name}")


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
