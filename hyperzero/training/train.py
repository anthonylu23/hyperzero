"""V1 training loop for the minimal neural AlphaZero agent."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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
from hyperzero.game.config import GameConfig
from hyperzero.game.symmetry import gravity_preserving_symmetries
from hyperzero.models import (
    NeuralEvaluator,
    build_policy_value_model,
    count_parameters,
)
from hyperzero.search.puct import PUCTConfig
from hyperzero.training.checkpoint import resolve_device
from hyperzero.training.replay_buffer import ReplayBuffer
from hyperzero.training.self_play import (
    SelfPlayExample,
    generate_game,
    generate_games_batched,
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Small v1 training configuration for end-to-end smoke runs."""

    game_config: GameConfig
    iterations: int = 1
    self_play_games_per_iteration: int = 2
    puct_simulations: int = 8
    c_puct: float = 1.5
    replay_capacity: int = 2_000
    batch_size: int = 32
    training_steps_per_iteration: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    hidden_size: int = 64
    residual_blocks: int = 1
    model_type: str = "mlp"
    symmetry_augmentation: str = "none"
    eval_score_weights: dict[str, float] | None = None
    seed: int = 0
    device: str = "cpu"
    checkpoint_dir: str | Path | None = None
    checkpoint_keep_last: int | None = None
    metrics_path: str | Path | None = None
    eval_games_per_iteration: int = 0
    eval_opponents: tuple[str, ...] = ()
    eval_simulations: int = 8
    eval_mcts_simulations: int = 32
    eval_interval: int = 1
    batched_self_play: bool = False
    max_active_self_play_games: int | None = None

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.self_play_games_per_iteration <= 0:
            raise ValueError("self_play_games_per_iteration must be positive")
        if self.puct_simulations <= 0:
            raise ValueError("puct_simulations must be positive")
        if self.c_puct < 0.0:
            raise ValueError("c_puct must be nonnegative")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive")
        if self.checkpoint_keep_last is not None and self.checkpoint_keep_last <= 0:
            raise ValueError("checkpoint_keep_last must be positive when set")
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
        if self.model_type not in ("mlp", "line_mlp", "cnn", "resnet", "transformer"):
            raise ValueError(f"unknown model_type: {self.model_type}")
        if self.symmetry_augmentation not in ("none", "random"):
            raise ValueError("symmetry_augmentation must be 'none' or 'random'")
        if self.eval_games_per_iteration < 0:
            raise ValueError("eval_games_per_iteration must be nonnegative")
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
class TrainingMetrics:
    """Aggregate metrics for one v1 training iteration."""

    iteration: int
    self_play_games: int
    self_play_examples: int
    replay_size: int
    training_steps: int
    batch_size: int
    puct_simulations: int
    eval_games: int
    eval_opponents: tuple[str, ...]
    eval_simulations: int
    eval_mcts_simulations: int
    eval_interval: int
    replay_capacity: int
    model_type: str
    model_parameters: int
    symmetry_augmentation: str
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
    average_game_length: float
    evaluations: dict[str, dict[str, float | int]]
    batched_self_play: bool
    max_active_self_play_games: int | None
    iteration_time_seconds: float
    total_training_time_seconds: float
    self_play_time_seconds: float
    self_play_time_per_game_seconds: float
    self_play_time_per_example_seconds: float
    self_play_inference_time_seconds: float
    self_play_inference_batches: int
    self_play_inference_states: int
    self_play_inference_time_per_state_seconds: float
    training_step_time_seconds: float
    training_time_per_step_seconds: float
    eval_time_seconds: float
    eval_inference_time_seconds: float
    eval_inference_batches: int
    eval_inference_states: int
    eval_inference_time_per_state_seconds: float
    total_inference_time_seconds: float
    total_inference_batches: int
    total_inference_states: int
    total_inference_time_per_state_seconds: float
    checkpoint_time_seconds: float
    checkpoint_path: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Output from a v1 training run."""

    model: nn.Module
    metrics: tuple[TrainingMetrics, ...]
    replay_size: int


def train_v1(
    config: TrainingConfig,
    *,
    model: nn.Module | None = None,
) -> TrainingResult:
    """Run a small self-play plus supervised update loop."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = resolve_device(config.device)
    model = (
        build_policy_value_model(
            config.game_config,
            model_type=config.model_type,
            hidden_size=config.hidden_size,
            residual_blocks=config.residual_blocks,
        )
        if model is None
        else model
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    replay = ReplayBuffer(config.replay_capacity, seed=config.seed)
    search_config = PUCTConfig(
        simulations=config.puct_simulations,
        c_puct=config.c_puct,
    )
    metrics: list[TrainingMetrics] = []
    model_parameters = count_parameters(model)
    augment_rng = np.random.default_rng(config.seed + 100_000)
    best_eval_score: float | None = None
    best_checkpoint_path: str | None = None
    training_start = time.perf_counter()

    for iteration in range(1, config.iterations + 1):
        iteration_start = time.perf_counter()
        evaluator = NeuralEvaluator(model, device=device)
        self_play_start = time.perf_counter()
        games = _generate_self_play_games(config, evaluator, search_config, rng)
        self_play_time_seconds = time.perf_counter() - self_play_start
        example_count = sum(len(game.examples) for game in games)
        for game in games:
            replay.add_many(game.examples)

        losses = []
        training_step_time_seconds = 0.0
        for _ in range(config.training_steps_per_iteration):
            _synchronize_if_cuda(device)
            step_start = time.perf_counter()
            losses.append(
                _training_step(
                    model,
                    optimizer,
                    replay.sample(config.batch_size),
                    value_weight=config.value_weight,
                    device=device,
                    config=config,
                    rng=augment_rng,
                )
            )
            _synchronize_if_cuda(device)
            training_step_time_seconds += time.perf_counter() - step_start
        eval_start = time.perf_counter()
        (
            evaluations,
            eval_inference_time_seconds,
            eval_inference_batches,
            eval_inference_states,
        ) = _evaluate_training_model(
            config,
            model,
            iteration=iteration,
            device=device,
        )
        eval_time_seconds = time.perf_counter() - eval_start
        loss_summary = _summarize_losses(losses)
        checkpoint_start = time.perf_counter()
        checkpoint_path = _save_checkpoint(config, model, optimizer, iteration, metrics)
        eval_score = _score_evaluations(config, evaluations)
        is_best_checkpoint = (
            checkpoint_path is not None
            and eval_score is not None
            and (best_eval_score is None or eval_score > best_eval_score)
        )
        if is_best_checkpoint:
            best_eval_score = eval_score
            best_checkpoint_path = _copy_best_checkpoint(checkpoint_path)
        if checkpoint_path is not None:
            _prune_checkpoints(
                Path(checkpoint_path).parent,
                keep_last=config.checkpoint_keep_last,
            )
        checkpoint_time_seconds = time.perf_counter() - checkpoint_start
        iteration_time_seconds = time.perf_counter() - iteration_start
        total_training_time_seconds = time.perf_counter() - training_start
        self_play_inference_time_seconds = evaluator.inference_time_seconds
        total_inference_time_seconds = (
            self_play_inference_time_seconds + eval_inference_time_seconds
        )
        total_inference_batches = evaluator.inference_batches + eval_inference_batches
        total_inference_states = evaluator.inference_states + eval_inference_states
        metric = TrainingMetrics(
            iteration=iteration,
            self_play_games=len(games),
            self_play_examples=example_count,
            replay_size=len(replay),
            training_steps=config.training_steps_per_iteration,
            batch_size=config.batch_size,
            puct_simulations=config.puct_simulations,
            eval_games=config.eval_games_per_iteration,
            eval_opponents=config.eval_opponents,
            eval_simulations=config.eval_simulations,
            eval_mcts_simulations=config.eval_mcts_simulations,
            eval_interval=config.eval_interval,
            replay_capacity=config.replay_capacity,
            model_type=config.model_type,
            model_parameters=model_parameters,
            symmetry_augmentation=config.symmetry_augmentation,
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
            average_game_length=float(np.mean([len(game.actions) for game in games])),
            evaluations=evaluations,
            batched_self_play=config.batched_self_play,
            max_active_self_play_games=config.max_active_self_play_games,
            iteration_time_seconds=iteration_time_seconds,
            total_training_time_seconds=total_training_time_seconds,
            self_play_time_seconds=self_play_time_seconds,
            self_play_time_per_game_seconds=_safe_rate(
                self_play_time_seconds,
                len(games),
            ),
            self_play_time_per_example_seconds=_safe_rate(
                self_play_time_seconds,
                example_count,
            ),
            self_play_inference_time_seconds=self_play_inference_time_seconds,
            self_play_inference_batches=evaluator.inference_batches,
            self_play_inference_states=evaluator.inference_states,
            self_play_inference_time_per_state_seconds=_safe_rate(
                self_play_inference_time_seconds,
                evaluator.inference_states,
            ),
            training_step_time_seconds=training_step_time_seconds,
            training_time_per_step_seconds=_safe_rate(
                training_step_time_seconds,
                config.training_steps_per_iteration,
            ),
            eval_time_seconds=eval_time_seconds,
            eval_inference_time_seconds=eval_inference_time_seconds,
            eval_inference_batches=eval_inference_batches,
            eval_inference_states=eval_inference_states,
            eval_inference_time_per_state_seconds=_safe_rate(
                eval_inference_time_seconds,
                eval_inference_states,
            ),
            total_inference_time_seconds=total_inference_time_seconds,
            total_inference_batches=total_inference_batches,
            total_inference_states=total_inference_states,
            total_inference_time_per_state_seconds=_safe_rate(
                total_inference_time_seconds,
                total_inference_states,
            ),
            checkpoint_time_seconds=checkpoint_time_seconds,
            checkpoint_path=checkpoint_path,
        )
        metrics.append(metric)
        _append_metrics_jsonl(config, metric)

    return TrainingResult(model=model, metrics=tuple(metrics), replay_size=len(replay))


def _training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    examples: list[SelfPlayExample],
    *,
    value_weight: float,
    device: torch.device,
    config: TrainingConfig,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    model.train()
    boards, policies, values, legal_masks = _batch_tensors(
        examples,
        device,
        config=config,
        rng=rng,
    )
    _validate_policy_targets(policies, legal_masks)
    optimizer.zero_grad(set_to_none=True)
    policy_logits, predicted_values = model(boards)
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


def _safe_rate(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return float(total / count)


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def _batch_tensors(
    examples: list[SelfPlayExample],
    device: torch.device,
    *,
    config: TrainingConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        config is not None
        and config.symmetry_augmentation == "random"
        and rng is not None
    ):
        examples = _augment_examples_with_random_symmetry(examples, config, rng)
    boards = torch.as_tensor(
        np.stack([example.board for example in examples]),
        dtype=torch.float32,
        device=device,
    )
    policies = torch.as_tensor(
        np.stack([example.policy for example in examples]),
        dtype=torch.float32,
        device=device,
    )
    values = torch.as_tensor(
        [example.value for example in examples],
        dtype=torch.float32,
        device=device,
    )
    legal_masks = torch.as_tensor(
        np.stack([example.legal_mask for example in examples]),
        dtype=torch.bool,
        device=device,
    )
    return boards, policies, values, legal_masks


def _augment_examples_with_random_symmetry(
    examples: list[SelfPlayExample],
    config: TrainingConfig,
    rng: np.random.Generator,
) -> list[SelfPlayExample]:
    symmetries = gravity_preserving_symmetries(config.game_config)
    if len(symmetries) <= 1:
        return examples
    augmented = []
    for example in examples:
        symmetry = symmetries[int(rng.integers(len(symmetries)))]
        augmented.append(
            SelfPlayExample(
                board=symmetry.transform_board(example.board).astype(np.float32),
                policy=symmetry.transform_policy(example.policy).astype(np.float32),
                value=example.value,
                legal_mask=symmetry.transform_policy(example.legal_mask).astype(bool),
                player_to_move=example.player_to_move,
                ply=example.ply,
            )
        )
    return augmented


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
    if not torch.allclose(
        policy_sums,
        torch.ones_like(policy_sums),
        atol=1e-5,
    ):
        raise ValueError("policy targets must sum to 1 for each example")


def _save_checkpoint(
    config: TrainingConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    previous_metrics: list[TrainingMetrics],
) -> str | None:
    if config.checkpoint_dir is None:
        return None

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"iteration_{iteration:04d}.pt"
    torch.save(
        {
            "checkpoint_version": 1,
            "iteration": iteration,
            "global_step": iteration * config.training_steps_per_iteration,
            "encoding_version": "canonical-flat-v1",
            "game_config": config.game_config.to_dict(),
            "training_config": _serializable_training_config(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.random.get_rng_state(),
            "previous_metrics": [asdict(metric) for metric in previous_metrics],
        },
        path,
    )
    return str(path)


def _copy_best_checkpoint(checkpoint_path: str) -> str:
    source = Path(checkpoint_path)
    target = source.parent / "best_by_eval_score.pt"
    shutil.copy2(source, target)
    return str(target)


def _prune_checkpoints(checkpoint_dir: Path, *, keep_last: int | None) -> None:
    if keep_last is None:
        return
    checkpoints = sorted(checkpoint_dir.glob("iteration_*.pt"))
    for checkpoint_path in checkpoints[:-keep_last]:
        checkpoint_path.unlink(missing_ok=True)


def _score_evaluations(
    config: TrainingConfig,
    evaluations: dict[str, dict[str, float | int]],
) -> float | None:
    if not evaluations:
        return None
    weights = config.eval_score_weights or {
        "heuristic": 0.35,
        "tactical": 0.30,
        "mcts": 0.20,
        "random": 0.15,
    }
    total_weight = 0.0
    score = 0.0
    for opponent, weight in weights.items():
        stats = evaluations.get(opponent)
        if stats is None:
            continue
        score += float(weight) * float(stats["agent_a_win_rate"])
        total_weight += float(weight)
    if total_weight == 0.0:
        return None
    return float(score / total_weight)


def _serializable_training_config(config: TrainingConfig) -> dict[str, object]:
    return {
        "game_config": config.game_config.to_dict(),
        "iterations": config.iterations,
        "self_play_games_per_iteration": config.self_play_games_per_iteration,
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
        "model_type": config.model_type,
        "symmetry_augmentation": config.symmetry_augmentation,
        "eval_score_weights": config.eval_score_weights,
        "seed": config.seed,
        "device": config.device,
        "checkpoint_dir": (
            None if config.checkpoint_dir is None else str(config.checkpoint_dir)
        ),
        "metrics_path": (
            None if config.metrics_path is None else str(config.metrics_path)
        ),
        "eval_games_per_iteration": config.eval_games_per_iteration,
        "eval_opponents": config.eval_opponents,
        "eval_simulations": config.eval_simulations,
        "eval_mcts_simulations": config.eval_mcts_simulations,
        "eval_interval": config.eval_interval,
        "batched_self_play": config.batched_self_play,
        "max_active_self_play_games": config.max_active_self_play_games,
    }


def _generate_self_play_games(
    config: TrainingConfig,
    evaluator: NeuralEvaluator,
    search_config: PUCTConfig,
    rng: np.random.Generator,
):
    if config.batched_self_play:
        return generate_games_batched(
            config.game_config,
            evaluator,
            games=config.self_play_games_per_iteration,
            search_config=search_config,
            rng=rng,
            max_active_games=config.max_active_self_play_games,
        )
    return tuple(
        generate_game(
            config.game_config,
            evaluator,
            search_config=search_config,
            rng=rng,
        )
        for _ in range(config.self_play_games_per_iteration)
    )


def _metrics_path(config: TrainingConfig) -> Path | None:
    if config.metrics_path is not None:
        return Path(config.metrics_path)
    if config.checkpoint_dir is not None:
        return Path(config.checkpoint_dir) / "metrics.jsonl"
    return None


def _append_metrics_jsonl(config: TrainingConfig, metric: TrainingMetrics) -> None:
    path = _metrics_path(config)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(metric), sort_keys=True) + "\n")


def _evaluate_training_model(
    config: TrainingConfig,
    model: nn.Module,
    *,
    iteration: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, float | int]], float, int, int]:
    if (
        config.eval_games_per_iteration == 0
        or not config.eval_opponents
        or iteration % config.eval_interval != 0
    ):
        return {}, 0.0, 0, 0

    results: dict[str, dict[str, float | int]] = {}
    inference_time_seconds = 0.0
    inference_batches = 0
    inference_states = 0
    for index, opponent_name in enumerate(config.eval_opponents):
        evaluator = NeuralEvaluator(model, device=device)
        agent = AlphaZeroAgent(
            evaluator,
            simulations=config.eval_simulations,
            c_puct=config.c_puct,
            seed=config.seed + 20_000 + iteration * 100 + index,
            name=f"train-v1-iteration-{iteration:04d}",
        )
        opponent = _build_baseline_agent(
            opponent_name,
            seed=config.seed + 10_000 + iteration * 100 + index,
            mcts_simulations=config.eval_mcts_simulations,
        )
        stats = evaluate_matchup(
            config.game_config,
            agent,
            opponent,
            games=config.eval_games_per_iteration,
        )
        results[opponent_name] = stats.to_dict()
        inference_time_seconds += evaluator.inference_time_seconds
        inference_batches += evaluator.inference_batches
        inference_states += evaluator.inference_states
    return results, inference_time_seconds, inference_batches, inference_states


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
