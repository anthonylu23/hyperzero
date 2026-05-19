from pathlib import Path

import pytest

from scripts.evaluate_checkpoint_series import select_checkpoints


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
