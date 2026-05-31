import pytest
from fastapi.testclient import TestClient

from hyperzero.server.agent_service import DEFAULT_CHECKPOINT
from hyperzero.server.modes import build_mode_spec
from services.api.main import app


def test_demo_api_uses_promoted_residual_recovery_checkpoint() -> None:
    assert DEFAULT_CHECKPOINT.parts[-5:] == (
        "runs",
        "universal_residual_followup_20260528",
        "residual_recovery_lr2e5_seed6603",
        "checkpoints",
        "best_by_eval_score.pt",
    )


def test_demo_api_reports_modes() -> None:
    client = TestClient(app)

    response = client.get("/modes")

    assert response.status_code == 200
    mode_ids = {mode["id"] for mode in response.json()["modes"]}
    assert {"2d_6x7_k4", "3d_4x4x4_k4", "4d_4x4x4x4_k4"} <= mode_ids


@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://[::1]:5175",
    ],
)
def test_demo_api_allows_local_vite_origins(origin: str) -> None:
    client = TestClient(app)

    response = client.options(
        "/games",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_demo_api_rejects_out_of_turn_agent_move() -> None:
    client = TestClient(app)
    created = client.post(
        "/games",
        json={"mode_id": "2d_6x7_k4", "human_mark": "X", "difficulty": "quick"},
    )
    game_id = created.json()["game"]["game_id"]

    response = client.post(f"/games/{game_id}/agent-move")

    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "wrong_turn"


def test_demo_api_plays_one_human_and_agent_move_per_mode() -> None:
    client = TestClient(app)

    for mode_id in ("2d_6x7_k4", "3d_4x4x4_k4", "4d_4x4x4x4_k4"):
        created = client.post(
            "/games",
            json={"mode_id": mode_id, "human_mark": "X", "difficulty": "quick"},
        )
        assert created.status_code == 200
        game = created.json()["game"]
        legal_actions = [action for action in game["actions"] if action["legal"]]
        assert legal_actions

        human_move = client.post(
            f"/games/{game['game_id']}/moves",
            json={"action": legal_actions[0]["action"]},
        )
        assert human_move.status_code == 200
        after_human = human_move.json()["game"]
        assert after_human["ply"] == 1

        if after_human["is_agent_turn"]:
            agent_move = client.post(f"/games/{game['game_id']}/agent-move")
            assert agent_move.status_code == 200
            payload = agent_move.json()
            assert payload["agent"]["simulations"] == 4
            assert payload["move"]["action"] == payload["agent"]["action"]
            assert payload["game"]["ply"] == 2


def test_demo_api_creates_game_from_custom_shape() -> None:
    client = TestClient(app)

    created = client.post(
        "/games",
        json={
            "shape": [8, 8],
            "connect_k": 5,
            "human_mark": "X",
            "difficulty": "quick",
        },
    )

    assert created.status_code == 200
    mode = created.json()["game"]["mode"]
    assert mode["shape"] == [8, 8]
    assert mode["connect_k"] == 5
    assert mode["dimensions"] == 2


def test_demo_api_rejects_invalid_custom_config() -> None:
    client = TestClient(app)

    # connect_k cannot exceed the largest board dimension.
    response = client.post(
        "/games",
        json={
            "shape": [4, 4],
            "connect_k": 6,
            "human_mark": "X",
            "difficulty": "quick",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["error"] == "invalid_request"


def test_build_mode_spec_validates_limits() -> None:
    spec = build_mode_spec((4, 5, 4, 3), connect_k=4)
    assert spec.dimensions == 4
    assert spec.game_config.shape == (4, 5, 4, 3)
    assert spec.game_config.connect_k == 4

    with pytest.raises(ValueError):
        build_mode_spec((4, 4, 4, 4, 4), connect_k=4)  # rank > MAX_RANK
    with pytest.raises(ValueError):
        build_mode_spec((9, 9), connect_k=4)  # extent > MAX_BOARD_EXTENT
    with pytest.raises(ValueError):
        build_mode_spec((4, 4), connect_k=5)  # connect_k > max(shape)
