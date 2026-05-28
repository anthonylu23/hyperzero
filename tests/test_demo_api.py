from fastapi.testclient import TestClient

from services.api.main import app


def test_demo_api_reports_modes() -> None:
    client = TestClient(app)

    response = client.get("/modes")

    assert response.status_code == 200
    mode_ids = {mode["id"] for mode in response.json()["modes"]}
    assert {"2d_6x7_k4", "3d_4x4x4_k4", "4d_4x4x4x4_k4"} <= mode_ids


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
