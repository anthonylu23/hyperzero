"""Smoke-test a deployed HyperZero web demo."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--web-url", required=True)
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    web_url = args.web_url.rstrip("/")

    health = request_json(f"{api_url}/health")
    model = health["model"]
    if not model["checkpoint_exists"]:
        raise SystemExit(f"checkpoint missing at {model['checkpoint_path']}")

    created = request_json(
        f"{api_url}/games",
        {
            "mode_id": "2d_6x7_k4",
            "human_mark": "X",
            "difficulty": "quick",
        },
    )
    game = created["game"]
    action = next(item["action"] for item in game["actions"] if item["legal"])
    after_human = request_json(
        f"{api_url}/games/{game['game_id']}/moves",
        {"action": action},
    )["game"]
    if after_human["is_agent_turn"]:
        agent_move = request_json(
            f"{api_url}/games/{game['game_id']}/agent-move",
            method="POST",
        )
        if agent_move["game"]["ply"] != 2:
            raise SystemExit("agent move did not advance the game")

    web_status = request_status(web_url)
    if web_status != 200:
        raise SystemExit(f"web URL returned HTTP {web_status}")

    print(
        json.dumps(
            {
                "api_url": api_url,
                "web_url": web_url,
                "checkpoint_path": model["checkpoint_path"],
                "checkpoint_exists": model["checkpoint_exists"],
                "model_loaded": model["loaded"],
                "model_iteration": model["iteration"],
            },
            indent=2,
        )
    )


def request_json(
    url: str,
    payload: dict[str, object] | None = None,
    *,
    method: str | None = None,
) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method or ("GET" if payload is None else "POST"),
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def request_status(url: str) -> int:
    request = urllib.request.Request(url, method="GET")
    last_error: TimeoutError | None = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status
        except TimeoutError as exc:
            last_error = exc
            time.sleep(2)
    raise TimeoutError(f"timed out fetching {url}") from last_error


if __name__ == "__main__":
    main()
