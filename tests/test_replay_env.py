import numpy as np

from hyperzero.game import ConnectKEnv, GameConfig, GameReplay, GameState


def test_replay_round_trips_state_actions() -> None:
    config = GameConfig(shape=(3, 3), connect_k=3, gravity_axis=0)
    state = GameState.new(config)

    for action in [0, 1, 0]:
        state.make_move(action)

    replay = GameReplay.from_state(state, metadata={"agent": "test"})
    restored = GameReplay.from_dict(replay.to_dict())
    final_state = restored.playback()

    assert restored.actions == (0, 1, 0)
    assert restored.metadata == {"agent": "test"}
    np.testing.assert_array_equal(final_state.board, state.board)
    assert final_state.zobrist_hash == state.zobrist_hash


def test_env_reset_and_step_return_observation_reward_done_info() -> None:
    config = GameConfig(shape=(1, 2), connect_k=2, gravity_axis=0)
    env = ConnectKEnv(config)

    obs = env.reset()
    assert obs["board"].shape == config.shape
    assert obs["legal_mask"].tolist() == [True, True]

    obs, reward, terminated, info = env.step(0)

    assert reward == 0
    assert not terminated
    assert info["winner"] is None
    assert obs["player_to_move"] == -1

    _, reward, terminated, info = env.step(1)

    assert reward == 0
    assert terminated
    assert info["winner"] == 0
