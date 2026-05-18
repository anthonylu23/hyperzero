from hyperzero.game import GameConfig, GameState
from hyperzero.game.encoding import canonical_board, legal_action_mask


def test_encoding_helpers_delegate_to_state() -> None:
    config = GameConfig(shape=(2, 3), connect_k=2, gravity_axis=0)
    state = GameState.new(config)
    move = state.make_move(2)

    encoded = canonical_board(state)
    mask = legal_action_mask(state)

    assert encoded.shape == config.shape
    assert encoded[move.cell_coord] == -1
    assert mask.tolist() == [True, True, True]
