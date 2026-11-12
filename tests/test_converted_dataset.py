
import numpy as np
import chess

from shared.move_encoder import (
    NUM_ACTIONS,
    move_to_index
)


def main():
    print("=== GeoChessZero Dataset Verification ===")

    dataset_path = "data/sample_games.npz"

    with np.load(dataset_path) as data:
        X = data["X"]
        y = data["y"]
        game_ids = data["game_ids"]

    print("Positions:", X.shape)
    print("Actions:", y.shape)
    print("Game IDs:", game_ids.shape)

    # Verify dimensions
    assert X.shape == (12, 64, 18)
    assert y.shape == (12,)
    assert game_ids.shape == (12,)

    # Verify numerical values
    assert np.isfinite(X).all()
    assert np.issubdtype(y.dtype, np.integer)
    assert np.all((y >= 0) & (y < NUM_ACTIONS))

    # Verify game identifiers
    assert np.array_equal(
        game_ids,
        np.array([0] * 6 + [1] * 6)
    )

    # Verify expected moves from both sample games
    expected_games = [
        ["e2e4", "e7e5", "g1f3",
         "b8c6", "f1b5", "a7a6"],

        ["d2d4", "d7d5", "c2c4",
         "e7e6", "b1c3", "g8f6"]
    ]

    expected_actions = []

    for moves in expected_games:
        board = chess.Board()

        for uci in moves:
            move = chess.Move.from_uci(uci)

            assert move in board.legal_moves
            expected_actions.append(move_to_index(move))
            board.push(move)

    assert np.array_equal(
        y,
        np.array(expected_actions)
    )

    print("All move indices verified!")
    print("All game IDs verified!")
    print("All dataset values are valid!")

    print("\nStep 14.21.3 passed!")


if __name__ == "__main__":
    main()
