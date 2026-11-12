
import numpy as np
import chess

from shared.move_encoder import (
    legal_move_mask,
    NUM_ACTIONS,
)


def main():
    print("=== GeoChessZero Legal Move Dataset Test ===")

    path = "data/splits/train.npz"

    with np.load(path) as data:
        X = data["X"]
        y = data["y"]

    print("Loaded positions:", len(X))

    # Verify the move encoding on a known position
    board = chess.Board()

    mask = np.asarray(
        legal_move_mask(board),
        dtype=bool
    )

    assert mask.shape == (NUM_ACTIONS,)
    assert mask.sum() == 20

    print("Starting position legal moves:", mask.sum())

    # Check that the first dataset position has the
    # expected feature dimensions and a valid action
    position = X[0]
    action = int(y[0])

    assert position.shape == (64, 18)
    assert 0 <= action < NUM_ACTIONS

    print("First dataset position:", position.shape)
    print("First dataset action:", action)

    print("Basic dataset checks passed!")
    print("Step 14.24.1 passed!")


if __name__ == "__main__":
    main()
