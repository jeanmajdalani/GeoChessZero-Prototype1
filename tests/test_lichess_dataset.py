
from pathlib import Path

import numpy as np

from shared.move_encoder import NUM_ACTIONS


def main():
    print("=== GeoChessZero Lichess Dataset Test ===")

    path = Path("data/lichess_sample_100.npz")

    assert path.exists(), "Dataset file not found!"

    with np.load(path) as data:
        X = data["X"]
        y = data["y"]
        game_ids = data["game_ids"]

    # Check dimensions
    assert X.ndim == 3
    assert X.shape[1:] == (64, 18)
    assert y.shape == (len(X),)
    assert game_ids.shape == (len(X),)

    # Check data quality
    assert np.isfinite(X).all()
    assert np.issubdtype(y.dtype, np.integer)
    assert np.all((y >= 0) & (y < NUM_ACTIONS))

    # Check number of games
    unique_games = np.unique(game_ids)
    assert len(unique_games) == 100

    # Check that all games have positions
    counts = np.bincount(game_ids)
    assert np.all(counts > 0)

    print("Positions:", X.shape)
    print("Actions:", y.shape)
    print("Game IDs:", game_ids.shape)
    print("Number of games:", len(unique_games))
    print("Minimum action:", y.min())
    print("Maximum action:", y.max())
    print("Average positions per game:", round(len(X) / 100, 2))

    print("\nAll dataset checks passed!")
    print("Step 14.23.2 passed!")


if __name__ == "__main__":
    main()
