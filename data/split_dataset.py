
from pathlib import Path
import numpy as np


INPUT_PATH = Path("data/lichess_sample_100.npz")
OUTPUT_DIR = Path("data/splits")


def split_dataset(seed=42):
    print("=== GeoChessZero Dataset Splitting ===")

    with np.load(INPUT_PATH) as data:
        X = data["X"]
        y = data["y"]
        game_ids = data["game_ids"]

    # Get unique game IDs
    unique_games = np.unique(game_ids)

    # Shuffle games reproducibly
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_games)

    n_games = len(unique_games)

    n_train = int(n_games * 0.8)
    n_val = int(n_games * 0.1)

    train_games = unique_games[:n_train]
    val_games = unique_games[n_train:n_train + n_val]
    test_games = unique_games[n_train + n_val:]

    # Verify no game appears in multiple splits
    assert not np.intersect1d(train_games, val_games).size
    assert not np.intersect1d(train_games, test_games).size
    assert not np.intersect1d(val_games, test_games).size

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_games,
        "val": val_games,
        "test": test_games,
    }

    for name, selected_games in splits.items():
        mask = np.isin(game_ids, selected_games)

        X_split = X[mask]
        y_split = y[mask]
        ids_split = game_ids[mask]

        output_path = OUTPUT_DIR / f"{name}.npz"

        np.savez_compressed(
            output_path,
            X=X_split,
            y=y_split,
            game_ids=ids_split,
        )

        print(
            f"{name.upper()}: "
            f"{len(selected_games)} games, "
            f"{len(X_split)} positions"
        )
        print("Saved:", output_path)

    print("\nNo game overlap between splits!")
    print("Step 14.23.3 passed!")


if __name__ == "__main__":
    split_dataset()
