
from pathlib import Path

import numpy as np


DATA_DIR = Path("data")
SPLIT_DIR = DATA_DIR / "splits"


def attach_legal_masks():
    print("=== GeoChessZero Attach Legal Masks ===")

    with np.load(DATA_DIR / "lichess_sample_100.npz") as data:
        X = data["X"]
        y = data["y"]
        game_ids = data["game_ids"]

    with np.load(DATA_DIR / "lichess_legal_masks.npz") as data:
        legal_masks = data["legal_masks"]

    assert len(X) == len(y) == len(game_ids) == len(legal_masks)
    assert legal_masks.shape == (len(X), 4272)

    # Verify every target is legal
    assert np.all(
        legal_masks[np.arange(len(y)), y]
    )

    print("Original positions:", len(X))
    print("Legal masks:", legal_masks.shape)

    # Process existing splits without changing game assignments
    for split_name in ["train", "val", "test"]:

        path = SPLIT_DIR / f"{split_name}.npz"

        with np.load(path) as data:
            X_split = data["X"]
            y_split = data["y"]
            ids_split = data["game_ids"]

        # Find original rows belonging to this split
        selected_games = np.unique(ids_split)
        indices = np.flatnonzero(
            np.isin(game_ids, selected_games)
        )

        # Verify exact row alignment
        assert np.array_equal(X[indices], X_split)
        assert np.array_equal(y[indices], y_split)
        assert np.array_equal(game_ids[indices], ids_split)

        split_masks = legal_masks[indices]

        assert np.all(
            split_masks[np.arange(len(y_split)), y_split]
        )

        # Preserve the original split and add legal masks
        np.savez_compressed(
            path,
            X=X_split,
            y=y_split,
            game_ids=ids_split,
            legal_masks=split_masks,
        )

        print(
            f"{split_name.upper()}: "
            f"{len(X_split)} positions, "
            f"{split_masks.shape[1]} possible actions"
        )

        print("Saved:", path)

    print("\nAll legal masks attached successfully!")
    print("Step 14.24.3 passed!")


if __name__ == "__main__":
    attach_legal_masks()
