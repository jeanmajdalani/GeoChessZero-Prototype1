
from pathlib import Path

import chess.pgn
import numpy as np

from shared.move_encoder import (
    legal_move_mask,
    move_to_index,
    NUM_ACTIONS,
)


PGN_PATH = Path(
    "data/lichess_db_standard_rated_2013-01.pgn"
)

DATASET_PATH = Path(
    "data/lichess_sample_100.npz"
)

OUTPUT_PATH = Path(
    "data/lichess_legal_masks.npz"
)


def generate_legal_masks():
    print("=== GeoChessZero Legal Masks Generation ===")

    with np.load(DATASET_PATH) as data:
        y = data["y"]
        game_ids = data["game_ids"]

    all_masks = []
    all_actions = []
    all_game_ids = []

    games_processed = 0

    with open(
        PGN_PATH,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as pgn_file:

        while games_processed < 100:
            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            if game.errors:
                continue

            moves = list(game.mainline_moves())

            if not moves:
                continue

            board = game.board()

            for move in moves:
                mask = np.asarray(
                    legal_move_mask(board),
                    dtype=np.bool_
                )

                assert mask.shape == (NUM_ACTIONS,)

                action = move_to_index(move)

                assert mask[action], (
                    "Target move is not legal!"
                )

                all_masks.append(mask)
                all_actions.append(action)
                all_game_ids.append(games_processed)

                board.push(move)

            games_processed += 1

    masks = np.stack(all_masks)
    actions = np.asarray(
        all_actions,
        dtype=np.int64
    )
    ids = np.asarray(
        all_game_ids,
        dtype=np.int32
    )

    # Verify exact alignment with converted dataset
    assert len(masks) == len(y)
    assert np.array_equal(actions, y)
    assert np.array_equal(ids, game_ids)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,
        legal_masks=masks
    )

    print("Games processed:", games_processed)
    print("Legal masks shape:", masks.shape)
    print("Actions verified:", len(actions))
    print("All target moves are legal!")
    print("Saved:", OUTPUT_PATH)
    print("\nStep 14.24.2 passed!")


if __name__ == "__main__":
    generate_legal_masks()
