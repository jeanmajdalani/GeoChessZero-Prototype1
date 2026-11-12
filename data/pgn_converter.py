
from pathlib import Path
import io

import chess.pgn
import numpy as np

from shared.chess_utils import encode_board_for_transformer
from shared.move_encoder import move_to_index


def convert_pgn_game(game):
    """
    Convert one PGN game into chess positions and
    target move indices.
    """

    board = game.board()

    positions = []
    actions = []

    for move in game.mainline_moves():

        # Encode position BEFORE playing the move
        position = encode_board_for_transformer(board)

        # Encode the move played in this position
        action = move_to_index(move)

        positions.append(position)
        actions.append(action)

        board.push(move)

    if not positions:
        return None, None

    X = np.asarray(positions, dtype=np.float32)
    y = np.asarray(actions, dtype=np.int64)

    return X, y


def test_converter():
    print("=== GeoChessZero PGN Converter Test ===")

    # Small example game stored directly in memory
    sample_pgn = """
[Event "Test Game"]
[Site "Local"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *
"""

    game = chess.pgn.read_game(
        io.StringIO(sample_pgn)
    )

    assert game is not None, "Could not read PGN"

    X, y = convert_pgn_game(game)

    assert X is not None
    assert X.shape == (6, 64, 18)
    assert y.shape == (6,)

    print("Positions shape:", X.shape)
    print("Actions shape:", y.shape)
    print("Action indices:", y.tolist())

    print("PGN conversion successful!")
    print("Step 14.21.1 passed!")

def convert_pgn_file(
    input_path,
    output_path,
    max_games=None
):
    """
    Convert multiple PGN games into an NPZ dataset.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    positions = []
    actions = []
    game_ids = []

    games_processed = 0

    with open(
        input_path,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as pgn_file:

        while max_games is None or games_processed < max_games:

            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            # Skip malformed games
            if game.errors:
                print("Skipping malformed game")
                continue

            X, y = convert_pgn_game(game)

            if X is None:
                continue

            positions.append(X)
            actions.append(y)

            game_ids.extend(
                [games_processed] * len(y)
            )

            games_processed += 1

    if not positions:
        raise ValueError("No valid games found")

    X = np.concatenate(positions, axis=0)
    y = np.concatenate(actions, axis=0)
    game_ids = np.asarray(
        game_ids,
        dtype=np.int32
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        game_ids=game_ids
    )

    print("Games processed:", games_processed)
    print("Positions:", X.shape)
    print("Actions:", y.shape)
    print("Game IDs:", game_ids.shape)
    print("Dataset saved:", output_path)

    return output_path



if __name__ == "__main__":
    test_converter()
