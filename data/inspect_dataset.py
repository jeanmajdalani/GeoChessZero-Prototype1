
from pathlib import Path

import chess.pgn


DATASET_PATH = Path(
    "data/lichess_db_standard_rated_2013-01.pgn"
)


def inspect_dataset(max_games=5):
    print("=== GeoChessZero Lichess Dataset Inspection ===")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    total_moves = 0
    valid_games = 0

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as pgn_file:

        for i in range(max_games):

            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            if game.errors:
                print(f"Game {i + 1}: invalid PGN")
                continue

            moves = list(game.mainline_moves())
            move_count = len(moves)

            total_moves += move_count
            valid_games += 1

            print(f"\n--- Game {i + 1} ---")
            print("White:", game.headers.get("White"))
            print("Black:", game.headers.get("Black"))
            print("White Elo:", game.headers.get("WhiteElo"))
            print("Black Elo:", game.headers.get("BlackElo"))
            print("Result:", game.headers.get("Result"))
            print("Moves (plies):", move_count)

    print("\n---------------------------")
    print("Valid games:", valid_games)
    print("Total positions:", total_moves)

    assert valid_games > 0
    assert total_moves > 0

    print("\nStep 14.22.2 passed!")


if __name__ == "__main__":
    inspect_dataset()
