
import chess
import torch

PROMOTIONS = [
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
]

# 64 possible origins x 64 possible destinations
BASE_ACTIONS = 64 * 64

# 22 possible promotion paths per color,
# 4 possible promotion pieces per path.
NUM_ACTIONS = BASE_ACTIONS + 44 * 4

PROMOTION_PATHS = []

for color in [chess.WHITE, chess.BLACK]:
    from_rank = 6 if color == chess.WHITE else 1
    to_rank = 7 if color == chess.WHITE else 0

    for file in range(8):
        from_square = chess.square(file, from_rank)

        for delta in [-1, 0, 1]:
            target_file = file + delta

            if 0 <= target_file < 8:
                to_square = chess.square(
                    target_file, to_rank
                )
                PROMOTION_PATHS.append(
                    (from_square, to_square)
                )

PROMOTION_PATH_TO_INDEX = {
    path: i
    for i, path in enumerate(PROMOTION_PATHS)
}


def move_to_index(move):
    """Convert a chess move to a unique action index."""

    if move.promotion is None:
        return move.from_square * 64 + move.to_square

    path = (move.from_square, move.to_square)

    path_index = PROMOTION_PATH_TO_INDEX[path]
    piece_index = PROMOTIONS.index(move.promotion)

    return BASE_ACTIONS + path_index * 4 + piece_index


def legal_move_mask(board):
    """Boolean mask containing all legal moves."""

    mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)

    for move in board.legal_moves:
        mask[move_to_index(move)] = True

    return mask
