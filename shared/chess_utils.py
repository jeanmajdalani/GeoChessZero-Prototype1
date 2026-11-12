
import chess
import numpy as np


def create_board():
    """Create a chessboard in its initial position."""
    return chess.Board()


def display_board(board):
    """Display the current chessboard."""
    print(board)


def get_legal_moves(board):
    """Return legal moves in UCI notation."""
    return [move.uci() for move in board.legal_moves]


def play_move(board, move_uci):
    """Play a legal move using UCI notation."""
    move = chess.Move.from_uci(move_uci)

    if move not in board.legal_moves:
        raise ValueError(f"Illegal move: {move_uci}")

    board.push(move)
    return board


def encode_board(board):
    """
    Encode the chessboard as an 8x8 matrix.

    Positive values: white pieces
    Negative values: black pieces
    Zero: empty square

    Piece IDs:
    Pawn=1, Knight=2, Bishop=3,
    Rook=4, Queen=5, King=6
    """
    encoded = np.zeros((8, 8), dtype=np.int8)

    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)

        value = piece.piece_type

        if piece.color == chess.BLACK:
            value = -value

        encoded[7 - rank, file] = value

    return encoded

def encode_board_for_transformer(board):
    """
    Encode a chess position as 64 tokens with 18 features.

    Channels 0-5: White pieces
    Channels 6-11: Black pieces
    Channel 12: Side to move
    Channels 13-16: Castling rights
    Channel 17: En passant target square
    """
    features = np.zeros((64, 18), dtype=np.float32)

    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        channel = color_offset + piece.piece_type - 1

        features[square, channel] = 1.0

    features[:, 12] = 1.0 if board.turn == chess.WHITE else 0.0

    features[:, 13] = float(
        board.has_kingside_castling_rights(chess.WHITE)
    )
    features[:, 14] = float(
        board.has_queenside_castling_rights(chess.WHITE)
    )
    features[:, 15] = float(
        board.has_kingside_castling_rights(chess.BLACK)
    )
    features[:, 16] = float(
        board.has_queenside_castling_rights(chess.BLACK)
    )

    if board.ep_square is not None:
        features[board.ep_square, 17] = 1.0

    return features



if __name__ == "__main__":
    board = create_board()

    print("Initial Chessboard:")
    display_board(board)

    print("\nLegal Moves:")
    print(get_legal_moves(board))

    print("\nEncoded Board:")
    print(encode_board(board))

    print("\nPlaying e2e4...")
    play_move(board, "e2e4")

    print("\nUpdated Board:")
    display_board(board)
