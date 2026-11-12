
from pathlib import Path
from importlib import import_module

import chess
import torch

from shared.chess_utils import encode_board_for_transformer
from shared.move_encoder import move_to_index, legal_move_mask


ROOT = Path(__file__).resolve().parents[2]

ChessPolicyModel = import_module(
    "prototypes.01_geochesszero.policy_value"
).ChessPolicyModel


def load_model(checkpoint_name, use_geometry, device):
    path = ROOT / "checkpoints" / checkpoint_name

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model = ChessPolicyModel(
        use_geometry=use_geometry
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Loaded: {checkpoint_name} "
        f"(epoch {checkpoint['epoch']})"
    )

    return model


@torch.no_grad()
def predict_moves(model, board, device, top_k=5):
    X = torch.as_tensor(
        encode_board_for_transformer(board),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    mask = torch.as_tensor(
        legal_move_mask(board),
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)

    logits = model(X, legal_mask=mask)
    probabilities = torch.softmax(logits, dim=-1)[0]

    legal_moves = list(board.legal_moves)

    predictions = []

    for move in legal_moves:
        index = move_to_index(move)

        predictions.append(
            (
                board.san(move),
                move.uci(),
                probabilities[index].item(),
            )
        )

    predictions.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    return predictions[:top_k]


def main():
    print("=== GeoChessZero Position Prediction ===")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    board = chess.Board()

    for move in ["e4", "e5", "Nf3", "Nc6"]:
        board.push_san(move)

    print("\nChess position:")
    print(board)
    print("\nSide to move:", "White" if board.turn else "Black")
    print("Legal moves:", board.legal_moves.count())

    experiments = [
        (
            "GeoChessZero",
            "geochesszero_lichess_best.pt",
            True,
        ),
        (
            "Standard Transformer",
            "standard_transformer_lichess_best.pt",
            False,
        ),
    ]

    for name, checkpoint_name, use_geometry in experiments:
        model = load_model(
            checkpoint_name,
            use_geometry,
            device,
        )

        predictions = predict_moves(
            model,
            board,
            device,
        )

        print(f"\n--- {name}: Top 5 moves ---")

        for rank, (san, uci, probability) in enumerate(
            predictions, start=1
        ):
            print(
                f"{rank}. {san:6s} "
                f"({uci}) - {probability:.2%}"
            )

        assert len(predictions) == 5
        assert all(0 <= p <= 1 for _, _, p in predictions)

    print("\nStep 14.25.7 passed!")


if __name__ == "__main__":
    main()
