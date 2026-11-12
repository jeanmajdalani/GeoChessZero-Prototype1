
import importlib

import chess
import numpy as np
import torch

from shared.chess_utils import encode_board_for_transformer
from shared.move_encoder import legal_move_mask, NUM_ACTIONS


def main():
    print("=== GeoChessZero GPU Prediction Test ===")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    module = importlib.import_module(
        "prototypes.01_geochesszero.policy_value"
    )

    ChessPolicyModel = module.ChessPolicyModel

    # Create a real chess position
    board = chess.Board()

    position = encode_board_for_transformer(board)

    X = torch.tensor(
        np.array(position),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    print("Input shape:", X.shape)

    # Legal moves from the initial position
    mask = torch.tensor(
        legal_move_mask(board),
        dtype=torch.bool
    ).unsqueeze(0).to(device)

    print("Legal moves:", mask.sum().item())

    for name, use_geometry in [
        ("Standard Transformer", False),
        ("GeoChessZero", True)
    ]:
        print("\n-------------------------")
        print("Model:", name)

        torch.manual_seed(42)

        model = ChessPolicyModel(
            use_geometry=use_geometry
        ).to(device)

        model.eval()

        with torch.no_grad():
            logits = model(X, legal_mask=mask)

        assert logits.shape == (1, NUM_ACTIONS)
        assert torch.isfinite(logits).all()

        predicted_action = logits.argmax(dim=1).item()

        assert mask[0, predicted_action].item(), (
            "Predicted move is illegal!"
        )

        print("Output shape:", logits.shape)
        print("Predicted action:", predicted_action)
        print("Prediction is legal!")
        print("Test passed!")

    print("\nStep 14.11 passed!")


if __name__ == "__main__":
    main()
