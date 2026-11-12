
import importlib

import chess
import torch
from torch.utils.data import DataLoader, TensorDataset

from shared.chess_utils import encode_board_for_transformer
from shared.move_encoder import move_to_index


def main():
    print("=== GeoChessZero Full Training Pipeline Test ===")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    policy_module = importlib.import_module(
        "prototypes.01_geochesszero.policy_value"
    )

    train_module = importlib.import_module(
        "prototypes.01_geochesszero.train"
    )

    ChessPolicyModel = policy_module.ChessPolicyModel

    # Create one chess position in memory
    board = chess.Board()

    position = torch.as_tensor(
        encode_board_for_transformer(board),
        dtype=torch.float32
    )

    target = move_to_index(
        chess.Move.from_uci("e2e4")
    )

    # Repeat the same example for a technical test
    X = position.unsqueeze(0).repeat(8, 1, 1)

    y = torch.full(
        (8,),
        target,
        dtype=torch.long
    )

    dataset = TensorDataset(X, y)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )

    for name, use_geometry in [
        ("Standard Transformer", False),
        ("GeoChessZero", True)
    ]:
        print("\n---------------------------")
        print("Model:", name)

        torch.manual_seed(42)

        model = ChessPolicyModel(
            use_geometry=use_geometry
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0003,
            weight_decay=0.01
        )

        # Training
        train_loss = train_module.train_one_epoch(
            model,
            dataloader,
            optimizer,
            device
        )

        # Validation
        val_loss, accuracy = train_module.validate_one_epoch(
            model,
            dataloader,
            device
        )

        assert torch.isfinite(
            torch.tensor(train_loss)
        ).item()

        assert torch.isfinite(
            torch.tensor(val_loss)
        ).item()

        assert 0.0 <= accuracy <= 1.0

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Accuracy: {accuracy:.2%}")
        print("Pipeline test passed!")

        del model, optimizer

    print("\nStep 14.14 passed!")


if __name__ == "__main__":
    main()
