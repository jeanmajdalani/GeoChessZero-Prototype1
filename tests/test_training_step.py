
import importlib

import chess
import torch
import torch.nn.functional as F

from shared.chess_utils import encode_board_for_transformer
from shared.move_encoder import legal_move_mask, move_to_index


def main():
    print("=== GeoChessZero Backpropagation Test ===")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    policy_module = importlib.import_module(
        "prototypes.01_geochesszero.policy_value"
    )

    ChessPolicyModel = policy_module.ChessPolicyModel

    board = chess.Board()

    position = torch.as_tensor(
        encode_board_for_transformer(board),
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    mask = torch.as_tensor(
        legal_move_mask(board),
        dtype=torch.bool,
        device=device
    ).unsqueeze(0)

    target = torch.tensor(
        [move_to_index(chess.Move.from_uci("e2e4"))],
        dtype=torch.long,
        device=device
    )

    for name, use_geometry in [
        ("Standard Transformer", False),
        ("GeoChessZero", True)
    ]:
        print("\nModel:", name)

        torch.manual_seed(42)

        model = ChessPolicyModel(
            use_geometry=use_geometry
        ).to(device)

        model.train()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0003,
            weight_decay=0.01
        )

        optimizer.zero_grad()

        logits = model(position, legal_mask=mask)

        loss = F.cross_entropy(logits, target)

        print("Loss before update:", loss.item())

        assert torch.isfinite(loss).item()

        loss.backward()

        gradients = [
            p.grad for p in model.parameters()
            if p.requires_grad
        ]

        assert all(g is not None for g in gradients)
        assert all(torch.isfinite(g).all().item() for g in gradients)

        first_parameter = next(model.parameters())
        before = first_parameter.detach().clone()

        optimizer.step()

        after = first_parameter.detach()

        assert not torch.equal(before, after), (
            "Optimizer did not update parameters"
        )

        print("Gradients computed successfully!")
        print("AdamW updated model parameters!")
        print("Test passed!")

        del model, optimizer

    print("\nStep 14.12 passed!")


if __name__ == "__main__":
    main()
