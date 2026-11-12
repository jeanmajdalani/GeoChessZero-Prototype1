
from importlib import import_module

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def main():
    print("=== GeoChessZero Real Training Batch ===")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    dataset_module = import_module(
        "prototypes.01_geochesszero.masked_dataset"
    )

    model_module = import_module(
        "prototypes.01_geochesszero.policy_value"
    )

    MaskedChessDataset = dataset_module.MaskedChessDataset
    ChessPolicyModel = model_module.ChessPolicyModel

    dataset = MaskedChessDataset(
        "data/splits/train.npz"
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    X, y, legal_masks = next(iter(loader))

    X = X.to(device)
    y = y.to(device)
    legal_masks = legal_masks.to(device)

    # Create GeoChessZero model
    model = ChessPolicyModel(
        use_geometry=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=0.01,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Forward pass with legal move masking
    logits = model(X, legal_mask=legal_masks)

    assert logits.shape == (16, 4272)

    loss = F.cross_entropy(logits, y)

    assert torch.isfinite(loss)

    print("Loss before update:", loss.item())

    # Backpropagation
    loss.backward()

    gradients_ok = any(
        p.grad is not None
        and torch.isfinite(p.grad).all().item()
        for p in model.parameters()
    )

    assert gradients_ok, "No valid gradients found!"

    print("Gradients computed successfully!")

    # Update parameters
    optimizer.step()

    print("AdamW updated model parameters!")
    print("Real Lichess training batch passed!")
    print("\nStep 14.24.5 passed!")


if __name__ == "__main__":
    main()
