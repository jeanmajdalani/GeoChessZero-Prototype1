
from importlib import import_module

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def main():
    print("=== GeoChessZero Overfit Batch Test ===")

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset_module = import_module(
        "prototypes.01_geochesszero.masked_dataset"
    )

    model_module = import_module(
        "prototypes.01_geochesszero.policy_value"
    )

    dataset = dataset_module.MaskedChessDataset(
        "data/splits/train.npz"
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    X, y, legal_masks = next(iter(loader))

    X = X.to(device)
    y = y.to(device)
    legal_masks = legal_masks.to(device)

    model = model_module.ChessPolicyModel(
        use_geometry=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=0.0,
    )

    initial_loss = None
    final_loss = None

    # Fixed batch: verify the model can memorize examples
    for step in range(1, 101):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(
            X,
            legal_mask=legal_masks
        )

        loss = F.cross_entropy(logits, y)

        if step == 1:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1:
            print(
                f"Step {step:3d}/100 | "
                f"Loss: {loss.item():.4f}"
            )

    # Evaluate without dropout
    model.eval()

    with torch.no_grad():
        logits = model(
            X,
            legal_mask=legal_masks
        )

        final_loss = F.cross_entropy(
            logits, y
        ).item()

        predictions = logits.argmax(dim=1)

        accuracy = (
            (predictions == y)
            .float()
            .mean()
            .item()
        )

    print("\nInitial loss:", round(initial_loss, 4))
    print("Final loss:", round(final_loss, 4))
    print("Training accuracy:", f"{accuracy:.2%}")

    assert final_loss < initial_loss, (
        "Loss did not decrease!"
    )

    print("\nLoss decreased successfully!")
    print("Step 14.24.6 passed!")


if __name__ == "__main__":
    main()
