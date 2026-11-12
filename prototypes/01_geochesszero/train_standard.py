
from pathlib import Path
from importlib import import_module

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]

model_module = import_module(
    "prototypes.01_geochesszero.policy_value"
)

dataset_module = import_module(
    "prototypes.01_geochesszero.masked_dataset"
)

training_module = import_module(
    "prototypes.01_geochesszero.train_lichess"
)

ChessPolicyModel = model_module.ChessPolicyModel
MaskedChessDataset = dataset_module.MaskedChessDataset
run_epoch = training_module.run_epoch


def main():
    print("=== Standard Transformer Lichess Training ===")

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_dataset = MaskedChessDataset(
        ROOT / "data/splits/train.npz"
    )

    val_dataset = MaskedChessDataset(
        ROOT / "data/splits/val.npz"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    # Standard Transformer: no geometry-aware attention
    model = ChessPolicyModel(
        use_geometry=False
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=0.01,
    )

    epochs = 10
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            device,
        )

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_acc:.2%}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_acc:.2%}")

        assert all(
            torch.isfinite(torch.tensor(v))
            for v in [train_loss, val_loss]
        ), "Non-finite loss detected!"

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_path = (
                ROOT / "checkpoints"
                / "standard_transformer_lichess_best.pt"
            )

            best_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "use_geometry": False,
                },
                best_path,
            )

            print(
                f"New best model saved! "
                f"Epoch {epoch}, "
                f"Val loss: {val_loss:.4f}"
            )

    checkpoint_path = (
        ROOT / "checkpoints"
        / f"standard_transformer_lichess_epoch{epochs}.pt"
    )

    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "use_geometry": False,
        },
        checkpoint_path,
    )

    print("\nCheckpoint saved:", checkpoint_path)
    print("Step 14.25.2 passed!")


if __name__ == "__main__":
    main()
