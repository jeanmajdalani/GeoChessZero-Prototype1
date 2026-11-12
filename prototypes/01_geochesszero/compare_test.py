
from pathlib import Path
from importlib import import_module

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]

policy_module = import_module(
    "prototypes.01_geochesszero.policy_value"
)
dataset_module = import_module(
    "prototypes.01_geochesszero.masked_dataset"
)

ChessPolicyModel = policy_module.ChessPolicyModel
MaskedChessDataset = dataset_module.MaskedChessDataset


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for X, y, legal_masks in loader:
        X = X.to(device)
        y = y.to(device)
        legal_masks = legal_masks.to(device).bool()

        logits = model(X, legal_mask=legal_masks)

        loss = F.cross_entropy(
            logits, y, reduction="sum"
        )

        top1 = logits.argmax(dim=1)
        top5 = logits.topk(k=5, dim=1).indices

        total_loss += loss.item()
        correct_top1 += (top1 == y).sum().item()
        correct_top5 += (
            (top5 == y.unsqueeze(1))
            .any(dim=1)
            .sum()
            .item()
        )
        total += y.size(0)

    return {
        "loss": total_loss / total,
        "top1": correct_top1 / total,
        "top5": correct_top5 / total,
    }


def main():
    print("=== GeoChessZero TEST Comparison ===")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    test_dataset = MaskedChessDataset(
        ROOT / "data/splits/test.npz"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    print("Test positions:", len(test_dataset))

    checkpoints = {
        "GeoChessZero":
            "geochesszero_lichess_best.pt",
        "Standard Transformer":
            "standard_transformer_lichess_best.pt",
    }

    results = {}

    for name, filename in checkpoints.items():
        path = ROOT / "checkpoints" / filename

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        model = ChessPolicyModel(
            use_geometry=checkpoint["use_geometry"]
        ).to(device)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        metrics = evaluate(model, test_loader, device)
        results[name] = metrics

        print(f"\n--- {name} ---")
        print("Best epoch:", checkpoint["epoch"])
        print("Geometry:", checkpoint["use_geometry"])
        print(f"Test loss: {metrics['loss']:.4f}")
        print(f"Top-1 accuracy: {metrics['top1']:.2%}")
        print(f"Top-5 accuracy: {metrics['top5']:.2%}")

        del model

    geo = results["GeoChessZero"]
    std = results["Standard Transformer"]

    print("\n=== Comparison ===")
    print(
        f"Loss difference: "
        f"{geo['loss'] - std['loss']:+.4f}"
    )
    print(
        f"Top-1 difference: "
        f"{(geo['top1'] - std['top1']) * 100:+.2f} points"
    )
    print(
        f"Top-5 difference: "
        f"{(geo['top5'] - std['top5']) * 100:+.2f} points"
    )

    print("\nStep 14.25.4 passed!")


if __name__ == "__main__":
    main()
