
import csv
import json
from pathlib import Path

from importlib import import_module

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]

comparison_module = import_module(
    "prototypes.01_geochesszero.compare_test"
)

dataset_module = import_module(
    "prototypes.01_geochesszero.masked_dataset"
)

ChessPolicyModel = comparison_module.ChessPolicyModel
evaluate = comparison_module.evaluate
MaskedChessDataset = dataset_module.MaskedChessDataset


def main():
    print("=== GeoChessZero Save Comparison ===")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    test_dataset = MaskedChessDataset(
        ROOT / "data/splits/test.npz"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    checkpoints = {
        "GeoChessZero": "geochesszero_lichess_best.pt",
        "Standard Transformer": "standard_transformer_lichess_best.pt",
    }

    results = []

    for name, filename in checkpoints.items():
        checkpoint = torch.load(
            ROOT / "checkpoints" / filename,
            map_location="cpu",
            weights_only=True,
        )

        model = ChessPolicyModel(
            use_geometry=checkpoint["use_geometry"]
        ).to(device)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        metrics = evaluate(
            model, test_loader, device
        )

        result = {
            "model": name,
            "best_epoch": checkpoint["epoch"],
            "use_geometry": checkpoint["use_geometry"],
            "test_positions": len(test_dataset),
            "test_loss": metrics["loss"],
            "top1_accuracy": metrics["top1"],
            "top5_accuracy": metrics["top5"],
        }

        results.append(result)

        print(
            f"{name}: "
            f"Loss={metrics['loss']:.4f}, "
            f"Top-1={metrics['top1']:.2%}, "
            f"Top-5={metrics['top5']:.2%}"
        )

        del model

    output_dir = ROOT / "results" / "geochesszero"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "test_comparison.json"
    csv_path = output_dir / "test_comparison.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:", json_path)
    print("Saved:", csv_path)
    print("Step 14.25.5 passed!")


if __name__ == "__main__":
    main()
