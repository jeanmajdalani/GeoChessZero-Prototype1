
from pathlib import Path
import json
import csv

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def check_file(relative_path):
    path = ROOT / relative_path
    assert path.is_file(), f"Missing file: {relative_path}"
    print(f"[OK] {relative_path}")
    return path


def main():
    print("=== GeoChessZero Final Prototype Verification ===")

    print("\n1. Checking model architecture")

    for name in [
        "shared/chess_utils.py",
        "shared/move_encoder.py",
        "prototypes/01_geochesszero/geometry_masks.py",
        "prototypes/01_geochesszero/attention.py",
        "prototypes/01_geochesszero/transformer.py",
        "prototypes/01_geochesszero/policy_value.py",
    ]:
        check_file(name)

    print("\n2. Checking datasets")

    expected_sizes = {
        "train": 4877,
        "val": 697,
        "test": 619,
    }

    for split, expected in expected_sizes.items():
        path = check_file(f"data/splits/{split}.npz")

        with np.load(path) as data:
            assert "X" in data.files
            assert "y" in data.files
            assert "legal_masks" in data.files

            X = data["X"]
            y = data["y"]
            masks = data["legal_masks"]

            assert X.shape == (expected, 64, 18)
            assert y.shape == (expected,)
            assert masks.shape == (expected, 4272)
            assert np.all((y >= 0) & (y < 4272))
            assert np.all(
                masks[np.arange(expected), y].astype(bool)
            ), f"Illegal target move in {split}"

        print(f"[OK] {split}: {expected} positions")

    print("\n3. Checking model checkpoints")

    checkpoint_names = [
        "geochesszero_lichess_best.pt",
        "standard_transformer_lichess_best.pt",
    ]

    for name in checkpoint_names:
        path = check_file(f"checkpoints/{name}")

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        assert "model_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "val_loss" in checkpoint

        expected_geometry = name.startswith("geochesszero")
        assert checkpoint["use_geometry"] == expected_geometry

        print(
            f"[OK] Epoch {checkpoint['epoch']}, "
            f"validation loss {checkpoint['val_loss']:.4f}"
        )

    print("\n4. Checking research results")

    result_dir = "results/geochesszero"

    json_path = check_file(
        f"{result_dir}/test_comparison.json"
    )

    csv_path = check_file(
        f"{result_dir}/test_comparison.csv"
    )

    with open(json_path, encoding="utf-8") as file:
        results = json.load(file)

    assert results, "Comparison JSON is empty"

    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) >= 2, "Expected two model results"

    print("[OK] Comparison results are available")

    print("\n5. Checking comparison plots")

    for name in [
        "test_loss_comparison.png",
        "top1_comparison.png",
        "top5_comparison.png",
    ]:
        check_file(f"{result_dir}/plots/{name}")

    print("\n6. Checking demo and report")

    check_file("demo/app.py")
    check_file(f"{result_dir}/final_report.md")

    print("\n====================================")
    print("ALL PROTOTYPE 1 CHECKS PASSED!")
    print("Step 14.25.10 passed!")
    print("====================================")


if __name__ == "__main__":
    main()
