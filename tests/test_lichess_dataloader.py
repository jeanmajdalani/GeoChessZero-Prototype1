
from pathlib import Path
from importlib import import_module

import torch
from torch.utils.data import DataLoader


def main():
    print("=== GeoChessZero Lichess DataLoader Test ===")

    dataset_module = import_module(
        "prototypes.01_geochesszero.dataset"
    )

    ChessDataset = dataset_module.ChessDataset

    split_dir = Path("data/splits")

    loaders = {}

    for split in ["train", "val", "test"]:

        dataset_path = split_dir / f"{split}.npz"

        assert dataset_path.exists(), (
            f"Missing dataset: {dataset_path}"
        )

        dataset = ChessDataset(dataset_path)

        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=(split == "train"),
            num_workers=0
        )

        loaders[split] = loader

        print(
            f"{split.upper()}: "
            f"{len(dataset)} positions, "
            f"{len(loader)} batches"
        )

    # Test one training batch
    batch = next(iter(loaders["train"]))

    X, y = batch[:2]

    assert X.shape == (16, 64, 18)
    assert y.shape == (16,)

    assert X.dtype == torch.float32
    assert y.dtype == torch.int64

    print("\nBatch positions:", X.shape)
    print("Batch actions:", y.shape)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    X = X.to(device)
    y = y.to(device)

    print("Device:", device)
    print("GPU batch transfer successful!")

    print("\nStep 14.23.4 passed!")


if __name__ == "__main__":
    main()
