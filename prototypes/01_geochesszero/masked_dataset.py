
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MaskedChessDataset(Dataset):
    def __init__(self, dataset_path):
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)

        with np.load(dataset_path) as data:
            self.X = torch.from_numpy(
                data["X"].astype(np.float32)
            )

            self.y = torch.from_numpy(
                data["y"].astype(np.int64)
            )

            self.legal_masks = torch.from_numpy(
                data["legal_masks"].astype(np.bool_)
            )

        assert self.X.shape[1:] == (64, 18)
        assert self.y.shape == (len(self.X),)
        assert self.legal_masks.shape == (
            len(self.X), 4272
        )

        # Every target action must be legal
        rows = torch.arange(len(self.y))
        assert self.legal_masks[rows, self.y].all()

        print(
            f"Loaded {len(self.X)} positions "
            f"with legal masks"
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (
            self.X[index],
            self.y[index],
            self.legal_masks[index],
        )


def main():
    print("=== GeoChessZero Masked Dataset Test ===")

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

    assert X.shape == (16, 64, 18)
    assert y.shape == (16,)
    assert legal_masks.shape == (16, 4272)

    assert X.dtype == torch.float32
    assert y.dtype == torch.int64
    assert legal_masks.dtype == torch.bool

    rows = torch.arange(len(y))
    assert legal_masks[rows, y].all()

    print("Batch positions:", X.shape)
    print("Batch actions:", y.shape)
    print("Batch legal masks:", legal_masks.shape)
    print("All target actions are legal!")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    X = X.to(device)
    y = y.to(device)
    legal_masks = legal_masks.to(device)

    print("Device:", device)
    print("GPU transfer successful!")

    print("\nStep 14.24.4 passed!")


if __name__ == "__main__":
    main()
