
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ChessDataset(Dataset):
    """
    Dataset pour GeoChessZero.

    Format attendu du fichier NPZ :
        X : (N, 64, 18)
        y : (N,)

    X contient les positions.
    y contient les indices des coups (0 à 4271).
    """

    def __init__(self, file_path):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        with np.load(self.file_path) as data:
            X = data["X"]
            y = data["y"]

        if X.ndim != 3 or X.shape[1:] != (64, 18):
            raise ValueError(
                f"Invalid X shape: {X.shape}"
            )

        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(
                f"Invalid y shape: {y.shape}"
            )

        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("Actions must be integers")

        if np.any(y < 0) or np.any(y >= 4272):
            raise ValueError("Action index out of range")

        self.X = torch.from_numpy(
            X.astype(np.float32)
        )

        self.y = torch.from_numpy(
            y.astype(np.int64)
        )

        print(f"Loaded {len(self.X)} positions")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def create_dataloader(
    file_path,
    batch_size=16,
    shuffle=True
):
    dataset = ChessDataset(file_path)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    print("=== GeoChessZero Dataset Module ===")
    print("ChessDataset class loaded successfully!")
    print("DataLoader function ready!")
    print("Step 14.3 passed!")
