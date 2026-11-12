
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrainConfig:
    # Dataset
    data_dir: str = "data"
    train_file: str = "train.npz"
    val_file: str = "val.npz"

    # Model
    input_dim: int = 18
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 4

    # Training
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 0.0003
    weight_decay: float = 0.01

    # Reproducibility
    seed: int = 42

    # Hardware
    device: str = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Output
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"


if __name__ == "__main__":
    config = TrainConfig()

    print("=== GeoChessZero Training Configuration ===")
    print("Device:", config.device)
    print("Batch size:", config.batch_size)
    print("Epochs:", config.epochs)
    print("Learning rate:", config.learning_rate)
    print("Model dimension:", config.d_model)
    print("Attention heads:", config.num_heads)
    print("Transformer layers:", config.num_layers)
    print("Configuration loaded successfully!")
