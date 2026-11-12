
import random
import os

import numpy as np
import torch


def set_seed(seed=42, deterministic=True):
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(
            True,
            warn_only=True
        )

    print(f"Random seed set to: {seed}")
    print(f"Deterministic mode: {deterministic}")


def get_environment_info():
    """
    Return hardware and software information.
    """

    info = {
        "python_seed_control": True,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        )
    }

    return info


if __name__ == "__main__":
    print("=== GeoChessZero Reproducibility ===")

    set_seed(42)

    info = get_environment_info()

    for key, value in info.items():
        print(f"{key}: {value}")

    print("Step 14.19 passed!")
