
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "prototypes" / "01_geochesszero"

sys.path.insert(0, str(MODEL_DIR))

from transformer import ChessTransformer


def test_transformers():
    standard = ChessTransformer(use_geometry=False)
    geometry = ChessTransformer(use_geometry=True)

    x = torch.randn(2, 64, 18)

    standard_output = standard(x)
    geometry_output = geometry(x)

    assert standard_output.shape == (2, 64, 128)
    assert geometry_output.shape == (2, 64, 128)

    standard_params = sum(
        p.numel() for p in standard.parameters()
    )

    geometry_params = sum(
        p.numel() for p in geometry.parameters()
    )

    assert standard_params == geometry_params

    assert torch.isfinite(standard_output).all()
    assert torch.isfinite(geometry_output).all()

    print("Standard parameters:", standard_params)
    print("Geometry parameters:", geometry_params)
    print("Both Transformer tests passed!")


if __name__ == "__main__":
    test_transformers()
