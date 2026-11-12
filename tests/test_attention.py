
import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "prototypes" / "01_geochesszero"

sys.path.insert(0, str(MODEL_DIR))

from attention import GeometryAwareAttention
from geometry_masks import create_geometry_masks


def test_geometry_attention():
    model = GeometryAwareAttention()
    x = torch.randn(2, 64, 128)

    output, weights = model(x, return_attention=True)

    assert output.shape == (2, 64, 128)
    assert weights.shape == (2, 8, 64, 64)

    masks = create_geometry_masks()
    names = ["rook", "bishop", "knight", "king"]

    for head_index, name in enumerate(names):
        allowed = masks[name]

        forbidden_weights = weights[:, head_index, :, :][
            :, ~allowed
        ]

        assert torch.all(forbidden_weights == 0), (
            f"Forbidden attention detected in {name} head"
        )

    # Each attention row must sum to 1.
    row_sums = weights.sum(dim=-1)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-5
    )

    print("All geometry attention tests passed!")

def test_standard_attention():
    model = GeometryAwareAttention(use_geometry=False)

    x = torch.randn(2, 64, 128)

    output, weights = model(x, return_attention=True)

    assert output.shape == (2, 64, 128)
    assert weights.shape == (2, 8, 64, 64)

    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()

    print("Standard attention test passed!")



if __name__ == "__main__":
    test_geometry_attention()
    test_standard_attention()