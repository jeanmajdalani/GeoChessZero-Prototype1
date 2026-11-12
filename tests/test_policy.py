
import sys
from pathlib import Path

import chess
import torch

# --------------------------------------------------
# 1. Configure project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(
    0, str(ROOT / "prototypes" / "01_geochesszero")
)

# --------------------------------------------------
# 2. Import project modules
# --------------------------------------------------

from policy_value import ChessPolicyModel

from shared.chess_utils import encode_board_for_transformer

from shared.move_encoder import (
    NUM_ACTIONS,
    move_to_index,
    legal_move_mask,
)


# --------------------------------------------------
# 3. Test policy output and legal moves
# --------------------------------------------------

def test_policy():

    board = chess.Board()

    x = torch.tensor(
        encode_board_for_transformer(board),
        dtype=torch.float32
    ).unsqueeze(0)

    mask = legal_move_mask(board).unsqueeze(0)

    model = ChessPolicyModel(use_geometry=True)
    model.eval()

    with torch.no_grad():
        logits = model(x, mask)

    assert logits.shape == (1, NUM_ACTIONS)
    assert NUM_ACTIONS == 4272

    # Initial position must have 20 legal moves.
    assert mask.sum().item() == 20

    predicted_index = logits.argmax(dim=-1).item()

    # Predicted action must be legal.
    assert mask[0, predicted_index].item()

    # Forbidden actions must not be selected.
    assert torch.isfinite(logits).all()

    print("Policy shape:", logits.shape)
    print("Legal moves:", mask.sum().item())
    print("Predicted action index:", predicted_index)
    print("Geometry policy test passed!")


# --------------------------------------------------
# 4. Test standard Transformer policy
# --------------------------------------------------

def test_standard_policy():

    model = ChessPolicyModel(use_geometry=False)
    model.eval()

    x = torch.randn(2, 64, 18)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, NUM_ACTIONS)
    assert torch.isfinite(logits).all()

    print("Standard policy test passed!")


# --------------------------------------------------
# 5. Test promotion encoding
# --------------------------------------------------

def test_promotions():

    promotion_moves = [
        chess.Move.from_uci("a7a8q"),
        chess.Move.from_uci("a7a8r"),
        chess.Move.from_uci("a7a8b"),
        chess.Move.from_uci("a7a8n"),
        chess.Move.from_uci("a7b8q"),
        chess.Move.from_uci("h2h1n"),
    ]

    indices = [
        move_to_index(move)
        for move in promotion_moves
    ]

    assert len(indices) == len(set(indices))

    for index in indices:
        assert 4096 <= index < NUM_ACTIONS

    print("Promotion encoding test passed!")


# --------------------------------------------------
# 6. Test GPU
# --------------------------------------------------

def test_gpu():

    if not torch.cuda.is_available():
        print("CUDA unavailable: GPU test skipped.")
        return

    device = torch.device("cuda")

    model = ChessPolicyModel(
        use_geometry=True
    ).to(device)

    model.eval()

    board = chess.Board()

    x = torch.tensor(
        encode_board_for_transformer(board),
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    mask = legal_move_mask(board).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x, mask)

    assert logits.shape == (1, NUM_ACTIONS)
    assert logits.device.type == "cuda"

    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU policy test passed!")


# --------------------------------------------------
# 7. Run all tests
# --------------------------------------------------

if __name__ == "__main__":

    print("\n=== Testing GeoChessZero Policy ===")

    test_policy()
    test_standard_policy()
    test_promotions()
    test_gpu()

    print("\nAll policy tests passed!")
