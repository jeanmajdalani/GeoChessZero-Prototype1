
import importlib

import torch

from shared.move_encoder import NUM_ACTIONS


def main():
    print("=== GeoChessZero Evaluation Test ===")

    module = importlib.import_module(
        "prototypes.01_geochesszero.evaluate"
    )

    compute_metrics = module.compute_metrics

    # Three artificial predictions
    logits = torch.zeros(3, NUM_ACTIONS)

    targets = torch.tensor([10, 20, 30])

    # Example 1: correct Top-1 prediction
    logits[0, 10] = 10.0

    # Example 2: correct target in Top-5
    logits[1, 21] = 10.0
    logits[1, 20] = 9.0

    # Example 3: incorrect prediction
    logits[2, 31] = 10.0
    logits[2, 32] = 9.0
    logits[2, 33] = 8.0
    logits[2, 34] = 7.0
    logits[2, 35] = 6.0

    # Artificial legal move masks
    legal_masks = torch.zeros(
        3, NUM_ACTIONS, dtype=torch.bool
    )

    legal_masks[0, 10] = True
    legal_masks[1, 21] = True
    legal_masks[2, 30] = True

    metrics = compute_metrics(
        logits,
        targets,
        legal_masks
    )

    print("Top-1 Accuracy:", metrics["top1_accuracy"])
    print("Top-5 Accuracy:", metrics["top5_accuracy"])
    print("Legal Move Rate:", metrics["legal_move_rate"])

    assert abs(
        metrics["top1_accuracy"] - 1/3
    ) < 1e-6

    assert abs(
        metrics["top5_accuracy"] - 2/3
    ) < 1e-6

    assert abs(
        metrics["legal_move_rate"] - 2/3
    ) < 1e-6

    print("All evaluation metrics are correct!")
    print("Step 14.15.1 passed!")


if __name__ == "__main__":
    main()
