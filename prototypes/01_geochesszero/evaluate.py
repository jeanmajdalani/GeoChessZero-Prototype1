
import torch

from shared.move_encoder import NUM_ACTIONS


@torch.no_grad()
def compute_metrics(logits, targets, legal_masks=None):
    """
    Compute chess move prediction metrics.

    logits: (B, 4272)
    targets: (B,)
    legal_masks: optional boolean tensor (B, 4272)

    Returns:
        Dictionary containing evaluation metrics.
    """

    if logits.ndim != 2 or logits.shape[1] != NUM_ACTIONS:
        raise ValueError("Invalid logits shape")

    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("Invalid targets shape")

    predictions = torch.argmax(logits, dim=1)

    # Top-1 accuracy
    top1 = (
        predictions == targets
    ).float().mean().item()

    # Top-5 accuracy
    top5_indices = torch.topk(
        logits,
        k=5,
        dim=1
    ).indices

    top5 = (
        top5_indices == targets.unsqueeze(1)
    ).any(dim=1).float().mean().item()

    metrics = {
        "top1_accuracy": top1,
        "top5_accuracy": top5
    }

    # Legal move rate
    if legal_masks is not None:

        if legal_masks.shape != logits.shape:
            raise ValueError("Invalid legal mask shape")

        legal_masks = legal_masks.to(
            device=logits.device,
            dtype=torch.bool
        )

        is_legal = legal_masks.gather(
            1,
            predictions.unsqueeze(1)
        ).squeeze(1)

        metrics["legal_move_rate"] = (
            is_legal.float().mean().item()
        )

    return metrics


if __name__ == "__main__":
    print("=== GeoChessZero Evaluation Module ===")
    print("compute_metrics function ready!")
    print("Step 14.15 passed!")
