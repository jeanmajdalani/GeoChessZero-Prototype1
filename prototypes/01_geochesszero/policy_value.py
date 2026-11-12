
import torch
import torch.nn as nn

from .transformer import ChessTransformer
from shared.move_encoder import (
    NUM_ACTIONS,
    PROMOTION_PATHS,
)


class ChessPolicyModel(nn.Module):
    def __init__(self, use_geometry=True):
        super().__init__()

        self.encoder = ChessTransformer(
            use_geometry=use_geometry
        )

        self.from_proj = nn.Linear(128, 64)
        self.to_proj = nn.Linear(128, 64)

        self.promotion_head = nn.Linear(128, 4)

        self.register_buffer(
            "promotion_from",
            torch.tensor(
                [p[0] for p in PROMOTION_PATHS],
                dtype=torch.long
            )
        )

        self.register_buffer(
            "promotion_to",
            torch.tensor(
                [p[1] for p in PROMOTION_PATHS],
                dtype=torch.long
            )
        )

    def forward(self, x, legal_mask=None):
        features = self.encoder(x)

        origins = self.from_proj(features)
        destinations = self.to_proj(features)

        # Score every origin-destination pair.
        move_scores = torch.matmul(
            origins,
            destinations.transpose(1, 2)
        )

        move_scores = move_scores.flatten(1)

        # Separate promotion scores.
        promotion_features = (
            features[:, self.promotion_from, :]
            + features[:, self.promotion_to, :]
        )

        promotion_scores = self.promotion_head(
            promotion_features
        ).flatten(1)

        logits = torch.cat(
            [move_scores, promotion_scores],
            dim=1
        )

        assert logits.shape[1] == NUM_ACTIONS

        if legal_mask is not None:
            logits = logits.masked_fill(
                ~legal_mask,
                torch.finfo(logits.dtype).min
            )

        return logits


if __name__ == "__main__":
    model = ChessPolicyModel()

    x = torch.randn(2, 64, 18)
    output = model(x)

    print("Policy logits:", output.shape)

    params = sum(
        p.numel() for p in model.parameters()
    )

    print("Trainable parameters:", params)
