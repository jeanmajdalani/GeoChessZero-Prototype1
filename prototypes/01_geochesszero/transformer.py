
import torch
import torch.nn as nn

from .attention import GeometryAwareAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model=128,
        num_heads=8,
        dropout=0.1,
        use_geometry=True
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attention = GeometryAwareAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_geometry=use_geometry
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-LayerNorm attention with residual connection
        x = x + self.attention(self.norm1(x))

        # Pre-LayerNorm feed-forward with residual connection
        x = x + self.feed_forward(self.norm2(x))

        return x


class ChessTransformer(nn.Module):
    def __init__(
        self,
        input_dim=18,
        d_model=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        use_geometry=True
    ):
        super().__init__()

        self.input_embedding = nn.Linear(input_dim, d_model)

        # Learnable positional embeddings for the 64 squares
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 64, d_model)
        )

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                use_geometry=use_geometry
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (64, 18):
            raise ValueError(
                "Expected input shape (batch, 64, 18)"
            )

        x = self.input_embedding(x)
        x = x + self.position_embedding

        for layer in self.layers:
            x = layer(x)

        return self.final_norm(x)


if __name__ == "__main__":
    model = ChessTransformer(use_geometry=True)

    x = torch.randn(2, 64, 18)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Trainable parameters:", parameters)
