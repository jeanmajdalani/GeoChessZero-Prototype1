
import math
import torch
import torch.nn as nn

from .geometry_masks import create_geometry_masks


class GeometryAwareAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=8, use_geometry=True):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_geometry = use_geometry

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        masks = create_geometry_masks()

        # True means attention is allowed.
        geometry_masks = torch.stack([
            masks["rook"],
            masks["bishop"],
            masks["knight"],
            masks["king"],
        ])

        self.register_buffer("geometry_masks", geometry_masks)

    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.shape

        if seq_len != 64:
            raise ValueError("Expected exactly 64 chessboard tokens.")

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        if self.use_geometry:
            # Restrict the first four attention heads.
            allowed = self.geometry_masks.to(x.device)

            scores[:, :4] = scores[:, :4].masked_fill(
                ~allowed.unsqueeze(0),
                torch.finfo(scores.dtype).min
            )

        attention = torch.softmax(scores, dim=-1)

        output = torch.matmul(attention, v)

        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_len, self.d_model)

        output = self.out_proj(output)

        if return_attention:
            return output, attention

        return output


if __name__ == "__main__":
    model = GeometryAwareAttention()

    x = torch.randn(2, 64, 128)

    output, weights = model(x, return_attention=True)

    print("Input:", x.shape)
    print("Output:", output.shape)
    print("Attention:", weights.shape)
