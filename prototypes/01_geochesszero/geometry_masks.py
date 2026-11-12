
import chess
import torch


def create_geometry_masks():
    """
    Create four 64x64 boolean attention masks.

    True = attention allowed
    False = attention blocked

    Square order follows python-chess:
    a1=0, b1=1, ..., h8=63.
    """
    masks = {
        "rook": torch.zeros(64, 64, dtype=torch.bool),
        "bishop": torch.zeros(64, 64, dtype=torch.bool),
        "knight": torch.zeros(64, 64, dtype=torch.bool),
        "king": torch.zeros(64, 64, dtype=torch.bool),
    }

    for source in chess.SQUARES:
        sx = chess.square_file(source)
        sy = chess.square_rank(source)

        for target in chess.SQUARES:
            tx = chess.square_file(target)
            ty = chess.square_rank(target)

            dx = abs(sx - tx)
            dy = abs(sy - ty)

            # Every square can attend to itself.
            if source == target:
                for mask in masks.values():
                    mask[source, target] = True
                continue

            masks["rook"][source, target] = (
                sx == tx or sy == ty
            )

            masks["bishop"][source, target] = (
                dx == dy
            )

            masks["knight"][source, target] = (
                (dx == 1 and dy == 2)
                or (dx == 2 and dy == 1)
            )

            masks["king"][source, target] = (
                max(dx, dy) == 1
            )

    return masks


if __name__ == "__main__":
    masks = create_geometry_masks()

    for name, mask in masks.items():
        print(f"{name}: {mask.shape}")

    source = chess.E4

    for name, mask in masks.items():
        targets = [
            chess.square_name(sq)
            for sq in chess.SQUARES
            if mask[source, sq] and sq != source
        ]
        print(f"{name} from e4: {targets}")
