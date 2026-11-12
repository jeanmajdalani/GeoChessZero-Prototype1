
from pathlib import Path
import zstandard as zstd


def decompress_dataset():
    print("=== GeoChessZero Dataset Decompression ===")

    input_path = Path(
        "data/lichess_db_standard_rated_2013-01.pgn.zst"
    )

    output_path = Path(
        "data/lichess_db_standard_rated_2013-01.pgn"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    print("Compressed dataset found!")
    print(
        "Compressed size:",
        round(input_path.stat().st_size / (1024 ** 2), 2),
        "MB"
    )

    decompressor = zstd.ZstdDecompressor()

    with open(input_path, "rb") as source:
        with open(output_path, "wb") as destination:
            decompressor.copy_stream(source, destination)

    print("Decompression completed!")
    print("Output:", output_path)
    print(
        "Decompressed size:",
        round(output_path.stat().st_size / (1024 ** 2), 2),
        "MB"
    )

    print("Step 14.22.1 passed!")


if __name__ == "__main__":
    decompress_dataset()
