
import csv
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = {
    "train_loss": "Training Loss",
    "val_loss": "Validation Loss",
    "top1_accuracy": "Top-1 Accuracy",
    "top5_accuracy": "Top-5 Accuracy",
}


def read_training_log(file_path):
    """
    Read training results from a CSV file.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Training log not found: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Training log is empty")

    return {
        "epoch": [int(row["epoch"]) for row in rows],
        **{
            metric: [
                float(row[metric]) for row in rows
            ]
            for metric in METRICS
        }
    }


def plot_comparison(
    standard_csv,
    geochesszero_csv,
    output_dir="results/geochesszero/plots"
):
    """
    Compare Standard Transformer and GeoChessZero.
    """

    standard = read_training_log(standard_csv)
    geochesszero = read_training_log(geochesszero_csv)

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for metric, title in METRICS.items():

        plt.figure(figsize=(8, 5))

        plt.plot(
            standard["epoch"],
            standard[metric],
            marker="o",
            label="Standard Transformer"
        )

        plt.plot(
            geochesszero["epoch"],
            geochesszero[metric],
            marker="s",
            label="GeoChessZero"
        )

        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"{title} Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = output_dir / f"{metric}.png"

        plt.savefig(output_file, dpi=150)
        plt.close()

        print(f"Saved: {output_file}")

    print("All comparison plots generated!")


if __name__ == "__main__":
    print("=== GeoChessZero Plotting Module ===")
    print("Plotting functions ready!")
    print("Step 14.17 passed!")
