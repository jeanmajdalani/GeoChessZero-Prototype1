
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "results" / "geochesszero"
JSON_PATH = RESULTS_DIR / "test_comparison.json"
PLOTS_DIR = RESULTS_DIR / "plots"


def plot_metric(results, key, title, ylabel, filename, percentage=False):
    names = [r["model"] for r in results]
    values = [r[key] for r in results]

    if percentage:
        values = [v * 100 for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(names))

    bars = ax.bar(
        x,
        values,
        width=0.55,
        color=["#2563eb", "#f59e0b"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    ax.set_ylim(0, max(values) * 1.2)

    for bar, value in zip(bars, values):
        label = (
            f"{value:.2f}%"
            if percentage
            else f"{value:.4f}"
        )

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()

    output_path = PLOTS_DIR / filename
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output_path)


def main():
    print("=== GeoChessZero Comparison Plots ===")

    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"Comparison results not found: {JSON_PATH}"
        )

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    assert len(results) == 2, "Expected exactly two models"

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_metric(
        results,
        key="test_loss",
        title="GeoChessZero vs Standard Transformer - Test Loss",
        ylabel="Cross-Entropy Loss (lower is better)",
        filename="test_loss_comparison.png",
    )

    plot_metric(
        results,
        key="top1_accuracy",
        title="GeoChessZero vs Standard Transformer - Top-1 Accuracy",
        ylabel="Top-1 Accuracy (%)",
        filename="top1_comparison.png",
        percentage=True,
    )

    plot_metric(
        results,
        key="top5_accuracy",
        title="GeoChessZero vs Standard Transformer - Top-5 Accuracy",
        ylabel="Top-5 Accuracy (%)",
        filename="top5_comparison.png",
        percentage=True,
    )

    expected_files = [
        "test_loss_comparison.png",
        "top1_comparison.png",
        "top5_comparison.png",
    ]

    for filename in expected_files:
        assert (PLOTS_DIR / filename).exists()
        print("Verified:", filename)

    print("\nAll comparison plots generated successfully!")
    print("Step 14.25.6 passed!")


if __name__ == "__main__":
    main()
