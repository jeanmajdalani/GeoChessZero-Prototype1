
import csv
from pathlib import Path


class TrainingLogger:
    """
    Save training metrics to a CSV file.
    """

    def __init__(
        self,
        experiment_name,
        output_dir="results/geochesszero"
    ):
        self.experiment_name = experiment_name

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.file_path = (
            self.output_dir /
            f"{experiment_name}_training.csv"
        )

        self.fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "top1_accuracy",
            "top5_accuracy",
            "learning_rate",
            "epoch_time_seconds"
        ]

        # Create CSV with header
        with open(
            self.file_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.fieldnames
            )
            writer.writeheader()

        print(f"Training logger created: {self.file_path}")

    def log_epoch(
        self,
        epoch,
        train_loss,
        val_loss,
        top1_accuracy,
        top5_accuracy,
        learning_rate,
        epoch_time_seconds
    ):
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "top1_accuracy": top1_accuracy,
            "top5_accuracy": top5_accuracy,
            "learning_rate": learning_rate,
            "epoch_time_seconds": epoch_time_seconds
        }

        with open(
            self.file_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.fieldnames
            )
            writer.writerow(row)

        print(
            f"Epoch {epoch} logged | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Top-1: {top1_accuracy:.2%}"
        )


if __name__ == "__main__":
    print("=== GeoChessZero Training Logger ===")
    print("TrainingLogger class ready!")
    print("Step 14.16 passed!")
