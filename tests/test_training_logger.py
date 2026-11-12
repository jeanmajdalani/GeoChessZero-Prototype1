
import csv
import importlib
import tempfile
from pathlib import Path


def main():
    print("=== GeoChessZero Training Logger Test ===")

    module = importlib.import_module(
        "prototypes.01_geochesszero.training_logger"
    )

    TrainingLogger = module.TrainingLogger

    # Temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:

        logger = TrainingLogger(
            experiment_name="geochesszero_test",
            output_dir=tmpdir
        )

        # Simulated training results
        results = [
            (1, 7.50, 7.20, 0.10, 0.25, 0.0003, 12.5),
            (2, 6.80, 6.50, 0.18, 0.35, 0.0003, 12.2),
            (3, 6.10, 5.90, 0.25, 0.45, 0.0003, 12.0)
        ]

        for result in results:
            logger.log_epoch(*result)

        # Check CSV exists
        csv_path = Path(logger.file_path)

        assert csv_path.exists(), "CSV file not created!"

        # Read CSV
        with open(
            csv_path,
            "r",
            encoding="utf-8",
            newline=""
        ) as file:
            rows = list(csv.DictReader(file))

        assert len(rows) == 3, (
            f"Expected 3 rows, got {len(rows)}"
        )

        assert int(rows[0]["epoch"]) == 1
        assert int(rows[-1]["epoch"]) == 3

        assert abs(
            float(rows[-1]["train_loss"]) - 6.10
        ) < 1e-6

        assert abs(
            float(rows[-1]["top5_accuracy"]) - 0.45
        ) < 1e-6

        print("\nCSV file verified successfully!")
        print("Number of epochs:", len(rows))
        print("Last training loss:", rows[-1]["train_loss"])
        print("Last Top-5 Accuracy:", rows[-1]["top5_accuracy"])

    print("\nStep 14.16.1 passed!")


if __name__ == "__main__":
    main()
