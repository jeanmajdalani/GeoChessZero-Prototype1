
import importlib
import tempfile
from pathlib import Path


def main():
    print("=== GeoChessZero Plotting Test ===")

    logger_module = importlib.import_module(
        "prototypes.01_geochesszero.training_logger"
    )

    plot_module = importlib.import_module(
        "prototypes.01_geochesszero.plot_results"
    )

    TrainingLogger = logger_module.TrainingLogger
    plot_comparison = plot_module.plot_comparison

    # Temporary directory for the test
    with tempfile.TemporaryDirectory() as tmpdir:

        output_dir = Path(tmpdir)

        standard_logger = TrainingLogger(
            "standard_transformer",
            output_dir=output_dir
        )

        geo_logger = TrainingLogger(
            "geochesszero",
            output_dir=output_dir
        )

        # Simulate 5 epochs
        for epoch in range(1, 6):

            standard_logger.log_epoch(
                epoch=epoch,
                train_loss=8.0 - epoch * 0.4,
                val_loss=8.2 - epoch * 0.3,
                top1_accuracy=0.05 * epoch,
                top5_accuracy=0.10 * epoch,
                learning_rate=0.0003,
                epoch_time_seconds=12.0
            )

            geo_logger.log_epoch(
                epoch=epoch,
                train_loss=8.0 - epoch * 0.5,
                val_loss=8.2 - epoch * 0.4,
                top1_accuracy=0.06 * epoch,
                top5_accuracy=0.12 * epoch,
                learning_rate=0.0003,
                epoch_time_seconds=13.0
            )

        # Generate comparison plots
        plots_dir = output_dir / "plots"

        plot_comparison(
            standard_csv=standard_logger.file_path,
            geochesszero_csv=geo_logger.file_path,
            output_dir=plots_dir
        )

        # Verify the four PNG files
        expected_files = [
            "train_loss.png",
            "val_loss.png",
            "top1_accuracy.png",
            "top5_accuracy.png"
        ]

        for filename in expected_files:

            file_path = plots_dir / filename

            assert file_path.exists(), (
                f"Missing plot: {filename}"
            )

            assert file_path.stat().st_size > 0, (
                f"Empty plot: {filename}"
            )

            print(f"Verified: {filename}")

    print("\nAll four plots verified!")
    print("Step 14.17.1 passed!")


if __name__ == "__main__":
    main()
