
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class ExperimentConfig:
    name: str
    use_geometry: bool

    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    seed: int = 42


def get_experiments():
    return [
        ExperimentConfig(
            name="standard_transformer",
            use_geometry=False
        ),
        ExperimentConfig(
            name="geochesszero",
            use_geometry=True
        )
    ]


def save_experiment_configs(
    output_dir="results/geochesszero"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = get_experiments()

    for experiment in experiments:
        output_file = output_dir / (
            experiment.name + "_config.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                asdict(experiment),
                file,
                indent=4
            )

        print(f"Saved: {output_file}")

    return experiments


if __name__ == "__main__":
    print("=== GeoChessZero Experiment Comparison ===")

    experiments = save_experiment_configs()

    for experiment in experiments:
        print(
            f"{experiment.name}: "
            f"use_geometry={experiment.use_geometry}"
        )

    print("Step 14.8 passed!")
