
from importlib import import_module

import torch

from .compare_config import get_experiments


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():
    print("=== GeoChessZero Experiment Orchestrator ===")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    policy_module = import_module(
        "prototypes.01_geochesszero.policy_value"
    )

    train_module = import_module(
        "prototypes.01_geochesszero.train"
    )

    ChessPolicyModel = policy_module.ChessPolicyModel
    create_optimizer = train_module.create_optimizer

    experiments = get_experiments()

    for experiment in experiments:

        print("\n------------------------------")
        print("Experiment:", experiment.name)

        # Same seed for fair initialization
        torch.manual_seed(experiment.seed)

        # Create model
        model = ChessPolicyModel(
            use_geometry=experiment.use_geometry
        ).to(device)

        # Create optimizer
        optimizer = create_optimizer(
            model,
            experiment
        )

        print("Geometry attention:", experiment.use_geometry)
        print("Parameters:", count_parameters(model))
        print("Optimizer:", type(optimizer).__name__)
        print("Learning rate:", optimizer.param_groups[0]["lr"])
        print("Weight decay:", optimizer.param_groups[0]["weight_decay"])

        assert count_parameters(model) == 820996
        assert isinstance(optimizer, torch.optim.AdamW)

        print("Model and optimizer ready!")

        del model, optimizer

    print("\nBoth experiments initialized successfully!")
    print("Step 14.18 passed!")


if __name__ == "__main__":
    main()
