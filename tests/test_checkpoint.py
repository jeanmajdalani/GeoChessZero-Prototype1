
import importlib
import tempfile
from pathlib import Path

import torch


def main():
    print("=== GeoChessZero Checkpoint Test ===")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Import our existing model
    policy_module = importlib.import_module(
        "prototypes.01_geochesszero.policy_value"
    )

    # Import checkpoint functions
    train_module = importlib.import_module(
        "prototypes.01_geochesszero.train"
    )

    ChessPolicyModel = policy_module.ChessPolicyModel

    # Create GeoChessZero model
    torch.manual_seed(42)

    model = ChessPolicyModel(
        use_geometry=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003
    )

    print("GeoChessZero model created!")

    # Save original model parameters
    original_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }

    # Temporary checkpoint directory
    with tempfile.TemporaryDirectory() as tmpdir:

        checkpoint_path = (
            Path(tmpdir) / "test_checkpoint.pt"
        )

        # STEP 1: Save checkpoint
        train_module.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=3,
            val_loss=1.25,
            file_path=checkpoint_path
        )

        assert checkpoint_path.exists()

        print("Checkpoint file created successfully!")

        # STEP 2: Modify model parameters
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)

        print("Model parameters modified!")

        # STEP 3: Reload checkpoint
        epoch, val_loss = train_module.load_checkpoint(
            model=model,
            optimizer=optimizer,
            file_path=checkpoint_path,
            device=device
        )

        # STEP 4: Verify metadata
        assert epoch == 3
        assert val_loss == 1.25

        # STEP 5: Verify restored parameters
        for name, tensor in model.state_dict().items():

            assert torch.equal(
                tensor,
                original_state[name]
            ), f"Parameter not restored: {name}"

    print("Model parameters restored correctly!")
    print("Epoch restored:", epoch)
    print("Validation loss restored:", val_loss)

    print("\nStep 14.13 passed!")


if __name__ == "__main__":
    main()
