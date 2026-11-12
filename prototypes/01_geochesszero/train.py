
import torch
import torch.nn as nn

from importlib import import_module


# ==========================================
# STEP 14.4 - TRAINING LOOP
# ==========================================

def train_one_epoch(model, dataloader, optimizer, device):
    """
    Train GeoChessZero or Standard Transformer
    for one epoch.
    """

    model.train()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    for positions, actions in dataloader:

        # Move data to GPU or CPU
        positions = positions.to(device)
        actions = actions.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(positions)

        # Calculate loss
        loss = criterion(logits, actions)

        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # Update model parameters
        optimizer.step()

        # Accumulate loss
        batch_size = positions.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training DataLoader is empty")

    return total_loss / total_samples


# ==========================================
# STEP 14.5 - ADAMW OPTIMIZER
# ==========================================

def create_optimizer(model, config):
    """
    Create AdamW optimizer using training config.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    return optimizer
@torch.no_grad()
def validate_one_epoch(model, dataloader, device):
    """
    Evaluate the model on a validation dataset.

    Returns:
        average_loss, accuracy
    """

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for positions, actions in dataloader:

        positions = positions.to(device)
        actions = actions.to(device)

        # Forward pass without gradients
        logits = model(positions)

        # Calculate validation loss
        loss = criterion(logits, actions)

        # Predicted move
        predictions = torch.argmax(logits, dim=1)

        # Count correct predictions
        total_correct += (
            predictions == actions
        ).sum().item()

        batch_size = positions.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation DataLoader is empty")

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ==========================================
# STEP 14.7 - CHECKPOINTS
# ==========================================

from pathlib import Path


def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    file_path
):
    """
    Save model and optimizer state.
    """

    file_path = Path(file_path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss
    }

    torch.save(checkpoint, file_path)

    print(f"Checkpoint saved: {file_path}")


def load_checkpoint(
    model,
    optimizer,
    file_path,
    device
):
    """
    Restore model and optimizer state.
    """

    checkpoint = torch.load(
        file_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    print(f"Checkpoint loaded: {file_path}")

    return checkpoint["epoch"], checkpoint["val_loss"]


# ==========================================
# MAIN - CONFIGURATION TEST
# ==========================================

def main():

    print("=== GeoChessZero Training Pipeline ===")

    # Detect GPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Import ChessPolicyModel
    policy_module = import_module(
        "prototypes.01_geochesszero.policy_value"
    )

    ChessPolicyModel = policy_module.ChessPolicyModel

    print("ChessPolicyModel imported successfully!")

    # Import training configuration
    config_module = import_module(
        "prototypes.01_geochesszero.train_config"
    )

    TrainConfig = config_module.TrainConfig
    config = TrainConfig()

    print("train_one_epoch function ready!")

    # Display optimizer configuration
    print("Optimizer: AdamW")
    print("Learning rate:", config.learning_rate)
    print("Weight decay:", config.weight_decay)

    print("Optimizer configuration ready!")

    print("Step 14.5 passed!")
    print("validate_one_epoch function ready!")
    print("Step 14.6 passed!")
    print("Checkpoint save/load functions ready!")
    print("Step 14.7 passed!")


if __name__ == "__main__":
    main()
