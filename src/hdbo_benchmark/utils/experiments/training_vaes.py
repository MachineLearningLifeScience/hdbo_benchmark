from typing import Callable

import torch
from torch.utils.data import DataLoader

from hdbo_benchmark.generative_models import VAEMario, VAESelfies
from hdbo_benchmark.utils.constants import DEVICE, MODELS_DIR

MODELS_DIR.mkdir(exist_ok=True, parents=True)


def training_loop(
    model: VAESelfies | VAEMario,
    training_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    Runs an epoch of training, returing the average training loss.
    """
    losses = []
    for (batch,) in training_loader:
        # Reset the gradients
        optimizer.zero_grad()

        # Compute the loss (forward pass is inside)
        loss = model.loss_function(batch)

        # Run a backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Append for logging
        losses.append(loss.item())

    return sum(losses) / len(losses)


def testing_loop(model: VAESelfies | VAEMario, testing_loader: DataLoader) -> float:
    """
    Runs an epoch of testing, returing the average testing loss.
    """
    losses = []
    with torch.no_grad():
        for (batch,) in testing_loader:
            # Compute the loss (forward pass is inside)
            loss = model.loss_function(batch)

            # Append for logging
            losses.append(loss.item())

    return sum(losses) / len(losses)


def train_model(
    model: VAESelfies | VAEMario,
    train_loader: DataLoader,
    test_loader: DataLoader,
    experiment_name: str,
    max_epochs: int = 1_000_000,
    lr: float = 1e-3,
    early_stopping_patience: int = 50,
    logger: Callable[[float, float], None] = lambda training_loss, testing_loss: None,
    model_name_for_saving: str | None = None,
):
    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Run the training loop
    best_testing_loss = float("inf")
    current_patience = 0

    # Saving the losses for plotting
    training_losses = []
    testing_losses = []

    # Create the directory to save the models
    if model_name_for_saving is not None:
        (MODELS_DIR / experiment_name).mkdir(exist_ok=True, parents=True)

    for epoch in range(max_epochs):
        model.to(DEVICE)

        # Run a training epoch
        training_loss = training_loop(model, train_loader, optimizer)

        # Run a testing epoch
        testing_loss = testing_loop(model, test_loader)

        # Save the losses for plotting
        training_losses.append(training_loss)
        testing_losses.append(testing_loss)

        # Log the results
        logger(training_loss, testing_loss)
        print(
            f"Epoch {epoch + 1}/{max_epochs}: Training loss: {training_loss:.4f}, Testing loss: {testing_loss:.4f}"
        )

        # Early stopping
        improvement_flag = testing_loss < best_testing_loss
        if improvement_flag:
            best_testing_loss = testing_loss
            current_patience = 0
        else:
            current_patience += 1

        # Save the best model so far
        if model_name_for_saving is not None:
            if epoch == 0 or improvement_flag:
                torch.save(
                    model.to("cpu").state_dict(),
                    MODELS_DIR / experiment_name / f"{model_name_for_saving}.pt",
                )

        if current_patience >= early_stopping_patience:
            print("Early stopping!")
            break

    print("Best testing loss: ", best_testing_loss)
