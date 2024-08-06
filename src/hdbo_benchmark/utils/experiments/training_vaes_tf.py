from typing import Callable

import tensorflow as tf
import wandb

from tensorflow.keras.optimizers import Adam

from torch.utils.data import DataLoader

from hdbo_benchmark.generative_models.vae_selfies_tf import VAESelfiesTF
from hdbo_benchmark.utils.constants import MODELS_DIR, DEVICE

MODELS_DIR.mkdir(exist_ok=True, parents=True)


def training_loop(
    model: VAESelfiesTF,
    training_data: tf.data.Dataset,
    optimizer: tf.keras.Optimizer,
) -> float:
    """
    Runs an epoch of training, returing the average training loss.
    """
    losses = []
    for batch in training_data:
        # setup the gradients
        with tf.GradientTape() as tape:
            # Compute the loss (forward pass is inside)
            loss = model.loss_function(batch)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Append for logging
        losses.append(loss.numpy())

    return sum(losses) / len(losses)


def testing_loop(model: VAESelfiesTF, testing_data: tf.data.Dataset) -> float:
    """
    Runs an epoch of testing, returing the average testing loss.
    """
    losses = []
    for batch in testing_data:
        # Compute the loss (forward pass is inside)
        loss = model.loss_function(batch)

        # Append for logging
        losses.append(loss.numpy())

    return sum(losses) / len(losses)


def train_model(
    model: VAESelfiesTF,
    train_loader: tf.data.Dataset,
    test_loader: tf.data.Dataset,
    experiment_name: str,
    max_epochs: int = 1_000_000,
    lr: float = 1e-3,
    early_stopping_patience: int = 200,
    logger: Callable[[float, float], None] = lambda training_loss, testing_loss: None,
    model_name_for_saving: str | None = None,
):
    # Define the optimizer
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)

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

        # Run a training epoch
        training_loss = training_loop(model, train_loader, optimizer)

        # Run a testing epoch
        testing_loss = testing_loop(model, test_loader)

        # Save the losses for plotting
        training_losses.append(training_loss)
        testing_losses.append(testing_loss)

        wandb.log({"training_loss": training_loss})
        wandb.log({"testing_loss": testing_loss})

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
                model.save(
                    MODELS_DIR / experiment_name / f"{model_name_for_saving}.h5",
                )

        if current_patience >= early_stopping_patience:
            print("Early stopping!")
            break

    print("Best testing loss: ", best_testing_loss)
