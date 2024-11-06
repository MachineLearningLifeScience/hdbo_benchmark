from pathlib import Path

import gpytorch  # type: ignore[import]
import torch

ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.resolve()
MODELS_DIR = ROOT_DIR / "data" / "trained_models" / "gps"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def filter_close_points(x, y, tolerance=1e-2):
    all_pairwise_distances_of_x = torch.cdist(x, x)
    mask = torch.triu(all_pairwise_distances_of_x < tolerance, diagonal=1)

    points_are_close_map = {}
    for i in range(len(x)):
        close_points = torch.nonzero(mask[i]).squeeze(1)
        points_are_close_map[i] = close_points.tolist()

    # Filtering one element of each equivalence class:
    indices_to_keep = set(range(len(x)))
    for close_points in points_are_close_map.values():
        indices_to_keep -= set(close_points)

    indices_to_keep = list(indices_to_keep)
    x_ = x[indices_to_keep]
    y_ = y[indices_to_keep]

    return x_, y_


def train_exact_gp_model(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    n_points: int,
    n_dimensions: int,
    patience: int = 50,
    lr: float = 0.05,
    model_name: str = "model",
    max_nr_iterations: int = 1000,
    verbose: bool = True,
) -> tuple[gpytorch.models.ExactGP, gpytorch.likelihoods.GaussianLikelihood, int, bool]:
    """Trains an ExactGP using early stopping and ADAM."""
    # print(f"training {model}")
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    # This training loop uses early stopping.
    training_had_nans = False
    best_likelihood_state_dict = None
    best_model_state_dict = None
    best_loss = float("inf")
    patience_counter = 0
    for i in range(max_nr_iterations):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Output from model
        output = model(train_x)

        # Calculate loss and backpropagate gradients
        loss = -mll(output, train_y)
        loss.backward()

        # Computing the test loss
        with torch.no_grad():
            model.eval()
            likelihood.eval()
            mll.eval()

            try:
                test_output = model(test_x)
                test_loss = -mll(test_output, test_y)
            except Exception as e:
                if verbose:
                    print(f"Exception {e} occurred at test iteration {i + 1}.")
                test_loss = torch.tensor(float("inf"))

                pass

            model.train()
            likelihood.train()
            mll.train()

        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0

            # Save the model
            best_model_state_dict = model.state_dict()
            best_likelihood_state_dict = likelihood.state_dict()

            torch.save(
                best_model_state_dict,
                MODELS_DIR
                / f"model_{model_name}_npoints_{n_points}_ndim_{n_dimensions}.pt",
            )
            # torch.save(
            #     best_likelihood_state_dict,
            #     MODELS_DIR
            #     / f"likelihood_{model_name}_npoints_{n_points}_ndim_{n_dimensions}",
            # )

        else:
            patience_counter += 1

        if patience_counter > patience:
            if verbose:
                print(
                    f"Patience reached at iteration {i + 1}. Stopping training early."
                )
            break

        if torch.isnan(loss):
            if verbose:
                print(f"Loss is NaN at iteration {i + 1}.")
            training_had_nans = True
            break

        optimizer.step()

        if verbose:
            print(
                f"Nr. points: {n_points} - Dim: {n_dimensions} - Iteration {i + 1}/{max_nr_iterations} - Train loss: {loss.item()} - Test loss: {test_loss.item()}"
            )

    # Load the best model
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        likelihood.load_state_dict(best_likelihood_state_dict)

    # Set the model and likelihood to evaluation mode
    model.eval()
    likelihood.eval()

    return model, likelihood, i, training_had_nans
