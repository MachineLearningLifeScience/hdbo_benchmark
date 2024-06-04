import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from hdbo_benchmark.utils.constants import ROOT_DIR


def load_mario_tensors(
    training_percentage=0.8,
    shuffle_seed=0,
    device="cpu",
):
    """Returns two tensors with training and testing data"""
    # Loading the data.
    # This data is structured [b, c, i, j], where c corresponds to the class.
    data = np.load(ROOT_DIR / "data" / "mario" / "all_levels_onehot.npz")["levels"]
    np.random.seed(shuffle_seed)
    np.random.shuffle(data)

    # Separating into training and test.
    n_data, _, _, _ = data.shape
    training_index = int(n_data * training_percentage)
    training_data = data[:training_index, :, :, :]
    testing_data = data[training_index:, :, :, :]
    training_tensors = torch.from_numpy(training_data).type(torch.float)
    test_tensors = torch.from_numpy(testing_data).type(torch.float)

    return training_tensors.to(device), test_tensors.to(device)


def load_mario_dataloaders(
    batch_size: int = 64,
    training_percentage: float = 0.8,
    shuffle_seed: int = 0,
    device: str | torch.device = "cpu",
    overfit_to_a_single_batch: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Returns train and test dataloaders for the Mario dataset."""
    # Loading the data.
    training_tensors, test_tensors = load_mario_tensors(
        training_percentage=training_percentage,
        shuffle_seed=shuffle_seed,
        device=device,
    )

    if overfit_to_a_single_batch:
        training_tensors = training_tensors[:batch_size]
        test_tensors = test_tensors[:batch_size]

    # Creating datasets.
    dataset = TensorDataset(training_tensors)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_tensors)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return data_loader, test_loader
