import json

import click

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

import esm

import lightning as L

from hdbo_benchmark.generative_models.ae_for_esm import LitAutoEncoder
from hdbo_benchmark.utils.constants import ROOT_DIR


@click.command()
@click.option("--latent-dim", type=int, default=2)
@click.option("--max-epochs", type=int, default=100)
@click.option("--seed", type=int, default=None)
@click.option("--batch-size", type=int, default=256)
@click.option("--strict-on-hash/--no-strict-on-hash", default=True)
def main(latent_dim, max_epochs, seed, batch_size, strict_on_hash):
    ESM_DATA_DIR = ROOT_DIR / "data" / "esm_embeddings"
    MODELS_DIR = (
        ROOT_DIR / "data" / "trained_models" / "ae_for_esm" / f"latent_dim_{latent_dim}"
    )

    if seed is None:
        seed = np.random.randint(0, 1_000)

    torch.manual_seed(seed)

    # Defining the training data and dataloaders
    with open(ESM_DATA_DIR / "esm_embeddings_pool.json") as f:
        embeddings_and_sequences = json.load(f)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    tokens = torch.Tensor([e["tokens"] for e in embeddings_and_sequences])
    embeddings = torch.Tensor([e["embedding"] for e in embeddings_and_sequences])

    # Define the dataset and dataloader
    # Splitting the dataset into train and test
    train_index = (4 * len(embeddings)) // 5
    train_embeddings = embeddings[:train_index]
    train_tokens = tokens[:train_index]
    test_embeddings = embeddings[train_index:]
    test_tokens = tokens[train_index:]

    train_dataset = TensorDataset(train_embeddings, train_tokens)
    test_dataset = TensorDataset(test_embeddings, test_tokens)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Define the model
    model = LitAutoEncoder(alphabet=alphabet, latent_dim=latent_dim)

    # Train the model
    trainer = L.Trainer(max_epochs=max_epochs, default_root_dir=MODELS_DIR)
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader
    )
