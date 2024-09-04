import json

import click

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

import esm

import lightning as L

from hdbo_benchmark.generative_models.ae_for_esm import LitAutoEncoder
from hdbo_benchmark.utils.constants import ROOT_DIR
from hdbo_benchmark.utils.download.from_drive import download_file_from_google_drive


@click.command()
@click.option("--latent-dim", type=int, default=2)
@click.option("--max-epochs", type=int, default=100)
@click.option("--seed", type=int, default=None)
@click.option("--batch-size", type=int, default=256)
def main(latent_dim, max_epochs, seed, batch_size):
    ESM_DATA_DIR = ROOT_DIR / "data" / "esm_embeddings"
    MODELS_DIR = (
        ROOT_DIR / "data" / "trained_models" / "ae_for_esm" / f"latent_dim_{latent_dim}"
    )

    if seed is None:
        seed = np.random.randint(0, 1_000)

    L.seed_everything(seed)

    # Defining the training data and dataloaders
    POOL_FILE = ESM_DATA_DIR / "esm_embeddings_pool.json"
    if not POOL_FILE.exists():
        download_file_from_google_drive(
            "1xHZ9P48u6a2PCxe8EUfF3vA3EpVyYMhQ",  # The ID of the file on drive.
            POOL_FILE,
            md5_checksum="d5cfa674d621d6e54359ff185ab6fd69",
        )

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

    # Predicting for some embeddings
    latent_embeddings = model.encode(embeddings)
    decoded_tokens = model.decode(latent_embeddings)

    print(decoded_tokens.argmax(dim=1)[:5])
    print(tokens[0:5])

    print(
        [
            "".join(x)
            for x in model.decode_to_string_array(
                latent_embeddings[:5].numpy(force=True)
            )
        ]
    )
    print(["".join(x) for x in model._from_token_ids_to_strings(tokens[:5])])

    # Calculate the accuracy
    print(
        "accuracy:",
        torch.sum(decoded_tokens.argmax(dim=1) == tokens)
        / (len(tokens) * model.max_sequence_length),
    )


if __name__ == "__main__":
    main()
