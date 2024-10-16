"""
This module implements an autoencoder for aggregated
ESM embeddings that returns the original sequence.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import esm
from esm import Alphabet

import lightning as L

from hdbo_benchmark.utils.constants import ROOT_DIR

THIS_DIR = Path(__file__).parent.resolve()


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, alphabet: Alphabet, latent_dim: int = 32):
        super().__init__()
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.embedding_size = 1280
        self.max_sequence_length = 230
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, self.alphabet_size * self.max_sequence_length),
        )

        ESM_EMBEDDINGS_DIR = ROOT_DIR / "data" / "esm_embeddings"
        with open(ESM_EMBEDDINGS_DIR / "esm_embeddings.json") as fp:
            embeddings_and_sequences = json.load(fp)
        df = pd.DataFrame(embeddings_and_sequences)
        df.set_index("label", inplace=True)
        self.esm_df_of_precomputed_embeddings = df

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        embeddings, tokens = batch
        z = self.encoder(embeddings)

        tokens_onehot_hat = self.decoder(z)
        x_hat = tokens_onehot_hat.view(-1, self.alphabet_size, self.max_sequence_length)
        loss = nn.functional.cross_entropy(x_hat, tokens)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def encode(self, embeddings):
        return self.encoder(embeddings)

    def decode(self, z):
        tokens_onehot_hat = self.decoder(z)
        x_hat = tokens_onehot_hat.view(-1, self.alphabet_size, self.max_sequence_length)
        return x_hat

    def _from_token_ids_to_strings(self, token_ids: torch.Tensor) -> list[list[str]]:
        return [
            [
                self.alphabet.get_tok(int(token_idx_i.item()))
                for token_idx_i in token_idx
            ]
            for token_idx in token_ids
        ]

    def decode_to_string_array(self, z: np.ndarray) -> np.ndarray:
        assert z.ndim == 2
        tokens_onehot_hat = self.decoder(
            torch.from_numpy(z).to(torch.get_default_dtype()).to(self.device)
        )
        x_hat = tokens_onehot_hat.view(-1, self.alphabet_size, self.max_sequence_length)
        x_hat = x_hat.argmax(dim=1)

        return np.array(self._from_token_ids_to_strings(x_hat))

    def encode_from_string_array(self, x: np.ndarray) -> np.ndarray:
        embeddigs = []
        for x_i in x:
            sequence = "".join(x_i)
            df = self.esm_df_of_precomputed_embeddings
            embedding = df[df["sequence"] == sequence]["embedding"].values
        
            embeddigs.append([embedding])

        return np.array(embeddigs)


    def clean_up_special_tokens(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                ["".join([c for c in p if c not in ["<pad>", "<cls>", "<eos>"]])]
                for p in x
            ]
        )


if __name__ == "__main__":
    # Defining the training data and dataloaders
    with open(THIS_DIR / "esm_embeddings_pool.json") as f:
        embeddings_and_sequences = json.load(f)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    sequences = [e["sequence"] for e in embeddings_and_sequences]
    tokens = torch.Tensor([e["tokens"] for e in embeddings_and_sequences])
    embeddings = torch.Tensor([e["embedding"] for e in embeddings_and_sequences])

    alphabet_size = len(alphabet)
    max_sequence_length = len(sequences[0])

    # Define the dataset and dataloader
    # Splitting the dataset into train and test
    train_index = (4 * len(embeddings)) // 5
    train_embeddings = embeddings[:train_index]
    train_tokens = tokens[:train_index]
    test_embeddings = embeddings[train_index:]
    test_tokens = tokens[train_index:]

    train_dataset = TensorDataset(train_embeddings, train_tokens)
    test_dataset = TensorDataset(test_embeddings, test_tokens)
    train_dataloader = DataLoader(train_dataset, batch_size=256)
    test_dataloader = DataLoader(test_dataset, batch_size=256)

    # Define the model
    model = LitAutoEncoder(alphabet=alphabet)

    # Train the model
    trainer = L.Trainer(max_epochs=100)
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader
    )

    # Predicting for some embeddings
    latent_embeddings = model.encode(embeddings)
    decoded_tokens = model.decode(latent_embeddings)

    print(decoded_tokens.argmax(dim=1)[:5])
    print(tokens[0:5])

    print(["".join(x) for x in model.decode_to_string_array(latent_embeddings[:5])])
    print(["".join(x) for x in model._from_token_ids_to_strings(tokens[:5])])

    # Calculate the accuracy
    print(
        "accuracy:",
        torch.sum(decoded_tokens.argmax(dim=1) == tokens)
        / (len(tokens) * max_sequence_length),
    )

    # Let's take a look at the latent space
    import matplotlib.pyplot as plt

    plt.scatter(
        latent_embeddings.detach().numpy()[:, 0],
        latent_embeddings.detach().numpy()[:, 1],
    )

    plt.show()
