"""
Implements a Variational Autoencoder that can be trained on
SELFIES data from zinc250k.
"""
from __future__ import annotations
from typing import Tuple, Dict, Optional
from pathlib import Path
from itertools import product
import json

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.distributions import Normal, Categorical

# from hdbo_benchmark.utils.selfies.visualization import (
#     selfie_to_numpy_image_array,
# )
from hdbo_benchmark.utils.selfies.tokens import from_selfie_to_tensor
from hdbo_benchmark.generative_models.vae import VAE

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()


class VAESelfies(VAE):
    def __init__(
        self,
        latent_dim: int = 64,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        with open(
            ROOT_DIR
            / "data"
            / "small_molecule_datasets"
            / "processed"
            / "zinc250k_alphabet_stoi.json",
            "r",
        ) as fp:
            alphabet_s_to_i = json.load(fp)

        super().__init__(
            latent_dim=latent_dim,
            alphabet_s_to_i=alphabet_s_to_i,
            device=device,
        )

        # Load the metadata to find the maximum sequence length
        with open(
            ROOT_DIR
            / "data"
            / "small_molecule_datasets"
            / "processed"
            / "zinc250k_metadata.json",
            "r",
        ) as fp:
            metadata = json.load(fp)

        self.max_sequence_length = metadata["max_sequence_length"]

        # Define the input length: length of a given SELFIES
        # (always padded to be {max_length}), times the number of tokens
        self.input_length = self.max_sequence_length * len(self.alphabet_s_to_i)

        # Define the model
        self.encoder = nn.Sequential(
            nn.Linear(self.input_length, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )
        self.encoder_mu = nn.Linear(256, latent_dim)
        self.encoder_log_var = nn.Linear(256, latent_dim)

        # The decoder, which outputs the logits of the categorical
        # distribution over the vocabulary.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.input_length),
        )

        # Defines the prior
        self.p_z = Normal(
            loc=torch.zeros(latent_dim, device=device),
            scale=torch.ones(latent_dim, device=device),
        )

        # Moves to device
        self.to(device)

    def encode(self, x: torch.Tensor) -> Normal:
        """
        Computes the approximate posterior q(z|x) over
        the latent variable z.
        """
        hidden = self.encoder(x.flatten(start_dim=1).to(self.device))
        mu = self.encoder_mu(hidden)
        log_var = self.encoder_log_var(hidden)

        return Normal(loc=mu, scale=torch.exp(0.5 * log_var))

    def decode(self, z: torch.Tensor) -> Categorical:
        """
        Returns a categorical likelihood over the vocabulary
        """
        logits = self.decoder(z.to(self.device))

        # The categorical distribution expects (batch_size, ..., num_classes)
        return Categorical(
            logits=logits.reshape(
                -1, self.max_sequence_length, len(self.alphabet_s_to_i)
            )
        )

    def forward(self, x: torch.Tensor) -> Tuple[Normal, Categorical]:
        """
        Computes a forward pass through the VAE, returning
        the distributions q_z_given_x and p_x_given_z.
        """
        q_z_given_x = self.encode(x)
        z = q_z_given_x.rsample()

        p_x_given_z = self.decode(z)

        return q_z_given_x, p_x_given_z

    def loss_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the ELBO loss for a given batch {x}.
        """
        q_z_given_x, p_x_given_z = self.forward(x)

        # Computes the KL divergence between q(z|x) and p(z)
        kl_div = torch.distributions.kl_divergence(q_z_given_x, self.p_z).sum(dim=-1)
        kl_div = kl_div*0.001

        # Computes the reconstruction loss
        recon_loss = -p_x_given_z.log_prob(x.argmax(dim=-1).to(self.device)).sum(dim=-1)

        # Computes the ELBO loss
        loss: torch.Tensor = (kl_div + recon_loss).mean()

        return loss

    def plot_grid(
        self,
        x_lims: tuple[float, float] = (-5, 5),
        y_lims: tuple[float, float] = (-5, 5),
        n_rows: int = 10,
        n_cols: int = 10,
        sample: bool = False,
        ax: Optional[plt.Axes] = None,
    ) -> np.ndarray:
        """
        A helper function which plots, as images, the levels in a
        fine grid in latent space, specified by the provided limits,
        number of rows and number of columns.

        The figure can be plotted in a given axis; if none is passed,
        a new figure is created.

        This function also returns the final image (which is the result
        of concatenating all the individual decoded images) as a numpy
        array.
        """
        img_width_and_height = 200
        z1 = np.linspace(*x_lims, n_cols)
        z2 = np.linspace(*y_lims, n_rows)

        zs = np.array([[a, b] for a, b in product(z1, z2)])

        selfies_dist = self.decode(torch.from_numpy(zs).type(torch.float))
        if sample:
            selfies_as_ints = selfies_dist.sample()
        else:
            selfies_as_ints = selfies_dist.probs.argmax(dim=-1)

        inverse_alphabet: Dict[int, str] = {
            v: k for k, v in self.alphabet_s_to_i.items()
        }
        selfies_strings = [
            "".join([inverse_alphabet[i] for i in row])
            for row in selfies_as_ints.numpy(force=True)
        ]

        selfies_as_images = np.array(
            [
                selfie_to_numpy_image_array(
                    selfie, width=img_width_and_height, height=img_width_and_height
                )
                for selfie in selfies_strings
            ]
        )
        img_dict = {(z[0], z[1]): img for z, img in zip(zs, selfies_as_images)}

        positions = {
            (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
        }

        pixels = img_width_and_height
        final_img = np.zeros((n_cols * pixels, n_rows * pixels, 3))
        for z, (i, j) in positions.items():
            final_img[i * pixels : (i + 1) * pixels, j * pixels : (j + 1) * pixels] = (
                img_dict[z]
            )

        final_img = final_img.astype(int)

        if ax is not None:
            ax.imshow(final_img, extent=(*x_lims, *y_lims))

        return final_img

    def encode_from_string_array(self, x: np.ndarray) -> np.ndarray:
        # Assuming x is an array of strings [b, L] or [b,]
        selfies_strings: list[str] = ["".join(x_i) for x_i in x]
        onehot_x = torch.cat(
            [
                from_selfie_to_tensor(
                    selfie,
                    self.alphabet_s_to_i,
                    sequence_length=self.max_sequence_length,
                )
                for selfie in selfies_strings
            ]
        )

        z_dist = self.encode(onehot_x)
        z_ = z_dist.mean
        z: np.ndarray = z_.cpu().detach().numpy()

        return z

    def decode_to_string_array(self, z: np.ndarray) -> np.ndarray:
        z_tensor = torch.tensor(z, device=self.device, dtype=torch.float32)
        categorical = self.decode(z_tensor)
        logits = categorical.logits.detach().cpu().numpy()
        return self._logits_to_string_array(logits)

    def _logits_to_string_array(self, logits: np.ndarray) -> np.ndarray:
        indices = np.argmax(logits, axis=-1)
        sequences = []
        for idx_seq in indices:
            sequence = "".join(self.alphabet_i_to_s[idx] for idx in idx_seq)
            sequences.append(sequence)
        return np.array(sequences)
