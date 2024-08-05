"""
Implements a Variational Autoencoder in TF that can be trained on
SELFIES data from zinc250k.
"""
from __future__ import annotations
from typing import Tuple, Dict, Optional
from pathlib import Path
from itertools import product
import json

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, models, losses


from hdbo_benchmark.utils.selfies.tokens import from_selfie_to_tensor
from hdbo_benchmark.generative_models.vae import VAE

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()


class VAESelfiesTF(VAE):
    def __init__(
        self,
        latent_dim: int = 64,
        device: tf.device = tf.device("cpu"),
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
        self.encoder = models.Sequential(layers=[
            layers.InputLayer(input_shape=(self.input_length,)),
            layers.Dense(2048),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
        ])
        self.encoder_mu = layers.Linear(latent_dim)
        self.encoder_log_var = layers.Linear(latent_dim)

        # The decoder, which outputs the logits of the categorical
        # distribution over the vocabulary.
        self.decoder = models.Sequential(layers=[
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            layers.Dense(2048),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.2),
            layers.Dense(self.input_length),
        ])

        # Defines the prior
        self.p_z = tfp.distributions.Normal(
            loc=tf.zeros(latent_dim, device=device),
            scale=tf.ones(latent_dim, device=device),
        )

        # Moves to device
        self.device = device

    def encode(self, x: tf.Tensor) -> tfp.distributions.Normal:
        """
        Computes the approximate posterior q(z|x) over
        the latent variable z.
        """
        hidden = self.encoder(tf.reshape(x, (x.shape[0], -1)))
        mu = self.encoder_mu(hidden)
        log_var = self.encoder_log_var(hidden)

        return tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))

    def decode(self, z: tf.Tensor) -> tfp.distributions.Categorical:
        """
        Returns a categorical likelihood over the vocabulary
        """
        logits = self.decoder(z.to(self.device))

        # The categorical distribution expects (batch_size, ..., num_classes)
        return tfp.distributions.Categorical(
            logits=logits.reshape(
                -1, self.max_sequence_length, len(self.alphabet_s_to_i)
            )
        )

    def call(self, x: tf.Tensor) -> Tuple[tfp.distributions.Normal, tfp.distributions.Categorical]:
        """
        Computes a forward pass through the VAE, returning
        the distributions q_z_given_x and p_x_given_z.
        """
        q_z_given_x = self.encode(x)
        z = q_z_given_x.sample()

        p_x_given_z = self.decode(z)

        return q_z_given_x, p_x_given_z

    def loss_function(self, x: tf.Tensor) -> tf.Tensor:
        """
        Computes the ELBO loss for a given batch {x}.
        """
        q_z_given_x, p_x_given_z = self.forward(x)

        # Computes the KL divergence between q(z|x) and p(z)
        kl_div = tfp.distributions.kl_divergence(q_z_given_x, self.p_z).sum(dim=-1)
        kl_div = kl_div*0.01 # KLD contribution 1%

        # Computes the reconstruction loss
        recon_loss = -p_x_given_z.log_prob(x.argmax(dim=-1).to(self.device)).sum(dim=-1)

        # Computes the ELBO loss
        loss: tf.Tensor = (kl_div + recon_loss).mean()

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
        raise NotImplementedError

    def encode_from_string_array(self, x: np.ndarray) -> np.ndarray:
        # Assuming x is an array of strings [b, L] or [b,]
        selfies_strings: list[str] = ["".join(x_i) for x_i in x]
        onehot_x = tf.concat(
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
        z_tensor = tf.tensor(z, device=self.device, dtype=tf.float32)
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
