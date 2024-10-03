"""
A categorical VAE that can train on Mario.

The notation very much follows the original VAE paper.
"""

from itertools import product
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Distribution, Normal, Categorical, kl_divergence
import torch.nn as nn

from hdbo_benchmark.generative_models.vae import VAE
from hdbo_benchmark.utils.data.mario import load_mario_tensors
# from hdbo_benchmark.utils.visualization.levels import get_img_from_level


class VAEMario(VAE):
    """
    A VAE that decodes to the Categorical distribution
    on "sentences" of shape (h, w).
    """

    def __init__(
        self,
        latent_dim: int,
        device: torch.device,
        w: int = 14,
        h: int = 14,
        n_sprites: int = 11,
    ):
        alphabet_s_to_i = {
            "X": 0,
            "S": 1,
            "-": 2,
            "?": 3,
            "Q": 4,
            "E": 5,
            "<": 6,
            ">": 7,
            "[": 8,
            "]": 9,
            "o": 10,
        }

        super(VAEMario, self).__init__(
            latent_dim=latent_dim, alphabet_s_to_i=alphabet_s_to_i, device=device
        )
        self.w = w
        self.h = h
        self.n_sprites = n_sprites
        self.input_dim = w * h * n_sprites  # for flattening

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, max(512, latent_dim)),
            nn.Tanh(),
            nn.Linear(max(512, latent_dim), max(256, latent_dim)),
            nn.Tanh(),
            nn.Linear(max(256, latent_dim), max(128, latent_dim)),
            nn.Tanh(),
        ).to(self.device)
        self.enc_mu = nn.Sequential(nn.Linear(max(128, latent_dim), latent_dim)).to(
            self.device
        )
        self.enc_var = nn.Sequential(nn.Linear(max(128, latent_dim), latent_dim)).to(
            self.device
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, max(256, latent_dim)),
            nn.Tanh(),
            nn.Linear(max(256, latent_dim), max(512, latent_dim)),
            nn.Tanh(),
            nn.Linear(max(512, latent_dim), self.input_dim),
        ).to(self.device)

        self.train_data, self.test_data = load_mario_tensors(device=self.device)

    def encode(self, x: torch.Tensor) -> Normal:
        """
        An encoding function that returns the normal distribution
        q(z|x) for some data x.

        It flattens x after the first dimension, passes it through
        the encoder networks which parametrize the mean and log-variance
        of the Normal, and returns the distribution.
        """
        x = x.view(-1, self.input_dim).to(self.device)
        result = self.encoder(x)
        mu = self.enc_mu(result)
        log_var = self.enc_var(result)

        return Normal(mu, torch.exp(0.5 * log_var))

    def decode(self, z: torch.Tensor) -> Categorical:
        """
        A decoding function that returns the Categorical distribution
        p(x|z) for some latent codes z.

        It passes it through the decoder network, which parametrizes
        the logits of the Categorical distribution of shape (h, w).
        """
        logits = self.decoder(z)
        p_x_given_z = Categorical(
            logits=logits.reshape(-1, self.h, self.w, self.n_sprites)
        )

        return p_x_given_z

    def encode_from_string_array(self, x: np.ndarray) -> np.ndarray:
        # Assuming that x is an array of strings [b, L], or [b,]
        assert len(x.shape) in [1, 2]
        levels: list[str] = ["".join(x_i) for x_i in x]

        # We need to move from 14*14 strings to (14, 14) strings
        # and then to onehots of (14, 14, 11)
        onehot_x = torch.cat([self._from_level_to_onehot(level) for level in levels])

        z_dist = self.encode(onehot_x)
        z_ = z_dist.mean
        z: np.ndarray = z_.cpu().detach().numpy()

        return z

    def _from_level_to_onehot(self, level: str):
        assert len(level) == 14 * 14
        onehot = torch.zeros(self.h, self.w, self.n_sprites)
        for i, c in enumerate(level):
            onehot[i // self.w, i % self.w, self.alphabet_s_to_i[c]] = 1

        return onehot

    def decode_to_string_array(self, z: np.ndarray) -> np.ndarray: ...

    def plot_grid(
        self,
        x_lims: tuple[float, float] = (-5.0, 5.0),
        y_lims: tuple[float, float] = (-5.0, 5.0),
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
        z1 = np.linspace(*x_lims, n_cols)
        z2 = np.linspace(*y_lims, n_rows)

        zs = np.array([[a, b] for a, b in product(z1, z2)])

        images_dist = self.decode(torch.from_numpy(zs).type(torch.float))
        if sample:
            images = images_dist.sample()
        else:
            images = images_dist.probs.argmax(dim=-1)

        images = np.array(
            [get_img_from_level(im) for im in images.cpu().detach().numpy()]
        )
        img_dict = {(z[0], z[1]): img for z, img in zip(zs, images)}

        positions = {
            (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
        }

        pixels = 16 * 14
        final_img = np.zeros((n_cols * pixels, n_rows * pixels, 3))
        for z, (i, j) in positions.items():
            final_img[i * pixels : (i + 1) * pixels, j * pixels : (j + 1) * pixels] = (
                img_dict[z]
            )

        final_img = final_img.astype(int)

        if ax is not None:
            ax.imshow(final_img, extent=(*x_lims, *y_lims))

        return final_img


if __name__ == "__main__":
    vae = VAEMario()
    print(vae)
