from typing import Tuple

import numpy as np
import torch
from torch.distributions import Normal, Categorical


class VAE(torch.nn.Module):
    """
    A common interface for VAEs
    """

    def __init__(
        self,
        latent_dim: int,
        alphabet_s_to_i: dict[str, int],
        device: torch.device,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.alphabet_s_to_i = alphabet_s_to_i
        self.alphabet_i_to_s = {v: k for k, v in alphabet_s_to_i.items()}

        # Defines the prior
        self.p_z = Normal(
            loc=torch.zeros(latent_dim, device=device),
            scale=torch.ones(latent_dim, device=device),
        )

    def encode(self, x: torch.Tensor) -> Normal:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> Categorical:
        raise NotImplementedError

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

        # Computes the reconstruction loss
        recon_loss = -p_x_given_z.log_prob(x.argmax(dim=-1).to(self.device)).sum(dim=-1)

        # Computes the ELBO loss
        loss: torch.Tensor = (kl_div + recon_loss).mean()

        return loss

    def encode_from_string_array(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decode_to_string_array(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError
