"""
Implements a Variational Autoencoder that loads MolOpt VAE model. 
It is originally designed to handle the
SELFIES.
"""

from typing import Tuple, Dict, Optional
from pathlib import Path
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Categorical

from hdbo_benchmark.utils.selfies.tokens import from_selfie_to_tensor
from hdbo_benchmark.generative_models.vae import VAE

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()

# add model, utils, etc of molopt to the namespace
sys.path.append(
    str(
        ROOT_DIR
        / "src"
        / "hdbo_benchmark"
        / "generative_models"
        / "molopt"
        / "selfies_vae"
    )
)
import models
import utils


class VAEMolOpt(VAE):
    def __init__(
        self,
        max_len: int = 100,
        model_path: Path = None,
        vocab_path: Path = None,
        config_path: Path = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        molopt_assets_path = (
            ROOT_DIR
            / "src"
            / "hdbo_benchmark"
            / "generative_models"
            / "molopt"
            / "selfies_vae"
            / "checkpoint"
        )
        if not model_path:
            model_path = molopt_assets_path / "selfies_vae_model_020.pt"
        if not vocab_path:
            vocab_path = molopt_assets_path / "selfies_vae_vocab.txt"
        if not config_path:
            config_path = molopt_assets_path / "selfies_vae_config.pt"
        self.vocab = torch.load(vocab_path)
        for ss in ("bos", "eos", "unk", "pad"):
            setattr(self, ss, getattr(self.vocab, ss))
        self._config = torch.load(config_path)  # TODO: check if latent dim is d_z
        alphabet_s_to_i = self.vocab.c2i  # TODO: assess if that is the same object
        self.max_len = max_len

        super().__init__(
            latent_dim=self._config.d_z,
            alphabet_s_to_i=alphabet_s_to_i,
            device=device,
        )
        self._vae = torch.load(model_path)
        # Moves to device
        self.to(device)

    def _forward(self, x: torch.Tensor) -> torch.TensorType:
        """
        NOTE: molopt builtin forward returns z and kl_loss, NOT mu or logvar.
        Custom shorter forward_encoder implementation below.
        """
        x = [self._vae.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)

        _, h = self._vae.encoder_rnn(x, None)

        h = h[-(1 + int(self._vae.encoder_rnn.bidirectional)) :]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)

        mu, logvar = self._vae.q_mu(h), self._vae.q_logvar(h)
        return mu, logvar

    def encode(self, x: torch.Tensor) -> Normal:
        """
        Computes the approximate posterior q(z|x) over
        the latent variable z.
        """
        mu, log_var = self._forward(x)

        return Normal(loc=mu, scale=torch.exp(0.5 * log_var))

    def decode(self, z: torch.Tensor, temp: float = 1.0) -> Categorical:
        """
        Returns a categorical likelihood over the vocabulary
        """
        n_batch = 2
        z = torch.cat([z, z], dim=0)
        probits = []
        with torch.no_grad():
            z = z.to(self.device)
            z_0 = z.unsqueeze(1)
            # Initial values
            h = self._vae.decoder_lat(z)
            # print('decode', h.shape)
            h = h.unsqueeze(0).repeat(self._vae.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(
                n_batch, self.max_len
            )
            x[:, 0] = self.bos

            # Generating cycle
            for i in range(0, self.max_len):
                x_emb = self._vae.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)
                o, h = self._vae.decoder_rnn(x_input, h)
                y = self._vae.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)
                probits.append(y)

        probs = torch.stack(probits, axis=1)[
            0
        ]  # select first entry of the n_batch=2, see original implementation
        # The categorical distribution expects (batch_size, ..., num_classes)
        return Categorical(probs=probs)

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
        # TODO: use reference loss implementation
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
        # Assuming x is an array of strings [b, L] or [b,]
        # TODO: assumes tokenization
        # selfies_strings: list[str] = ["".join(x_i) for x_i in x]
        encoded_selfies = np.array(
            [[self.vocab.c2i.get(s) for s in selfie] for selfie in x]
        )
        enc_x = torch.from_numpy(encoded_selfies)

        z_dist = self.encode(enc_x)
        z_ = z_dist.mean
        z: np.ndarray = z_.cpu().detach().numpy()

        return z

    def decode_to_string_array(self, z: np.ndarray, sample=False) -> np.ndarray:
        selfie_strs = []
        for z_i in z:
            decoder_cat = self.decode(
                torch.from_numpy(np.atleast_1d(z_i.reshape(1, -1))).to(
                    torch.get_default_dtype()
                )
            )
            if not sample:
                id_seqs = decoder_cat.probs.argmax(0).detach().numpy()
            else:
                id_seqs = decoder_cat.sample().detach().numpy()
            selfie_str = np.array([self.vocab.i2c.get(s) for s in id_seqs])
            selfie_strs.append("".join(selfie_str))
        return np.array(selfie_strs)
