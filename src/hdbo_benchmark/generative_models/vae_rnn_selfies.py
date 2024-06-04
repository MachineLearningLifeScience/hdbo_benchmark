import json

import numpy as np
import torch
from torch.distributions import Normal, Categorical

import torch.nn as nn

from hdbo_benchmark.generative_models.vae import VAE
from hdbo_benchmark.utils.constants import ROOT_DIR, DEVICE


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True).to(
            DEVICE
        )
        self.hidden2mean = nn.Linear(hidden_dim, latent_dim).to(DEVICE)
        self.hidden2logvar = nn.Linear(hidden_dim, latent_dim).to(DEVICE)

        self.to(DEVICE)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = h_n[-1]  # Use the last layer's hidden state
        mean = self.hidden2mean(h_n)
        logvar = self.hidden2logvar(h_n)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True).to(
            DEVICE
        )
        self.hidden2output = nn.Linear(hidden_dim, output_dim)

        self.to(DEVICE)

    def forward(self, z, seq_len):
        hidden = self.latent2hidden(z)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm(hidden)
        out = self.hidden2output(out)
        return out


class VAERNNSelfies(VAE):
    def __init__(
        self,
        latent_dim: int,
        device: torch.device,
        hidden_dim: int = 256,
        num_layers: int = 1,
        sequence_length: int = 70,
    ):
        with open(
            ROOT_DIR
            / "data"
            / "small_molecule_datasets"
            / "processed"
            / "zinc250k_alphabet_stoi.json",
            "r",
        ) as fp:
            alphabet_s_to_i = json.load(fp)

        super().__init__(latent_dim, alphabet_s_to_i, device)
        self.input_dim = len(alphabet_s_to_i)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        self.encoder = Encoder(
            self.input_dim, self.hidden_dim, latent_dim, self.num_layers
        ).to(device)
        self.decoder = Decoder(
            latent_dim, self.hidden_dim, self.input_dim, self.num_layers
        ).to(device)

        self.to(device)

    def encode(self, x: torch.Tensor) -> Normal:
        mean, logvar = self.encoder(x)
        return Normal(mean, torch.exp(0.5 * logvar))

    def decode(self, z: torch.Tensor) -> Categorical:
        logits = self.decoder(z, self.sequence_length)
        return Categorical(logits=logits)

    def encode_from_string_array(self, x: np.ndarray) -> np.ndarray:
        x_tensor = self._one_hot_encode(x).to(self.device)
        q = self.encode(x_tensor)
        mean: np.ndarray = q.loc.detach().cpu().numpy()
        return mean

    def decode_to_string_array(self, z: np.ndarray) -> np.ndarray:
        z_tensor = torch.tensor(z, device=self.device, dtype=torch.float32)
        categorical = self.decode(z_tensor)
        logits = categorical.logits.detach().cpu().numpy()
        return self._logits_to_string_array(logits)

    def _one_hot_encode(self, x: np.ndarray) -> torch.Tensor:
        batch_size, seq_len = x.shape
        one_hot = np.zeros((batch_size, seq_len, self.input_dim), dtype=np.float32)
        for i, sequence in enumerate(x):
            for j, char in enumerate(sequence):
                one_hot[i, j, self.alphabet_s_to_i[char]] = 1.0
        return torch.tensor(one_hot, device=self.device)

    def _logits_to_string_array(self, logits: np.ndarray) -> np.ndarray:
        indices = np.argmax(logits, axis=-1)
        sequences = []
        for idx_seq in indices:
            sequence = "".join(self.alphabet_i_to_s[idx] for idx in idx_seq)
            sequences.append(sequence)
        return np.array(sequences)

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


if __name__ == "__main__":
    from hdbo_benchmark.utils.constants import MODELS_DIR, DEVICE

    vae = VAERNNSelfies(
        latent_dim=256,
        device=DEVICE,
        hidden_dim=512,
        num_layers=1,
    )
    opt_vae = torch.compile(vae)
    opt_vae.load_state_dict(
        torch.load(
            MODELS_DIR
            / "training_vae_on_zinc_250k"
            / "rnn-latent_dim-256-batch_size-512-lr-0.0005-seed-49.pt"
        )
    )

    z = opt_vae.p_z.rsample((10,))
    print(z)

    print(opt_vae.decode_to_string_array(z.numpy()))
