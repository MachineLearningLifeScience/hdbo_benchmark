"""
Defines a common way of loading VAEs from just the name
of the experiment and the latent dimension.
"""

from typing import Literal

import torch

from hdbo_benchmark.generative_models.vae import VAE
from hdbo_benchmark.generative_models.vae_selfies import VAESelfies
from hdbo_benchmark.generative_models.vae_rnn_selfies import VAERNNSelfies
from hdbo_benchmark.generative_models.vae_mario import VAEMario
from hdbo_benchmark.utils.constants import MODELS_DIR, DEVICE


class VAEFactory:
    def create(
        self,
        experiment_name: Literal["benchmark_on_mario", "benchmark_on_pmo"],
        latent_dim: int,
    ) -> VAE:
        match experiment_name:
            case "benchmark_on_pmo":
                vae = self._create_vae_on_molecules(latent_dim)
            case "benchmark_on_mario":
                vae = self._create_vae_on_mario(latent_dim)
            case _:
                raise ValueError("...")

        return vae

    def _create_vae_on_mario(self, latent_dim: int) -> VAEMario:
        match latent_dim:
            case 2:
                weights_path = (
                    MODELS_DIR
                    / "vae_mario-latent_dim-2-seed-2"
                    / "latent_dim-2-batch_size-256-lr-0.001-seed-2.pt"
                )
            case 16:
                weights_path = (
                    MODELS_DIR
                    / "vae_mario-latent_dim-16-seed-2"
                    / "latent_dim-16-batch_size-256-lr-0.001-seed-2.pt"
                )
            case 64:
                weights_path = (
                    MODELS_DIR
                    / "vae_mario-latent_dim-64-seed-3"
                    / "latent_dim-64-batch_size-256-lr-0.001-seed-3.pt"
                )
            case 256:
                weights_path = (
                    MODELS_DIR
                    / "vae_mario-latent_dim-256-seed-3"
                    / "latent_dim-256-batch_size-256-lr-0.001-seed-3.pt"
                )
            case 512:
                weights_path = (
                    MODELS_DIR
                    / "vae_mario-latent_dim-512-seed-2"
                    / "latent_dim-512-batch_size-256-lr-0.001-seed-2.pt"
                )
            case 1024:
                weights_path = (
                    MODELS_DIR
                    / "vae_mario-latent_dim-1024-seed-1"
                    / "latent_dim-1024-batch_size-256-lr-0.001-seed-1.pt"
                )
            case _:
                raise NotImplementedError

        vae = VAEMario(latent_dim=latent_dim, device=DEVICE)
        opt_vae: VAEMario = torch.compile(vae)
        opt_vae.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        return opt_vae

    def _create_vae_on_molecules(self, latent_dim: int) -> VAESelfies:
        match latent_dim:
            case 2:
                weights_path = ( # TODO: d=2 OLD ARCHITECTURE -> RETRAIN required
                    MODELS_DIR
                    / "training_vae_on_zinc_250k"
                    / "latent_dim-2-batch_size-512-lr-0.0005-seed-86.pt"
                )
                vae = VAESelfies(latent_dim=latent_dim, device=DEVICE)
                opt_vae: VAESelfies = torch.compile(vae)  # type: ignore
                opt_vae.load_state_dict(torch.load(weights_path, map_location=DEVICE))
                return opt_vae
            case 128:
                # We return our MLP VAE.
                weights_path = (
                    MODELS_DIR
                    / "training_vae_on_zinc_250k"
                    / "latent_dim-128-batch_size-512-lr-0.0005-seed-0.pt" # UPDATED ARCHITECTURE
                )
                vae: VAESelfies = VAESelfies(latent_dim=latent_dim, device=DEVICE)
                opt_vae: VAESelfies = torch.compile(vae)  # type: ignore
                opt_vae.load_state_dict(torch.load(weights_path, map_location=DEVICE))
                return opt_vae
            case 256:
                # We return our trained RNNVAE.
                vae: VAERNNSelfies = VAERNNSelfies(
                    latent_dim=256,
                    device=DEVICE,
                    hidden_dim=512,
                    num_layers=1,
                )
                weights_path = (
                    MODELS_DIR
                    / "training_vae_on_zinc_250k"
                    / "rnn-latent_dim-256-batch_size-512-lr-0.0005-seed-49.pt"
                )
                opt_vae: VAERNNSelfies = torch.compile(vae)
                opt_vae.load_state_dict(torch.load(weights_path, map_location=DEVICE))
                return opt_vae
            case _:
                raise NotImplementedError
