"""
Defines a common way of loading VAEs from just the name
of the experiment and the latent dimension.
"""

from pathlib import Path
from typing import Literal

import torch

from hdbo_benchmark.generative_models.vae import VAE
from hdbo_benchmark.generative_models.vae_selfies import VAESelfies
from hdbo_benchmark.generative_models.vae_mario import VAEMario
from hdbo_benchmark.utils.constants import DEVICE
from hdbo_benchmark.assets import __file__ as ASSETS_PATH

ASSETS_DIR = Path(ASSETS_PATH).parent


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
        MODELS_DIR = ASSETS_DIR / "training_vae_on_mario"
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
        MODELS_DIR = ASSETS_DIR / "training_vae_on_zinc_250k"
        match latent_dim:
            case 2:
                weights_path = (
                    MODELS_DIR / "latent_dim-2-batch_size-512-lr-0.0005-seed-1.pt"
                )
            case 64:
                weights_path = (
                    MODELS_DIR / "latent_dim-64-batch_size-512-lr-0.0005-seed-0.pt"
                )
            case 128:
                # We return our MLP VAE.
                weights_path = (
                    MODELS_DIR / "latent_dim-128-batch_size-512-lr-0.0005-seed-1.pt"
                )
            case _:
                raise NotImplementedError

        vae = VAESelfies(latent_dim=latent_dim, device=DEVICE)
        opt_vae: VAESelfies = torch.compile(vae)  # type: ignore
        opt_vae.load_state_dict(
            torch.load(weights_path, map_location=DEVICE, weights_only=True)
        )
        opt_vae.eval()
        return opt_vae
