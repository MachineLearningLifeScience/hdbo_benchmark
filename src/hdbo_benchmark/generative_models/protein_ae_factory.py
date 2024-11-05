import esm

from hdbo_benchmark.generative_models.ae_for_esm import LitAutoEncoder
from hdbo_benchmark.utils.constants import ROOT_DIR, DEVICE


class ProteinAEFactory:
    def create(
        self,
        latent_dim: int,
    ) -> LitAutoEncoder:
        match latent_dim:
            case 32:
                MODELS_DIR = (
                    ROOT_DIR
                    / "data"
                    / "trained_models"
                    / "ae_for_esm"
                    / f"latent_dim_{latent_dim}"
                )
                _, alphabet_for_ae = esm.pretrained.esm2_t33_650M_UR50D()

                ae = LitAutoEncoder.load_from_checkpoint(
                    MODELS_DIR
                    / "version_23"
                    / "checkpoints"
                    / "epoch=99-step=700.ckpt",
                    alphabet=alphabet_for_ae,
                    latent_dim=latent_dim,
                ).to(DEVICE)

                return ae
            case _:
                raise ValueError(
                    f"Latent dim {latent_dim} not recognized for the Protein AE factory."
                )
