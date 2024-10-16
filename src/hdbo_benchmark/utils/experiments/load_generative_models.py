from poli.benchmarks import PMOBenchmark

from hdbo_benchmark.generative_models.vae_factory import VAEFactory, VAE
from hdbo_benchmark.generative_models.protein_ae_factory import (
    ProteinAEFactory,
    LitAutoEncoder,
)


def load_generative_model_and_bounds(
    function_name: str,
    latent_dim: int,
) -> tuple[VAE | LitAutoEncoder, tuple[float, float]]:
    match function_name:
        case function_name if function_name in PMOBenchmark(
            string_representation="SELFIES"
        ).problem_names:
            experiment_name = "benchmark_on_pmo"
            latent_space_bounds = (-5.0, 5.0)
            vae = VAEFactory().create(
                experiment_name=experiment_name,
                latent_dim=latent_dim,
            )

            return vae, latent_space_bounds
        case "rfp_rasp":
            experiment_name = "benchmark_on_rasp"
            latent_space_bounds = (-15.0, 15.0)  # By inspecting z0.
            ae = ProteinAEFactory().create(latent_dim=latent_dim)

            return ae, latent_space_bounds
        case "rfp_foldx_stability":
            latent_space_bounds = (-15.0, 15.0)  # By inspecting z0.
            ae = ProteinAEFactory().create(latent_dim=latent_dim)

            return ae, latent_space_bounds
        case _:
            raise ValueError(f"Function {function_name} not recognized.")
