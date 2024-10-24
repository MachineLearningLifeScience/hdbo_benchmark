from poli.benchmarks import PMOBenchmark
from poli.core.problem import Problem

from hdbo_benchmark.generative_models.vae_factory import VAEFactory, VAE
from hdbo_benchmark.generative_models.protein_ae_factory import (
    ProteinAEFactory,
    LitAutoEncoder,
)
from hdbo_benchmark.generative_models.onehot import OneHot


def load_generative_model_and_bounds(
    function_name: str,
    latent_dim: int,
    problem: Problem,
) -> tuple[VAE | LitAutoEncoder, tuple[float, float]]:
    match function_name:
        case function_name if function_name in PMOBenchmark(
            string_representation="SELFIES"
        ).problem_names:
            experiment_name = "benchmark_on_pmo"
            latent_space_bounds = (-10.0, 10.0)
            vae = VAEFactory().create(
                experiment_name=experiment_name,
                latent_dim=latent_dim,
            )
            vae.eval()

            return vae, latent_space_bounds
        case "rfp_rasp":
            experiment_name = "benchmark_on_rasp"
            latent_space_bounds = (-15.0, 15.0)  # By inspecting z0.
            ae = ProteinAEFactory().create(latent_dim=latent_dim)
            ae.eval()

            return ae, latent_space_bounds
        case "rfp_foldx_stability":
            latent_space_bounds = (-15.0, 15.0)  # By inspecting z0.
            ae = ProteinAEFactory().create(latent_dim=latent_dim)
            ae.eval()

            return ae, latent_space_bounds
        case function_name if "ehrlich" in function_name:
            latent_space_bounds = (0.0, 1.0)
            alphabet = problem.info.alphabet
            alphabet_s_to_i = {s: i for i, s in enumerate(alphabet)}
            sequence_length = problem.info.max_sequence_length

            return (
                OneHot(
                    alphabet_s_to_i=alphabet_s_to_i,
                    max_sequence_length=sequence_length,
                ),
                latent_space_bounds,
            )
        case "pest_control_equivalent":
            latent_space_bounds = (0.0, 1.0)
            alphabet = problem.info.alphabet
            alphabet_s_to_i = {s: i for i, s in enumerate(alphabet)}
            sequence_length = problem.info.max_sequence_length

            return (
                OneHot(
                    alphabet_s_to_i=alphabet_s_to_i,
                    max_sequence_length=sequence_length,
                ),
                latent_space_bounds,
            )
        case _:
            raise ValueError(f"Function {function_name} not recognized.")
