import click
import numpy as np
import torch
import wandb

import hdbo_benchmark
from hdbo_benchmark.generative_models.vae_selfies import VAESelfies
from hdbo_benchmark.utils.constants import DEVICE, WANDB_ENTITY
from hdbo_benchmark.utils.data.zinc_250k import load_zinc_250k_dataloaders
from hdbo_benchmark.utils.experiments.training_vaes import train_model
from hdbo_benchmark.utils.logging import has_uncommitted_changes


@click.command()
@click.option("--latent-dim", type=int, default=2)
@click.option("--seed", type=int, default=None)
@click.option("--batch-size", type=int, default=256)
@click.option("--lr", type=float, default=1e-3)
@click.option("--overfit/--no-overfit", default=False)
@click.option("--strict-on-hash/--no-strict-on-hash", default=True)
def main(
    latent_dim: int,
    seed: int,
    batch_size: int,
    lr: float,
    overfit: bool,
    strict_on_hash: bool,
):
    for module in [hdbo_benchmark]:
        if has_uncommitted_changes(module) and strict_on_hash:
            raise Exception(
                f"There are uncommitted changes in the repositories in {module.__name__}"
            )
    print(f"Training from: {DEVICE}")
    if seed is None:
        seed = np.random.randint(0, 1_000)

    torch.manual_seed(seed)

    run_name = f"latent_dim-{latent_dim}-batch_size-{batch_size}-lr-{lr}-seed-{seed}"
    run = wandb.init(
        project="training_vae_on_zinc250k",
        entity=WANDB_ENTITY,
        name=run_name,
        config={
            "latent_dim": latent_dim,
            "batch_size": batch_size,
            "overfit": overfit,
            "device": DEVICE,
            "seed": seed,
            "lr": lr,
        },
    )

    def logger(training_loss, testing_loss):
        run.log(
            {
                "training_loss": training_loss,
                "testing_loss": testing_loss,
            }
        )

    model = VAESelfies(
        latent_dim=latent_dim,
        device=DEVICE,
    )
    opt_model = torch.compile(model)

    wandb.watch(opt_model, log="all", log_freq=100)

    # Defining the experiment's name
    experiment_name = "training_vae_on_zinc_250k"

    # Loading up the dataloaders
    train_loader, test_loader = load_zinc_250k_dataloaders(
        random_seed=seed,
        batch_size=batch_size,
        overfit_to_a_single_batch=overfit,
    )

    train_model(
        opt_model,  # type: ignore
        max_epochs=1049,  # 750,
        train_loader=train_loader,
        test_loader=test_loader,
        experiment_name=experiment_name,
        lr=lr,
        logger=logger,
        model_name_for_saving=run_name,
    )


if __name__ == "__main__":
    main()

# srun -p boomsma --nodes=1 --job-name=training-vaes-on-selfies --cpus-per-task=4 --ntasks-per-node=1 --time=01:00:00 --gres=gpu:titanx:2 --pty bash -i
