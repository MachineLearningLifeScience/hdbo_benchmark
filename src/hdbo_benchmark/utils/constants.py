from pathlib import Path

import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
MODELS_DIR = ROOT_DIR / "data" / "trained_models"

WANDB_PROJECT = "hdbo-embeddings-benchmark"
WANDB_ENTITY = "hdbo-benchmark"

PENALIZE_UNFEASIBLE_WITH = -100.0
