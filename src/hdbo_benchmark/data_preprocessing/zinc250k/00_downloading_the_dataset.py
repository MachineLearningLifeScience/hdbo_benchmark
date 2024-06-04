"""Downloads Zinc250k using torch drug

Downloads the Zinc250k dataset using torch drug, and
saves it on the assets folder.
"""

import pickle
from pathlib import Path

from torchdrug import datasets  # type: ignore

if __name__ == "__main__":
    # The root of the project.
    ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.resolve()

    DATASET_DIR = ROOT_DIR / "data" / "small_molecule_datasets" / "raw"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Download the dataset.
    zinc250k = datasets.ZINC250k(DATASET_DIR, kekulize=True, atom_feature="symbol")

    # Save the dataset.
    with open(DATASET_DIR / "zinc250k.pkl", "wb") as fout:
        pickle.dump(zinc250k, fout)
