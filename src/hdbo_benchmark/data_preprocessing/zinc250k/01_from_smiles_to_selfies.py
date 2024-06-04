"""Transforms the SMILES representation to SELFIES

Using the tools from inside `poli`'s utilities, this
script transforms the SMILES representation of the
molecules in the ZINC dataset to SELFIES.
"""

import pickle
from pathlib import Path

from poli.core.util.chemistry.string_to_molecule import translate_smiles_to_selfies  # type: ignore

if __name__ == "__main__":
    # We get the path to the ZINC dataset
    ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.resolve()
    ASSETS_DIR = ROOT_DIR / "data" / "small_molecule_datasets"
    SAVED_DATASET_PATH = ASSETS_DIR / "raw" / "zinc250k.pkl"

    assert (
        SAVED_DATASET_PATH.exists()
    ), "The ZINC dataset was not found. Please run the script at ./00_downloading_the_dataset.py first."

    # We define the path to the transformed dataset
    TRANSFORMED_DATASET_DIR = ASSETS_DIR / "processed"
    TRANSFORMED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # We load the dataset
    with open(ASSETS_DIR / SAVED_DATASET_PATH, "rb") as fin:
        zinc250k = pickle.load(fin)

    # We transform the dataset
    zinc250k = [molecule.to_smiles() for molecule in zinc250k.data]

    # We save these as a new dataset
    with open(TRANSFORMED_DATASET_DIR / "zinc250k_smiles.pkl", "wb") as fout:
        pickle.dump(zinc250k, fout)

    # We transform the dataset, and save it again
    zinc250k = translate_smiles_to_selfies(zinc250k)
    with open(TRANSFORMED_DATASET_DIR / "zinc250k_selfies.pkl", "wb") as fout:
        pickle.dump(zinc250k, fout)
