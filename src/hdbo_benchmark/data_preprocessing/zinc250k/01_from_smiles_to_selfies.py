"""Transforms the SMILES representation to SELFIES

Using the tools from inside `poli`'s utilities, this
script transforms the SMILES representation of the
molecules in the ZINC dataset to SELFIES.
"""

import pickle
from pathlib import Path
from typing import List

import selfies as sf


def translate_smiles_to_selfies(
    smiles_strings: List[str], strict: bool = False
) -> List[str]:
    """Translates a list of SMILES strings to SELFIES strings.

    Given a list of SMILES strings, returns the translation
    into SELFIES strings. If strict is True, it raises an error
    if a SMILES string in the list cannot be parsed. Else, it
    returns None for those.

    This function uses the `selfies` package from Aspuru-Guzik's
    lab. See https://github.com/aspuru-guzik-group/selfies


    Parameters
    ----------
    smiles_strings : List[str]
        A list of SMILES strings.
    strict : bool, optional
        If True, raise an error if a SMILES string in the list cannot be parsed.

    Returns
    -------
    List[str]
        A list of SELFIES strings.
    """
    selfies_strings = []
    for smile in smiles_strings:
        try:
            selfies_strings.append(sf.encoder(smile))
        except sf.EncoderError:
            if strict:
                raise ValueError("Failed to encode SMILES to SELFIES.")
            else:
                selfies_strings.append(None)

    return selfies_strings


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
