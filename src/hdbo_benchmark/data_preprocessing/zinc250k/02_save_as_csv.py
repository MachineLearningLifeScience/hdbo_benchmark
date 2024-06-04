"""Loads the processed datasets, and saves them as a csv."""
import pickle
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.resolve()
    PROCESSED_DIR = ROOT_DIR / "data" / "small_molecule_datasets" / "processed"

    # We load the smiles dataset
    with open(PROCESSED_DIR / "zinc250k_smiles.pkl", "rb") as fin:
        zinc250k = pickle.load(fin)

    df = pd.DataFrame(zinc250k, columns=["SMILES"])

    # We load the selfies dataset
    with open(PROCESSED_DIR / "zinc250k_selfies.pkl", "rb") as fin:
        zinc250k = pickle.load(fin)

    df["SELFIES"] = zinc250k

    df.to_csv(PROCESSED_DIR / "zinc250k.csv", index=False)
