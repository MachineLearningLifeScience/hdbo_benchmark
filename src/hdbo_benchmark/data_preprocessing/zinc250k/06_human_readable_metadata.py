"""Computes readable metadata for the dataset

This script saves a metadata.json file on the processed
directory, containing information about e.g. the maximum
sequence length, and the length of the alphabet.
"""

import json
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.resolve()
    PROCESSED_DIR = ROOT_DIR / "data" / "small_molecule_datasets" / "processed"

    # We load the sequence lengths
    sequence_lengths = pd.read_csv(PROCESSED_DIR / "zinc250k_sequence_lengths.csv")[
        "SELFIES"
    ]
    max_sequence_length = max(sequence_lengths)

    # We compute the length of the alphabet
    with open(PROCESSED_DIR / "zinc250k_alphabet_stoi.json", "r") as fin:
        alphabet = json.load(fin)

    alphabet_length = len(alphabet)

    metadata = {
        "max_sequence_length": max_sequence_length,
        "alphabet_length": alphabet_length,
    }

    # We save the metadata
    with open(PROCESSED_DIR / "zinc250k_metadata.json", "w") as fout:
        json.dump(metadata, fout, indent=4)
