"""Loads the SELFIES, and computes their integer and one-hot representations.

This script loads the SELFIES strings from the ZINC250k dataset,
and computes their integer and one-hot representations w.r.t. the alphabet
we computed.

These representations are saved to disk as .npz files at
experiments/assets/data/small_molecules/processed/
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

import selfies as sf  # type: ignore

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.resolve()
    PROCESSED_DIR = ROOT_DIR / "data" / "small_molecule_datasets" / "processed"

    dataset_path = PROCESSED_DIR / "zinc250k.csv"
    alphabet_path = PROCESSED_DIR / "alphabet_stoi.json"

    assert (
        dataset_path.exists()
    ), "The dataset does not exist. Run the previous files for preprocessing."

    assert (
        alphabet_path.exists()
    ), "The alphabet does not exist. Run the previous files for preprocessing."

    selfies = pd.read_csv(dataset_path, index_col=False)["SELFIES"]
    tokens = selfies.map(lambda x: list(sf.split_selfies(x)))
    with open(alphabet_path, "r") as f:
        alphabet = json.load(f)

    # Computing the integer representation of the SELFIES strings
    tokens_as_int = tokens.map(lambda x: [alphabet[token] for token in x])

    # Adding padding
    # First, we comput the maximum length.
    max_length = max(tokens_as_int.map(lambda x: len(x)))

    # Then, we pad the tokens
    padded_tokens_as_int = []
    for token_list in tokens_as_int:
        padded_tokens_as_int.append(
            token_list + [alphabet["[nop]"]] * (max_length - len(token_list))
        )

    # One-hot encode the SELFIES strings
    one_hot_tokens = np.zeros(
        (len(padded_tokens_as_int), max_length, len(alphabet)), dtype=np.int32
    )
    for i, token_list in enumerate(padded_tokens_as_int):
        one_hot_tokens[i, np.arange(len(token_list)), token_list] = 1

    # Saving the integer and one-hot representations
    np.savez(
        PROCESSED_DIR / "zinc250k_onehot_and_integers.npz",
        onehot=one_hot_tokens,
        integers=padded_tokens_as_int,
    )
