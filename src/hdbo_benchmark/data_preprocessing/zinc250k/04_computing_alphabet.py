"""Computes the alphabet by counting the tokens in the dataset."""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

import selfies as sf  # type: ignore

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.resolve()
    PROCESSED_DIR = ROOT_DIR / "data" / "small_molecule_datasets" / "processed"

    # We load the selfies dataset
    filepath = PROCESSED_DIR / "zinc250k.csv"

    # We only load the SELFIES column
    df = pd.read_csv(filepath, index_col=False)

    # We compute the alphabet
    token_lists = df["SELFIES"].apply(lambda x: list(sf.split_selfies(x)))

    # For a small analysis, we plot the distribution of the number of tokens per SELFIES
    # Computing the sequence lengths, and saving
    sequence_lengths = token_lists.apply(lambda x: len(x))
    sequence_lengths.to_csv(
        PROCESSED_DIR / "zinc250k_sequence_lengths.csv", index=False
    )

    # Plotting the sequence lengths as a histogram
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.hist(sequence_lengths, bins=max(sequence_lengths) - min(sequence_lengths) + 1)
    ax.set_xlabel("Number of tokens")
    ax.set_ylabel("Number of SELFIES")
    fig.tight_layout()

    # Saving the figure
    fig.savefig(PROCESSED_DIR / "zinc250k_sequence_lengths.jpg")

    # Counting the tokens
    token_counts: dict[str, int] = defaultdict(int)
    for token_list in token_lists:
        for token in token_list:
            token_counts[token] += 1

    # Sorting the tokens by their frequency
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    # Appending [nop] at the start
    tokens = ["[nop]"] + [k for k, _ in sorted_tokens]

    # Saving the alphabet as a json file
    with open(PROCESSED_DIR / "zinc250k_alphabet_stoi.json", "w") as fout:
        json.dump({token: i for i, token in enumerate(tokens)}, fout)
