import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from hdbo_benchmark.utils.constants import ROOT_DIR
from hdbo_benchmark.utils.selfies.visualization import selfie_to_numpy_image_array

PROCESSED_DIR = ROOT_DIR / "data" / "small_molecule_datasets" / "processed"

with open(PROCESSED_DIR / "zinc250k_alphabet_stoi.json", "r") as fin:
    alphabet = json.load(fin)

print(f"Alphabet: {alphabet}")
print(f"Alphabet: {alphabet.keys()}")
print(f"Alphabet size: {len(alphabet)}")

df_lengths = pd.read_csv(PROCESSED_DIR / "zinc250k_sequence_lengths.csv")

smallest_molecule_index = df_lengths["SELFIES"].idxmin()
largest_molecule_index = df_lengths["SELFIES"].idxmax()

with open(PROCESSED_DIR / "zinc250k_smiles.pkl", "rb") as fin:
    all_smiles = pickle.load(fin)

with open(PROCESSED_DIR / "zinc250k_selfies.pkl", "rb") as fin:
    all_selfies = pickle.load(fin)

print(f"Smallest molecule as SMILES: {all_smiles[smallest_molecule_index]}")
print(f"Largest molecule as SMILES: {all_smiles[largest_molecule_index]}")

print(
    f"Smallest molecule as SELFIES: {all_selfies[smallest_molecule_index]} (size: {df_lengths['SELFIES'].min()})"
)
print(
    f"Largest molecule as SELFIES: {all_selfies[largest_molecule_index]} (size: {df_lengths['SELFIES'].max()})"
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(
    selfie_to_numpy_image_array(
        all_selfies[smallest_molecule_index], width=600, height=600
    ),
    # cmap="gray",
)
ax1.set_title("\nSmallest molecule (8 SELFIES tokens)")
ax1.axis("off")
ax2.imshow(
    selfie_to_numpy_image_array(
        all_selfies[largest_molecule_index], width=600, height=600
    ),
    # cmap="gray",
)
ax2.set_title("\nLargest molecule (70 SELFIES tokens)")
ax2.axis("off")

plt.tight_layout()
fig.savefig(
    ROOT_DIR / "reports" / "figures" / "smallest_and_largest_molecules.jpg", dpi=300
)

# Sampling random molecules from the dataset
n_molecules = 3
random_molecule_indices = df_lengths.sample(3).index

fig, axes = plt.subplots(1, n_molecules, figsize=(5 * n_molecules, 5))

for i, ax in zip(random_molecule_indices, axes):
    ax.imshow(
        selfie_to_numpy_image_array(all_selfies[i], width=600, height=600),
        # cmap="gray",
    )
    # ax.set_title(f"\nMolecule {i}")
    ax.axis("off")

plt.tight_layout()
fig.savefig(ROOT_DIR / "reports" / "figures" / "random_molecules.jpg", dpi=300)


# plt.show()
