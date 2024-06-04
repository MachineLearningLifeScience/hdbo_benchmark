from pathlib import Path
import json

from hdbo_benchmark.utils.constants import ROOT_DIR


def load_alphabet_for_pmo(n_dimensions: int | None = None) -> dict[str, int]:
    METADATA_PATH = ROOT_DIR / "data" / "small_molecule_datasets" / "processed"

    with open(METADATA_PATH / "zinc250k_alphabet_stoi.json") as fp_alphabet:
        alphabet: dict[str, int] = json.load(fp_alphabet)

    return alphabet


def load_sequence_length_for_pmo(n_dimensions: int | None = None) -> int:
    METADATA_PATH = ROOT_DIR / "data" / "small_molecule_datasets" / "processed"
    with open(METADATA_PATH / "zinc250k_metadata.json") as fp_metadata:
        metadata: dict[str, int] = json.load(fp_metadata)

    return metadata["max_sequence_length"]
