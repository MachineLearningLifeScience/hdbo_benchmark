import json
from pathlib import Path

from hdbo_benchmark.assets import __file__ as assets_file

ASSETS_DIR = Path(assets_file).parent


def load_alphabet_for_pmo() -> dict[str, int]:
    with open(ASSETS_DIR / "zinc250k_alphabet_stoi.json") as fp_alphabet:
        alphabet: dict[str, int] = json.load(fp_alphabet)

    return alphabet


def load_sequence_length_for_pmo() -> int:
    with open(ASSETS_DIR / "zinc250k_metadata.json") as fp_metadata:
        metadata: dict[str, int] = json.load(fp_metadata)

    return metadata["max_sequence_length"]
