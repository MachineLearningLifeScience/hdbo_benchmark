from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.resolve()
    filepath = (
        ROOT_DIR / "data" / "small_molecule_datasets" / "processed" / "zinc250k.csv"
    )

    df = pd.read_csv(filepath, index_col=False)
    print(df.head())
