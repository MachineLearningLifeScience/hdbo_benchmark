# Preprocessing for small molecules

## Some dependencies

**You will need to install TorchDrug**, which is a little bit picky. [Check this documentation](https://torchdrug.ai/docs/installation.html#installation).

These are other dependencies:
```bash
pip install rdkit selfies pandas matplotlib
```

## Downloading the dataset

We use TorchDrug to download the dataset (see e.g. [this tutorial](https://torchdrug.ai/docs/tutorials/generation.html#prepare-the-pretraining-dataset)).


## Casting to `selfies`

We use `selfies`, then, to transform these small molecules to their `SELFIES` representation.

## A single csv file

Finally, we save a csv file with both `SMILES` and `SELFIES` representations at `corel/assets/data/small_molecules/processed/zinc250k.csv`. To load it up, you can use

```python
df = pd.read_csv(filepath, index_col=False)
print(df.head())
```

## Building a vocabulary/alphabet

The `selfies` library contains two utilities:
- `split_selfies`, which takes a `SELFIES` string and splits it into all the tokens in it.
- `get_alphabet_from_selfies`, which takes an iterator and outputs an alphabet (`set[str]`).

I have decided to use `split_selfies` instead, and to sort the alphabet by frequency. The code was adapted from the `minimal_vae_on_selfies`.

The script `04_computing_alphabet.py` saves the string-to-index version of the alphabet on the processed directory, sorted by frequency (but starting with a `[nop]` token, which is the standard padding in `selfies`).

