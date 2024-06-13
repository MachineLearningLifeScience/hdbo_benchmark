# A survey and benchmark of High-Dimensional Bayesian Optimization for discrete sequence optimization

[![Link to Project website](https://img.shields.io/badge/GitHub-Project_Website-100000?logo=github&logoColor=white)](https://machinelearninglifescience.github.io/hdbo_benchmark)
[![Link to Project website](https://img.shields.io/badge/GitHub-poli_docs-100000?logo=github&logoColor=white)](https://machinelearninglifescience.github.io/poli-docs)
[![Tests on hdbo (conda, python 3.10)](https://github.com/MachineLearningLifeScience/hdbo_benchmark/actions/workflows/tox-lint-and-pytest.yml/badge.svg)](https://github.com/MachineLearningLifeScience/hdbo_benchmark/actions/workflows/tox-lint-and-pytest.yml)

This repository contains the code for our survey and benchmark of **high-dimensional Bayesian optimization** of discrete sequences using [poli](https://github.com/MachineLearningLifeScience/poli) and [poli-baselines](https://github.com/MachineLearningLifeScience/poli-baselines).

## Checking ongoing results

[Check our leaderboards in our project website.](https://machinelearninglifescience.github.io/hdbo_benchmark)

## Adding a new solver

### Adding necessary files

We expect contributions to this benchmark to be implemented as solvers in [`poli-baselines`](https://github.com/MachineLearningLifeScience/poli-baselines). [Follow the documentation therein](https://machinelearninglifescience.github.io/poli-docs/contributing/a_new_solver.html).

In a few words, we expect you to provide the following folder structure:

```bash
# In poli-baselines' solvers folder
solvers
├── your_solver_name
│   ├── __init__.py
│   ├── environment.your_solver_name.yml
│   └── your_solver_name.py
```

We expect `environment.your_solver_name.yml` to create a conda environment in which `your_solver_name.py` could be imported. See a template here:

```yml
name: poli__your_solver_name
channels:
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - your
      - dependencies
      - here
      - "git+https://github.com/MachineLearningLifeScience/poli.git@dev"
      - "git+https://github.com/MachineLearningLifeScience/poli-baselines.git@main"
```

Provide said code as **as a pull request to poli-baselines.** Afterwards, we will register it, run it, and add its reports to our ongoing benchmarks.


### (Optional) Running your solver locally

If you feel eager to test it in our problems, you could prepare for local testing here. We provide a `requirements.txt`/`environment.yml` you can use to create an environment for running the benchmarks. Afterwards, install this package:

```bash
conda create -n hdbo_benchmark python=3.10
conda activate hdbo_benchmark
pip install -r requirements.txt
pip install -e .
```

Change the `WANDB_PROJECT` and `WANDB_ENTITY` in `src/hdbo_benchmark/utils/constants.py`.

After implementing a solver in `poli-baselines`, you can **register it** in `src/hdbo_benchmark/utils/experiments/load_solvers.py`.

The scripts used to run the benchmarks can be found in `src/hdbo_benchmark/experiments`. To run e.g. `albuterol_similarity` [of the PMO benchmark](https://openreview.net/forum?id=yCZRdI0Y7G) you can run:

```bash
conda run -n hdbo_benchmark python src/hdbo_benchmark/experiments/benchmark_on_pmo/run.py \
    --function-name=albuterol_similarity \
    --solver-name=line_bo \
    --latent-dim=128 \
    --max-iter=300 \
```

assuming `hdbo_benchmark` is an environment in which you can run your solver, and in which this package is installed. Examples of environments where solvers have been tested to run can be found in `poli-baselines`.

## Replicating the data preprocessing for downloading zinc250k

We use [torchdrug](https://torchdrug.ai/docs/installation.html) to download the dataset. It has very picky dependencies, but you should be able to install it by running

```bash
conda env create --file environment.vae_training.yml
```

and following the scripts in `src/hdbo_benchmark/data_preprocessing/zinc250k`.

## Citing all the relevant work

Depending on the black box you use within `poli`, we expect you to cite a set of references. [Check the documentation of the black box for a list (including `bibtex`)](https://machinelearninglifescience.github.io/poli-docs/using_poli/objective_repository/all_objectives.html).

