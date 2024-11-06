"""
This file takes a .sh script, and runs it in parallel using the multiprocessing library.
"""

import subprocess
from multiprocessing import Pool

import click


def run(line: str):
    """
    Function to run a given line in a script.
    """
    subprocess.run(line, shell=True)


@click.command()
@click.argument("script_file", type=click.Path(exists=True))
@click.argument("num_processes", type=int)
def run_parallel(script_file, num_processes):
    """
    Function to run the script in parallel using multiprocessing.
    """
    # Read the script file
    with open(script_file, "r") as file:
        script = file.read()

    # Split the script into lines
    lines = script.split("\n")

    # Create a pool of processes
    with Pool(processes=num_processes) as pool:
        # Run each line in parallel
        pool.map(run, lines)


if __name__ == "__main__":
    run_parallel()
