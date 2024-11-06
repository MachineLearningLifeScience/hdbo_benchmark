import re
import subprocess


def get_git_hash_of_library(library):
    library_path = library.__path__[0]

    # Get the current Git hash
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=library_path,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .decode("utf-8")
        )
    except subprocess.CalledProcessError:
        git_hash = None

    if git_hash is None:
        try:
            git_hash = get_git_hash_of_library_from_pip(library)
        except subprocess.CalledProcessError:
            git_hash = None

    return git_hash


def get_git_hash_of_library_from_pip(library):
    library_name = library.__name__

    # Get the output of pip freeze
    try:
        pip_freeze_output = (
            subprocess.check_output(["pip", "freeze"]).strip().decode("utf-8")
        )
    except subprocess.CalledProcessError:
        return None

    # Find the line for the library
    match = re.search(
        f"{library_name}.*@.*#egg={library_name}", pip_freeze_output, re.IGNORECASE
    )
    if match:
        # Extract the commit hash
        git_url = match.group(0)
        git_hash = git_url.split("@")[1].split("#")[0]
        return git_hash
    else:
        return None
