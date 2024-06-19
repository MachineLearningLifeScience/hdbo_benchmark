from types import ModuleType
import subprocess


def has_uncommitted_changes(library: ModuleType):
    library_path = library.__path__[0]
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=library_path,
            check=True,
        ).stdout
        return len(status) > 0
    except subprocess.CalledProcessError:
        return False
