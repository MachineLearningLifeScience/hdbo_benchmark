import poli
import poli_baselines
import hdbo_benchmark

from hdbo_benchmark.utils.logging import has_uncommitted_changes


def repos_have_uncommited_changes() -> bool:
    return (
        has_uncommitted_changes(hdbo_benchmark)
        or has_uncommitted_changes(poli)
        or has_uncommitted_changes(poli_baselines)
    )


def verify_repos_are_clean(strict_on_hash: bool):
    if not strict_on_hash:
        return

    if repos_have_uncommited_changes():
        raise Exception(
            "There are uncommitted changes in the repositories in hdbo_benchmark, poli, or poli_baselines"
        )
