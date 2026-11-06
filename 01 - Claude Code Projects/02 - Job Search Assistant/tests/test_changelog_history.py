"""The Timeline reads this project's whole history from inside mraottth/projects (D-028): the
imported commits up to IMPORTED_TIP, at their original root-level paths, plus later commits that
touch this folder, with paths read relative to it."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_changelog", ROOT / "scripts" / "build_changelog.py")
build_changelog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_changelog)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


@pytest.fixture(scope="module")
def commits():
    if _git("rev-parse").returncode != 0:
        pytest.skip("not a git checkout")
    if _git("cat-file", "-e", f"{build_changelog.IMPORTED_TIP}^{{commit}}").returncode != 0:
        pytest.skip("a checkout without the imported history (e.g. shallow)")
    return build_changelog.git_commits()


def test_imported_history_and_later_commits_are_both_read(commits):
    hashes = {c["hash"] for c in commits}
    assert build_changelog.IMPORTED_TIP in hashes
    imported = int(_git("rev-list", "--no-merges", "--count", build_changelog.IMPORTED_TIP).stdout)
    assert len(commits) >= imported
    folder = _git("rev-parse", "--show-prefix").stdout.strip().rstrip("/")
    if folder:  # inside the projects repo: commits after the import are found too
        later = _git(
            "log",
            "--no-merges",
            "--format=%H",
            f"{build_changelog.IMPORTED_TIP}..HEAD",
            "--",
            f":(top){folder}",
        ).stdout.split()
        assert set(later) <= hashes


def test_paths_are_relative_to_this_folder(commits):
    for c in commits:
        assert not any(p.startswith("01 - Claude Code Projects/") for p in c["_paths"]), c["short"]
        assert c["url"].startswith("https://github.com/mraottth/projects/commit/")
