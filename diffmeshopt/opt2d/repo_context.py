import os
import subprocess
from pathlib import Path


def resolve_repo_root(start: str | Path | None = None) -> Path:
    """Resolve repository root robustly.

    Priority:
    1) DIFFMESHOPT_REPO_ROOT env var
    2) git toplevel from start path
    3) nearest parent containing pyproject.toml
    4) current working directory
    """
    env_root = os.getenv("DIFFMESHOPT_REPO_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.exists():
            return p

    start_path = Path(start).resolve() if start is not None else Path.cwd().resolve()

    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start_path),
            text=True,
        ).strip()
        if git_root:
            return Path(git_root).resolve()
    except Exception:
        pass

    for p in [start_path, *start_path.parents]:
        if (p / "pyproject.toml").exists():
            return p

    return Path.cwd().resolve()


def get_git_branch_and_commit(repo_root: str | Path | None = None) -> str:
    """Get '<branch> @ <short_sha>' for the repository."""
    root = resolve_repo_root(repo_root)

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(root),
            text=True,
        ).strip()
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            text=True,
        ).strip()
    except Exception:
        branch, sha = "unknown", "unknown"

    return f"{branch} @ {sha}"
