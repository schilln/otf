"""Run with `uv run python scripts/run_examples_notebooks.py` from the repo
root with project dev dependencies installed (`uv sync --dev`)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"
    notebooks = sorted(
        path
        for path in examples_dir.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )

    if not notebooks:
        print(f"No notebooks found under {examples_dir}")
        return 0

    failures: list[Path] = []

    for notebook in notebooks:
        relative_path = notebook.relative_to(repo_root)
        print(f"Executing {relative_path}")
        result = subprocess.run(
            [
                "uv",
                "run",
                "jupyter",
                "nbconvert",
                "--execute",
                "--inplace",
                str(relative_path),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failures.append(relative_path)
            print(f"  failed with exit code {result.returncode}")
            if result.stderr:
                stderr_lines = result.stderr.strip().splitlines()
                print(f"  {stderr_lines[-1]}")
        else:
            print("  ok")

    if failures:
        print("\nNotebooks with errors:")
        for notebook in failures:
            print(f"- {notebook}")
        return 1

    print("\nAll notebooks executed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
