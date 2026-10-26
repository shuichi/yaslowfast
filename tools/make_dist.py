#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = []
# ///
"""Build a source ZIP using an explicit allowlist (Python 3.11+, no dependencies)."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import tomllib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# Add new distributable files here deliberately; .gitignore is not the ZIP policy.
REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".gitignore",
    "tools/make_dist.py",
)
SOURCE_TREES = {
    "src/yaslowfast": {".py", ".pyi"},
    "configs": {".yaml", ".yml"},
    "tests": {".py"},
}
EXCLUDED_DIRS = {
    "__pycache__",
    "build",
    "data",
    "datasets",
    "dist",
    "env",
    "exports",
    "node_modules",
    "results",
    "runs",
    "site-packages",
    "venv",
}


def has_symlink(root: Path, relative: Path) -> bool:
    return any((root / part).is_symlink() for part in (relative, *relative.parents))


def fail_on_walk_error(error: OSError) -> None:
    raise error


def collect_files(root: Path) -> list[Path]:
    files = []
    for name in REQUIRED_FILES:
        relative = Path(name)
        if has_symlink(root, relative) or not (root / relative).is_file():
            raise ValueError(f"Required regular file is missing or is a symlink: {name}")
        files.append(relative)

    for tree, suffixes in SOURCE_TREES.items():
        if has_symlink(root, Path(tree)) or not (root / tree).is_dir():
            raise ValueError(f"Required source directory is missing or is a symlink: {tree}")
        tree_files = []
        for directory, dirs, names in os.walk(
            root / tree, followlinks=False, onerror=fail_on_walk_error
        ):
            folder = Path(directory)
            dirs[:] = sorted(
                name
                for name in dirs
                if not name.startswith(".")
                and name not in EXCLUDED_DIRS
                and not name.endswith(".egg-info")
                and not (folder / name).is_symlink()
            )
            for name in names:
                path = folder / name
                if name.startswith(".") or path.is_symlink() or not path.is_file():
                    continue
                if path.suffix in suffixes or (tree == "src/yaslowfast" and name == "py.typed"):
                    tree_files.append(path.relative_to(root))
        if not tree_files:
            raise ValueError(f"No distributable files found in: {tree}")
        files.extend(tree_files)
    return sorted(files)


def create_zip(root: Path, files: list[Path], prefix: str, output: Path, force: bool) -> None:
    if output.suffix.lower() != ".zip":
        raise ValueError("Output filename must end in .zip")
    if output.is_symlink():
        raise ValueError(f"Output must not be a symlink: {output}")
    if output.exists() and not force:
        raise ValueError(f"Output already exists: {output} (use --force to replace it)")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep any previous ZIP intact if creating the new archive fails.
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for relative in files:
                archive.write(root / relative, arcname=f"{prefix}/{relative.as_posix()}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", type=Path, help="Output ZIP (relative to current directory)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List included files without writing"
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output ZIP")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        files = collect_files(root)
        with (root / "pyproject.toml").open("rb") as project_file:
            version = tomllib.load(project_file)["project"]["version"]
        if not isinstance(version, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._+-]*", version
        ):
            raise ValueError("Project version must be safe to use in a filename")
        prefix = f"yaslowfast-{version}"
        output = args.output or root / "dist" / f"{prefix}-source.zip"
        if args.dry_run:
            for relative in files:
                print(f"{prefix}/{relative.as_posix()}")
            print(f"Dry run: {len(files)} files -> {output.absolute()}")
            return
        create_zip(root, files, prefix, output, args.force)
        print(f"Created: {output.absolute()}")
        print(f"{len(files)} files, {output.stat().st_size:,} bytes")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
