from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def distribution_project(tmp_path):
    root = tmp_path / "project"
    files = {
        "README.md": "# Distribution example\n",
        "LICENSE": "Project license\n",
        "THIRD_PARTY_NOTICES.md": "Third-party notices\n",
        "pyproject.toml": '[project]\nname = "yaslowfast"\nversion = "0.1.0"\n',
        "uv.lock": "version = 1\n",
        ".python-version": "3.12\n",
        ".gitignore": ".venv/\ndist/\n",
        "licenses/PYTORCHVIDEO-LICENSE": "Third-party license\n",
        "src/yaslowfast/__init__.py": "__version__ = '0.1.0'\n",
        "src/yaslowfast/nested/module.py": "VALUE = 1\n",
        "src/yaslowfast/module.pyi": "VALUE: int\n",
        "src/yaslowfast/py.typed": "",
        "configs/default.yaml": "batch_size: 2\n",
        "configs/nested/smoke.yml": "batch_size: 1\n",
        "tests/test_example.py": "def test_example():\n    assert True\n",
    }
    for name, contents in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    script = root / "tools/make_dist.py"
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "tools/make_dist.py", script)
    return root, set(files) | {"tools/make_dist.py"}


def run_distribution(root, *arguments, cwd=None):
    return subprocess.run(
        [sys.executable, str(root / "tools/make_dist.py"), *arguments],
        cwd=cwd or root,
        text=True,
        capture_output=True,
        timeout=30,
    )


def archive_contents(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        return {name: archive.read(name) for name in archive.namelist()}


def test_distribution_includes_source_and_excludes_local_artifacts(distribution_project, tmp_path):
    root, expected = distribution_project
    excluded = [
        ".venv/lib/site-packages/library.py",
        ".git/config",
        ".env",
        "data/dataset.json",
        "datasets/shard.tar",
        "runs/model.pt",
        "results/report.json",
        "exports/model.onnx",
        "dist/old.zip",
        "tools/local_only.py",
        "src/yaslowfast/weights.pt",
        "src/yaslowfast/video.mp4",
        "src/yaslowfast/shard.tar",
        "src/yaslowfast/module.pyc",
        "configs/local.json",
        "tests/fixture.npz",
    ]
    for directory in (
        ".private",
        "data",
        "datasets",
        "runs",
        "results",
        "exports",
        "build",
        "dist",
        "venv",
        "env",
        "__pycache__",
        "site-packages",
        "node_modules",
    ):
        excluded.extend([f"src/yaslowfast/{directory}/junk.py", f"configs/{directory}/junk.yaml"])
    for name in excluded:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Must not be distributed", encoding="utf-8")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.py").write_text("SECRET = 'private'\n", encoding="utf-8")
    (root / "src/yaslowfast/linked.py").symlink_to(outside / "private.py")
    (root / "src/yaslowfast/linked_directory").symlink_to(outside, target_is_directory=True)

    result = run_distribution(root, cwd=outside)

    assert result.returncode == 0, result.stderr
    contents = archive_contents(root / "dist/yaslowfast-0.1.0-source.zip")
    assert set(contents) == {f"yaslowfast-0.1.0/{name}" for name in expected}
    assert contents["yaslowfast-0.1.0/uv.lock"] == b"version = 1\n"
    assert contents["yaslowfast-0.1.0/LICENSE"] == b"Project license\n"
    assert contents["yaslowfast-0.1.0/licenses/PYTORCHVIDEO-LICENSE"] == b"Third-party license\n"


@pytest.mark.parametrize("custom_output", [False, True])
def test_distribution_dry_run_does_not_create_files(distribution_project, tmp_path, custom_output):
    root, expected = distribution_project
    arguments = ["--dry-run"]
    if custom_output:
        arguments += ["--output", "new-output/archive.zip"]
    before = set(tmp_path.rglob("*"))

    result = run_distribution(root, *arguments, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert set(tmp_path.rglob("*")) == before
    for name in expected:
        assert name in result.stdout


def test_distribution_requires_force_to_replace_archive(distribution_project, tmp_path):
    root, _ = distribution_project
    output = tmp_path / "release.zip"
    first = run_distribution(root, "-o", "release.zip", cwd=tmp_path)
    assert first.returncode == 0, first.stderr
    original = output.read_bytes()
    (root / "README.md").write_text("Updated documentation\n", encoding="utf-8")

    refused = run_distribution(root, "--output", "release.zip", cwd=tmp_path)

    assert refused.returncode != 0
    assert output.read_bytes() == original

    replaced = run_distribution(root, "--output", "release.zip", "--force", cwd=tmp_path)

    assert replaced.returncode == 0, replaced.stderr
    assert archive_contents(output)["yaslowfast-0.1.0/README.md"] == b"Updated documentation\n"


def test_extracted_distribution_can_repackage_without_git(distribution_project, tmp_path):
    root, _ = distribution_project
    result = run_distribution(root)
    assert result.returncode == 0, result.stderr
    original_path = root / "dist/yaslowfast-0.1.0-source.zip"
    original = archive_contents(original_path)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(original_path) as archive:
        archive.extractall(extracted)
    extracted_root = extracted / "yaslowfast-0.1.0"
    assert not (extracted_root / ".git").exists()

    rebuilt = run_distribution(extracted_root, cwd=tmp_path)

    assert rebuilt.returncode == 0, rebuilt.stderr
    assert archive_contents(extracted_root / "dist/yaslowfast-0.1.0-source.zip") == original


@pytest.mark.parametrize("license_file", ["LICENSE", "licenses/PYTORCHVIDEO-LICENSE"])
def test_distribution_refuses_missing_required_license(distribution_project, license_file):
    root, _ = distribution_project
    (root / license_file).unlink()

    result = run_distribution(root)

    assert result.returncode != 0
    assert not (root / "dist/yaslowfast-0.1.0-source.zip").exists()
