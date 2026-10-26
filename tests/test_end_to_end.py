"""Real video -> WebDataset -> training -> resume -> distributed integration."""

import json
import os
import shutil
import socket
import subprocess
import sys

import pytest
import torch
import yaml

from yaslowfast.checkpoint import atomic_save, load_checkpoint
from yaslowfast.config import config_from_dict
from yaslowfast.prepare import prepare_dataset, verify_dataset
from yaslowfast.stats import compute_statistics
from yaslowfast.train import run_training


@pytest.fixture(scope="module")
def video_dataset(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is not installed")
    root = tmp_path_factory.mktemp("training-videos")
    records = []
    for index, color in enumerate(
        ("0xf02020", "0xd03030", "0x2020f0", "0x3030d0", "0xe82828", "0x2828e8")
    ):
        path = root / f"{index}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=48x40:r=8:d=1.5",
                "-vf",
                f"drawbox=x={index * 2}:y=2:w=4:h=4:color=white:t=fill",
                "-c:v",
                "mpeg4",
                "-threads",
                "1",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        label = "red" if index in (0, 1, 4) else "blue"
        records.append(
            {"video": str(path), "label": label, "split": "train" if index < 4 else "val"}
        )
    manifest = root / "videos.json"
    manifest.write_text(json.dumps(records), encoding="utf-8")
    dataset = prepare_dataset(
        manifest,
        root / "wds",
        fps=8,
        short_side=40,
        window_frames=8,
        window_stride=4,
        shard_size_mb=0.04,
    )
    assert verify_dataset(dataset)["videos"] == 6
    return dataset


def tiny_config(dataset, output, *, epochs=2, steps=2, dropout=0.2):
    return config_from_dict(
        {
            "model": {
                "num_frames": 8,
                "crop_size": 32,
                "width_per_group": 8,
                "stage_depths": [1, 1, 1, 1],
                "dropout": dropout,
            },
            "data": {
                "dataset": str(dataset),
                "batch_size": 2,
                "num_workers": 0,
                "shuffle_buffer": 4,
                "train_min_scale": 32,
                "train_max_scale": 40,
                "pin_memory": False,
            },
            "train": {
                "output_dir": str(output),
                "device": "cpu",
                "epochs": epochs,
                "steps_per_epoch": steps,
                "warmup_epochs": 0,
                "learning_rate": 0.03,
                "scale_learning_rate": False,
                "precision": "fp32",
                "deterministic": True,
                "log_every": 100,
                "accumulation_steps": 2,
            },
        }
    )


@pytest.mark.integration
def test_real_video_training_resume_matches_uninterrupted(video_dataset, tmp_path):
    torch.set_num_threads(2)
    full = tiny_config(video_dataset, tmp_path / "full")
    run_training(full)
    interrupted = tiny_config(video_dataset, tmp_path / "resumed")
    run_training(interrupted, stop_after_epoch=1)
    path = tmp_path / "resumed" / "latest.pt"
    boundary = load_checkpoint(path)
    assert boundary["epoch"] == 1
    assert boundary["task"] == "binary_classification" and boundary["num_classes"] == 2
    assert boundary["model"]["blocks.6.proj.weight"].shape[0] == 2
    statistics = json.loads(video_dataset.read_text())["normalization"]
    saved_config = json.loads((tmp_path / "resumed" / "config.json").read_text())
    for key in ("mean", "std"):
        assert boundary["config"]["data"][key] == pytest.approx(statistics[key])
        assert saved_config["data"][key] == pytest.approx(statistics[key])
    # Older two-class checkpoints have the same head, but omit these metadata fields.
    boundary.pop("task")
    boundary.pop("num_classes")
    atomic_save(boundary, path)
    # A new process starts from the original YAML defaults, not the mutated config.
    resumed = tiny_config(video_dataset, tmp_path / "resumed")
    run_training(resumed, resume=str(path))
    original = load_checkpoint(tmp_path / "full" / "latest.pt")
    restarted = load_checkpoint(path)
    assert restarted["task"] == "binary_classification" and restarted["num_classes"] == 2
    for name, value in original["model"].items():
        torch.testing.assert_close(restarted["model"][name], value, rtol=0, atol=0, msg=name)
    assert original["scheduler"] == restarted["scheduler"]
    assert original["global_step"] == restarted["global_step"] == 4
    assert original["best_metric"] == restarted["best_metric"]
    for index, state in original["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                restarted["optimizer"]["state"][index][name], value, rtol=0, atol=0
            )
    report = json.loads((tmp_path / "resumed" / "validation.json").read_text())
    assert report["num_videos"] == 2 and report["num_windows"] == 4


@pytest.mark.integration
def test_normalization_excludes_validation_and_overlapping_padding(video_dataset):
    report = compute_statistics(video_dataset)
    assert report["split"] == "train"
    assert report["unique_frames"] == 4 * 12
    assert all(0 < value < 1 for value in report["std"])
    embedded = json.loads(video_dataset.read_text())["normalization"]
    assert embedded["split"] == "train"
    assert embedded["unique_frames"] == report["unique_frames"]
    assert embedded["pixels"] == report["pixels"]
    for key in ("mean", "std"):
        assert embedded[key] == pytest.approx(report[key], rel=1e-12, abs=1e-12)


@pytest.mark.integration
def test_legacy_dataset_and_checkpoint_keep_configured_normalization(video_dataset, tmp_path):
    torch.set_num_threads(2)
    legacy_root = tmp_path / "legacy"
    shutil.copytree(video_dataset.parent, legacy_root)
    legacy_path = legacy_root / "dataset.json"
    metadata = json.loads(legacy_path.read_text())
    metadata.pop("normalization")
    legacy_path.write_text(json.dumps(metadata), encoding="utf-8")
    output = tmp_path / "legacy-run"
    config = tiny_config(legacy_path, output, steps=1)
    config.data.mean = (0.1, 0.2, 0.3)
    config.data.std = (0.4, 0.5, 0.6)
    run_training(config, stop_after_epoch=1)
    checkpoint_path = output / "latest.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    # Simulate a checkpoint written before the normalization selector existed.
    checkpoint["config"]["data"].pop("normalization", None)
    checkpoint["resume_contract"]["data"].pop("normalization", None)
    atomic_save(checkpoint, checkpoint_path)
    restored = config_from_dict(checkpoint["config"])
    run_training(restored, resume=str(checkpoint_path))
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint["epoch"] == 2
    assert checkpoint["config"]["data"]["mean"] == pytest.approx((0.1, 0.2, 0.3))
    assert checkpoint["config"]["data"]["std"] == pytest.approx((0.4, 0.5, 0.6))


@pytest.mark.integration
def test_evaluation_and_prediction_use_checkpoint_normalization(video_dataset, tmp_path):
    from yaslowfast.evaluate import main as evaluate_main
    from yaslowfast.predict import main as predict_main

    torch.set_num_threads(2)
    output = tmp_path / "training"
    config = tiny_config(video_dataset, output, epochs=1, steps=1)
    run_training(config)
    expected = json.loads((output / "validation.json").read_text())["predictions"]
    checkpoint_path = output / "latest.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    # Inference must remain possible after the training dataset has been removed.
    checkpoint["config"]["data"]["dataset"] = str(tmp_path / "missing-dataset.json")
    atomic_save(checkpoint, checkpoint_path)
    evaluation_root = tmp_path / "evaluation-data"
    shutil.copytree(video_dataset.parent, evaluation_root)
    evaluation_dataset = evaluation_root / "dataset.json"
    metadata = json.loads(evaluation_dataset.read_text())
    # Evaluation data may have different statistics; the checkpoint values must win.
    metadata["normalization"]["mean"] = [0.0, 0.0, 0.0]
    metadata["normalization"]["std"] = [1.0, 1.0, 1.0]
    evaluation_dataset.write_text(json.dumps(metadata), encoding="utf-8")
    evaluation_path = tmp_path / "evaluation.json"
    evaluate_main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--dataset",
            str(evaluation_dataset),
            "--output",
            str(evaluation_path),
            "--device",
            "cpu",
        ]
    )
    actual = json.loads(evaluation_path.read_text())["predictions"]
    for reference, prediction in zip(expected, actual, strict=True):
        assert prediction["video_id"] == reference["video_id"]
        assert prediction["probabilities"] == pytest.approx(reference["probabilities"], abs=1e-7)
    selected = expected[0]
    prediction_path = tmp_path / "prediction.json"
    predict_main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--video",
            metadata["videos"][selected["video_id"]]["video"],
            "--output",
            str(prediction_path),
            "--device",
            "cpu",
        ]
    )
    prediction = json.loads(prediction_path.read_text())["predictions"][0]
    probabilities = [
        prediction["probabilities"][label]
        for label, _ in sorted(metadata["class_to_idx"].items(), key=lambda item: item[1])
    ]
    assert probabilities == pytest.approx(selected["probabilities"], abs=1e-7)


@pytest.mark.integration
def test_two_rank_training_checkpoint_and_resume(video_dataset, tmp_path):
    config = tiny_config(video_dataset, tmp_path / "ddp", dropout=0.2)
    config.data.num_workers = 1
    config_path = tmp_path / "ddp.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    if sys.platform == "darwin":
        environment["GLOO_SOCKET_IFNAME"] = "lo0"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--master-addr=127.0.0.1",
        f"--master-port={port}",
        "-m",
        "yaslowfast.train",
        "--config",
        str(config_path),
        "--num-threads",
        "1",
    ]
    result = subprocess.run(
        command + ["--stop-after-epoch", "1"],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint_path = tmp_path / "ddp" / "latest.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint["world_size"] == 2 and len(checkpoint["rng_states"]) == 2
    assert checkpoint["global_step"] == 2
    statistics = json.loads(video_dataset.read_text())["normalization"]
    for key in ("mean", "std"):
        assert checkpoint["config"]["data"][key] == pytest.approx(statistics[key])
    result = subprocess.run(
        command + ["--resume", str(checkpoint_path)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint["epoch"] == 2 and checkpoint["global_step"] == 4
    report = json.loads((tmp_path / "ddp" / "validation.json").read_text())
    assert report["num_videos"] == 2 and report["num_windows"] == 4
    # Compare against an uninterrupted two-rank run, including rank RNG and worker seeds.
    result = subprocess.run(
        command + ["--output-dir", str(tmp_path / "ddp-full")],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    uninterrupted = load_checkpoint(tmp_path / "ddp-full" / "latest.pt")
    for name, value in uninterrupted["model"].items():
        torch.testing.assert_close(checkpoint["model"][name], value, rtol=0, atol=0, msg=name)
    assert checkpoint["scheduler"] == uninterrupted["scheduler"]


@pytest.mark.integration
def test_small_slowfast_can_fit_the_synthetic_video_classes(video_dataset, tmp_path):
    torch.set_num_threads(2)
    config = tiny_config(video_dataset, tmp_path / "overfit", epochs=1, steps=60, dropout=0)
    config.data.batch_size = 4
    config.train.accumulation_steps = 1
    config.train.learning_rate = 0.08
    run_training(config)
    report = json.loads((tmp_path / "overfit" / "validation.json").read_text())
    assert report["accuracy"] == 1.0
    assert report["video_loss"] < 0.15
