import datetime
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from yaslowfast.benchmark import benchmark_model
from yaslowfast.evaluate import evaluate_model, write_json


class EchoLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(2))
        self.register_buffer("offset", torch.zeros(2))

    def forward(self, slow, fast):
        return fast + self.bias + self.offset


def batch(probabilities, targets, videos, keys):
    return {
        "slow": torch.zeros(len(targets), 1),
        "fast": torch.tensor(probabilities).log(),
        "target": torch.tensor(targets, dtype=torch.long),
        "video_id": videos,
        "key": keys,
        "weight": torch.ones(len(targets)),
    }


def test_evaluate_preserves_training_mode_and_uses_all_examples():
    model = EchoLogits().train()
    loader = [
        batch([[0.8, 0.2], [0.1, 0.9]], [0, 1], ["a", "b"], ["a0", "b0"]),
        batch([[0.6, 0.4]], [0], ["a"], ["a1"]),
    ]
    report = evaluate_model(model, loader, "cpu", 2)
    assert model.training
    assert report["accuracy"] == 1.0
    assert report["num_videos"] == 2
    assert report["num_windows"] == 3
    assert report["predictions"][0]["probabilities"] == pytest.approx([0.7, 0.3])
    expected_loss = -(torch.tensor([0.8, 0.9, 0.6]).log()).mean().item()
    assert report["loss"] == pytest.approx(expected_loss)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_evaluate_failure_restores_training_mode():
    model = EchoLogits().train()
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_model(
            model, [batch([[0.8, 0.2]] * 2, [0, 0], ["a", "a"], ["same", "same"])], "cpu", 2
        )
    assert model.training


def test_evaluation_rejects_nonbinary_contract_without_changing_model_mode():
    model = EchoLogits().train()
    with pytest.raises(ValueError, match="num_classes=2"):
        evaluate_model(model, [], "cpu", 3)
    assert model.training


@pytest.mark.parametrize(
    "class_to_idx",
    [
        {"a": 0},
        {"a": 0, "b": 1, "c": 2},
        {"a": 0, "b": 0},
        {"a": False, "b": True},
    ],
)
def test_checkpoint_and_prediction_reject_nonbinary_class_mapping(tmp_path, class_to_idx):
    from yaslowfast.checkpoint import load_checkpoint
    from yaslowfast.predict import predict_video

    path = tmp_path / "invalid.pt"
    torch.save(
        {
            "format_version": 1,
            "model": {},
            "config": {},
            "data_preprocessing": {},
            "class_to_idx": class_to_idx,
        },
        path,
    )
    with pytest.raises(ValueError, match="Checkpoint class_to_idx"):
        load_checkpoint(path)
    with pytest.raises(ValueError, match="Prediction class_to_idx"):
        predict_video(
            EchoLogits(),
            tmp_path / "unused.mp4",
            config=None,
            preprocessing={},
            class_to_idx=class_to_idx,
            device="cpu",
        )


def test_checkpoint_accepts_legacy_binary_mapping_and_rejects_conflicting_metadata(tmp_path):
    from yaslowfast.checkpoint import load_checkpoint

    path = tmp_path / "binary.pt"
    checkpoint = {
        "format_version": 1,
        "model": {},
        "config": {},
        "data_preprocessing": {},
        "class_to_idx": {"negative": 0, "positive": 1},
    }
    torch.save(checkpoint, path)
    assert load_checkpoint(path)["class_to_idx"] == checkpoint["class_to_idx"]
    for field, value in (("task", "multiclass"), ("num_classes", 3), ("num_classes", 2.0)):
        torch.save(checkpoint | {field: value}, path)
        with pytest.raises(ValueError, match="Checkpoint"):
            load_checkpoint(path)


def test_benchmark_reports_model_only_latency():
    model = EchoLogits()
    report = benchmark_model(
        model, torch.zeros(2, 1), torch.zeros(2, 2), device="cpu", warmup=1, iterations=3
    )
    assert report["batch_size"] == 2
    assert report["num_classes"] == 2
    assert report["latency_ms_mean"] > 0
    assert report["clips_per_second"] > 0
    assert report["peak_cuda_memory_bytes"] is None
    assert model.training


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
def test_predict_and_eval_cli_share_video_preprocessing(tmp_path):
    from yaslowfast.config import config_from_dict
    from yaslowfast.data import build_loader
    from yaslowfast.evaluate import main as evaluate_main
    from yaslowfast.models import build_model
    from yaslowfast.predict import predict_video
    from yaslowfast.prepare import prepare_dataset

    torch.set_num_threads(2)
    for name, source in (
        ("red", "color=c=red:s=64x48:r=4:d=1,drawbox=x=0:y=0:w=16:h=16:color=white:t=fill"),
        ("motion", "testsrc2=s=64x48:r=4:d=1"),
    ):
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
                source,
                "-c:v",
                "mpeg4",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "1",
                str(tmp_path / f"{name}.mp4"),
            ],
            check=True,
            capture_output=True,
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"video": "red.mp4", "label": "red", "split": "train"},
                {"video": "motion.mp4", "label": "motion", "split": "val"},
            ]
        )
    )
    dataset_path = prepare_dataset(
        manifest, tmp_path / "dataset", fps=4, short_side=48, window_frames=8, window_stride=4
    )
    metadata = json.loads(dataset_path.read_text())
    config = config_from_dict(
        {
            "model": {
                "num_frames": 4,
                "crop_size": 32,
                "width_per_group": 8,
                "stage_depths": [1, 1, 1, 1],
                "dropout": 0,
            },
            "data": {
                "dataset": str(dataset_path),
                "train_min_scale": 48,
                "train_max_scale": 48,
                "num_workers": 0,
                "batch_size": 1,
            },
        }
    )
    model = build_model(config.model, 2).eval()
    loader = build_loader(
        dataset_path,
        "val",
        config=config.data,
        model_config=config.model,
        batch_size=1,
        num_workers=0,
        seed=42,
        training=False,
    )
    expected = evaluate_model(model, loader, "cpu", 2)
    predicted = predict_video(
        model,
        tmp_path / "motion.mp4",
        config=config,
        preprocessing=metadata["preprocessing"],
        class_to_idx=metadata["class_to_idx"],
        device="cpu",
        return_windows=True,
    )
    actual = [
        predicted["probabilities"][label]
        for label, _ in sorted(metadata["class_to_idx"].items(), key=lambda item: item[1])
    ]
    assert actual == pytest.approx(expected["predictions"][0]["probabilities"], abs=1e-7)
    assert predicted["windows"][0]["valid_frames"] == 4

    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "config": config.to_dict(),
            "class_to_idx": metadata["class_to_idx"],
            "data_preprocessing": metadata["preprocessing"],
        },
        checkpoint,
    )
    output = tmp_path / "evaluation.json"
    evaluate_main(["--checkpoint", str(checkpoint), "--output", str(output), "--device", "cpu"])
    saved = json.loads(output.read_text())
    assert saved["num_windows"] == saved["num_videos"] == 1
    assert saved["preprocessing_mismatch"] == {}

    metadata["preprocessing"]["fps"] = 8
    dataset_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="preprocessing differs"):
        evaluate_main(["--checkpoint", str(checkpoint), "--output", str(output), "--device", "cpu"])


def _distributed_worker(rank, directory, scenario):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=(directory / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        module = EchoLogits()
        model = torch.nn.parallel.DistributedDataParallel(module)
        # Rank-zero buffer broadcast must remove this deliberately unequal bias.
        module.offset[0] = rank * 100
        if scenario == "empty_rank":
            loader = (
                [batch([[0.9, 0.1]], [0], ["a"], ["a0"]), batch([[0.2, 0.8]], [1], ["b"], ["b0"])]
                if rank == 0
                else []
            )
        elif scenario == "duplicate":
            loader = [batch([[0.9, 0.1]], [0], ["a"], ["a0"])]
        elif scenario == "uneven_video_windows":
            loader = (
                [batch([[0.8, 0.2]] * 3, [0] * 3, ["a"] * 3, ["a0", "a1", "a2"])]
                if rank == 0
                else [batch([[0.1, 0.9]], [0], ["a"], ["a3"])]
            )
        else:
            raise AssertionError(scenario)
        if scenario == "duplicate":
            try:
                evaluate_model(model, loader, "cpu", 2, distributed=True)
            except RuntimeError as error:
                assert "Duplicate" in str(error)
                write_json(directory / f"rank{rank}.json", {"error": str(error)})
            else:
                raise AssertionError("Duplicate windows were not rejected")
        else:
            report = evaluate_model(model, loader, "cpu", 2, distributed=True)
            write_json(directory / f"rank{rank}.json", report)
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend unavailable")
@pytest.mark.parametrize("scenario", ["empty_rank", "uneven_video_windows", "duplicate"])
def test_distributed_evaluation_handles_uneven_and_duplicate_windows(tmp_path, scenario):
    mp.spawn(_distributed_worker, args=(str(tmp_path), scenario), nprocs=2, join=True)
    rank0 = json.loads((tmp_path / "rank0.json").read_text())
    rank1 = json.loads((tmp_path / "rank1.json").read_text())
    if scenario == "duplicate":
        assert "Duplicate" in rank0["error"]
        assert rank0["error"] == rank1["error"]
    else:
        assert rank0["accuracy"] == rank1["accuracy"] == 1.0
        assert "predictions" in rank0
        assert "predictions" not in rank1
        if scenario == "uneven_video_windows":
            assert rank0["predictions"][0]["probabilities"] == pytest.approx([0.625, 0.375])
        else:
            assert rank0["num_videos"] == 2
