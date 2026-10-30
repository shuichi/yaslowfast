from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
import torch

from yaslowfast.config import config_from_dict


class EchoLogits(torch.nn.Module):
    def forward(self, slow, fast):
        return fast


def _archive(path, entries):
    with tarfile.open(path, "w") as archive:
        for name, content in entries:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return path


def _video_content(identifier, probabilities):
    return json.dumps({"id": identifier, "probabilities": probabilities}).encode()


def _config():
    return config_from_dict(
        {
            "model": {
                "num_frames": 4,
                "crop_size": 32,
                "width_per_group": 8,
                "stage_depths": [1, 1, 1, 1],
                "dropout": 0,
            },
            "data": {
                "dataset": "missing-training-dataset.json",
                "train_min_scale": 48,
                "train_max_scale": 48,
                "num_workers": 0,
                "batch_size": 2,
                "mean": [0.1, 0.2, 0.3],
                "std": [0.2, 0.3, 0.4],
            },
        }
    )


PREPROCESSING = {
    "fps": 4,
    "short_side": 48,
    "window_frames": 4,
    "window_stride": 2,
    "jpeg_quality": 95,
}


def _evaluate(positive, negative, *, model=None, class_to_idx=None, **kwargs):
    from yaslowfast.evaluate import evaluate_archives

    return evaluate_archives(
        model if model is not None else EchoLogits(),
        positive,
        negative,
        config=_config(),
        preprocessing=PREPROCESSING,
        class_to_idx=class_to_idx or {"positive": 0, "negative": 1},
        device="cpu",
        **kwargs,
    )


@pytest.fixture
def fake_batches(monkeypatch):
    from yaslowfast import video

    sources = []

    def batches(video_path, *, config, preprocessing, batch_size=1):
        source = Path(video_path)
        sources.append(source)
        payload = json.loads(source.read_bytes())
        if payload.get("broken"):
            raise RuntimeError("decoder failed")
        probabilities = payload["probabilities"]
        for offset in range(0, len(probabilities), batch_size):
            selected = probabilities[offset : offset + batch_size]
            yield {
                "slow": torch.zeros(len(selected), 1),
                "fast": torch.tensor(selected, dtype=torch.float32).log(),
                "windows": [
                    {"start_seconds": float(offset + index), "valid_frames": 4}
                    for index in range(len(selected))
                ],
            }

    monkeypatch.setattr(video, "iter_video_batches", batches)
    return sources


@pytest.mark.parametrize("positive_id", [0, 1])
def test_archive_evaluation_counts_videos_and_preserves_window_loss(
    tmp_path, fake_batches, positive_id
):
    mapping = {"positive": positive_id, "negative": 1 - positive_id}
    entries = {"positive": [], "negative": []}
    # Deliberately asymmetric confusion cells expose transposed axes and wrong class IDs.
    cases = [
        ("positive", "tp0", [0.9, 0.7, 0.8]),
        ("positive", "tp1", [0.8]),
        ("positive", "fn", [0.2]),
        *[("negative", f"fp{index}", [0.8]) for index in range(3)],
        *[("negative", f"tn{index}", [0.2]) for index in range(4)],
    ]
    for label, identifier, positive_probabilities in cases:
        probabilities = [
            [value, 1 - value] if positive_id == 0 else [1 - value, value]
            for value in positive_probabilities
        ]
        entries[label].append(
            (f"nested/{identifier}.MP4", _video_content(identifier, probabilities))
        )
    # Finder resource forks may have video extensions; neither they nor CSV are samples.
    entries["positive"].extend(
        [("nested/._tp0.MP4", b"resource fork"), ("labels.csv", b"metadata")]
    )
    positive = _archive(tmp_path / "positive.tar", entries["positive"])
    negative = _archive(tmp_path / "negative.tar", entries["negative"])
    model = EchoLogits().train()
    report = _evaluate(positive, negative, model=model, class_to_idx=mapping, batch_size=2)

    assert model.training
    assert report["num_videos"] == 10
    assert report["num_windows"] == 12
    assert report["accuracy"] == pytest.approx(0.6)
    assert report["confusion_matrix"] == (
        [[2, 1], [3, 4]] if positive_id == 0 else [[4, 3], [1, 2]]
    )
    expected_loss = -sum(math.log(p) for p in [0.9, 0.7, 0.8, 0.8, 0.2] + [0.2] * 3 + [0.8] * 4)
    assert report["window_loss"] == pytest.approx(expected_loss / 12)
    expected_video_loss = -sum(math.log(p) for p in [0.8, 0.8, 0.2] + [0.2] * 3 + [0.8] * 4)
    assert report["video_loss"] == pytest.approx(expected_video_loss / 10)
    rows = {row["source_member"]: row for row in report["predictions"]}
    assert rows["nested/tp0.MP4"]["num_windows"] == 3
    assert rows["nested/tp0.MP4"]["probabilities"][positive_id] == pytest.approx(0.8)
    for label, identifier, _ in cases:
        row = rows[f"nested/{identifier}.MP4"]
        assert row["target"] == mapping[label]
        assert row["source_archive"] == str(tmp_path / f"{label}.tar")
        content = dict(entries[label])[row["source_member"]]
        assert row["sha256"] == hashlib.sha256(content).hexdigest()
    assert len(fake_batches) == 10
    assert all(not source.exists() for source in fake_batches)


@pytest.mark.parametrize("positive_id", [0, 1])
def test_equal_probabilities_use_checkpoint_class_zero(tmp_path, fake_batches, positive_id):
    positive = _archive(tmp_path / "positive.tar", [("p.mp4", _video_content("p", [[0.5, 0.5]]))])
    negative = _archive(tmp_path / "negative.tar", [("n.mp4", _video_content("n", [[0.5, 0.5]]))])
    report = _evaluate(
        positive, negative, class_to_idx={"positive": positive_id, "negative": 1 - positive_id}
    )
    assert report["confusion_matrix"] == [[1, 0], [1, 0]]
    assert all(row["prediction"] == 0 for row in report["predictions"])


@pytest.mark.parametrize(
    "entries,match",
    [
        ([], "no supported videos"),
        ([("labels.csv", b"notes")], "no supported videos"),
        ([("../outside.mp4", b"unsafe")], "Unsafe archive path"),
        ([("same.mp4", b"a"), ("same.mp4", b"b")], "Duplicate archive member"),
        ([("first.mp4", b"same"), ("second.mp4", b"same")], "Duplicate video contents"),
    ],
)
def test_invalid_archive_fails_before_inference(tmp_path, fake_batches, entries, match):
    positive = _archive(tmp_path / "positive.tar", entries)
    negative = _archive(tmp_path / "negative.tar", [("n.mp4", b"negative")])
    with pytest.raises(ValueError, match=match):
        _evaluate(positive, negative)
    assert fake_batches == []
    assert not (tmp_path / "outside.mp4").exists()


def test_duplicate_contents_across_labels_are_rejected(tmp_path, fake_batches):
    positive = _archive(tmp_path / "positive.tar", [("p.mp4", b"same video")])
    negative = _archive(tmp_path / "negative.tar", [("n.mp4", b"same video")])
    with pytest.raises(ValueError, match="Duplicate video contents"):
        _evaluate(positive, negative)
    assert fake_batches == []


def test_corrupt_tar_is_an_error(tmp_path, fake_batches):
    positive = tmp_path / "positive.tar"
    positive.write_bytes(b"not a tar file")
    negative = _archive(tmp_path / "negative.tar", [("n.mp4", b"negative")])
    with pytest.raises((tarfile.ReadError, ValueError)):
        _evaluate(positive, negative)
    assert fake_batches == []


def test_decode_failure_names_original_source_and_cleans_up(tmp_path, fake_batches):
    positive = _archive(tmp_path / "positive.tar", [("p.mp4", _video_content("p", [[0.8, 0.2]]))])
    negative = _archive(tmp_path / "negative.tar", [("nested/broken.mp4", b'{"broken": true}')])
    model = EchoLogits().train()
    with pytest.raises((ValueError, RuntimeError)) as error:
        _evaluate(positive, negative, model=model)
    assert f"{negative}::nested/broken.mp4" in str(error.value)
    assert "decoder failed" in str(error.value)
    assert model.training
    assert all(not source.exists() for source in fake_batches)


def test_nonfinite_model_output_fails_instead_of_omitting_video(tmp_path, fake_batches):
    positive = _archive(
        tmp_path / "positive.tar", [("p.mp4", _video_content("p", [[float("nan"), 0.2]]))]
    )
    negative = _archive(tmp_path / "negative.tar", [("n.mp4", _video_content("n", [[0.2, 0.8]]))])
    with pytest.raises((ValueError, RuntimeError), match="finite"):
        _evaluate(positive, negative)
    assert all(not source.exists() for source in fake_batches)


def test_unknown_checkpoint_class_names_are_rejected(tmp_path, fake_batches):
    with pytest.raises(ValueError, match="positive|negative"):
        _evaluate(
            tmp_path / "positive.tar",
            tmp_path / "negative.tar",
            class_to_idx={"real": 0, "fake": 1},
        )
    assert fake_batches == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--positive", "positive.tar"],
        ["--negative", "negative.tar"],
        ["--positive", "positive.tar", "--negative", "negative.tar", "--dataset", "dataset.json"],
        ["--positive", "positive.tar", "--negative", "negative.tar", "--split", "test"],
        ["--positive", "positive.tar", "--negative", "negative.tar", "--num-workers", "0"],
        [
            "--positive",
            "positive.tar",
            "--negative",
            "negative.tar",
            "--allow-preprocessing-mismatch",
        ],
        ["--positive", "positive.tar", "--negative", "negative.tar", "--batch-size", "0"],
    ],
)
def test_cli_rejects_ambiguous_or_invalid_archive_options(tmp_path, extra):
    from yaslowfast.evaluate import main

    output = tmp_path / "evaluation.json"
    with pytest.raises(SystemExit) as error:
        main(["--checkpoint", "unused.pt", "--output", str(output), *extra])
    assert error.value.code == 2
    assert not output.exists()


def test_archive_cli_rejects_distributed_launch_before_initializing(monkeypatch):
    from yaslowfast.distributed import DistributedContext
    from yaslowfast.evaluate import main

    def unexpected_context(*args, **kwargs):
        pytest.fail("Archive mode must reject distributed launches before process-group setup")

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(DistributedContext, "from_environment", unexpected_context)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--checkpoint",
                "unused.pt",
                "--positive",
                "positive.tar",
                "--negative",
                "negative.tar",
            ]
        )
    assert error.value.code == 2


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
@pytest.mark.parametrize("with_report", [False, True])
def test_archive_cli_matches_video_prediction_and_writes_report(tmp_path, capsys, with_report):
    from yaslowfast.evaluate import main
    from yaslowfast.models import build_model
    from yaslowfast.predict import predict_video

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        sources = {}
        for label, source in (
            ("positive", "color=c=red:s=64x48:r=4:d=0.5"),
            ("negative", "testsrc2=s=64x48:r=4:d=2.5"),
        ):
            video = tmp_path / f"{label}.mp4"
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
                    str(video),
                ],
                check=True,
                capture_output=True,
            )
            _archive(tmp_path / f"{label}.tar", [(f"nested/{label}.mp4", video.read_bytes())])
            sources[label] = video
        config = _config()
        mapping = {"negative": 0, "positive": 1}
        model = build_model(config.model, 2).eval()
        checkpoint = tmp_path / "model.pt"
        torch.save(
            {
                "format_version": 1,
                "model": model.state_dict(),
                "config": config.to_dict(),
                "class_to_idx": mapping,
                "data_preprocessing": PREPROCESSING,
            },
            checkpoint,
        )
        references = {
            label: predict_video(
                model,
                source,
                config=config,
                preprocessing=PREPROCESSING,
                class_to_idx=mapping,
                device="cpu",
                batch_size=2,
            )
            for label, source in sources.items()
        }
        output = tmp_path / "evaluation.json"
        report_directory = tmp_path / "report"
        arguments = [
            "--checkpoint",
            str(checkpoint),
            "--positive",
            str(tmp_path / "positive.tar"),
            "--negative",
            str(tmp_path / "negative.tar"),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--batch-size",
            "2",
        ]
        if with_report:
            arguments.extend(["--report", str(report_directory)])
        main(arguments)
        saved = json.loads(output.read_text())
        console = capsys.readouterr().out
        assert "positive" in console and "negative" in console
        assert "TP" in console and "FN" in console and "FP" in console and "TN" in console
        assert saved["num_videos"] == 2
        assert saved["num_windows"] == sum(row["num_windows"] for row in references.values())
        assert saved["num_windows"] > saved["num_videos"]
        assert saved["class_to_idx"] == mapping
        for row in saved["predictions"]:
            label = Path(row["source_member"]).stem
            expected = references[label]
            assert row["target"] == mapping[label]
            assert row["prediction"] == expected["prediction"]
            assert row["num_windows"] == expected["num_windows"]
            assert row["probabilities"] == pytest.approx(
                [expected["probabilities"]["negative"], expected["probabilities"]["positive"]],
                abs=1e-7,
            )
        if with_report:
            html = (report_directory / "index.html").read_text()
            assert "nested/positive.mp4" in html and "nested/negative.mp4" in html
            assert (report_directory / "predictions.csv").is_file()
            assert json.loads((report_directory / "metrics.json").read_text()) == saved
        else:
            assert not report_directory.exists()
    finally:
        torch.set_num_threads(previous_threads)
