"""Compatibility checks against the existing two-input binary inference contract."""

import io
import json
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from yaslowfast.compat_slowfast import (
    CompatSlowFast,
    finalize_compat_graph,
    make_compat_inputs,
    validate_compat_graph,
)
from yaslowfast.config import Config, DataConfig, ModelConfig
from yaslowfast.data import transform_frames
from yaslowfast.export import export_checkpoint
from yaslowfast.export import main as export_main
from yaslowfast.models import build_model


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_config():
    return Config(
        model=ModelConfig(
            num_frames=32,
            crop_size=32,
            width_per_group=8,
            stage_depths=(1, 1, 1, 1),
        ),
        data=DataConfig(
            mean=(0.12, 0.34, 0.56),
            std=(0.19, 0.27, 0.38),
            train_min_scale=40,
            train_max_scale=48,
        ),
    )


class InspectPathways(torch.nn.Module):
    def forward(self, slow, fast):
        self.slow = slow.detach().clone()
        self.fast = fast.detach().clone()
        return torch.stack((fast[:, 0].mean((1, 2, 3)), slow[:, 2].mean((1, 2, 3))), dim=1)


def test_compat_preprocessing_matches_dataset_evaluation():
    config = small_config()
    rows, columns = np.indices((430, 295))
    frames = [
        np.stack(
            (
                (columns + index * 7) % 256,
                (rows * 3 + index * 11) % 256,
                (rows + columns * 2 + index * 5) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        for index in range(32)
    ]
    encoded = []
    for frame in frames:
        stream = io.BytesIO()
        Image.fromarray(frame).save(stream, format="PNG")
        encoded.append(stream.getvalue())
    expected_slow, expected_fast = transform_frames(
        encoded, config=config.data, model_config=config.model, training=False
    )
    fast = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float().div(255).unsqueeze(0)
    probe = InspectPathways()
    wrapper = CompatSlowFast(probe, config.model, config.data)
    with torch.inference_mode():
        wrapper(fast[:, :, ::4], fast)
    torch.testing.assert_close(probe.fast[0], expected_fast, rtol=0, atol=1e-6)
    torch.testing.assert_close(probe.slow[0], expected_slow, rtol=0, atol=1e-6)


def test_compat_regenerates_slow_from_fast_and_preserves_output_columns():
    config = small_config()
    probe = InspectPathways()
    wrapper = CompatSlowFast(probe, config.model, config.data)
    values = torch.arange(32).view(1, 1, 32, 1, 1) / 64
    channels = torch.arange(3).view(1, 3, 1, 1, 1) / 4
    fast = (values + channels).expand(1, 3, 32, 430, 295)
    unused_slow = torch.full((1, 3, 8, 430, 295), float("nan"))
    with torch.inference_mode():
        result = wrapper(unused_slow, fast)
    assert torch.isfinite(result).all()
    expected = values + channels - torch.tensor(config.data.mean).view(1, 3, 1, 1, 1)
    expected = expected / torch.tensor(config.data.std).view(1, 3, 1, 1, 1)
    torch.testing.assert_close(probe.fast, expected.expand(1, 3, 32, 32, 32))
    torch.testing.assert_close(probe.slow, probe.fast[:, :, [0, 4, 8, 13, 17, 22, 26, 31]])
    assert not torch.allclose(probe.slow, probe.fast[:, :, ::4])
    expected_logits = torch.stack(
        (probe.fast[:, 0].mean((1, 2, 3)), probe.slow[:, 2].mean((1, 2, 3))), dim=1
    )
    torch.testing.assert_close(result, expected_logits)
    metadata = wrapper.metadata()
    assert metadata["output_class_indices"] == [0, 1]
    assert "class_to_idx" not in metadata
    assert "training_class_to_idx" not in metadata
    assert "output_semantics" not in metadata


def test_compat_cli_has_no_label_mapping_option(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["yaslowfast-export", "--help"])
    with pytest.raises(SystemExit) as error:
        export_main()
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--compat-mode" in help_text
    assert "--compat-class-order" not in help_text


def test_compat_cli_rejects_removed_label_mapping_option(monkeypatch, capsys, tmp_path):
    output = tmp_path / "invalid.onnx"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "yaslowfast-export",
            "--checkpoint",
            str(tmp_path / "unused.pt"),
            "--output",
            str(output),
            "--format",
            "onnx",
            "--compat-mode",
            "slowfast",
            "--compat-class-order",
            "positive",
            "negative",
        ],
    )
    with pytest.raises(SystemExit) as error:
        export_main()
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("num_frames,alpha", [(8, 4), (64, 4), (32, 8)])
def test_compat_rejects_incompatible_temporal_configuration(num_frames, alpha):
    config = small_config()
    config.model.num_frames, config.model.alpha = num_frames, alpha
    with pytest.raises(ValueError):
        CompatSlowFast(InspectPathways(), config.model, config.data)


@pytest.fixture(scope="module")
def exported_compat_model(tmp_path_factory):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    root = tmp_path_factory.mktemp("compat-export")
    config = small_config()
    model = build_model(config.model, 2).eval()
    # Distinct known logits detect any column swap without trusting the wrapper.
    with torch.no_grad():
        model.blocks[-1].proj.weight.zero_()
        model.blocks[-1].proj.bias.copy_(torch.tensor([2.0, -3.0]))
    checkpoint = root / "model.pt"
    torch.save(
        {
            "format_version": 1,
            "config": config.to_dict(),
            "model": model.state_dict(),
            "class_to_idx": {"任意のクラス甲": 0, "任意のクラス乙": 1},
            "data_preprocessing": {"fps": 15, "window_frames": 64},
        },
        checkpoint,
    )
    output = root / "model.onnx"
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = export_checkpoint(checkpoint, output, format="onnx", compat_mode="slowfast")
    finally:
        torch.set_num_threads(previous)
    graph = onnx.load(output)
    session = ort.InferenceSession(output.read_bytes(), providers=["CPUExecutionProvider"])
    wrapper = CompatSlowFast(model, config.model, config.data).eval()
    return graph, session, wrapper, result, checkpoint


def test_compat_single_file_signature_and_versions(exported_compat_model):
    import onnx
    import onnxruntime as ort

    graph, session, _, result, _ = exported_compat_model
    assert onnx.__version__ == "1.17.0"
    assert ort.__version__ == "1.22.0"
    onnx.checker.check_model(graph)
    validate_compat_graph(graph)
    assert graph.ir_version <= 10
    assert {entry.domain: entry.version for entry in graph.opset_import}[""] == 18
    assert all(not tensor.external_data for tensor in graph.graph.initializer)
    assert [(value.name, value.shape, value.type) for value in session.get_inputs()] == [
        ("input_slow", ["batch_size", 3, 8, 430, 295], "tensor(float)"),
        ("input_fast", ["batch_size", 3, 32, 430, 295], "tensor(float)"),
    ]
    assert [(value.name, value.shape, value.type) for value in session.get_outputs()] == [
        ("output", ["batch_size", 2], "tensor(float)")
    ]
    with open(result["metadata"]) as stream:
        metadata = json.load(stream)
    assert metadata["compat_mode"] == "slowfast"
    assert metadata["class_to_idx"] == {"任意のクラス甲": 0, "任意のクラス乙": 1}
    assert metadata["output_class_indices"] == [0, 1]
    assert "training_class_to_idx" not in metadata
    assert "output_semantics" not in metadata
    assert metadata["output"]["activation"] == "none"
    assert metadata["verification"]["loaded_from_bytes"] is True
    assert metadata["verification"]["external_data"] is False
    assert set(metadata["verification"]["batch_max_absolute_errors"]) == {"1", "2", "4"}


def test_compat_restores_pruned_slow_input(exported_compat_model):
    import onnx

    source = exported_compat_model[0]
    graph = onnx.ModelProto.FromString(source.SerializeToString())
    del graph.graph.input[0]
    restored = finalize_compat_graph(graph)
    onnx.checker.check_model(restored)
    validate_compat_graph(restored)
    assert tuple(value.name for value in restored.graph.input) == ("input_slow", "input_fast")


@pytest.mark.parametrize("invalid_field", ["static_batch", "dynamic_spatial", "wrong_dtype"])
def test_compat_rejects_incompatible_graph_signature(exported_compat_model, invalid_field):
    import onnx

    graph = onnx.ModelProto.FromString(exported_compat_model[0].SerializeToString())
    tensor = graph.graph.input[1].type.tensor_type
    if invalid_field == "static_batch":
        tensor.shape.dim[0].dim_value = 1
    elif invalid_field == "dynamic_spatial":
        tensor.shape.dim[-1].dim_param = "width"
    else:
        tensor.elem_type = onnx.TensorProto.FLOAT16
    with pytest.raises(ValueError):
        finalize_compat_graph(graph)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_compat_dynamic_batch_and_full_feed_matches_pytorch(exported_compat_model, batch_size):
    _, session, wrapper, _, _ = exported_compat_model
    inputs = make_compat_inputs(batch_size, generator=torch.Generator().manual_seed(batch_size))
    with torch.inference_mode():
        expected = wrapper(*inputs)
    actual = session.run(
        ["output"],
        {name: value.numpy() for name, value in zip(("input_slow", "input_fast"), inputs)},
    )[0]
    assert actual.shape == (batch_size, 2)
    np.testing.assert_array_equal(actual, np.tile(np.array([2.0, -3.0]), (batch_size, 1)))
    torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=1e-4, atol=1e-5)
    feeds = {"input_slow": np.full_like(inputs[0].numpy(), np.nan), "input_fast": inputs[1].numpy()}
    np.testing.assert_array_equal(session.run(["output"], feeds)[0], actual)


@pytest.mark.parametrize(
    "class_to_idx",
    [
        {"positive": 0, "negative": 1},
        {"negative": 0, "positive": 1},
        {"fake": 0, "genuine": 1},
    ],
)
def test_compat_checkpoint_labels_do_not_change_onnx_output_columns(
    exported_compat_model, tmp_path, class_to_idx
):
    import onnxruntime as ort

    checkpoint = torch.load(exported_compat_model[-1], map_location="cpu", weights_only=True)
    checkpoint["class_to_idx"] = class_to_idx
    checkpoint_path = tmp_path / "labels.pt"
    torch.save(checkpoint, checkpoint_path)
    output = tmp_path / "labels.onnx"
    result = export_checkpoint(checkpoint_path, output, format="onnx", compat_mode="slowfast")
    session = ort.InferenceSession(output.read_bytes(), providers=["CPUExecutionProvider"])
    slow, fast = make_compat_inputs(1, generator=torch.Generator().manual_seed(9))
    actual = session.run(["output"], {"input_slow": slow.numpy(), "input_fast": fast.numpy()})[0]
    np.testing.assert_array_equal(actual, [[2.0, -3.0]])
    with open(result["metadata"]) as stream:
        metadata = json.load(stream)
    assert metadata["class_to_idx"] == class_to_idx
    assert metadata["output_class_indices"] == [0, 1]
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["class_to_idx"] == class_to_idx


@pytest.mark.parametrize("opset", [0, 23])
def test_export_rejects_opsets_outside_supported_range(exported_compat_model, tmp_path, opset):
    checkpoint = exported_compat_model[-1]
    output = tmp_path / "invalid.onnx"
    with pytest.raises(ValueError, match="opset"):
        export_checkpoint(
            checkpoint, output, format="onnx", compat_mode="slowfast", opset_version=opset
        )
    assert not output.exists()


def test_compat_requires_onnx_format(exported_compat_model, tmp_path):
    checkpoint = exported_compat_model[-1]
    output = tmp_path / "invalid.pt2"
    with pytest.raises(ValueError):
        export_checkpoint(checkpoint, output, format="torch", compat_mode="slowfast")
    assert not output.exists()
