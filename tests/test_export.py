import json

import pytest
import torch

from yaslowfast.export import export_model
from yaslowfast.models import SlowFast, pack_pathways


@pytest.mark.parametrize(
    "format,suffix,opset", [("torch", ".pt2", 18), ("onnx", ".onnx", 18), ("onnx", ".onnx", 21)]
)
def test_fixed_shape_export_roundtrip(tmp_path, format, suffix, opset):
    if format == "onnx":
        pytest.importorskip("onnxruntime")
        pytest.importorskip("onnxscript")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = SlowFast(2, width_per_group=8, stage_depths=(1, 1, 1, 1)).eval()
        inputs = pack_pathways(torch.randn(1, 3, 8, 32, 32))
        result = export_model(
            model,
            inputs,
            tmp_path / ("model" + suffix),
            format=format,
            opset_version=opset,
            metadata={"class_to_idx": {"a": 0, "b": 1}},
        )
        with open(result["metadata"]) as stream:
            metadata = json.load(stream)
        assert metadata["class_to_idx"] == {"a": 0, "b": 1}
        assert metadata["task"] == "binary_classification"
        assert metadata["num_classes"] == 2
        assert metadata["inputs"][0]["shape"] == [1, 3, 2, 32, 32]
        assert metadata["inputs"][1]["shape"] == [1, 3, 8, 32, 32]
        assert metadata["output"]["activation"] == "none"
        assert metadata["output"]["shape"] == [1, 2]
        assert metadata["output"]["class_ids"] == [0, 1]
        assert metadata["output"]["probability_activation"] == "softmax(dim=1)"
        assert metadata["verification"]["max_absolute_error"] < 1e-5
        if format == "onnx":
            assert metadata["verification"]["onnxruntime_version"] == "1.20.1"
            assert metadata["verification"]["opset_version"] == opset
    finally:
        torch.set_num_threads(previous)


def test_export_rejects_nonbinary_output_before_publishing(tmp_path):
    class ThreeLogits(torch.nn.Module):
        def forward(self, slow, fast):
            return fast.new_zeros((fast.shape[0], 3))

    output = tmp_path / "invalid.pt2"
    inputs = (torch.ones(1, 3, 2, 32, 32), torch.ones(1, 3, 8, 32, 32))
    with pytest.raises(ValueError, match="Binary model output"):
        export_model(ThreeLogits(), inputs, output)
    assert not output.exists()


def test_export_rejects_nonbinary_class_mapping(tmp_path):
    with pytest.raises(ValueError, match="Export class_to_idx"):
        export_model(
            torch.nn.Identity(),
            (),
            tmp_path / "invalid.pt2",
            metadata={"class_to_idx": {"a": 0, "b": 1, "c": 2}},
        )


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("format,suffix", [("torch", ".pt2"), ("onnx", ".onnx")])
def test_nonfinite_output_preserves_existing_artifact_and_metadata(tmp_path, value, format, suffix):
    class NonfiniteLogits(torch.nn.Module):
        def forward(self, slow, fast):
            return fast.new_full((fast.shape[0], 2), value)

    output = tmp_path / ("existing" + suffix)
    metadata = output.with_suffix(output.suffix + ".json")
    original_artifact = b"previously published artifact"
    original_metadata = b'{"previously_published": true}\n'
    output.write_bytes(original_artifact)
    metadata.write_bytes(original_metadata)
    inputs = (torch.ones(1, 3, 1, 1, 1), torch.ones(1, 3, 4, 1, 1))

    with pytest.raises(ValueError, match="finite"):
        export_model(NonfiniteLogits(), inputs, output, format=format)

    assert output.read_bytes() == original_artifact
    assert metadata.read_bytes() == original_metadata
    assert set(tmp_path.iterdir()) == {output, metadata}
