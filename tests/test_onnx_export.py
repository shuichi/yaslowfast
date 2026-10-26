"""Check the exported artifact policy without running the model exporter."""

import pytest
import torch

from yaslowfast.onnx_export import export_onnx, validate_onnx_model, validate_opset

onnx = pytest.importorskip("onnx")


def identity_model(*, opset=18, ir=10):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["output"])],
        "identity",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])],
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)], ir_version=ir
    )


def function_model(*, function_opset=18):
    model = identity_model()
    function = onnx.helper.make_function(
        "local.test",
        "PassThrough",
        ["x"],
        ["y"],
        [onnx.helper.make_node("Identity", ["x"], ["y"])],
        [onnx.helper.make_opsetid("", function_opset)],
    )
    model.functions.append(function)
    model.opset_import.append(onnx.helper.make_opsetid("local.test", 1))
    model.graph.node[0].CopyFrom(
        onnx.helper.make_node("PassThrough", ["input"], ["output"], domain="local.test")
    )
    return model


def test_valid_embedded_model_passes_policy_and_checker():
    validate_onnx_model(identity_model(), opset_version=18)


def test_rejects_newer_ir_instead_of_relabeling_it():
    model = identity_model(ir=11)
    with pytest.raises(ValueError, match="IR version"):
        validate_onnx_model(model, opset_version=18)
    assert model.ir_version == 11


@pytest.mark.parametrize("actual_opset", [19, 23])
def test_rejects_actual_opset_different_from_requested(actual_opset):
    model = identity_model(opset=actual_opset)
    with pytest.raises(ValueError, match="does not match requested"):
        validate_onnx_model(model, opset_version=18)
    assert model.opset_import[0].version == actual_opset


def test_rejects_newer_operator_set_in_local_function():
    with pytest.raises(ValueError, match="operator set exceeds 22"):
        validate_onnx_model(function_model(function_opset=23), opset_version=18)


@pytest.mark.parametrize("location", ["initializer", "constant_attribute"])
@pytest.mark.parametrize("metadata_only", [False, True])
def test_rejects_external_tensors_including_node_attributes(location, metadata_only):
    model = identity_model()
    tensor = onnx.helper.make_tensor("weights", onnx.TensorProto.FLOAT, [1, 2], [1.0, 2.0])
    tensor.external_data.add(key="location", value="weights.bin")
    if not metadata_only:
        tensor.data_location = onnx.TensorProto.EXTERNAL
    if location == "initializer":
        model.graph.initializer.append(tensor)
        model.graph.node[0].input[0] = "weights"
    else:
        model.graph.node[0].CopyFrom(
            onnx.helper.make_node("Constant", [], ["output"], value=tensor)
        )
    with pytest.raises(ValueError, match="must embed all tensors"):
        validate_onnx_model(model, opset_version=18)


def test_rejects_custom_operator_without_a_local_function():
    model = identity_model()
    model.opset_import.append(onnx.helper.make_opsetid("custom.test", 1))
    model.graph.node[0].domain = "custom.test"
    with pytest.raises(ValueError, match="Custom ONNX operator"):
        validate_onnx_model(model, opset_version=18)


def test_local_function_using_standard_operators_is_self_contained():
    model = function_model()
    validate_onnx_model(model, opset_version=18)
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    value = torch.tensor([[1.0, 2.0]])
    observed = session.run(["output"], {"input": value.numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(observed), value)


def test_custom_operator_inside_local_function_is_rejected():
    model = function_model()
    model.functions[0].node[0].domain = "custom.test"
    model.functions[0].opset_import.append(onnx.helper.make_opsetid("custom.test", 1))
    with pytest.raises(ValueError, match="Custom ONNX operator"):
        validate_onnx_model(model, opset_version=18)


@pytest.mark.parametrize("opset", [17, 23, True, 18.0, "18"])
def test_rejects_unsupported_requested_opset(opset):
    with pytest.raises(ValueError, match="integer from 18 to 22"):
        validate_opset(opset)


@pytest.mark.parametrize("package", ["onnx", "onnxruntime"])
def test_dependency_mismatch_fails_before_export_or_publishing(tmp_path, monkeypatch, package):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    monkeypatch.setattr(onnx if package == "onnx" else ort, "__version__", "99.0.0")

    def unexpected_export(*args, **kwargs):
        pytest.fail("The exporter must not run with unverified dependencies")

    monkeypatch.setattr(torch.onnx, "export", unexpected_export)
    destination = tmp_path / "model.onnx"
    with pytest.raises(RuntimeError, match="requires onnx==1.17.0 and onnxruntime==1.22.0"):
        export_onnx(
            torch.nn.Identity(),
            (torch.ones(1), torch.ones(1)),
            destination,
            input_names=("slow", "fast"),
            output_name="logits",
        )
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
