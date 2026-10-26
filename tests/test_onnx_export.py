"""Check the exported artifact policy without running the model exporter."""

import pytest
import torch

from yaslowfast.onnx_export import (
    export_onnx,
    normalize_node_names,
    validate_onnx_model,
    validate_opset,
)

onnx = pytest.importorskip("onnx")


def identity_model(*, opset=18, ir=10):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["output"], name="identity")],
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
        [onnx.helper.make_node("Identity", ["x"], ["y"], name="function_identity")],
        [onnx.helper.make_opsetid("", function_opset)],
    )
    model.functions.append(function)
    model.opset_import.append(onnx.helper.make_opsetid("local.test", 1))
    model.graph.node[0].CopyFrom(
        onnx.helper.make_node(
            "PassThrough", ["input"], ["output"], domain="local.test", name="call"
        )
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
    with pytest.raises(ValueError, match="operator set exceeds 21"):
        validate_onnx_model(function_model(function_opset=22), opset_version=18)


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
            onnx.helper.make_node("Constant", [], ["output"], value=tensor, name="constant")
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


@pytest.mark.parametrize("opset", [17, 22, 23, True, 18.0, "18"])
def test_rejects_unsupported_requested_opset(opset):
    with pytest.raises(ValueError, match="integer from 18 to 21"):
        validate_opset(opset)


@pytest.mark.parametrize("opset", [18, 21])
def test_supported_opset_boundaries(opset):
    validate_onnx_model(identity_model(opset=opset), opset_version=opset)


@pytest.mark.parametrize("package", ["onnx", "onnxruntime"])
def test_dependency_mismatch_fails_before_export_or_publishing(tmp_path, monkeypatch, package):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    monkeypatch.setattr(onnx if package == "onnx" else ort, "__version__", "99.0.0")

    def unexpected_export(*args, **kwargs):
        pytest.fail("The exporter must not run with unverified dependencies")

    monkeypatch.setattr(torch.onnx, "export", unexpected_export)
    destination = tmp_path / "model.onnx"
    with pytest.raises(RuntimeError, match="requires onnx==1.17.0 and onnxruntime==1.20.1"):
        export_onnx(
            torch.nn.Identity(),
            (torch.ones(1), torch.ones(1)),
            destination,
            input_names=("slow", "fast"),
            output_name="logits",
        )
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def model_nodes(model):
    def descend(nodes):
        for node in nodes:
            yield node
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    yield from descend(attribute.g.node)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph in attribute.graphs:
                        yield from descend(graph.node)

    yield from descend(model.graph.node)
    for function in model.functions:
        yield from descend(function.node)


def nested_name_model():
    model = function_model()
    branch_output = onnx.helper.make_tensor_value_info(
        "branch_result", onnx.TensorProto.FLOAT, [1, 2]
    )
    then_branch = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["branch_result"], name="shared")],
        "then",
        [],
        [branch_output],
    )
    else_branch = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["branch_result"])],
        "else",
        [],
        [branch_output],
    )
    model.graph.node.insert(
        0,
        onnx.helper.make_node(
            "If",
            ["condition"],
            ["selected"],
            name="shared",
            then_branch=then_branch,
            else_branch=else_branch,
        ),
    )
    model.graph.initializer.append(
        onnx.helper.make_tensor("condition", onnx.TensorProto.BOOL, [], [True])
    )
    model.graph.node[1].input[0] = "selected"
    # This preexisting function name collides with the unnamed branch's first
    # generated candidate, despite occurring later in traversal order.
    model.functions[0].node[0].name = "yaslowfast_Identity_1"
    model.functions[0].node[0].output[0] = "intermediate"
    model.functions[0].node.append(onnx.helper.make_node("Identity", ["intermediate"], ["y"]))
    return model


def test_normalize_names_preserves_semantics_across_graphs_and_functions():
    model = nested_name_model()
    original = onnx.ModelProto.FromString(model.SerializeToString())
    normalize_node_names(model)
    nodes = list(model_nodes(model))
    names = [node.name for node in nodes]
    assert all(names)
    assert len(names) == len(set(names))
    assert model.graph.node[0].name == "shared"
    assert model.graph.node[1].name == "call"
    assert model.functions[0].node[0].name == "yaslowfast_Identity_1"
    validate_onnx_model(model, opset_version=18)

    # Restoring only node names must restore the entire original protobuf.
    restored = onnx.ModelProto.FromString(model.SerializeToString())
    for before, after in zip(model_nodes(original), model_nodes(restored), strict=True):
        if before.HasField("name"):
            after.name = before.name
        else:
            after.ClearField("name")
    assert restored == original

    serialized = model.SerializeToString()
    normalize_node_names(model)
    assert model.SerializeToString() == serialized
    normalize_node_names(original)
    assert original == model


@pytest.mark.parametrize("invalid_name", ["", "call"])
def test_rejects_unnamed_or_duplicate_nodes_in_local_functions(invalid_name):
    model = function_model()
    model.functions[0].node[0].name = invalid_name
    with pytest.raises(ValueError, match="node names must be non-empty and globally unique"):
        validate_onnx_model(model, opset_version=18)


def gather_fusion_model():
    # Each input is partitioned into two scalar-index Gathers. ORT 1.20.1
    # fuses each pair to a Split; two anonymous pairs create the same Split name.
    nodes = [
        onnx.helper.make_node("Gather", [source, f"index_{index}"], [f"{source}_{index}"], axis=1)
        for source in ("slow", "fast")
        for index in (0, 1)
    ]
    nodes.extend(
        [
            onnx.helper.make_node("Add", ["slow_0", "slow_1"], ["slow_sum"], name="slow_sum"),
            onnx.helper.make_node("Add", ["fast_0", "fast_1"], ["fast_sum"], name="fast_sum"),
            onnx.helper.make_node("Add", ["slow_sum", "fast_sum"], ["logits"], name="sum"),
        ]
    )
    graph = onnx.helper.make_graph(
        nodes,
        "gather_fusion_regression",
        [
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 2])
            for name in ("slow", "fast")
        ],
        [onnx.helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1])],
        [
            onnx.helper.make_tensor(f"index_{index}", onnx.TensorProto.INT64, [], [index])
            for index in (0, 1)
        ],
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10
    )


def test_export_names_finalized_nodes_before_runtime_1201_optimization(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    assert ort.__version__ == "1.20.1"
    model = gather_fusion_model()
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail, match="same node name"):
        ort.InferenceSession(
            model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
        )

    # Start with named nodes and remove their names in finalize to exercise the
    # actual export ordering, not just the standalone normalizer.
    normalize_node_names(model)

    def fake_export(_model, _inputs, destination, **kwargs):
        onnx.save_model(model, destination)

    def finalize(graph):
        for node in graph.graph.node:
            if node.op_type == "Gather":
                node.ClearField("name")
        return graph

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    destination = tmp_path / "model.onnx"
    session, details = export_onnx(
        torch.nn.Identity(),
        (torch.ones(1, 2), torch.ones(1, 2)),
        destination,
        input_names=("slow", "fast"),
        output_name="logits",
        finalize=finalize,
    )
    actual = session.run(
        ["logits"],
        {"slow": torch.tensor([[1.0, 2.0]]).numpy(), "fast": torch.tensor([[3.0, 4.0]]).numpy()},
    )[0]
    torch.testing.assert_close(torch.from_numpy(actual), torch.tensor([10.0]))
    assert details["onnxruntime_version"] == "1.20.1"
    validate_onnx_model(onnx.load_model(destination), opset_version=18)
