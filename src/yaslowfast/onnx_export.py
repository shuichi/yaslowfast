"""Shared ONNX export policy and checks for the supported runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

ONNX_VERSION = "1.17.0"
RUNTIME_VERSION = "1.22.0"
DEFAULT_OPSET = 18
MAX_OPSET = 22
MAX_IR_VERSION = 10


def validate_opset(opset_version: int) -> None:
    if type(opset_version) is not int or not DEFAULT_OPSET <= opset_version <= MAX_OPSET:
        raise ValueError(f"ONNX opset must be an integer from {DEFAULT_OPSET} to {MAX_OPSET}")


def _messages(message: Any) -> Iterator[Any]:
    """Visit nested protobuf messages, including tensor-valued node attributes."""
    yield message
    for field, value in message.ListFields():
        if field.message_type is not None:
            if field.is_repeated:
                for child in value:
                    yield from _messages(child)
            else:
                yield from _messages(value)


def validate_onnx_model(model: Any, *, opset_version: int) -> None:
    """Inspect the actual graph; never disguise an unsupported IR or operator set."""
    import onnx

    validate_opset(opset_version)
    if not 1 <= model.ir_version <= MAX_IR_VERSION:
        raise ValueError(f"ONNX IR version must be <= {MAX_IR_VERSION}, got {model.ir_version}")
    imports = [item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}]
    if imports != [opset_version]:
        raise ValueError(f"Exported ONNX opset {imports} does not match requested {opset_version}")
    local_functions = {(f.domain, f.name) for f in model.functions}
    for message in _messages(model):
        if isinstance(message, onnx.TensorProto):
            if message.data_location == onnx.TensorProto.EXTERNAL or message.external_data:
                raise ValueError("ONNX must embed all tensors; external data is not supported")
        elif isinstance(message, onnx.OperatorSetIdProto):
            if message.domain in {"", "ai.onnx"} and not 1 <= message.version <= MAX_OPSET:
                raise ValueError(f"ONNX operator set exceeds {MAX_OPSET}: {message.version}")
        elif isinstance(message, onnx.NodeProto):
            if (
                message.domain not in {"", "ai.onnx"}
                and (message.domain, message.op_type) not in local_functions
            ):
                raise ValueError(
                    f"Custom ONNX operator is not supported: {message.domain}::{message.op_type}"
                )
    onnx.checker.check_model(model, full_check=True)


def export_onnx(
    model: nn.Module,
    inputs: tuple[Tensor, Tensor],
    destination: Path,
    *,
    input_names: tuple[str, str],
    output_name: str,
    opset_version: int = DEFAULT_OPSET,
    dynamic_batch: bool = False,
    finalize: Callable[[Any], Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Write, check and load a self-contained ONNX artifact from bytes."""
    validate_opset(opset_version)
    try:
        import onnx
        import onnxruntime as ort
        import onnxscript
    except ImportError as exc:
        raise RuntimeError("ONNX export requires: pip install 'yaslowfast[export]'") from exc
    if onnx.__version__ != ONNX_VERSION or ort.__version__ != RUNTIME_VERSION:
        raise RuntimeError(
            f"ONNX export requires onnx=={ONNX_VERSION} and onnxruntime=={RUNTIME_VERSION}; "
            f"found {onnx.__version__} and {ort.__version__}"
        )
    dynamic_shapes = None
    if dynamic_batch:
        batch = torch.export.Dim("batch_size", min=1)
        dynamic_shapes = ({0: batch}, {0: batch})
    torch.onnx.export(
        model,
        inputs,
        str(destination),
        dynamo=True,
        opset_version=opset_version,
        input_names=list(input_names),
        output_names=[output_name],
        dynamic_shapes=dynamic_shapes,
        external_data=False,
    )
    graph = onnx.load_model(destination, load_external_data=False)
    if finalize is not None:
        graph = finalize(graph)
    validate_onnx_model(graph, opset_version=opset_version)
    onnx.save_model(graph, destination, save_as_external_data=False)
    # A mobile caller may only supply bytes, so do not rely on adjacent files.
    options = ort.SessionOptions()
    options.intra_op_num_threads = torch.get_num_threads()
    session = ort.InferenceSession(
        destination.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    details = {
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "onnxscript_version": onnxscript.__version__,
        "onnx_ir_package_version": version("onnx-ir"),
        "opset_version": opset_version,
        "ir_version": graph.ir_version,
        "external_data": False,
        "loaded_from_bytes": True,
    }
    return session, details
