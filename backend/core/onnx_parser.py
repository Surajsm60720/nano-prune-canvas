"""
ONNX model parser and operations — load, parse, prune, and quantize ONNX models.
Supports models with Gemm (fully connected) and MatMul nodes.
"""

import uuid
import random
import copy
import numpy as np
import onnx
from onnx import numpy_helper
from typing import Any

MAX_SAMPLED_EDGES = 50


def _parse_graph(
    graph, weights: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], int, float]:
    """Parse an ONNX graph and return (layers, total_params, max_weight)."""
    layers: list[dict[str, Any]] = []
    total_params = 0
    max_weight = 0.0

    for idx, node in enumerate(graph.node):
        if node.op_type not in ("Gemm", "MatMul"):
            continue

        W = None
        weight_name = None
        for inp_name in node.input:
            if inp_name in weights:
                arr = weights[inp_name]
                if arr.ndim == 2:
                    W = arr
                    weight_name = inp_name
                    break

        if W is None:
            continue

        transB = False
        if node.op_type == "Gemm":
            for attr in node.attribute:
                if attr.name == "transB" and attr.i == 1:
                    transB = True
                    break

        if transB:
            out_f, in_f = W.shape
        else:
            in_f, out_f = W.shape

        param_count = W.size
        for inp_name in node.input:
            if inp_name in weights and inp_name != weight_name:
                bias = weights[inp_name]
                if bias.ndim == 1:
                    param_count += bias.size
                    break

        total_params += param_count

        # Sparsity
        sparsity = round(float(np.sum(W == 0) / W.size * 100), 2)

        # Max weight
        layer_max = float(np.abs(W).max())
        max_weight = max(max_weight, layer_max)

        # Sample edges (non-zero if there is sparsity, otherwise all)
        flat = W.flatten()
        non_zero_indices = np.nonzero(flat != 0)[0].tolist()

        if non_zero_indices:
            sample_count = min(MAX_SAMPLED_EDGES, len(non_zero_indices))
            sampled_indices = random.sample(non_zero_indices, sample_count)
        elif len(flat) > 0:
            sample_count = min(MAX_SAMPLED_EDGES, len(flat))
            sampled_indices = random.sample(range(len(flat)), sample_count)
        else:
            sampled_indices = []

        sampled_weights = [round(float(flat[i]), 6) for i in sampled_indices]

        layers.append({
            "layer_id": f"layer_{len(layers)}",
            "type": "Gemm" if node.op_type == "Gemm" else "MatMul",
            "in_features": int(in_f),
            "out_features": int(out_f),
            "weights_shape": list(W.shape),
            "layer_sparsity": sparsity,
            "param_count": int(param_count),
            "sampled_edges": sampled_weights,
        })

    return layers, total_params, max_weight


def parse_onnx_proto(
    model_proto: onnx.ModelProto,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Parse an in-memory ONNX model proto and return topology dict."""
    graph = model_proto.graph
    weights: dict[str, np.ndarray] = {}
    for initializer in graph.initializer:
        weights[initializer.name] = numpy_helper.to_array(initializer)

    layers, total_params, max_weight = _parse_graph(graph, weights)

    if not layers:
        raise ValueError(
            "No fully-connected layers (Gemm/MatMul) found in ONNX model."
        )

    original_size_kb = round((total_params * 4) / 1024, 2)

    # Global sparsity
    total_el = 0
    total_zeros = 0
    for l in layers:
        weight_el = 1
        for d in l["weights_shape"]:
            weight_el *= d
        total_el += weight_el
        total_zeros += int(weight_el * l["layer_sparsity"] / 100)
    global_sparsity = (
        round(total_zeros / total_el * 100, 2) if total_el > 0 else 0.0
    )

    return {
        "model_id": model_id or str(uuid.uuid4()),
        "original_size_kb": original_size_kb,
        "current_size_kb": original_size_kb,
        "global_sparsity": global_sparsity,
        "max_weight": round(max_weight, 6),
        "model_format": "onnx",
        "layers": layers,
    }


def parse_onnx_model(file_path: str) -> dict[str, Any]:
    """Load an ONNX model from file and parse it."""
    model = onnx.load(file_path, load_external_data=True)
    onnx.checker.check_model(model)
    return parse_onnx_proto(model)


# ── ONNX operations (in-place) ─────────────────────────────


def prune_onnx_weights(
    model_proto: onnx.ModelProto, threshold: float,
) -> None:
    """Prune ONNX model weights **in place** by zeroing values below threshold."""
    for initializer in model_proto.graph.initializer:
        arr = numpy_helper.to_array(initializer)
        if arr.ndim == 2:
            mask = np.abs(arr) >= threshold
            arr_pruned = (arr * mask).astype(arr.dtype)
            new_tensor = numpy_helper.from_array(arr_pruned, name=initializer.name)
            initializer.CopyFrom(new_tensor)


def quantize_onnx_weights(
    model_proto: onnx.ModelProto,
    target_bitwidth: int,
) -> None:
    """Apply per-tensor affine fake quantization to ONNX weights **in place**."""
    if target_bitwidth >= 32:
        return

    qmin = -(2 ** (target_bitwidth - 1))
    qmax = 2 ** (target_bitwidth - 1) - 1

    for initializer in model_proto.graph.initializer:
        arr = numpy_helper.to_array(initializer)
        if arr.ndim != 2:
            continue

        w_min = float(arr.min())
        w_max = float(arr.max())

        if w_max == w_min:
            continue

        scale = (w_max - w_min) / (qmax - qmin)
        zero_point = int(round(qmin - w_min / scale))
        zero_point = max(qmin, min(qmax, zero_point))

        arr_q = np.clip(np.round(arr / scale + zero_point), qmin, qmax)
        arr_deq = ((arr_q - zero_point) * scale).astype(np.float32)

        new_tensor = numpy_helper.from_array(arr_deq, name=initializer.name)
        initializer.CopyFrom(new_tensor)
