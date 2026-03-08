"""
API routes — upload, prune, quantize, export.
"""

import os
import io
import copy
import tempfile
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.parser import parse_model
from core.pruner import prune_model
from core.quantizer import quantize_model
from core.onnx_parser import (
    parse_onnx_proto,
    prune_onnx_weights,
    quantize_onnx_weights,
)

router = APIRouter()

# In-memory store: model_id -> entry dict
model_store: dict[str, dict] = {}


class PruneRequest(BaseModel):
    model_id: str
    threshold: float


class QuantizeRequest(BaseModel):
    model_id: str
    target_bitwidth: int


class ExportRequest(BaseModel):
    model_id: str


ALLOWED_EXTENSIONS = (".pt", ".pth", ".onnx")


# ── Helpers ────────────────────────────────────────────────


def _compute_effective_size(
    original_kb: float, sparsity: float, bitwidth: int,
) -> float:
    """Effective size considering both pruning (sparse storage) and quantization."""
    sparsity_frac = sparsity / 100.0
    size = original_kb * (bitwidth / 32) * (1.0 - sparsity_frac * 0.95)
    return round(size, 2)


def _global_sparsity_from_layers(layers: list[dict]) -> float:
    """Compute global sparsity from per-layer info."""
    total_el = 0
    total_zeros = 0
    for l in layers:
        weight_el = 1
        for d in l["weights_shape"]:
            weight_el *= d
        total_el += weight_el
        total_zeros += int(weight_el * l["layer_sparsity"] / 100)
    return round(total_zeros / total_el * 100, 2) if total_el > 0 else 0.0


def _recompress_pytorch(entry: dict, model_id: str) -> dict:
    """Recompute compressed PyTorch model from original using current settings."""
    threshold = entry["current_threshold"]
    bitwidth = entry["current_bitwidth"]

    # Step 1: Prune from original (creates a fresh deep copy)
    model, layers, global_sparsity = prune_model(
        entry["original_model"], threshold
    )

    # Step 2: Quantize in place if needed (model is already a copy)
    if bitwidth < 32:
        _, layers = quantize_model(model, bitwidth, entry["original_size_kb"])
        global_sparsity = _global_sparsity_from_layers(layers)

    entry["model"] = model

    effective_size = _compute_effective_size(
        entry["original_size_kb"], global_sparsity, bitwidth,
    )

    return {
        "model_id": model_id,
        "original_size_kb": entry["original_size_kb"],
        "current_size_kb": effective_size,
        "global_sparsity": global_sparsity,
        "max_weight": entry["max_weight"],
        "layers": layers,
    }


def _recompress_onnx(entry: dict, model_id: str) -> dict:
    """Recompute compressed ONNX model from original using current settings."""
    threshold = entry["current_threshold"]
    bitwidth = entry["current_bitwidth"]

    proto = copy.deepcopy(entry["onnx_original"])

    if threshold > 0:
        prune_onnx_weights(proto, threshold)
    if bitwidth < 32:
        quantize_onnx_weights(proto, bitwidth)

    entry["onnx_model"] = proto
    topo = parse_onnx_proto(proto, model_id=model_id)

    global_sparsity = topo["global_sparsity"]
    effective_size = _compute_effective_size(
        entry["original_size_kb"], global_sparsity, bitwidth,
    )
    topo["original_size_kb"] = entry["original_size_kb"]
    topo["current_size_kb"] = effective_size
    topo["max_weight"] = entry["max_weight"]

    return topo


# ── Endpoints ──────────────────────────────────────────────


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(400, "Only .pt, .pth, or .onnx files accepted.")

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        is_onnx = file.filename.endswith(".onnx")

        if is_onnx:
            import onnx

            onnx_model = onnx.load(tmp_path, load_external_data=True)
            onnx.checker.check_model(onnx_model)
            topology = parse_onnx_proto(onnx_model)
            model_id = topology["model_id"]

            model_store[model_id] = {
                "original_model": None,
                "model": None,
                "onnx_original": onnx_model,
                "onnx_model": onnx_model,
                "original_size_kb": topology["original_size_kb"],
                "max_weight": topology["max_weight"],
                "format": "onnx",
                "current_threshold": 0.0,
                "current_bitwidth": 32,
            }
        else:
            model, topology = parse_model(tmp_path)
            model_id = topology["model_id"]

            model_store[model_id] = {
                "original_model": copy.deepcopy(model),
                "model": model,
                "onnx_original": None,
                "onnx_model": None,
                "original_size_kb": topology["original_size_kb"],
                "max_weight": topology["max_weight"],
                "format": "pytorch",
                "current_threshold": 0.0,
                "current_bitwidth": 32,
            }

        return topology

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to parse model: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)


@router.post("/prune")
async def prune(req: PruneRequest):
    if req.model_id not in model_store:
        raise HTTPException(404, "Model not found.")

    entry = model_store[req.model_id]
    entry["current_threshold"] = req.threshold

    if entry["format"] == "onnx":
        return _recompress_onnx(entry, req.model_id)

    return _recompress_pytorch(entry, req.model_id)


@router.post("/quantize")
async def quantize(req: QuantizeRequest):
    if req.model_id not in model_store:
        raise HTTPException(404, "Model not found.")

    entry = model_store[req.model_id]
    entry["current_bitwidth"] = req.target_bitwidth

    if entry["format"] == "onnx":
        return _recompress_onnx(entry, req.model_id)

    return _recompress_pytorch(entry, req.model_id)


@router.post("/export")
async def export_model(req: ExportRequest):
    if req.model_id not in model_store:
        raise HTTPException(404, "Model not found.")

    entry = model_store[req.model_id]

    if entry["format"] == "pytorch":
        if entry["model"] is None:
            raise HTTPException(400, "No model available for export.")

        buffer = io.BytesIO()
        torch.save(entry["model"], buffer)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=compressed_model.pt"
            },
        )

    elif entry["format"] == "onnx":
        if entry.get("onnx_model") is None:
            raise HTTPException(400, "No ONNX model available for export.")

        buffer = io.BytesIO()
        buffer.write(entry["onnx_model"].SerializeToString())
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=compressed_model.onnx"
            },
        )

    raise HTTPException(400, "Unknown model format.")
