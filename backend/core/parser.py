"""
Model parser — loads PyTorch models, extracts Linear and Conv2d layer topology.
Supports arbitrary nn.Module architectures (not just nn.Sequential).
"""

import uuid
import random
import torch
import torch.nn as nn
from typing import Any

MAX_SAMPLED_EDGES = 50

# Layer types we can extract and operate on
SUPPORTED_LAYERS = (nn.Linear, nn.Conv2d)


def _sample_weights(W: torch.Tensor, non_zero_only: bool = False) -> list[float]:
    """Sample up to MAX_SAMPLED_EDGES weights from a tensor."""
    flat = W.flatten()
    if non_zero_only:
        indices = torch.nonzero(flat != 0).squeeze(-1).tolist()
        if isinstance(indices, int):
            indices = [indices]
    else:
        indices = list(range(len(flat)))

    sample_count = min(MAX_SAMPLED_EDGES, len(indices))
    sampled = random.sample(indices, sample_count) if indices else []
    return [round(float(flat[i]), 6) for i in sampled]


def _layer_info(
    name: str,
    module: nn.Module,
    idx: int,
    sparsity: float = 0.0,
    non_zero_sample: bool = False,
) -> dict[str, Any] | None:
    """Extract topology info from a single supported layer."""
    if not isinstance(module, SUPPORTED_LAYERS):
        return None

    W = module.weight.data
    param_count = W.numel() + (module.bias.numel() if module.bias is not None else 0)
    sampled = _sample_weights(W, non_zero_only=non_zero_sample)

    info: dict[str, Any] = {
        "layer_id": f"layer_{idx}",
        "type": type(module).__name__,
        "in_features": 0,
        "out_features": 0,
        "weights_shape": list(W.shape),
        "layer_sparsity": sparsity,
        "param_count": int(param_count),
        "sampled_edges": sampled,
    }

    if isinstance(module, nn.Linear):
        info["in_features"] = int(module.in_features)
        info["out_features"] = int(module.out_features)
    elif isinstance(module, nn.Conv2d):
        info["in_features"] = int(module.in_channels)
        info["out_features"] = int(module.out_channels)
        info["kernel_size"] = list(module.kernel_size)
        info["stride"] = list(module.stride)
        padding = module.padding
        if isinstance(padding, str):
            info["padding"] = [0, 0]
        else:
            info["padding"] = list(padding)

    return info


def parse_model(file_path: str) -> tuple[nn.Module, dict[str, Any]]:
    """Load a PyTorch model and extract its topology.

    Supports any nn.Module containing Linear and/or Conv2d layers.
    """
    model = torch.load(file_path, map_location="cpu", weights_only=False)

    if not isinstance(model, nn.Module):
        raise ValueError("File must contain a PyTorch nn.Module model.")

    layers: list[dict[str, Any]] = []
    idx = 0

    for name, module in model.named_modules():
        info = _layer_info(name, module, idx)
        if info:
            layers.append(info)
            idx += 1

    if not layers:
        raise ValueError("No supported layers (Linear, Conv2d) found in model.")

    total_params = sum(p.numel() for p in model.parameters())
    original_size_kb = round((total_params * 4) / 1024, 2)

    # Max absolute weight across all supported layers
    max_weight = 0.0
    for _, module in model.named_modules():
        if isinstance(module, SUPPORTED_LAYERS):
            layer_max = float(module.weight.data.abs().max().item())
            max_weight = max(max_weight, layer_max)

    topology = {
        "model_id": str(uuid.uuid4()),
        "original_size_kb": original_size_kb,
        "current_size_kb": original_size_kb,
        "global_sparsity": 0.0,
        "max_weight": round(max_weight, 6),
        "layers": layers,
    }

    return model, topology
