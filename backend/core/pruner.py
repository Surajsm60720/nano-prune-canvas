"""
Magnitude-based pruner — mask generation and sparsity calculation.
Supports Linear and Conv2d layers in arbitrary nn.Module architectures.
"""

import copy
import torch
import torch.nn as nn
from typing import Any

from core.parser import SUPPORTED_LAYERS, _layer_info


def prune_model(
    original_model: nn.Module,
    threshold: float,
) -> tuple[nn.Module, list[dict[str, Any]], float]:
    """Prune all supported layers by magnitude threshold.

    Returns a deep copy with weights whose |value| < threshold zeroed.
    """
    model = copy.deepcopy(original_model)

    layers: list[dict[str, Any]] = []
    total_elements = 0
    total_zeros = 0
    idx = 0

    for name, module in model.named_modules():
        if not isinstance(module, SUPPORTED_LAYERS):
            continue

        W = module.weight.data
        mask = torch.abs(W) >= threshold
        W_pruned = W * mask.float()
        module.weight.data = W_pruned

        sparsity = round(float((W_pruned == 0).sum() / W.numel() * 100), 2)
        total_elements += W.numel()
        total_zeros += int((W_pruned == 0).sum().item())

        info = _layer_info(name, module, idx, sparsity=sparsity, non_zero_sample=True)
        if info:
            layers.append(info)
        idx += 1

    global_sparsity = (
        round((total_zeros / total_elements) * 100, 2) if total_elements > 0 else 0.0
    )

    return model, layers, global_sparsity
