"""
Post-training quantizer — per-tensor affine fake quantization.
Supports Linear and Conv2d layers in arbitrary nn.Module architectures.

Weights are quantized and then dequantized (fake quantization), so they
remain as float tensors but only take values representable at the target
bitwidth.  This lets downstream code (canvas visualisation, sampled_edges)
work unchanged while accurately modelling the precision loss.
"""

import torch
import torch.nn as nn
from typing import Any

from core.parser import SUPPORTED_LAYERS, _layer_info


def quantize_model(
    model: nn.Module,
    target_bitwidth: int,
    original_size_kb: float,
) -> tuple[float, list[dict[str, Any]]]:
    """Apply per-tensor affine fake quantization **in place**.

    Parameters
    ----------
    model : nn.Module
        The model whose weights will be modified *in place*.
        Caller should pass a deep-copy if the original must be preserved.
    target_bitwidth : int
        Target precision (4, 8, 16, or 32).
    original_size_kb : float
        Original FP32 model size used for the size-ratio calculation.

    Returns
    -------
    new_size_kb : float
        Estimated model size at the target bitwidth.
    layers : list[dict]
        Per-layer topology info (with updated sampled_edges reflecting
        the quantised values).
    """
    qmin = -(2 ** (target_bitwidth - 1))
    qmax = 2 ** (target_bitwidth - 1) - 1

    layers: list[dict[str, Any]] = []
    idx = 0

    for name, module in model.named_modules():
        if not isinstance(module, SUPPORTED_LAYERS):
            continue

        W = module.weight.data

        if target_bitwidth < 32:
            w_min = W.min().item()
            w_max = W.max().item()

            if w_max == w_min:
                scale = 1.0
                zero_point = 0
            else:
                scale = (w_max - w_min) / (qmax - qmin)
                zero_point = int(round(qmin - w_min / scale))
                zero_point = max(qmin, min(qmax, zero_point))

            # Quantize then dequantize (fake quantization)
            W_q = torch.clamp(torch.round(W / scale + zero_point), qmin, qmax)
            W_deq = (W_q - zero_point) * scale
            module.weight.data = W_deq

        sparsity = round(
            float((module.weight.data == 0).sum() / W.numel() * 100), 2
        )

        info = _layer_info(name, module, idx, sparsity=sparsity, non_zero_sample=True)
        if info:
            layers.append(info)
        idx += 1

    new_size_kb = round(original_size_kb * (target_bitwidth / 32), 2)
    return new_size_kb, layers
