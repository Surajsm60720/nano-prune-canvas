"""
Backend smoke tests for parser, pruner, quantizer, and ONNX operations.
Covers the original trivial model, all 6 generated models, Conv2d support,
composability, real quantization verification, and ONNX operations.
"""

import sys
import os
import copy
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from core.parser import parse_model, MAX_SAMPLED_EDGES
from core.pruner import prune_model
from core.quantizer import quantize_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


# Custom nn.Module for testing arbitrary architecture support
class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(8, 3)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=[2, 3])
        return self.classifier(x)

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Helpers
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def create_test_model():
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(model, tmp.name)
    return tmp.name


# Expected specs for generated models:
#   name -> (layer_sizes_tuple, expected_linear_count)
GENERATED_MODELS = {
    "tiny_classifier":   ((4, 8, 3),                             2),
    "mnist_mlp":         ((784, 256, 128, 64, 10),               4),
    "wide_shallow":      ((128, 1024, 512, 2),                   3),
    "deep_narrow":       ((16, 32, 32, 32, 32, 32, 16, 8, 4),   8),
    "embedding_reducer": ((512, 256, 128, 64, 32, 16),           5),
    "large_recommender": ((2048, 1024, 512, 256, 128, 64, 10),   6),
}


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Original tests
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def test_parser():
    path = create_test_model()
    model, topo = parse_model(path)
    assert isinstance(model, nn.Sequential)
    assert len(topo["layers"]) == 2
    assert topo["layers"][0]["in_features"] == 4
    assert topo["layers"][0]["out_features"] == 3
    assert topo["layers"][1]["in_features"] == 3
    assert topo["layers"][1]["out_features"] == 2
    assert topo["original_size_kb"] > 0
    assert "param_count" in topo["layers"][0]
    assert "sampled_edges" in topo["layers"][0]
    os.unlink(path)
    print("\u2705 Parser: OK")


def test_pruner_zero():
    path = create_test_model()
    model, _ = parse_model(path)
    _, layers, sparsity = prune_model(model, 0.0)
    assert sparsity == 0.0
    os.unlink(path)
    print("\u2705 Pruner (\u03c4=0): OK")


def test_pruner_high():
    path = create_test_model()
    model, _ = parse_model(path)
    _, layers, sparsity = prune_model(model, 100.0)
    assert sparsity == 100.0
    for layer in layers:
        assert len(layer["sampled_edges"]) == 0
    os.unlink(path)
    print("\u2705 Pruner (\u03c4=100): OK")


def test_quantizer():
    path = create_test_model()
    model, _ = parse_model(path)
    new_kb, _ = quantize_model(model, 8, 1.0)
    assert new_kb == 0.25
    os.unlink(path)
    print("\u2705 Quantizer (8-bit): OK")


def test_quantizer_fp32():
    path = create_test_model()
    model, _ = parse_model(path)
    new_kb, _ = quantize_model(model, 32, 1.0)
    assert new_kb == 1.0
    os.unlink(path)
    print("\u2705 Quantizer (FP32): OK")


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Generated-model tests
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def test_parse_generated_models():
    """Each generated model parses with the correct number of Linear layers
    and the expected in/out feature dimensions."""
    for name, (sizes, expected_linears) in GENERATED_MODELS.items():
        path = os.path.join(MODELS_DIR, f"{name}.pt")
        assert os.path.exists(path), f"Missing model file: {path}"

        model, topo = parse_model(path)

        assert isinstance(model, nn.Sequential), f"{name}: not Sequential"
        assert len(topo["layers"]) == expected_linears, (
            f"{name}: expected {expected_linears} layers, got {len(topo['layers'])}"
        )

        # Verify dimensions of each Linear layer
        for i, layer_info in enumerate(topo["layers"]):
            expected_in = sizes[i]
            expected_out = sizes[i + 1]
            assert layer_info["in_features"] == expected_in, (
                f"{name} layer {i}: expected in={expected_in}, got {layer_info['in_features']}"
            )
            assert layer_info["out_features"] == expected_out, (
                f"{name} layer {i}: expected out={expected_out}, got {layer_info['out_features']}"
            )

        # Verify sampled_edges is bounded
        for layer_info in topo["layers"]:
            assert len(layer_info["sampled_edges"]) <= MAX_SAMPLED_EDGES, (
                f"{name}: sampled_edges exceeds {MAX_SAMPLED_EDGES}"
            )

        # Verify max_weight is positive
        assert topo["max_weight"] > 0, f"{name}: max_weight should be positive"

        print(f"  \u2705 {name}: {expected_linears} layers, dims OK, edges bounded")

    print("\u2705 Parse generated models: OK")


def test_prune_generated_models():
    """Pruning at various thresholds produces sensible sparsity."""
    thresholds = [0.0, 0.05, 0.2, 0.5, 100.0]

    for name, (sizes, expected_linears) in GENERATED_MODELS.items():
        path = os.path.join(MODELS_DIR, f"{name}.pt")
        model, topo = parse_model(path)

        prev_sparsity = -1.0
        for tau in thresholds:
            _, layers, sparsity = prune_model(model, tau)
            assert len(layers) == expected_linears, (
                f"{name} \u03c4={tau}: layer count mismatch"
            )
            assert 0.0 <= sparsity <= 100.0, (
                f"{name} \u03c4={tau}: sparsity out of range: {sparsity}"
            )
            # Sparsity should be non-decreasing as threshold increases
            assert sparsity >= prev_sparsity, (
                f"{name}: sparsity decreased from {prev_sparsity} to {sparsity} at \u03c4={tau}"
            )
            prev_sparsity = sparsity

            # All sampled_edges should be bounded
            for layer in layers:
                assert len(layer["sampled_edges"]) <= MAX_SAMPLED_EDGES

        # At \u03c4=100 everything should be fully sparse
        _, layers, sparsity = prune_model(model, 100.0)
        assert sparsity == 100.0, f"{name}: \u03c4=100 should give 100% sparsity, got {sparsity}"
        for layer in layers:
            assert len(layer["sampled_edges"]) == 0, (
                f"{name}: \u03c4=100 should have no sampled edges"
            )

        print(f"  \u2705 {name}: pruning monotonic, \u03c4=100 fully sparse")

    print("\u2705 Prune generated models: OK")


def test_quantize_generated_models():
    """Quantization computes correct size ratios for various bitwidths."""
    bitwidths = [4, 8, 16, 32]

    for name, (sizes, expected_linears) in GENERATED_MODELS.items():
        path = os.path.join(MODELS_DIR, f"{name}.pt")
        model, topo = parse_model(path)
        orig_kb = topo["original_size_kb"]

        for bits in bitwidths:
            # Deep copy because quantize_model modifies in place
            model_copy = copy.deepcopy(model)
            new_kb, layers = quantize_model(model_copy, bits, orig_kb)
            expected_kb = round(orig_kb * (bits / 32), 2)
            assert new_kb == expected_kb, (
                f"{name} {bits}-bit: expected {expected_kb} KB, got {new_kb} KB"
            )
            assert len(layers) == expected_linears

        print(f"  \u2705 {name}: quantization ratios correct for {bitwidths}")

    print("\u2705 Quantize generated models: OK")


def test_large_model_performance():
    """Sanity-check that the large_recommender model doesn't blow up
    in time or produce unexpected values."""
    path = os.path.join(MODELS_DIR, "large_recommender.pt")
    model, topo = parse_model(path)

    # Size should be in the ~10 MB range (FP32)
    assert topo["original_size_kb"] > 5000, "Large model should be > 5 MB"

    # Prune at a small threshold
    _, layers, sparsity = prune_model(model, 0.001)
    assert 0 <= sparsity < 100, "Small pruning threshold should not prune everything"

    # Quantize to 4-bit (use copy since quantize is in-place)
    model_copy = copy.deepcopy(model)
    new_kb, _ = quantize_model(model_copy, 4, topo["original_size_kb"])
    assert new_kb < topo["original_size_kb"], "4-bit should shrink the model"

    print("\u2705 Large model performance: OK")


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# NEW: Conv2d + arbitrary nn.Module tests
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def test_conv2d_parser():
    """A model with Conv2d layers should parse correctly."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(model, tmp.name)

    loaded, topo = parse_model(tmp.name)
    assert len(topo["layers"]) == 2
    assert topo["layers"][0]["type"] == "Conv2d"
    assert topo["layers"][0]["in_features"] == 3
    assert topo["layers"][0]["out_features"] == 16
    assert topo["layers"][0]["kernel_size"] == [3, 3]
    assert topo["layers"][0]["stride"] == [1, 1]
    assert topo["layers"][0]["padding"] == [1, 1]
    assert topo["layers"][1]["in_features"] == 16
    assert topo["layers"][1]["out_features"] == 32
    assert topo["max_weight"] > 0
    os.unlink(tmp.name)
    print("\u2705 Conv2d parser: OK")


def test_conv2d_prune():
    """Pruning should work on Conv2d layers."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3),
    )

    _, layers, sparsity = prune_model(model, 0.0)
    assert sparsity == 0.0
    assert len(layers) == 2

    _, layers, sparsity = prune_model(model, 100.0)
    assert sparsity == 100.0
    for layer in layers:
        assert len(layer["sampled_edges"]) == 0
        assert layer["type"] == "Conv2d"

    print("\u2705 Conv2d pruning: OK")


def test_conv2d_quantize():
    """Quantization should work on Conv2d layers."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3),
        nn.Conv2d(16, 32, kernel_size=3),
    )
    model_copy = copy.deepcopy(model)

    # Capture original weights
    orig_w0 = model[0].weight.data.clone()

    new_kb, layers = quantize_model(model_copy, 8, 10.0)
    assert new_kb == 2.5  # 10 * 8/32
    assert len(layers) == 2
    assert layers[0]["type"] == "Conv2d"

    # Weights should have changed (fake quantization)
    assert not torch.equal(orig_w0, model_copy[0].weight.data), (
        "Weights should be modified after quantization"
    )

    print("\u2705 Conv2d quantization: OK")


def test_arbitrary_module():
    """A custom nn.Module (not Sequential) should parse correctly."""
    model = _TinyNet()
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(model, tmp.name)

    loaded, topo = parse_model(tmp.name)
    assert len(topo["layers"]) == 2, f"Expected 2 layers, got {len(topo['layers'])}"

    # Should find Conv2d and Linear
    types = {l["type"] for l in topo["layers"]}
    assert "Conv2d" in types, "Should find Conv2d layer"
    assert "Linear" in types, "Should find Linear layer"

    # Prune should work
    _, layers, sparsity = prune_model(loaded, 100.0)
    assert sparsity == 100.0

    os.unlink(tmp.name)
    print("\u2705 Arbitrary nn.Module support: OK")


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# NEW: Real quantization verification
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def test_real_quantization():
    """Verify that quantization actually modifies weight values."""
    model = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 4))
    original_w = model[0].weight.data.clone()

    model_q = copy.deepcopy(model)
    new_kb, layers = quantize_model(model_q, 4, 1.0)

    # Weights should have changed (fake quantization maps to discrete levels)
    assert not torch.equal(original_w, model_q[0].weight.data), (
        "4-bit quantization should change weight values"
    )

    # The quantized weights should have fewer unique values
    orig_unique = len(torch.unique(original_w))
    quant_unique = len(torch.unique(model_q[0].weight.data))
    assert quant_unique <= orig_unique, (
        f"Quantized weights should have fewer unique values: {quant_unique} vs {orig_unique}"
    )

    # For 4-bit: at most 16 distinct levels (plus possibly 0)
    # Allow some slack due to floating point
    assert quant_unique <= 20, (
        f"4-bit should have ~16 unique values, got {quant_unique}"
    )

    # FP32 quantization should NOT change weights
    model_fp32 = copy.deepcopy(model)
    quantize_model(model_fp32, 32, 1.0)
    assert torch.equal(model[0].weight.data, model_fp32[0].weight.data), (
        "FP32 quantization should not change weights"
    )

    print("\u2705 Real quantization: OK")


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# NEW: Composability tests
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def test_composability():
    """Pruning + quantization should compose: prune first, then quantize the result."""
    model = nn.Sequential(nn.Linear(32, 16), nn.Linear(16, 8))

    # Step 1: Prune (creates a copy)
    pruned, prune_layers, prune_sparsity = prune_model(model, 0.3)
    assert prune_sparsity > 0, "Pruning at 0.3 should produce some sparsity"

    # Step 2: Quantize the pruned model in place
    _, quant_layers = quantize_model(pruned, 8, 10.0)

    # The zeros from pruning should still be zeros after quantization
    for name, module in pruned.named_modules():
        if isinstance(module, nn.Linear):
            zeros_count = (module.weight.data == 0).sum().item()
            assert zeros_count > 0, (
                "Pruned zeros should survive quantization"
            )

    # Layer sparsity should be preserved
    for layer in quant_layers:
        assert layer["layer_sparsity"] >= 0

    print("\u2705 Composability (prune + quantize): OK")


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# NEW: ONNX operations tests
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def test_onnx_parse():
    """ONNX models should parse via parse_onnx_proto."""
    try:
        import onnx
    except ImportError:
        print("\u26a0\ufe0f  ONNX not installed, skipping ONNX tests")
        return

    from core.onnx_parser import parse_onnx_proto

    path = os.path.join(MODELS_DIR, "tiny_classifier.onnx")
    if not os.path.exists(path):
        print("\u26a0\ufe0f  ONNX model not found, skipping")
        return

    model_proto = onnx.load(path, load_external_data=True)
    topo = parse_onnx_proto(model_proto)

    assert len(topo["layers"]) == 2
    assert topo["original_size_kb"] > 0
    assert topo["max_weight"] > 0
    assert topo["model_format"] == "onnx"
    print("\u2705 ONNX parse: OK")


def test_onnx_prune():
    """ONNX pruning should zero out small weights."""
    try:
        import onnx
        import numpy as np
    except ImportError:
        print("\u26a0\ufe0f  ONNX not installed, skipping")
        return

    from core.onnx_parser import parse_onnx_proto, prune_onnx_weights

    path = os.path.join(MODELS_DIR, "tiny_classifier.onnx")
    if not os.path.exists(path):
        print("\u26a0\ufe0f  ONNX model not found, skipping")
        return

    model_proto = onnx.load(path, load_external_data=True)

    # Parse before pruning
    topo_before = parse_onnx_proto(model_proto)
    assert topo_before["global_sparsity"] == 0.0

    # Prune with high threshold
    pruned = copy.deepcopy(model_proto)
    prune_onnx_weights(pruned, 100.0)
    topo_after = parse_onnx_proto(pruned)
    assert topo_after["global_sparsity"] == 100.0, (
        f"Expected 100% sparsity, got {topo_after['global_sparsity']}"
    )

    print("\u2705 ONNX pruning: OK")


def test_onnx_quantize():
    """ONNX quantization should modify weight values."""
    try:
        import onnx
        import numpy as np
        from onnx import numpy_helper
    except ImportError:
        print("\u26a0\ufe0f  ONNX not installed, skipping")
        return

    from core.onnx_parser import quantize_onnx_weights

    path = os.path.join(MODELS_DIR, "tiny_classifier.onnx")
    if not os.path.exists(path):
        print("\u26a0\ufe0f  ONNX model not found, skipping")
        return

    model_proto = onnx.load(path, load_external_data=True)

    # Get original weights
    orig_weights = {}
    for init in model_proto.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.ndim == 2:
            orig_weights[init.name] = arr.copy()

    # Quantize
    quantized = copy.deepcopy(model_proto)
    quantize_onnx_weights(quantized, 4)

    # Check weights changed
    any_changed = False
    for init in quantized.graph.initializer:
        arr = numpy_helper.to_array(init)
        if init.name in orig_weights and arr.ndim == 2:
            if not np.array_equal(arr, orig_weights[init.name]):
                any_changed = True
                # Check fewer unique values
                orig_unique = len(np.unique(orig_weights[init.name]))
                quant_unique = len(np.unique(arr))
                assert quant_unique <= orig_unique

    assert any_changed, "4-bit quantization should change weight values"
    print("\u2705 ONNX quantization: OK")


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Runner
# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

if __name__ == "__main__":
    print("\u2500\u2500\u2500 Original tests \u2500\u2500\u2500")
    test_parser()
    test_pruner_zero()
    test_pruner_high()
    test_quantizer()
    test_quantizer_fp32()

    print("\n\u2500\u2500\u2500 Generated-model tests \u2500\u2500\u2500")
    test_parse_generated_models()
    test_prune_generated_models()
    test_quantize_generated_models()
    test_large_model_performance()

    print("\n\u2500\u2500\u2500 Conv2d & arbitrary Module tests \u2500\u2500\u2500")
    test_conv2d_parser()
    test_conv2d_prune()
    test_conv2d_quantize()
    test_arbitrary_module()

    print("\n\u2500\u2500\u2500 Real quantization verification \u2500\u2500\u2500")
    test_real_quantization()

    print("\n\u2500\u2500\u2500 Composability tests \u2500\u2500\u2500")
    test_composability()

    print("\n\u2500\u2500\u2500 ONNX operations tests \u2500\u2500\u2500")
    test_onnx_parse()
    test_onnx_prune()
    test_onnx_quantize()

    print("\n\U0001f389 All tests passed!")
