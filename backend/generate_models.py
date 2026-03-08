"""
Generate a suite of diverse nn.Sequential test models for NanoPrune Canvas.
Exports both PyTorch (.pt) and ONNX (.onnx) formats.

Run:  python generate_models.py
Creates model files in backend/models/
"""

import os
import torch
import torch.nn as nn

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def make_sequential(*layer_sizes: int) -> nn.Sequential:
    """Build nn.Sequential(Linear, ReLU, Linear, ReLU, ..., Linear)."""
    layers: list[nn.Module] = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:          # no ReLU after the last Linear
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


MODELS: dict[str, tuple[int, ...]] = {
    # 1. Minimal baseline
    "tiny_classifier":    (4, 8, 3),
    # 2. Classic MNIST-style MLP
    "mnist_mlp":          (784, 256, 128, 64, 10),
    # 3. Wide shallow — stress-tests edge sampling
    "wide_shallow":       (128, 1024, 512, 2),
    # 4. Deep narrow — many nodes on the canvas
    "deep_narrow":        (16, 32, 32, 32, 32, 32, 16, 8, 4),
    # 5. Progressive dimension reduction
    "embedding_reducer":  (512, 256, 128, 64, 32, 16),
    # 6. Large recommender
    "large_recommender":  (2048, 1024, 512, 256, 128, 64, 10),
}


def export_onnx(model: nn.Sequential, input_dim: int, path: str) -> None:
    """Export a PyTorch model to ONNX format with all data embedded inline."""
    import onnx
    dummy_input = torch.randn(1, input_dim)

    # Export to a temp path first
    tmp_path = path + ".tmp"
    torch.onnx.export(
        model,
        dummy_input,
        tmp_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )

    # Re-save with all external data converted to inline tensors
    onnx_model = onnx.load(tmp_path, load_external_data=True)
    onnx.save_model(onnx_model, path, save_as_external_data=False)

    # Cleanup temp file and any .data files
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    data_file = tmp_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=== PyTorch Models (.pt) ===")
    for name, sizes in MODELS.items():
        model = make_sequential(*sizes)

        total_params = sum(p.numel() for p in model.parameters())
        size_kb = round(total_params * 4 / 1024, 2)

        pt_path = os.path.join(MODELS_DIR, f"{name}.pt")
        torch.save(model, pt_path)

        n_linear = sum(1 for m in model if isinstance(m, nn.Linear))
        print(f"  ✅ {name:24s}  layers={n_linear}  params={total_params:>10,}  ~{size_kb:>10.2f} KB  → {pt_path}")

    print("\n=== ONNX Models (.onnx) ===")
    for name, sizes in MODELS.items():
        model = make_sequential(*sizes)
        model.eval()

        onnx_path = os.path.join(MODELS_DIR, f"{name}.onnx")
        input_dim = sizes[0]

        try:
            export_onnx(model, input_dim, onnx_path)
            file_size_kb = round(os.path.getsize(onnx_path) / 1024, 2)
            print(f"  ✅ {name:24s}  input_dim={input_dim:>5}  file={file_size_kb:>10.2f} KB  → {onnx_path}")
        except Exception as e:
            print(f"  ❌ {name:24s}  ONNX export failed: {e}")

    print(f"\n🎉 Generated {len(MODELS)} models (PT + ONNX) in {MODELS_DIR}")


if __name__ == "__main__":
    main()
