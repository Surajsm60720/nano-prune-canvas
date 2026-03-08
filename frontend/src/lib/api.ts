import { ModelTopology } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function uploadModel(file: File): Promise<ModelTopology> {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Upload failed");
    }

    return res.json();
}

export async function pruneModel(
    modelId: string,
    threshold: number
): Promise<ModelTopology> {
    const res = await fetch(`${API_BASE}/api/prune`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId, threshold }),
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Prune failed");
    }

    return res.json();
}

export async function quantizeModel(
    modelId: string,
    targetBitwidth: number
): Promise<ModelTopology> {
    const res = await fetch(`${API_BASE}/api/quantize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId, target_bitwidth: targetBitwidth }),
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Quantize failed");
    }

    return res.json();
}

export async function downloadModel(modelId: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId }),
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Export failed");
    }

    const blob = await res.blob();
    const disposition = res.headers.get("Content-Disposition");
    const filename = disposition?.match(/filename=(.+)/)?.[1] || "compressed_model.pt";

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
