"use client";

import React, { useCallback, useRef, useState } from "react";

interface ControlPanelProps {
    onUpload: (file: File) => void;
    threshold: number;
    onThresholdChange: (value: number) => void;
    bitwidth: number;
    onBitwidthChange: (value: number) => void;
    isLoading: boolean;
    hasModel: boolean;
    maxWeight: number;
    sparsity: number;
    onOptimize?: () => void;
}

const BITWIDTH_OPTIONS = [
    { value: 32, label: "FP32", desc: "32-bit" },
    { value: 16, label: "FP16", desc: "16-bit" },
    { value: 8, label: "INT8", desc: "8-bit" },
    { value: 4, label: "INT4", desc: "4-bit" },
];

export default function ControlPanel({
    onUpload,
    threshold,
    onThresholdChange,
    bitwidth,
    onBitwidthChange,
    isLoading,
    hasModel,
    maxWeight,
    sparsity,
    onOptimize,
}: ControlPanelProps) {
    const sliderMax = maxWeight > 0 ? maxWeight : 1;
    const sliderStep = sliderMax / 200;
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [dragOver, setDragOver] = useState(false);

    const handleFileSelect = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            const file = e.target.files?.[0];
            if (file) onUpload(file);
        },
        [onUpload]
    );

    const handleDrop = useCallback(
        (e: React.DragEvent) => {
            e.preventDefault();
            setDragOver(false);
            const file = e.dataTransfer.files?.[0];
            if (file && (file.name.endsWith(".pt") || file.name.endsWith(".pth") || file.name.endsWith(".onnx"))) {
                onUpload(file);
            }
        },
        [onUpload]
    );

    const compressionPct = bitwidth < 32 ? ((1 - bitwidth / 32) * 100).toFixed(0) : "0";

    return (
        <div className="w-[280px] shrink-0 bg-[var(--bg-surface)] flex flex-col overflow-y-auto">
            {/* Header */}
            <div className="px-5 pt-5 pb-4">
                <h2 className="text-[13px] font-bold text-[var(--text-primary)] tracking-tight">Controls</h2>
                <p className="text-[11px] text-[var(--text-faint)] mt-1">Configure compression settings</p>
            </div>

            <div className="px-5 pb-5 flex flex-col gap-5 flex-1">
                {/* Upload zone */}
                <div>
                    <div
                        onClick={() => fileInputRef.current?.click()}
                        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={handleDrop}
                        className={`cursor-pointer rounded-xl border-2 border-dashed p-5 text-center transition-all duration-200 ${dragOver
                            ? "border-[var(--accent)] bg-[var(--accent-subtle)] shadow-[0_0_20px_var(--accent-glow)]"
                            : "border-[var(--border-default)] hover:border-[var(--text-faint)] bg-[var(--bg-inset)]"
                            }`}
                    >
                        {isLoading ? (
                            <div className="flex flex-col items-center gap-2.5 py-1">
                                <div className="w-6 h-6 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
                                <span className="text-xs font-medium text-[var(--accent)]">Parsing model…</span>
                            </div>
                        ) : (
                            <>
                                <div className="w-10 h-10 mx-auto mb-3 rounded-xl bg-[var(--bg-elevated)] border border-[var(--border-default)] flex items-center justify-center">
                                    <svg className="w-5 h-5 text-[var(--text-faint)]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                                    </svg>
                                </div>
                                <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                                    Drop <code className="text-[var(--accent)] font-semibold">.pt</code> / <code className="text-[var(--accent)] font-semibold">.onnx</code>
                                </p>
                                <p className="text-[11px] text-[var(--text-faint)] mt-1">or click to browse</p>
                            </>
                        )}
                    </div>
                    <input ref={fileInputRef} type="file" accept=".pt,.pth,.onnx" onChange={handleFileSelect} className="hidden" />
                </div>

                {/* ── Pruning ── */}
                <div className={!hasModel ? "opacity-25 pointer-events-none" : ""}>
                    <div className="flex items-center justify-between mb-3">
                        <label className="text-xs font-semibold text-[var(--text-secondary)]">Pruning</label>
                        <span className="text-[11px] font-mono font-semibold text-[var(--accent)] bg-[var(--accent-subtle)] px-2 py-0.5 rounded-full">
                            {sparsity.toFixed(1)}%
                        </span>
                    </div>
                    <div className="flex items-center gap-3">
                        <input
                            type="range"
                            min="0"
                            max={sliderMax}
                            step={sliderStep}
                            value={threshold}
                            onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
                            className="flex-1"
                        />
                        <span className="text-xs font-mono text-[var(--text-primary)] w-12 text-right tabular-nums font-semibold">
                            {threshold.toFixed(3)}
                        </span>
                    </div>
                    {/* Progress bar */}
                    <div className="mt-2.5 h-1.5 bg-[var(--border-default)] rounded-full overflow-hidden">
                        <div
                            className="h-full rounded-full transition-all duration-300"
                            style={{
                                width: `${Math.min(sparsity, 100)}%`,
                                background: "linear-gradient(90deg, var(--accent) 0%, #00cc7a 100%)",
                            }}
                        />
                    </div>
                    <p className="text-[10px] text-[var(--text-faint)] mt-1.5 font-mono">
                        τ = {threshold.toFixed(4)} · max|W| = {sliderMax.toFixed(4)}
                    </p>
                </div>

                {/* ── Quantization ── */}
                <div className={!hasModel ? "opacity-25 pointer-events-none" : ""}>
                    <div className="flex items-center justify-between mb-3">
                        <label className="text-xs font-semibold text-[var(--text-secondary)]">Quantization</label>
                        {bitwidth < 32 && (
                            <span className="text-[11px] font-semibold text-[var(--success)] bg-[var(--accent-subtle)] px-2 py-0.5 rounded-full">
                                -{compressionPct}%
                            </span>
                        )}
                    </div>
                    <div className="grid grid-cols-4 gap-1.5">
                        {BITWIDTH_OPTIONS.map((opt) => (
                            <button
                                key={opt.value}
                                onClick={() => onBitwidthChange(opt.value)}
                                className={`py-2.5 rounded-xl text-center transition-all duration-200 cursor-pointer ${bitwidth === opt.value
                                    ? "bg-gradient-to-b from-[var(--accent)] to-[#00cc7a] text-[#080a12] shadow-[0_2px_12px_var(--accent-glow)]"
                                    : "bg-[var(--bg-inset)] text-[var(--text-muted)] hover:text-[var(--text-secondary)] border border-[var(--border-default)] hover:border-[var(--text-faint)]"
                                    }`}
                            >
                                <div className="text-xs font-bold">{opt.label}</div>
                                <div className={`text-[10px] mt-0.5 ${bitwidth === opt.value ? "opacity-75" : "opacity-50"}`}>{opt.desc}</div>
                            </button>
                        ))}
                    </div>
                </div>

                {/* ── Optimize CTA ── */}
                {hasModel && (
                    <button
                        onClick={onOptimize}
                        disabled={isLoading || (sparsity === 0 && bitwidth === 32)}
                        className="w-full py-3 rounded-xl text-sm font-bold transition-all duration-200 cursor-pointer disabled:opacity-25 disabled:cursor-not-allowed btn-accent flex items-center justify-center gap-2 group"
                    >
                        {isLoading ? (
                            <>
                                <div className="w-4 h-4 border-2 border-[#080a12] border-t-transparent rounded-full animate-spin" />
                                Processing…
                            </>
                        ) : (
                            <>
                                <svg className="w-4 h-4 transition-transform group-hover:rotate-12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                                </svg>
                                Optimize Model
                            </>
                        )}
                    </button>
                )}

                {/* ── Summary card ── */}
                {hasModel && (
                    <div className="bg-[var(--bg-card)] rounded-xl p-4 border border-[var(--border-default)] shadow-[var(--shadow-card)] hover-lift">
                        <h3 className="text-xs font-bold text-[var(--text-secondary)] mb-3 tracking-wide uppercase">Summary</h3>
                        <div className="space-y-2.5">
                            <div className="flex justify-between text-xs items-center">
                                <span className="text-[var(--text-faint)]">Pruning</span>
                                <span className="text-[var(--text-primary)] font-mono font-semibold">{sparsity.toFixed(1)}%</span>
                            </div>
                            <div className="h-px bg-[var(--border-subtle)]" />
                            <div className="flex justify-between text-xs items-center">
                                <span className="text-[var(--text-faint)]">Precision</span>
                                <span className="text-[var(--text-primary)] font-mono font-semibold">
                                    {BITWIDTH_OPTIONS.find(b => b.value === bitwidth)?.label}
                                </span>
                            </div>
                            <div className="h-px bg-[var(--border-subtle)]" />
                            <div className="flex justify-between text-xs items-center">
                                <span className="text-[var(--text-faint)]">Size reduction</span>
                                <span className="text-[var(--accent)] font-mono font-bold">{compressionPct}%</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Footer */}
            <div className="px-5 py-3.5 border-t border-[var(--border-subtle)]">
                <p className="text-[11px] text-[var(--text-faint)] leading-relaxed">
                    Magnitude pruning & post-training quantization for edge deployment.
                </p>
            </div>
        </div>
    );
}
