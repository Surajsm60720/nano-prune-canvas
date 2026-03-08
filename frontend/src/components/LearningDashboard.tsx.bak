"use client";

import React, { useState } from "react";

interface LearningDashboardProps {
    onClose: () => void;
}

type SectionId = "pruning" | "quantization" | "tradeoff" | "pipeline";

const SECTIONS: { id: SectionId; title: string; subtitle: string }[] = [
    { id: "pruning", title: "What is Pruning?", subtitle: "Remove insignificant weights to shrink models" },
    { id: "quantization", title: "What is Quantization?", subtitle: "Reduce numerical precision for efficiency" },
    { id: "tradeoff", title: "Accuracy vs Compression", subtitle: "Understanding the optimization tradeoff" },
    { id: "pipeline", title: "TinyML Pipeline", subtitle: "From sensor data to edge inference" },
];

export default function LearningDashboard({ onClose }: LearningDashboardProps) {
    const [expanded, setExpanded] = useState<SectionId | null>("pruning");

    return (
        <div className="fixed inset-0 z-50 flex justify-end" onClick={onClose}>
            <div className="absolute inset-0 overlay-backdrop" />

            <div
                className="relative w-[480px] h-full bg-[var(--bg-surface)] border-l border-[var(--border-default)] overflow-y-auto animate-slide-in-right"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className="sticky top-0 z-10 px-6 py-5 bg-[var(--bg-surface)] border-b border-[var(--border-subtle)]">
                    <div className="flex items-center justify-between">
                        <div>
                            <h2 className="text-[15px] font-bold text-[var(--text-primary)] tracking-tight">Learn</h2>
                            <p className="text-xs text-[var(--text-faint)] mt-1">Model compression explained</p>
                        </div>
                        <button
                            onClick={onClose}
                            className="w-8 h-8 rounded-xl bg-[var(--bg-elevated)] border border-[var(--border-default)] flex items-center justify-center text-[var(--text-faint)] hover:text-[var(--text-primary)] transition-colors cursor-pointer"
                        >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                            </svg>
                        </button>
                    </div>
                </div>

                {/* Sections */}
                <div className="p-6 space-y-3">
                    {SECTIONS.map((section) => (
                        <div key={section.id} className="border border-[var(--border-default)] rounded-xl overflow-hidden bg-[var(--bg-card)]">
                            <button
                                onClick={() => setExpanded(expanded === section.id ? null : section.id)}
                                className="w-full px-5 py-4 flex items-center justify-between text-left hover:bg-[var(--bg-elevated)] transition-colors cursor-pointer"
                            >
                                <div>
                                    <h3 className="text-xs font-bold text-[var(--text-primary)]">{section.title}</h3>
                                    <p className="text-[11px] text-[var(--text-faint)] mt-0.5">{section.subtitle}</p>
                                </div>
                                <svg
                                    width="16" height="16" viewBox="0 0 24 24" fill="none"
                                    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                                    className={`text-[var(--text-faint)] transition-transform duration-200 shrink-0 ml-3 ${expanded === section.id ? "rotate-180" : ""}`}
                                >
                                    <polyline points="6 9 12 15 18 9" />
                                </svg>
                            </button>

                            {expanded === section.id && (
                                <div className="px-5 pb-5 animate-fade-in">
                                    {section.id === "pruning" && <PruningExplainer />}
                                    {section.id === "quantization" && <QuantizationExplainer />}
                                    {section.id === "tradeoff" && <TradeoffExplainer />}
                                    {section.id === "pipeline" && <PipelineExplainer />}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

/* ── Section 1: Pruning Explainer ────────────────────── */

function PruningExplainer() {
    return (
        <div className="space-y-3">
            <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                <strong className="text-[var(--text-primary)]">Magnitude-based pruning</strong> removes weights
                whose absolute values fall below a threshold. Small weights contribute
                little to the output, so zeroing them has minimal impact on accuracy.
            </p>

            {/* Weight matrix */}
            <div className="bg-[var(--bg-card)] rounded-xl p-4 border border-[var(--border-default)]">
                <p className="text-[11px] font-bold text-[var(--text-secondary)] mb-3 uppercase tracking-wider">Weight matrix visualization</p>
                <div className="grid grid-cols-6 gap-1">
                    {[0.92, 0.03, -0.87, 0.01, 0.65, -0.02,
                        0.04, -0.91, 0.02, 0.78, -0.01, 0.43,
                        -0.56, 0.01, 0.89, -0.03, 0.02, -0.71,
                        0.02, 0.67, -0.01, -0.95, 0.04, 0.01].map((w, i) => {
                            const isPruned = Math.abs(w) < 0.05;
                            return (
                                <div
                                    key={i}
                                    className={`h-8 rounded-lg flex items-center justify-center text-[10px] font-mono font-semibold ${isPruned
                                        ? "bg-[var(--border-default)] text-[var(--text-faint)] line-through"
                                        : w > 0
                                            ? "bg-[var(--accent-subtle)] text-[var(--accent)] border border-[var(--accent)]/15"
                                            : "bg-[var(--danger)]/8 text-[var(--danger)] border border-[var(--danger)]/15"
                                        }`}
                                >
                                    {isPruned ? "0" : w.toFixed(2)}
                                </div>
                            );
                        })}
                </div>
                <div className="flex items-center gap-4 mt-2.5">
                    <Legend color="var(--accent)" label="Kept (positive)" />
                    <Legend color="var(--danger)" label="Kept (negative)" />
                    <Legend color="var(--text-faint)" label="Pruned" />
                </div>
            </div>

            <Callout
                title="Key Insight"
                text="Unstructured pruning sets weights to zero but keeps the tensor shape. Real size savings come from sparse storage formats (CSR/CSC) or compression."
            />
        </div>
    );
}

/* ── Section 2: Quantization Explainer ───────────────── */

function QuantizationExplainer() {
    const levels = [
        { label: "FP32", bits: 32, desc: "Full precision (32 bits per weight)", example: "0.123456789" },
        { label: "FP16", bits: 16, desc: "Half precision (16 bits per weight)", example: "0.12346" },
        { label: "INT8", bits: 8, desc: "8-bit integer (256 levels)", example: "31" },
        { label: "INT4", bits: 4, desc: "4-bit integer (16 levels)", example: "8" },
    ];

    return (
        <div className="space-y-3">
            <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                <strong className="text-[var(--text-primary)]">Quantization</strong> reduces the number of bits
                used to represent each weight. Fewer bits means smaller models and faster
                inference, with a small accuracy tradeoff.
            </p>

            <div className="space-y-2">
                {levels.map((lvl) => (
                    <div key={lvl.label} className="flex items-center gap-3">
                        <div className="w-12 py-1.5 rounded-lg text-center text-[11px] font-mono font-bold bg-[var(--bg-elevated)] border border-[var(--border-default)] text-[var(--text-primary)]">
                            {lvl.label}
                        </div>
                        <div className="flex-1">
                            <div className="h-1.5 bg-[var(--border-default)] rounded-full overflow-hidden mb-1">
                                <div
                                    className="h-full rounded-full bg-[var(--accent)] transition-all duration-500"
                                    style={{ width: `${(lvl.bits / 32) * 100}%` }}
                                />
                            </div>
                            <div className="flex justify-between">
                                <span className="text-[11px] text-[var(--text-faint)]">{lvl.desc}</span>
                                <span className="text-[11px] font-mono text-[var(--text-secondary)]">{lvl.example}</span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            <Callout
                title="Size Reduction"
                text="INT8 gives 4x smaller model vs FP32. INT4 gives 8x reduction. Combined with pruning, 10-50x reduction is achievable."
            />
        </div>
    );
}

/* ── Section 3: Tradeoff Explainer ───────────────────── */

function TradeoffExplainer() {
    const points = [
        { compression: 0, accuracy: 100 }, { compression: 10, accuracy: 99.8 },
        { compression: 20, accuracy: 99.5 }, { compression: 30, accuracy: 99.0 },
        { compression: 40, accuracy: 98.2 }, { compression: 50, accuracy: 97.0 },
        { compression: 60, accuracy: 95.2 }, { compression: 70, accuracy: 92.0 },
        { compression: 80, accuracy: 86.5 }, { compression: 90, accuracy: 75.0 },
        { compression: 95, accuracy: 60.0 },
    ];

    return (
        <div className="space-y-3">
            <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                There is an inherent tradeoff between <strong className="text-[var(--text-primary)]">model compression</strong> and{" "}
                <strong className="text-[var(--text-primary)]">accuracy retention</strong>. Moderate compression (30-50%) typically
                preserves most accuracy, while aggressive compression (&gt;80%) causes significant degradation.
            </p>

            <div className="bg-[var(--bg-card)] rounded-xl p-4 border border-[var(--border-default)]">
                <p className="text-[11px] font-bold text-[var(--text-secondary)] mb-3 uppercase tracking-wider">Accuracy vs Compression</p>
                <svg viewBox="0 0 300 160" className="w-full">
                    {/* Grid */}
                    {[0, 25, 50, 75, 100].map((v) => (
                        <g key={v}>
                            <line x1="30" y1={140 - v * 1.3} x2="290" y2={140 - v * 1.3} stroke="var(--border-muted)" strokeWidth="0.5" />
                            <text x="26" y={143 - v * 1.3} textAnchor="end" fill="var(--text-faint)" fontSize="7" fontFamily="var(--font-mono)">
                                {v}%
                            </text>
                        </g>
                    ))}

                    {/* Area */}
                    <path
                        d={`M ${points.map((p) => `${30 + p.compression * 2.6},${140 - p.accuracy * 1.3}`).join(" L ")} L 277,140 L 30,140 Z`}
                        fill="var(--accent)" fillOpacity="0.06"
                    />

                    {/* Line */}
                    <polyline
                        points={points.map((p) => `${30 + p.compression * 2.6},${140 - p.accuracy * 1.3}`).join(" ")}
                        fill="none" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
                    />

                    {/* Sweet spot */}
                    <rect x="108" y="10" width="78" height="130" fill="var(--success)" fillOpacity="0.04" rx="3" />
                    <text x="147" y="22" textAnchor="middle" fill="var(--success)" fontSize="7" fontWeight="600" fontFamily="var(--font-mono)">
                        Sweet Spot
                    </text>

                    {/* Points */}
                    {points.map((p, i) => (
                        <circle key={i} cx={30 + p.compression * 2.6} cy={140 - p.accuracy * 1.3} r="2" fill="var(--accent)" />
                    ))}

                    <text x="160" y="156" textAnchor="middle" fill="var(--text-faint)" fontSize="7">Compression %</text>
                </svg>
            </div>

            <Callout
                title="Best Practice"
                text="Start with 30-50% pruning + INT8 quantization. This typically achieves 4-6x compression with less than 1% accuracy drop."
            />
        </div>
    );
}

/* ── Section 4: Pipeline Explainer ───────────────────── */

function PipelineExplainer() {
    const stages = [
        { label: "Sensor", desc: "Camera, mic, IMU" },
        { label: "Preprocess", desc: "Normalize, filter" },
        { label: "MCU", desc: "STM32, ESP32" },
        { label: "Inference", desc: "Quantized model" },
        { label: "Action", desc: "Alert, classify" },
    ];

    return (
        <div className="space-y-3">
            <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                <strong className="text-[var(--text-primary)]">TinyML</strong> runs machine learning models directly on
                microcontrollers, enabling on-device inference without cloud connectivity — reducing
                latency, cost, and power consumption.
            </p>

            <div className="bg-[var(--bg-card)] rounded-xl p-4 border border-[var(--border-default)]">
                <p className="text-[11px] font-bold text-[var(--text-secondary)] mb-3 uppercase tracking-wider">Edge inference pipeline</p>
                <div className="flex items-center gap-1">
                    {stages.map((stage, i) => (
                        <React.Fragment key={stage.label}>
                            <div className="flex-1 rounded-xl p-2.5 text-center bg-[var(--bg-elevated)] border border-[var(--border-default)]">
                                <div className="text-[11px] font-medium text-[var(--text-primary)]">{stage.label}</div>
                                <div className="text-[10px] text-[var(--text-faint)] mt-0.5">{stage.desc}</div>
                            </div>
                            {i < stages.length - 1 && (
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="var(--text-faint)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
                                    <polyline points="9 18 15 12 9 6" />
                                </svg>
                            )}
                        </React.Fragment>
                    ))}
                </div>
            </div>

            <Callout
                title="Why Compress?"
                text="MCUs have 256KB-8MB flash and 64KB-512KB SRAM. Models must fit within these constraints while maintaining acceptable accuracy."
            />
        </div>
    );
}

/* ── Shared Components ───────────────────────────────── */

function Callout({ title, text }: { title: string; text: string }) {
    return (
        <div className="rounded-xl p-4 bg-[var(--accent-subtle)] border border-[var(--accent)]/10">
            <p className="text-[11px] font-bold text-[var(--accent)] mb-1">{title}</p>
            <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">{text}</p>
        </div>
    );
}

function Legend({ color, label }: { color: string; label: string }) {
    return (
        <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded" style={{ background: color, opacity: 0.6 }} />
            <span className="text-[10px] text-[var(--text-faint)] font-medium">{label}</span>
        </div>
    );
}
