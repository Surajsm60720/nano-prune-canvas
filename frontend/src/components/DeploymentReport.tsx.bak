"use client";

import React, { useRef, useCallback, useState } from "react";
import {
    ModelTopology,
    HardwareDevice,
    HARDWARE_DEVICES,
    HARDWARE_DEVICE_CATEGORIES,
    computeModelMetrics,
    computeHardwareAnalysis,
    computeReadinessScore,
} from "@/lib/types";
import { downloadModel } from "@/lib/api";

interface DeploymentReportProps {
    topology: ModelTopology;
    selectedDevice: HardwareDevice;
    bitwidth: number;
    modelId: string | null;
    onClose: () => void;
}

export default function DeploymentReport({
    topology,
    selectedDevice,
    bitwidth,
    modelId,
    onClose,
}: DeploymentReportProps) {
    const reportRef = useRef<HTMLDivElement>(null);
    const [downloading, setDownloading] = useState(false);

    const metrics = computeModelMetrics(topology, bitwidth);
    const hw = computeHardwareAnalysis(topology, selectedDevice, bitwidth);
    const score = computeReadinessScore(
        hw.flashFit, hw.ramFit, hw.flashUtil, hw.ramUtil,
        hw.inferenceMs, hw.energyMj, 100 - metrics.estimatedAccuracyDrop
    );

    const handleExportPDF = useCallback(async () => {
        if (!reportRef.current) return;

        const html2canvas = (await import("html2canvas")).default;
        const { jsPDF } = await import("jspdf");

        const canvas = await html2canvas(reportRef.current, {
            backgroundColor: "#0e1019",
            scale: 2,
        });

        const imgData = canvas.toDataURL("image/png");
        const pdf = new jsPDF("p", "mm", "a4");
        const pdfWidth = pdf.internal.pageSize.getWidth();
        const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

        pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
        pdf.save(`nanoprune_report_${new Date().toISOString().slice(0, 10)}.pdf`);
    }, []);

    const handleDownloadModel = useCallback(async () => {
        if (!modelId) return;
        setDownloading(true);
        try {
            await downloadModel(modelId);
        } catch (err) {
            console.error("Model download failed:", err);
        } finally {
            setDownloading(false);
        }
    }, [modelId]);

    const compressionPct = bitwidth < 32
        ? ((1 - bitwidth / 32) * 100).toFixed(0)
        : topology.global_sparsity > 0
            ? ((1 - topology.current_size_kb / topology.original_size_kb) * 100).toFixed(1)
            : "0";

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-6" onClick={onClose}>
            <div className="absolute inset-0 overlay-backdrop" />

            <div
                className="relative w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-[var(--bg-surface)] border border-[var(--border-default)] rounded-2xl shadow-[var(--shadow-elevated)] animate-fade-in-scale"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Action bar */}
                <div className="sticky top-0 z-10 flex items-center justify-between px-6 py-4 bg-[var(--bg-surface)] border-b border-[var(--border-subtle)] rounded-t-2xl">
                    <h2 className="text-[15px] font-bold text-[var(--text-primary)] tracking-tight">Deployment Report</h2>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={handleDownloadModel}
                            disabled={!modelId || downloading}
                            className="h-8 px-3.5 text-xs font-bold rounded-xl bg-[var(--accent-subtle)] text-[var(--accent)] border border-[var(--accent)]/15 hover:border-[var(--accent)]/30 transition-all flex items-center gap-1.5 cursor-pointer disabled:opacity-25 disabled:cursor-not-allowed"
                        >
                            {downloading ? (
                                <div className="w-3 h-3 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
                            ) : (
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                                    <polyline points="7 10 12 15 17 10" />
                                    <line x1="12" y1="15" x2="12" y2="3" />
                                </svg>
                            )}
                            Model
                        </button>
                        <button
                            onClick={handleExportPDF}
                            className="h-8 px-3.5 text-xs font-bold rounded-xl bg-[var(--purple-subtle)] text-[var(--purple)] border border-[var(--purple)]/15 hover:border-[var(--purple)]/30 transition-all flex items-center gap-1.5 cursor-pointer"
                        >
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                                <polyline points="7 10 12 15 17 10" />
                                <line x1="12" y1="15" x2="12" y2="3" />
                            </svg>
                            PDF
                        </button>
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

                {/* Report content */}
                <div ref={reportRef} className="p-6 space-y-6" style={{ background: "var(--bg-surface)" }}>
                    {/* Report header */}
                    <div className="text-center pb-5 border-b border-[var(--border-subtle)]">
                        <h1 className="text-base font-bold text-[var(--text-primary)] tracking-tight">NanoPrune Deployment Report</h1>
                        <p className="text-xs text-[var(--text-faint)] mt-1.5">
                            Generated {new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
                        </p>
                    </div>

                    {/* Model Summary */}
                    <ReportSection title="Model Summary">
                        <div className="grid grid-cols-3 gap-2">
                            <ReportStat label="Layers" value={`${topology.layers.length}`} />
                            <ReportStat label="Parameters" value={fmtNum(metrics.totalParams)} />
                            <ReportStat label="Total FLOPs" value={fmtNum(metrics.totalFlops)} />
                            <ReportStat label="Original Size" value={`${topology.original_size_kb.toFixed(1)} KB`} />
                            <ReportStat label="Current Size" value={`${topology.current_size_kb.toFixed(1)} KB`} />
                            <ReportStat label="Memory Footprint" value={`${metrics.memoryFootprintKb.toFixed(1)} KB`} />
                        </div>
                    </ReportSection>

                    {/* Compression Applied */}
                    <ReportSection title="Compression Applied">
                        <div className="grid grid-cols-2 gap-3">
                            <div className="bg-[var(--bg-card)] rounded-xl p-4 border border-[var(--border-default)]">
                                <p className="text-[11px] text-[var(--text-faint)]">Pruning</p>
                                <p className="text-xl font-bold text-[var(--text-primary)] mt-1.5 font-mono">
                                    {topology.global_sparsity.toFixed(1)}%
                                </p>
                                <p className="text-[11px] text-[var(--text-faint)] mt-1">global sparsity</p>
                            </div>
                            <div className="bg-[var(--bg-card)] rounded-xl p-4 border border-[var(--border-default)]">
                                <p className="text-[11px] text-[var(--text-faint)]">Quantization</p>
                                <p className="text-xl font-bold text-[var(--text-primary)] mt-1.5 font-mono">
                                    {bitwidth < 32 ? `INT${bitwidth}` : "FP32"}
                                </p>
                                <p className="text-[11px] text-[var(--text-faint)] mt-1">{compressionPct}% size reduction</p>
                            </div>
                        </div>
                        <div className="mt-3 bg-[var(--bg-card)] rounded-xl p-4 border border-[var(--border-default)]">
                            <div className="flex justify-between items-center mb-2">
                                <span className="text-[11px] text-[var(--text-faint)]">Est. Accuracy Retention</span>
                                <span className="text-xs font-bold text-[var(--accent)] font-mono">~{(100 - metrics.estimatedAccuracyDrop).toFixed(1)}%</span>
                            </div>
                            <div className="h-2 bg-[var(--border-default)] rounded-full overflow-hidden">
                                <div
                                    className="h-full rounded-full"
                                    style={{
                                        width: `${100 - metrics.estimatedAccuracyDrop}%`,
                                        background: "linear-gradient(90deg, var(--accent) 0%, #00cc7a 100%)",
                                    }}
                                />
                            </div>
                        </div>
                    </ReportSection>

                    {/* Hardware Compatibility */}
                    <ReportSection title="Hardware Compatibility">
                        <div className="overflow-hidden rounded-xl border border-[var(--border-default)]">
                            <table className="w-full text-xs">
                                <thead>
                                    <tr className="bg-[var(--bg-card)]">
                                        <th className="text-left px-3 py-2.5 text-[var(--text-faint)] font-bold text-[11px] uppercase tracking-wider">Device</th>
                                        <th className="text-center px-2 py-2.5 text-[var(--text-faint)] font-bold text-[11px] uppercase tracking-wider">Flash</th>
                                        <th className="text-center px-2 py-2.5 text-[var(--text-faint)] font-bold text-[11px] uppercase tracking-wider">SRAM</th>
                                        <th className="text-center px-2 py-2.5 text-[var(--text-faint)] font-bold text-[11px] uppercase tracking-wider">Inference</th>
                                        <th className="text-center px-2 py-2.5 text-[var(--text-faint)] font-bold text-[11px] uppercase tracking-wider">Fit</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {HARDWARE_DEVICE_CATEGORIES.map((cat) => {
                                        const devices = HARDWARE_DEVICES.filter((d) => d.category === cat);
                                        return (
                                            <React.Fragment key={cat}>
                                                <tr className="border-t border-[var(--border-default)]">
                                                    <td colSpan={5} className="px-3 py-2 text-[11px] font-bold text-[var(--accent)] bg-[var(--accent-subtle)] uppercase tracking-wider">{cat}</td>
                                                </tr>
                                                {devices.map((device) => {
                                                    const hw = computeHardwareAnalysis(topology, device, bitwidth);
                                                    const fits = hw.flashFit && hw.ramFit;
                                                    return (
                                                        <tr key={device.name} className="border-t border-[var(--border-subtle)]">
                                                            <td className="px-3 py-2.5 text-[var(--text-primary)] font-semibold">{device.name}</td>
                                                            <td className="text-center px-2 py-2.5">
                                                                <span className={hw.flashFit ? "text-[var(--accent)]" : "text-[var(--danger)]"}>
                                                                    {hw.flashFit ? "OK" : "Fail"}
                                                                </span>
                                                            </td>
                                                            <td className="text-center px-2 py-2.5">
                                                                <span className={hw.ramFit ? "text-[var(--accent)]" : "text-[var(--danger)]"}>
                                                                    {hw.ramFit ? "OK" : "Fail"}
                                                                </span>
                                                            </td>
                                                            <td className="text-center px-2 py-2.5 text-[var(--text-secondary)] font-mono">
                                                                {hw.inferenceMs < 1 ? `${(hw.inferenceMs * 1000).toFixed(0)}µs` : `${hw.inferenceMs.toFixed(1)}ms`}
                                                            </td>
                                                            <td className="text-center px-2 py-2.5">
                                                                <span className={`w-2 h-2 rounded-full inline-block ${fits ? "bg-[var(--accent)]" : "bg-[var(--danger)]"}`} />
                                                            </td>
                                                        </tr>
                                                    );
                                                })}
                                            </React.Fragment>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </ReportSection>

                    {/* Edge Readiness */}
                    <ReportSection title="Edge Readiness Score">
                        <div className="flex items-center gap-5">
                            <div className="relative w-24 h-24 shrink-0">
                                <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                                    <circle cx="50" cy="50" r="42" fill="none" stroke="var(--border-default)" strokeWidth="5" />
                                    <circle
                                        cx="50" cy="50" r="42" fill="none"
                                        stroke={score.total >= 70 ? "var(--accent)" : score.total >= 40 ? "var(--warning)" : "var(--danger)"}
                                        strokeWidth="5"
                                        strokeLinecap="round"
                                        strokeDasharray={`${(score.total / 100) * 264} 264`}
                                        style={{ filter: `drop-shadow(0 0 4px ${score.total >= 70 ? "var(--accent-glow)" : "transparent"})` }}
                                    />
                                </svg>
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <span className="text-xl font-bold font-mono text-[var(--text-primary)]">{score.total}</span>
                                </div>
                            </div>
                            <div className="flex-1 space-y-2.5">
                                <ScoreRow label="Memory" value={score.memory.score} max={30} />
                                <ScoreRow label="Latency" value={score.latency.score} max={30} />
                                <ScoreRow label="Energy" value={score.energy.score} max={20} />
                                <ScoreRow label="Accuracy" value={score.accuracy.score} max={20} />
                            </div>
                        </div>
                    </ReportSection>

                    {/* Recommendations */}
                    <ReportSection title="Recommendations">
                        <div className="space-y-2.5">
                            {generateRecommendations(topology, metrics, score, bitwidth).map((rec, i) => (
                                <div key={i} className="flex items-start gap-2.5 text-xs">
                                    <span className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${rec.type === "good" ? "bg-[var(--accent)]" : "bg-[var(--warning)]"}`} />
                                    <span className="text-[var(--text-secondary)]">{rec.text}</span>
                                </div>
                            ))}
                        </div>
                    </ReportSection>
                </div>
            </div>
        </div>
    );
}

/* ── Sub-components ──────────────────────────────────── */

function ReportSection({ title, children }: { title: string; children: React.ReactNode }) {
    return (
        <div>
            <h3 className="text-xs font-bold text-[var(--text-secondary)] mb-3 uppercase tracking-wider">{title}</h3>
            {children}
        </div>
    );
}

function ReportStat({ label, value }: { label: string; value: string }) {
    return (
        <div className="bg-[var(--bg-card)] rounded-xl p-3 border border-[var(--border-default)]">
            <p className="text-[11px] text-[var(--text-faint)]">{label}</p>
            <p className="text-xs font-bold text-[var(--text-primary)] mt-1 font-mono">{value}</p>
        </div>
    );
}

function ScoreRow({ label, value, max }: { label: string; value: number; max: number }) {
    return (
        <div>
            <div className="flex justify-between mb-1">
                <span className="text-xs text-[var(--text-secondary)] font-medium">{label}</span>
                <span className="text-xs font-mono font-bold text-[var(--text-primary)]">{value}/{max}</span>
            </div>
            <div className="h-1.5 bg-[var(--border-default)] rounded-full overflow-hidden">
                <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{
                        width: `${(value / max) * 100}%`,
                        background: "linear-gradient(90deg, var(--accent) 0%, #00cc7a 100%)",
                    }}
                />
            </div>
        </div>
    );
}

function fmtNum(n: number): string {
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
    return `${n}`;
}

function generateRecommendations(
    topology: ModelTopology,
    metrics: ReturnType<typeof computeModelMetrics>,
    score: ReturnType<typeof computeReadinessScore>,
    bitwidth: number,
) {
    const recs: { type: "good" | "warn"; text: string }[] = [];

    if (score.total >= 70) {
        recs.push({ type: "good", text: "Model is edge-ready with good overall score." });
    }

    if (bitwidth === 32) {
        recs.push({ type: "warn", text: "Consider INT8 quantization for 4x size reduction with minimal accuracy loss." });
    } else if (bitwidth === 16) {
        recs.push({ type: "warn", text: "INT8 would further halve model size compared to FP16." });
    } else if (bitwidth <= 8) {
        recs.push({ type: "good", text: `Quantization to INT${bitwidth} applied — good compression.` });
    }

    if (topology.global_sparsity < 10) {
        recs.push({ type: "warn", text: "Apply 30-50% pruning to reduce model size with minimal accuracy impact." });
    } else if (topology.global_sparsity >= 50) {
        recs.push({ type: "good", text: `${topology.global_sparsity.toFixed(0)}% sparsity — significant compression achieved.` });
    }

    if (metrics.estimatedAccuracyDrop > 5) {
        recs.push({ type: "warn", text: "High estimated accuracy drop. Consider reducing pruning threshold or using INT8 instead of INT4." });
    } else {
        recs.push({ type: "good", text: `Estimated accuracy retention ~${(100 - metrics.estimatedAccuracyDrop).toFixed(1)}% — acceptable for most applications.` });
    }

    if (score.memory.score < 15) {
        recs.push({ type: "warn", text: "Model may not fit on smaller MCUs. Consider more aggressive compression." });
    }

    return recs;
}
