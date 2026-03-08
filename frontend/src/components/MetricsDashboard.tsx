"use client";

import React, { useMemo, useState, useEffect, useRef } from "react";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    Cell,
} from "recharts";
import {
    ModelTopology,
    HardwareDevice,
    HARDWARE_DEVICES,
    HARDWARE_DEVICE_CATEGORIES,
    computeModelMetrics,
    computeHardwareAnalysis,
    computeReadinessScore,
} from "@/lib/types";

interface MetricsDashboardProps {
    topology: ModelTopology | null;
    selectedDevice: HardwareDevice;
    onDeviceChange: (device: HardwareDevice) => void;
    onShowReport?: () => void;
}

type TabId = "model" | "hardware" | "readiness";
const TABS: { id: TabId; label: string }[] = [
    { id: "model", label: "Model" },
    { id: "hardware", label: "Hardware" },
    { id: "readiness", label: "Readiness" },
];

function fmt(n: number): string {
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
    return `${n}`;
}

function fmtKb(kb: number): string {
    if (kb >= 1048576) return `${(kb / 1048576).toFixed(1)} GB`;
    if (kb >= 1024) return `${(kb / 1024).toFixed(1)} MB`;
    return `${kb.toFixed(1)} KB`;
}

function fmtBattery(hrs: number): string {
    if (hrs >= 8760) return `~${(hrs / 8760).toFixed(1)} yrs`;
    if (hrs >= 720) return `~${(hrs / 720).toFixed(0)} mo`;
    if (hrs >= 48) return `~${(hrs / 24).toFixed(0)} days`;
    return `~${hrs.toFixed(0)} hrs`;
}

export default function MetricsDashboard({
    topology,
    selectedDevice,
    onDeviceChange,
    onShowReport,
}: MetricsDashboardProps) {
    const [tab, setTab] = useState<TabId>("model");

    // Compression history tracker
    const historyRef = useRef<{ sparsity: number; sizeKb: number; bitwidth: number }[]>([]);
    useEffect(() => {
        if (topology) {
            const last = historyRef.current[historyRef.current.length - 1];
            const cur = { sparsity: topology.global_sparsity, sizeKb: topology.current_size_kb, bitwidth: 32 };
            if (!last || Math.abs(last.sparsity - cur.sparsity) > 0.5 || Math.abs(last.sizeKb - cur.sizeKb) > 0.5) {
                historyRef.current = [...historyRef.current.slice(-19), cur];
            }
        }
    }, [topology]);

    const metrics = useMemo(
        () => (topology ? computeModelMetrics(topology) : null),
        [topology]
    );

    const hw = useMemo(() => {
        if (!topology || !metrics) return null;
        return computeHardwareAnalysis(
            topology,
            selectedDevice,
            metrics.totalFlops,
            metrics.peakActivationKb
        );
    }, [topology, selectedDevice, metrics]);

    const readiness = useMemo(() => {
        if (!hw || !metrics) return null;
        return computeReadinessScore(
            hw.flashFit,
            hw.ramFit,
            hw.flashUtil,
            hw.ramUtil,
            hw.inferenceMs,
            hw.energyMj,
            100 - metrics.estimatedAccuracyDrop
        );
    }, [hw, metrics]);

    const chartData = useMemo(() => {
        if (!topology) return [];
        return topology.layers.map((layer) => ({
            name: layer.layer_id.replace("layer_", "L"),
            sparsity: layer.layer_sparsity,
        }));
    }, [topology]);

    return (
        <div className="w-[320px] shrink-0 bg-[var(--bg-surface)] flex flex-col overflow-y-auto">
            {/* Header */}
            <div className="px-5 pt-5 pb-4">
                <h2 className="text-[13px] font-bold text-[var(--text-primary)] tracking-tight">Analytics</h2>
            </div>

            {/* Tabs */}
            <div className="px-5 pb-0">
                <div className="flex gap-1 p-1 bg-[var(--bg-inset)] rounded-xl">
                    {TABS.map((t) => (
                        <button
                            key={t.id}
                            onClick={() => setTab(t.id)}
                            className={`flex-1 py-1.5 text-xs font-semibold rounded-lg transition-all cursor-pointer ${tab === t.id
                                ? "bg-[var(--bg-card)] text-[var(--accent)] shadow-[var(--shadow-card)]"
                                : "text-[var(--text-faint)] hover:text-[var(--text-secondary)]"
                                }`}
                        >
                            {t.label}
                        </button>
                    ))}
                </div>
            </div>

            {!topology || !metrics ? (
                <div className="flex-1 flex items-center justify-center py-12">
                    <p className="text-xs text-[var(--text-faint)]">Upload a model to view analytics</p>
                </div>
            ) : (
                <div className="px-5 py-5 flex flex-col gap-4">
                    {/* ═══ MODEL TAB ═══ */}
                    {tab === "model" && (
                        <>
                            {/* Key metrics */}
                            <div className="grid grid-cols-3 gap-2">
                                <StatCell label="Params" value={fmt(metrics.totalParams)} />
                                <StatCell label="FLOPs" value={fmt(metrics.totalFlops)} />
                                <StatCell label="Memory" value={fmtKb(metrics.memoryFootprintKb)} />
                            </div>

                            {/* Size comparison */}
                            <div className="grid grid-cols-2 gap-2">
                                <StatCell label="Original" value={fmtKb(topology.original_size_kb)} />
                                <StatCell label="Current" value={fmtKb(topology.current_size_kb)} accent />
                            </div>

                            {/* Compression / Sparsity */}
                            <div className="grid grid-cols-2 gap-2">
                                <StatCell label="Sparsity" value={`${topology.global_sparsity.toFixed(1)}%`} />
                                <StatCell
                                    label="Compression"
                                    value={`${(topology.original_size_kb > 0
                                        ? ((1 - topology.current_size_kb / topology.original_size_kb) * 100)
                                        : 0
                                    ).toFixed(1)}%`}
                                    accent
                                />
                            </div>

                            {/* Accuracy & Latency */}
                            <div className="grid grid-cols-2 gap-2">
                                <div className="bg-[var(--bg-card)] rounded-xl p-3 border border-[var(--border-default)]">
                                    <p className="text-[11px] text-[var(--text-faint)] mb-1">Est. Accuracy Drop</p>
                                    <p className={`text-base font-bold font-mono ${metrics.estimatedAccuracyDrop > 3 ? "text-[var(--danger)]" :
                                        metrics.estimatedAccuracyDrop > 1 ? "text-[var(--warning)]" :
                                            "text-[var(--accent)]"
                                        }`}>
                                        ~{metrics.estimatedAccuracyDrop.toFixed(1)}%
                                    </p>
                                </div>
                                <div className="bg-[var(--bg-card)] rounded-xl p-3 border border-[var(--border-default)]">
                                    <p className="text-[11px] text-[var(--text-faint)] mb-1">Latency Factor</p>
                                    <p className={`text-base font-bold font-mono ${metrics.latencyFactor < 0.5 ? "text-[var(--accent)]" :
                                        metrics.latencyFactor < 0.8 ? "text-[var(--purple)]" :
                                            "text-[var(--text-secondary)]"
                                        }`}>
                                        {metrics.latencyFactor.toFixed(2)}x
                                    </p>
                                </div>
                            </div>

                            {/* Layer Sparsity Chart */}
                            <div>
                                <h3 className="text-[11px] font-semibold text-[var(--text-faint)] mb-2 uppercase tracking-wider">Layer Sparsity</h3>
                                <div className="h-32 bg-[var(--bg-card)] rounded-xl p-3 border border-[var(--border-default)]">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={chartData}>
                                            <XAxis dataKey="name" tick={{ fill: "var(--text-faint)", fontSize: 9 }} axisLine={false} tickLine={false} />
                                            <YAxis domain={[0, 100]} tick={{ fill: "var(--text-faint)", fontSize: 9 }} axisLine={false} tickLine={false} width={24} />
                                            <Tooltip
                                                contentStyle={{
                                                    background: "var(--bg-elevated)",
                                                    border: "1px solid var(--border-default)",
                                                    borderRadius: "12px",
                                                    fontSize: "11px",
                                                    color: "var(--text-secondary)",
                                                    boxShadow: "var(--shadow-elevated)",
                                                }}
                                                labelStyle={{ color: "var(--text-primary)" }}
                                                cursor={{ fill: "var(--accent-subtle)" }}
                                            />
                                            <Bar dataKey="sparsity" radius={[4, 4, 0, 0]}>
                                                {chartData.map((entry, i) => (
                                                    <Cell key={i} fill={entry.sparsity > 50 ? "var(--warning)" : "var(--accent)"} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* Compression Timeline */}
                            {historyRef.current.length > 1 && (
                                <div>
                                    <h3 className="text-[11px] font-semibold text-[var(--text-faint)] mb-2 uppercase tracking-wider">Compression Timeline</h3>
                                    <div className="bg-[var(--bg-card)] rounded-xl p-3 border border-[var(--border-default)]">
                                        <CompressionSparkline
                                            data={historyRef.current}
                                            originalSizeKb={topology.original_size_kb}
                                        />
                                    </div>
                                </div>
                            )}
                        </>
                    )}

                    {/* ═══ HARDWARE TAB ═══ */}
                    {tab === "hardware" && hw && (
                        <>
                            <div>
                                <label className="text-xs font-semibold text-[var(--text-secondary)] mb-2 block">Target Device</label>
                                <select
                                    value={selectedDevice.name}
                                    onChange={(e) => {
                                        const d = HARDWARE_DEVICES.find((x) => x.name === e.target.value);
                                        if (d) onDeviceChange(d);
                                    }}
                                    className="w-full bg-[var(--bg-inset)] border border-[var(--border-default)] rounded-xl px-3 py-2.5 text-xs text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/30 focus:border-[var(--accent)]/50 transition-all"
                                >
                                    {HARDWARE_DEVICE_CATEGORIES.map((cat) => (
                                        <optgroup key={cat} label={cat}>
                                            {HARDWARE_DEVICES.filter((d) => d.category === cat).map((d) => (
                                                <option key={d.name} value={d.name}>{d.name}</option>
                                            ))}
                                        </optgroup>
                                    ))}
                                </select>
                            </div>

                            <div className="flex items-center gap-2 flex-wrap">
                                <span className="text-[11px] font-bold px-2.5 py-1 rounded-lg bg-[var(--accent-subtle)] text-[var(--accent)]">{selectedDevice.category}</span>
                                {selectedDevice.macs_per_cycle > 1 && (
                                    <span className="text-[11px] font-bold px-2.5 py-1 rounded-lg bg-[var(--purple-subtle)] text-[var(--purple)]">{selectedDevice.macs_per_cycle} MACs/clk</span>
                                )}
                                <span className="text-[11px] text-[var(--text-faint)] ml-auto font-mono">{selectedDevice.power_mw < 1000 ? `${selectedDevice.power_mw} mW` : `${(selectedDevice.power_mw / 1000).toFixed(1)} W`}</span>
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                                <StatCell label="SRAM" value={fmtKb(selectedDevice.sram_kb)} />
                                <StatCell label="Flash" value={fmtKb(selectedDevice.flash_kb)} />
                                <StatCell label="Clock" value={`${selectedDevice.clock_mhz} MHz`} />
                            </div>

                            <FitRow label="Flash" fits={hw.flashFit} utilization={hw.flashUtil}
                                detail={`${fmtKb(topology.current_size_kb)} / ${fmtKb(selectedDevice.flash_kb)}`} />
                            <FitRow label="SRAM" fits={hw.ramFit} utilization={hw.ramUtil}
                                detail={`${fmtKb(metrics.peakActivationKb)} / ${fmtKb(selectedDevice.sram_kb)}`} />

                            <div className="grid grid-cols-2 gap-2">
                                <StatCell label="Inference"
                                    value={hw.inferenceMs < 1 ? `${(hw.inferenceMs * 1000).toFixed(0)} µs` : `${hw.inferenceMs.toFixed(1)} ms`} />
                                <StatCell label="Throughput"
                                    value={hw.throughput > 1000 ? `${(hw.throughput / 1000).toFixed(1)}K/s` : `${hw.throughput.toFixed(0)}/s`} />
                                <StatCell label="Energy"
                                    value={hw.energyMj < 0.01 ? `${(hw.energyMj * 1000).toFixed(1)} µJ` : `${hw.energyMj.toFixed(3)} mJ`} />
                                <StatCell label="Battery (1Ah)"
                                    value={fmtBattery(hw.batteryHrs)} />
                            </div>
                        </>
                    )}

                    {/* ═══ READINESS TAB ═══ */}
                    {tab === "readiness" && readiness && hw && metrics && (
                        <>
                            {/* Gauge */}
                            <div className="flex justify-center">
                                <ReadinessGauge score={readiness.total} />
                            </div>

                            {/* Score breakdown */}
                            <div className="space-y-3">
                                <ScoreBar label="Memory" score={readiness.memory.score} max={readiness.memory.max} />
                                <ScoreBar label="Latency" score={readiness.latency.score} max={readiness.latency.max} />
                                <ScoreBar label="Energy" score={readiness.energy.score} max={readiness.energy.max} />
                                <ScoreBar label="Accuracy" score={readiness.accuracy.score} max={readiness.accuracy.max} />
                            </div>

                            {/* Summary */}
                            <div className="bg-[var(--bg-card)] rounded-xl p-4 border border-[var(--border-default)] space-y-2.5">
                                <h4 className="text-[11px] font-bold text-[var(--text-secondary)] mb-2 uppercase tracking-wider">Deploy Summary</h4>
                                <SummaryRow ok={hw.flashFit} text={`Flash: ${fmtKb(topology.current_size_kb)} / ${fmtKb(selectedDevice.flash_kb)}`} />
                                <SummaryRow ok={hw.ramFit} text={`SRAM: ${fmtKb(metrics.peakActivationKb)} / ${selectedDevice.sram_kb} KB`} />
                                <SummaryRow ok={hw.inferenceMs < 100} text={`Inference: ${hw.inferenceMs < 1 ? `${(hw.inferenceMs * 1000).toFixed(0)} µs` : `${hw.inferenceMs.toFixed(1)} ms`}`} />
                                <SummaryRow ok={true} text={`Battery (1Ah LiPo): ${fmtBattery(hw.batteryHrs)}`} />
                                <SummaryRow ok={metrics.estimatedAccuracyDrop < 3} text={`Est. accuracy: ~${(100 - metrics.estimatedAccuracyDrop).toFixed(1)}%`} />
                            </div>

                            {/* Generate Report CTA */}
                            {onShowReport && (
                                <button
                                    onClick={onShowReport}
                                    className="w-full py-2.5 rounded-xl text-xs font-bold transition-all cursor-pointer bg-[var(--purple-subtle)] text-[var(--purple)] border border-[var(--purple)]/15 hover:border-[var(--purple)]/30 hover:bg-[var(--purple-muted)] flex items-center justify-center gap-2"
                                >
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" /><polyline points="14 2 14 8 20 8" />
                                    </svg>
                                    Generate Report
                                </button>
                            )}
                        </>
                    )}
                </div>
            )}
        </div>
    );
}

/* ── Sub-components ────────────────────────────────────── */

function StatCell({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
    return (
        <div className="bg-[var(--bg-card)] rounded-xl p-3 border border-[var(--border-default)]">
            <p className="text-[11px] text-[var(--text-faint)] mb-1">{label}</p>
            <p className={`text-sm font-bold font-mono ${accent ? "text-[var(--accent)]" : "text-[var(--text-primary)]"}`}>{value}</p>
        </div>
    );
}

function FitRow({ label, fits, utilization, detail }: { label: string; fits: boolean; utilization: number; detail: string }) {
    return (
        <div className={`rounded-xl p-3 border ${fits
            ? "bg-[var(--accent-subtle)] border-[var(--accent)]/15"
            : "bg-[var(--danger)]/5 border-[var(--danger)]/15"
            }`}>
            <div className="flex items-center gap-2 mb-1.5">
                <span className={`w-2 h-2 rounded-full ${fits ? "bg-[var(--accent)]" : "bg-[var(--danger)]"}`} />
                <span className={`text-xs font-bold ${fits ? "text-[var(--accent)]" : "text-[var(--danger)]"}`}>
                    {label}: {fits ? "Fits" : "Exceeds"}
                </span>
            </div>
            <p className="text-[11px] text-[var(--text-faint)] mb-2">{detail}</p>
            <div className="h-1.5 bg-[var(--border-default)] rounded-full overflow-hidden">
                <div
                    className={`h-full rounded-full transition-all duration-300 ${fits ? "bg-[var(--accent)]" : "bg-[var(--danger)]"}`}
                    style={{ width: `${Math.min(utilization, 100)}%` }}
                />
            </div>
        </div>
    );
}

function ReadinessGauge({ score }: { score: number }) {
    const r = 46;
    const c = 2 * Math.PI * r;
    const progress = (score / 100) * c;
    const color = score >= 75 ? "var(--accent)" : score >= 50 ? "var(--warning)" : score >= 25 ? "#f97316" : "var(--danger)";
    const label = score >= 80 ? "Excellent" : score >= 60 ? "Good" : score >= 40 ? "Fair" : score >= 20 ? "Poor" : "Critical";

    return (
        <div className="relative w-[130px] h-[130px]">
            <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                <circle cx="50" cy="50" r={r} fill="none" stroke="var(--border-default)" strokeWidth="5" />
                <circle
                    cx="50" cy="50" r={r} fill="none" stroke={color} strokeWidth="5"
                    strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c - progress}
                    style={{ transition: "stroke-dashoffset 0.6s ease", filter: `drop-shadow(0 0 6px ${color})` }}
                />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-3xl font-bold font-mono" style={{ color }}>{Math.round(score)}</span>
                <span className="text-[11px] text-[var(--text-faint)] font-medium">{label}</span>
            </div>
        </div>
    );
}

function ScoreBar({ label, score, max }: { label: string; score: number; max: number }) {
    const pct = max > 0 ? (score / max) * 100 : 0;
    return (
        <div>
            <div className="flex justify-between mb-1.5">
                <span className="text-xs text-[var(--text-secondary)] font-medium">{label}</span>
                <span className="text-xs font-mono font-bold text-[var(--text-primary)]">{score}/{max}</span>
            </div>
            <div className="h-2 bg-[var(--border-default)] rounded-full overflow-hidden">
                <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{
                        width: `${Math.min(pct, 100)}%`,
                        background: "linear-gradient(90deg, var(--accent) 0%, #00cc7a 100%)",
                    }}
                />
            </div>
        </div>
    );
}

function SummaryRow({ ok, text }: { ok: boolean; text: string }) {
    return (
        <div className="flex items-center gap-2.5">
            <span className={`w-2 h-2 rounded-full shrink-0 ${ok ? "bg-[var(--accent)]" : "bg-[var(--danger)]"}`} />
            <span className="text-xs text-[var(--text-secondary)]">{text}</span>
        </div>
    );
}

function CompressionSparkline({ data, originalSizeKb }: { data: { sparsity: number; sizeKb: number }[]; originalSizeKb: number }) {
    if (data.length < 2) return null;

    const w = 240;
    const h = 48;
    const pad = 4;

    const maxSize = originalSizeKb;
    const points = data.map((d, i) => ({ 
        x: pad + (i / (data.length - 1)) * (w - pad * 2),
        y: pad + (1 - d.sizeKb / maxSize) * (h - pad * 2),
    }));

    const line = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
    const area = `${line} L ${points[points.length - 1].x.toFixed(1)} ${h} L ${points[0].x.toFixed(1)} ${h} Z`;

    return (
        <div>
            <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ height: 48 }}>
                <defs>
                    <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.2" />
                        <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
                    </linearGradient>
                </defs>
                <path d={area} fill="url(#sparkGrad)" />
                <path d={line} fill="none" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                {/* Current point */}
                <circle cx={points[points.length - 1].x} cy={points[points.length - 1].y} r="3" fill="var(--accent)" />
            </svg>
            <div className="flex justify-between mt-1.5">
                <span className="text-[10px] text-[var(--text-faint)]">{data.length} steps</span>
                <span className="text-[10px] font-mono font-bold text-[var(--accent)]">
                    {((1 - data[data.length - 1].sizeKb / originalSizeKb) * 100).toFixed(1)}% reduced
                </span>
            </div>
        </div>
    );
}
