"use client";

import React, { useMemo, useState, useCallback } from "react";
import {
    ReactFlow,
    Node,
    Edge,
    Background,
    Controls,
    BackgroundVariant,
    Handle,
    Position,
} from "reactflow";
import "reactflow/dist/style.css";
import { ModelTopology } from "@/lib/types";

const NUM_HANDLES = 8;
const LAYER_Y_GAP = 240;

/* ─── Helpers ──────────────────────────────────────────────── */

function sparsityColor(s: number): string {
    if (s < 20) return "var(--accent)";
    if (s < 50) return "var(--warning)";
    if (s < 80) return "#f97316";
    return "var(--danger)";
}

function fmtFlops(f: number): string {
    if (f >= 1e6) return `${(f / 1e6).toFixed(1)}M`;
    if (f >= 1e3) return `${(f / 1e3).toFixed(1)}K`;
    return `${f}`;
}

function handleOffset(i: number): string {
    return `${((i + 1) / (NUM_HANDLES + 1)) * 100}%`;
}

/* ─── Input Node ───────────────────────────────────────────── */

function InputNode({ data }: { data: { features: number } }) {
    return (
        <div className="bg-[var(--bg-card)] border border-[var(--accent)]/20 rounded-2xl px-10 py-5 min-w-[280px] text-center shadow-[0_0_20px_var(--accent-glow)]">
            {Array.from({ length: NUM_HANDLES }).map((_, i) => (
                <Handle
                    key={`s-${i}`}
                    type="source"
                    position={Position.Bottom}
                    id={`s-${i}`}
                    style={{ left: handleOffset(i), background: "var(--accent)", width: 6, height: 6, border: "none" }}
                />
            ))}
            <div className="text-[11px] font-bold text-[var(--accent)] mb-1 uppercase tracking-wider">Input</div>
            <div className="text-2xl font-bold text-[var(--text-primary)] font-mono">{data.features}</div>
            <div className="text-[11px] text-[var(--text-faint)]">features</div>
        </div>
    );
}

/* ─── Layer Node ───────────────────────────────────────────── */

interface LayerNodeData {
    type: string;
    in_features: number;
    out_features: number;
    param_count: number;
    layer_sparsity: number;
    flops: number;
    bitwidth: number;
    kernel_size?: number[];
}

function LayerNode({ data }: { data: LayerNodeData }) {
    const color = sparsityColor(data.layer_sparsity);
    const isQuantized = data.bitwidth < 32;

    return (
        <div
            className="bg-[var(--bg-card)] rounded-2xl px-6 py-5 min-w-[340px] relative shadow-[var(--shadow-card)]"
            style={{
                border: isQuantized ? `1.5px dashed color-mix(in srgb, ${color} 40%, transparent)` : `1px solid var(--border-default)`,
            }}
        >
            {Array.from({ length: NUM_HANDLES }).map((_, i) => (
                <Handle key={`t-${i}`} type="target" position={Position.Top} id={`t-${i}`}
                    style={{ left: handleOffset(i), background: color, width: 6, height: 6, border: "none" }} />
            ))}
            {Array.from({ length: NUM_HANDLES }).map((_, i) => (
                <Handle key={`s-${i}`} type="source" position={Position.Bottom} id={`s-${i}`}
                    style={{ left: handleOffset(i), background: color, width: 6, height: 6, border: "none" }} />
            ))}

            {/* Top row: badges */}
            <div className="flex items-center gap-2 mb-3 flex-wrap">
                <span
                    className="text-[11px] font-bold px-2.5 py-1 rounded-lg"
                    style={{ background: `color-mix(in srgb, ${color} 12%, transparent)`, color }}
                >
                    {data.type}
                </span>
                {isQuantized && (
                    <span className="text-[11px] font-bold px-2.5 py-1 rounded-lg bg-[var(--purple-subtle)] text-[var(--purple)]">
                        Q{data.bitwidth}
                    </span>
                )}
                {data.kernel_size && (
                    <span className="text-[11px] font-bold px-2.5 py-1 rounded-lg bg-[var(--accent-subtle)] text-[var(--accent)]">
                        {data.kernel_size.join("\u00d7")}
                    </span>
                )}
                <span className="ml-auto text-[11px] font-mono text-[var(--text-faint)] font-medium">
                    {fmtFlops(data.flops)} FLOPs
                </span>
            </div>

            {/* Dimensions */}
            <div className="flex items-baseline gap-3 mb-3">
                <span className="text-lg font-bold text-[var(--text-primary)] font-mono">
                    {data.in_features} <span className="text-[var(--text-faint)] mx-0.5">→</span> {data.out_features}
                </span>
                <span className="text-[11px] text-[var(--text-faint)]">
                    {data.param_count.toLocaleString()} params
                </span>
            </div>

            {/* Sparsity bar */}
            <div className="h-1.5 bg-[var(--border-default)] rounded-full overflow-hidden mb-1.5">
                <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{
                        width: `${Math.max(data.layer_sparsity, 0.5)}%`,
                        background: `linear-gradient(90deg, ${color}, color-mix(in srgb, ${color} 70%, transparent))`,
                    }}
                />
            </div>
            <div className="text-[11px] font-mono font-semibold" style={{ color }}>
                {data.layer_sparsity.toFixed(1)}% sparse
            </div>
        </div>
    );
}

/* ─── Output Node ──────────────────────────────────────────── */

function OutputNode({ data }: { data: { features: number } }) {
    return (
        <div className="bg-[var(--bg-card)] border border-[var(--purple)]/20 rounded-2xl px-10 py-5 min-w-[280px] text-center shadow-[0_0_20px_rgba(123,97,255,0.1)]">
            {Array.from({ length: NUM_HANDLES }).map((_, i) => (
                <Handle key={`t-${i}`} type="target" position={Position.Top} id={`t-${i}`}
                    style={{ left: handleOffset(i), background: "var(--purple)", width: 6, height: 6, border: "none" }} />
            ))}
            <div className="text-[11px] font-bold text-[var(--purple)] mb-1 uppercase tracking-wider">Output</div>
            <div className="text-2xl font-bold text-[var(--text-primary)] font-mono">{data.features}</div>
            <div className="text-[11px] text-[var(--text-faint)]">classes</div>
        </div>
    );
}

const nodeTypes = { inputNode: InputNode, layerNode: LayerNode, outputNode: OutputNode };

/* ─── Node Inspector Panel ─────────────────────────────────── */

interface InspectedNode {
    id: string;
    type: string;
    data: LayerNodeData | { features: number };
}

function NodeInspector({ node, onClose }: { node: InspectedNode; onClose: () => void }) {
    const isLayer = "in_features" in node.data && "out_features" in node.data && "layer_sparsity" in node.data;
    const data = node.data as LayerNodeData;

    return (
        <div className="absolute top-4 right-4 z-20 w-[280px] bg-[var(--bg-surface)] border border-[var(--border-default)] rounded-2xl shadow-[var(--shadow-elevated)] animate-fade-in-scale overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border-subtle)]">
                <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-[var(--accent)] shadow-[0_0_6px_var(--accent-glow)]" />
                    <h3 className="text-xs font-bold text-[var(--text-primary)]">Node Inspector</h3>
                </div>
                <button
                    onClick={onClose}
                    className="w-6 h-6 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border-default)] flex items-center justify-center text-[var(--text-faint)] hover:text-[var(--text-primary)] transition-colors cursor-pointer"
                >
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                </button>
            </div>

            {isLayer ? (
                <div className="p-4 space-y-3">
                    {/* Type badge */}
                    <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-[11px] font-bold px-2.5 py-1 rounded-lg bg-[var(--accent-subtle)] text-[var(--accent)]">
                            {data.type}
                        </span>
                        {data.bitwidth < 32 && (
                            <span className="text-[11px] font-bold px-2.5 py-1 rounded-lg bg-[var(--purple-subtle)] text-[var(--purple)]">
                                Q{data.bitwidth}
                            </span>
                        )}
                        {data.kernel_size && (
                            <span className="text-[11px] font-bold px-2.5 py-1 rounded-lg bg-[var(--bg-elevated)] text-[var(--text-secondary)]">
                                {data.kernel_size.join("×")}
                            </span>
                        )}
                    </div>

                    {/* Dimensions */}
                    <div className="bg-[var(--bg-card)] rounded-xl p-3 border border-[var(--border-default)]">
                        <div className="flex items-center justify-center gap-4 text-center">
                            <div>
                                <p className="text-lg font-bold font-mono text-[var(--text-primary)]">{data.in_features}</p>
                                <p className="text-[10px] text-[var(--text-faint)]">input</p>
                            </div>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--text-faint)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" />
                            </svg>
                            <div>
                                <p className="text-lg font-bold font-mono text-[var(--accent)]">{data.out_features}</p>
                                <p className="text-[10px] text-[var(--text-faint)]">output</p>
                            </div>
                        </div>
                    </div>

                    {/* Stats grid */}
                    <div className="grid grid-cols-2 gap-2">
                        <div className="bg-[var(--bg-card)] rounded-xl p-2.5 border border-[var(--border-default)]">
                            <p className="text-[10px] text-[var(--text-faint)]">Parameters</p>
                            <p className="text-xs font-bold font-mono text-[var(--text-primary)]">{data.param_count.toLocaleString()}</p>
                        </div>
                        <div className="bg-[var(--bg-card)] rounded-xl p-2.5 border border-[var(--border-default)]">
                            <p className="text-[10px] text-[var(--text-faint)]">FLOPs</p>
                            <p className="text-xs font-bold font-mono text-[var(--text-primary)]">{fmtFlops(data.flops)}</p>
                        </div>
                    </div>

                    {/* Sparsity */}
                    <div>
                        <div className="flex justify-between mb-1.5">
                            <span className="text-[11px] text-[var(--text-faint)]">Layer Sparsity</span>
                            <span className="text-[11px] font-bold font-mono" style={{ color: sparsityColor(data.layer_sparsity) }}>
                                {data.layer_sparsity.toFixed(1)}%
                            </span>
                        </div>
                        <div className="h-2 bg-[var(--border-default)] rounded-full overflow-hidden">
                            <div
                                className="h-full rounded-full transition-all duration-300"
                                style={{
                                    width: `${Math.max(data.layer_sparsity, 0.5)}%`,
                                    background: `linear-gradient(90deg, ${sparsityColor(data.layer_sparsity)}, color-mix(in srgb, ${sparsityColor(data.layer_sparsity)} 70%, transparent))`,
                                }}
                            />
                        </div>
                    </div>

                    {/* Weight distribution preview */}
                    <div className="bg-[var(--bg-card)] rounded-xl p-3 border border-[var(--border-default)]">
                        <p className="text-[10px] font-bold text-[var(--text-faint)] uppercase tracking-wider mb-2">Precision</p>
                        <div className="flex items-center gap-2">
                            <div className="flex-1 h-1.5 bg-[var(--border-default)] rounded-full overflow-hidden">
                                <div
                                    className="h-full rounded-full bg-[var(--purple)]"
                                    style={{ width: `${(data.bitwidth / 32) * 100}%` }}
                                />
                            </div>
                            <span className="text-[11px] font-mono font-bold text-[var(--purple)]">
                                {data.bitwidth < 32 ? `INT${data.bitwidth}` : "FP32"}
                            </span>
                        </div>
                    </div>
                </div>
            ) : (
                <div className="p-4">
                    <div className="bg-[var(--bg-card)] rounded-xl p-4 border border-[var(--border-default)] text-center">
                        <p className="text-2xl font-bold font-mono text-[var(--text-primary)]">
                            {(node.data as { features: number }).features}
                        </p>
                        <p className="text-[11px] text-[var(--text-faint)] mt-1">
                            {node.id === "input" ? "Input features" : "Output classes"}
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
}

/* ─── Main Component ───────────────────────────────────────── */

interface CanvasAreaProps {
    topology: ModelTopology | null;
    bitwidth: number;
    optimizeFlash?: boolean;
}

export default function CanvasArea({ topology, bitwidth, optimizeFlash }: CanvasAreaProps) {
    const [inspectedNode, setInspectedNode] = useState<InspectedNode | null>(null);

    const handleNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
        setInspectedNode({ id: node.id, type: node.type || "", data: node.data });
    }, []);

    const handlePaneClick = useCallback(() => {
        setInspectedNode(null);
    }, []);

    const { nodes, edges } = useMemo(() => {
        if (!topology || topology.layers.length === 0) {
            return { nodes: [], edges: [] };
        }

        const rfNodes: Node[] = [];
        const rfEdges: Edge[] = [];

        const firstLayer = topology.layers[0];
        rfNodes.push({
            id: "input",
            type: "inputNode",
            data: { features: firstLayer.in_features },
            position: { x: 0, y: 0 },
        });

        topology.layers.forEach((layer, idx) => {
            let flops = 2 * layer.in_features * layer.out_features;
            if (layer.kernel_size) {
                flops *= layer.kernel_size.reduce((a, b) => a * b, 1);
            }
            rfNodes.push({
                id: layer.layer_id,
                type: "layerNode",
                data: {
                    type: layer.type,
                    in_features: layer.in_features,
                    out_features: layer.out_features,
                    param_count: layer.param_count,
                    layer_sparsity: layer.layer_sparsity,
                    flops,
                    bitwidth,
                    kernel_size: layer.kernel_size,
                },
                position: { x: 0, y: (idx + 1) * LAYER_Y_GAP },
            });
        });

        const lastLayer = topology.layers[topology.layers.length - 1];
        rfNodes.push({
            id: "output",
            type: "outputNode",
            data: { features: lastLayer.out_features },
            position: { x: 0, y: (topology.layers.length + 1) * LAYER_Y_GAP },
        });

        const allNodeIds = ["input", ...topology.layers.map((l) => l.layer_id), "output"];

        topology.layers.forEach((layer, i) => {
            const sourceId = allNodeIds[i];
            const targetId = allNodeIds[i + 1];
            const samples = layer.sampled_edges;

            if (samples.length === 0) {
                rfEdges.push({
                    id: `e-${i}-empty`,
                    source: sourceId,
                    target: targetId,
                    sourceHandle: "s-4",
                    targetHandle: "t-4",
                    style: { stroke: "var(--border-default)", strokeWidth: 1, opacity: 0.3 },
                    animated: false,
                });
                return;
            }

            const maxW = Math.max(...samples.map((w) => Math.abs(w)), 0.001);
            const count = Math.min(NUM_HANDLES, samples.length);

            for (let j = 0; j < count; j++) {
                const w = samples[j];
                const absW = Math.abs(w);
                const opacity = Math.max(0.15, absW / maxW);

                rfEdges.push({
                    id: `e-${i}-${j}`,
                    source: sourceId,
                    target: targetId,
                    sourceHandle: `s-${j}`,
                    targetHandle: `t-${j}`,
                    style: {
                        stroke: w >= 0 ? "var(--accent)" : "var(--danger)",
                        strokeWidth: 1.5,
                        opacity,
                    },
                    animated: absW > maxW * 0.8,
                });
            }
        });

        const lastSamples = topology.layers[topology.layers.length - 1].sampled_edges;
        if (lastSamples.length > 0) {
            const lastIdx = topology.layers.length - 1;
            const maxW = Math.max(...lastSamples.map((w) => Math.abs(w)), 0.001);
            const count = Math.min(NUM_HANDLES, lastSamples.length);

            for (let j = 0; j < count; j++) {
                const w = lastSamples[j];
                const absW = Math.abs(w);
                const opacity = Math.max(0.15, absW / maxW);

                rfEdges.push({
                    id: `e-out-${j}`,
                    source: topology.layers[lastIdx].layer_id,
                    target: "output",
                    sourceHandle: `s-${j}`,
                    targetHandle: `t-${j}`,
                    style: {
                        stroke: "var(--purple)",
                        strokeWidth: 1.5,
                        opacity,
                    },
                    animated: false,
                });
            }
        }

        return { nodes: rfNodes, edges: rfEdges };
    }, [topology, bitwidth]);

    if (!topology) {
        return (
            <div className="flex-1 flex items-center justify-center bg-[var(--bg-root)]">
                <div className="text-center">
                    <div className="w-14 h-14 mx-auto mb-4 rounded-2xl bg-[var(--bg-card)] border border-[var(--border-default)] flex items-center justify-center shadow-[var(--shadow-card)]">
                        <svg className="w-6 h-6 text-[var(--text-faint)]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                        </svg>
                    </div>
                    <p className="text-sm text-[var(--text-secondary)] font-medium">
                        Upload a <code className="text-[var(--accent)] font-bold">.pt</code>,{" "}
                        <code className="text-[var(--accent)] font-bold">.pth</code>, or{" "}
                        <code className="text-[var(--accent)] font-bold">.onnx</code> model
                    </p>
                    <p className="text-xs text-[var(--text-faint)] mt-1.5">
                        PyTorch &amp; ONNX models supported
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex-1 bg-[var(--bg-root)] overflow-hidden relative">
            <ReactFlow
                nodes={nodes}
                edges={edges}
                nodeTypes={nodeTypes}
                onNodeClick={handleNodeClick}
                onPaneClick={handlePaneClick}
                fitView
                fitViewOptions={{ padding: 0.4, maxZoom: 1 }}
                minZoom={0.15}
                maxZoom={2.5}
                proOptions={{ hideAttribution: true }}
            >
                <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="var(--border-muted)" />
                <Controls position="bottom-right" />
            </ReactFlow>
            {inspectedNode && (
                <NodeInspector node={inspectedNode} onClose={() => setInspectedNode(null)} />
            )}
            {optimizeFlash && (
                <div className="absolute inset-0 pointer-events-none z-30 bg-[var(--accent)]/5 animate-fade-in" />
            )}
        </div>
    );
}
