"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
import CanvasArea from "@/components/CanvasArea";
import ControlPanel from "@/components/ControlPanel";
import MetricsDashboard from "@/components/MetricsDashboard";
import LearningDashboard from "@/components/LearningDashboard";
import DeploymentReport from "@/components/DeploymentReport";
import { ModelTopology, HardwareDevice, HARDWARE_DEVICES } from "@/lib/types";
import { uploadModel, pruneModel, quantizeModel, downloadModel } from "@/lib/api";

export default function Home() {
  const [topology, setTopology] = useState<ModelTopology | null>(null);
  const [threshold, setThreshold] = useState(0.0);
  const [bitwidth, setBitwidth] = useState(32);
  const [selectedDevice, setSelectedDevice] = useState<HardwareDevice>(
    HARDWARE_DEVICES[0]
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showLearn, setShowLearn] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [optimizeFlash, setOptimizeFlash] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastModelIdRef = useRef<string | null>(null);

  const handleUpload = useCallback(async (file: File) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await uploadModel(file);
      setTopology(result);
      lastModelIdRef.current = result.model_id;
      setThreshold(0.0);
      setBitwidth(32);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleThresholdChange = useCallback(
    (value: number) => {
      setThreshold(value);

      if (!lastModelIdRef.current) return;

      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(async () => {
        setIsLoading(true);
        setError(null);
        try {
          const result = await pruneModel(lastModelIdRef.current!, value);
          setTopology(result);
        } catch (err) {
          setError(err instanceof Error ? err.message : "Prune failed");
        } finally {
          setIsLoading(false);
        }
      }, 200);
    },
    []
  );

  const handleBitwidthChange = useCallback(
    async (value: number) => {
      setBitwidth(value);
      if (!lastModelIdRef.current) return;

      setIsLoading(true);
      setError(null);
      try {
        const result = await quantizeModel(lastModelIdRef.current!, value);
        setTopology(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Quantize failed");
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const handleOptimize = useCallback(() => {
    setOptimizeFlash(true);
    setTimeout(() => setOptimizeFlash(false), 1200);
  }, []);

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  return (
    <div className="h-screen flex flex-col bg-[var(--bg-root)] text-[var(--text-primary)] overflow-hidden">
      {/* ── Top Bar ── */}
      <header className="shrink-0 h-14 px-5 flex items-center justify-between border-b border-[var(--border-subtle)] bg-[var(--bg-surface)]">
        <div className="flex items-center gap-3.5">
          {/* Logo mark */}
          <div className="w-8 h-8 rounded-[10px] bg-gradient-to-br from-[var(--accent)] to-[#00cc7a] flex items-center justify-center shadow-[0_0_14px_var(--accent-glow)]">
            <svg className="w-4.5 h-4.5 text-[#080a12]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <div className="flex items-baseline gap-2">
            <h1 className="text-[15px] font-bold tracking-tight text-gradient-green">
              NanoPrune
            </h1>
            <span className="text-xs text-[var(--text-faint)] font-medium hidden sm:block">
              Compression Toolkit
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          {/* Model pill stats */}
          {topology && (
            <div className="flex items-center gap-4 px-4 py-1.5 rounded-full bg-[var(--bg-elevated)] border border-[var(--border-default)] text-xs animate-fade-in">
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent)]" />
                <span className="text-[var(--text-faint)]">Layers</span>
                <span className="font-mono font-semibold text-[var(--text-primary)]">{topology.layers.length}</span>
              </span>
              <span className="w-px h-3 bg-[var(--border-default)]" />
              <span className="flex items-center gap-1.5">
                <span className="text-[var(--text-faint)]">Size</span>
                <span className="font-mono font-semibold text-[var(--text-primary)]">{topology.current_size_kb.toFixed(0)} KB</span>
              </span>
              {topology.global_sparsity > 0 && (
                <>
                  <span className="w-px h-3 bg-[var(--border-default)]" />
                  <span className="flex items-center gap-1.5">
                    <span className="text-[var(--text-faint)]">Sparse</span>
                    <span className="font-mono font-semibold text-[var(--warning)]">{topology.global_sparsity.toFixed(1)}%</span>
                  </span>
                </>
              )}
            </div>
          )}

          <button
            onClick={() => setShowLearn(true)}
            className="btn-ghost h-8 px-3.5 text-xs rounded-[10px] flex items-center gap-1.5 cursor-pointer"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z" /><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z" />
            </svg>
            Learn
          </button>

          <button
            onClick={() => setShowReport(true)}
            disabled={!topology}
            className="h-8 px-3.5 text-xs font-semibold rounded-[10px] transition-all flex items-center gap-1.5 cursor-pointer disabled:opacity-25 disabled:cursor-not-allowed bg-[var(--accent-subtle)] text-[var(--accent)] border border-[var(--accent)]/20 hover:border-[var(--accent)]/40 hover:bg-[var(--accent-glow)] relative"
            onBlur={() => setTimeout(() => setShowExportMenu(false), 150)}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /><polyline points="10 9 9 9 8 9" />
            </svg>
            Export
          </button>

          {isLoading && (
            <div className="flex items-center gap-2 text-xs text-[var(--accent)] animate-fade-in ml-1">
              <div className="w-3.5 h-3.5 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin status-dot-pulse" />
              <span className="font-medium tracking-wide">Processing</span>
            </div>
          )}
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="shrink-0 px-5 py-2.5 bg-[var(--danger)]/5 border-b border-[var(--danger)]/15 animate-fade-in">
          <p className="text-xs text-[var(--danger)] flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
              <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            {error}
          </p>
        </div>
      )}

      {/* ── Main Content ── */}
      <main className="flex-1 flex min-h-0 bg-[var(--bg-root)]">
        <ControlPanel
          onUpload={handleUpload}
          threshold={threshold}
          onThresholdChange={handleThresholdChange}
          bitwidth={bitwidth}
          onBitwidthChange={handleBitwidthChange}
          isLoading={isLoading}
          hasModel={topology !== null}
          maxWeight={topology?.max_weight ?? 1}
          sparsity={topology?.global_sparsity ?? 0}
          onOptimize={handleOptimize}
        />

        <div className="w-px bg-[var(--border-subtle)]" />

        <CanvasArea topology={topology} bitwidth={bitwidth} optimizeFlash={optimizeFlash} />

        <div className="w-px bg-[var(--border-subtle)]" />

        <MetricsDashboard
          topology={topology}
          selectedDevice={selectedDevice}
          onDeviceChange={setSelectedDevice}
          onShowReport={topology ? () => setShowReport(true) : undefined}
        />
      </main>

      {/* Overlays */}
      {showLearn && <LearningDashboard onClose={() => setShowLearn(false)} />}
      {showReport && topology && (
        <DeploymentReport
          topology={topology}
          selectedDevice={selectedDevice}
          bitwidth={bitwidth}
          modelId={lastModelIdRef.current}
          onClose={() => setShowReport(false)}
        />
      )}
    </div>
  );
}
