"use client";

import React from "react";

interface IconProps {
    className?: string;
    size?: number;
}

function icon(className?: string, size = 14) {
    return { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, className };
}

/* ── Tab / Section Icons ─────────────────────────────── */

export function ChartBarIcon({ className, size }: IconProps) {
    return (
        <svg {...icon(className, size)}>
            <path d="M3 3v18h18" />
            <rect x="7" y="10" width="3" height="8" rx="0.5" fill="currentColor" opacity="0.3" />
            <rect x="12" y="6" width="3" height="12" rx="0.5" fill="currentColor" opacity="0.3" />
            <rect x="17" y="3" width="3" height="15" rx="0.5" fill="currentColor" opacity="0.3" />
        </svg>
    );
}

export function WrenchIcon({ className, size }: IconProps) {
    return (
        <svg {...icon(className, size)}>
            <path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z" />
        </svg>
    );
}

export function TargetIcon({ className, size }: IconProps) {
    return (
        <svg {...icon(className, size)}>
            <circle cx="12" cy="12" r="10" />
            <circle cx="12" cy="12" r="6" />
            <circle cx="12" cy="12" r="2" />
        </svg>
    );
}

export function ScissorsIcon({ className, size }: IconProps) {
    return (
        <svg {...icon(className, size)}>
            <circle cx="6" cy="6" r="3" />
            <circle cx="6" cy="18" r="3" />
            <line x1="20" y1="4" x2="8.12" y2="15.88" />
            <line x1="14.47" y1="14.48" x2="20" y2="20" />
            <line x1="8.12" y1="8.12" x2="12" y2="12" />
        </svg>
    );
}

export function GridIcon({ className, size }: IconProps) {
    return (
        <svg {...icon(className, size)}>
            <rect x="3" y="3" width="7" height="7" rx="1" />
            <rect x="14" y="3" width="7" height="7" rx="1" />
            <rect x="3" y="14" width="7" height="7" rx="1" />
            <rect x="14" y="14" width="7" height="7" rx="1" />
        </svg>
    );
}

export function TrendingUpIcon({ className, size }: IconProps) {
    return (
        <svg {...icon(className, size)}>
            <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
            <polyline points="17 6 23 6 23 12" />
        </svg>
    );
}

export function SlidersIcon({ className, size }: IconProps) {
    return (
        <svg {...icon(className, size)}>
            <line x1="4" y1="21" x2="4" y2="14" />
            <line x1="4" y1="10" x2="4" y2="3" />
            <line x1="12" y1="21" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12" y2="3" />
            <line x1="20" y1="21" x2="20" y2="16" />
            <line x1="20" y1="12" x2="20" y2="3" />
            <line x1="1" y1="14" x2="7" y2="14" />
            <line x1="9" y1="8" x2="15" y2="8" />
            <line x1="17" y1="16" x2="23" y2="16" />
        </svg>
    );
}

/* ── Status Icons (for summary lines) ─────────────────── */

export function CheckCircleIcon({ className, size }: IconProps) {
    return (
        <svg {...icon(className, size)}>
            <path d="M22 11.08V12a10 10 0 11-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
        </svg>
    );
}

export function XCircleIcon({ className, size }: IconProps) {
    return (
        <svg {...icon(className, size)}>
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
        </svg>
    );
}

export function ZapIcon({ className, size }: IconProps) {
    return (
        <svg {...icon(className, size)}>
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
        </svg>
    );
}

export function ClockIcon({ className, size }: IconProps) {
    return (
        <svg {...icon(className, size)}>
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
        </svg>
    );
}

export function BatteryIcon({ className, size }: IconProps) {
    return (
        <svg {...icon(className, size)}>
            <rect x="1" y="6" width="18" height="12" rx="2" ry="2" />
            <line x1="23" y1="13" x2="23" y2="11" />
            <rect x="3" y="8" width="10" height="8" rx="1" fill="currentColor" opacity="0.3" />
        </svg>
    );
}

export function CrosshairIcon({ className, size }: IconProps) {
    return (
        <svg {...icon(className, size)}>
            <circle cx="12" cy="12" r="10" />
            <line x1="22" y1="12" x2="18" y2="12" />
            <line x1="6" y1="12" x2="2" y2="12" />
            <line x1="12" y1="6" x2="12" y2="2" />
            <line x1="12" y1="22" x2="12" y2="18" />
        </svg>
    );
}

export function CpuIcon({ className, size }: IconProps) {
    return (
        <svg {...icon(className, size)}>
            <rect x="4" y="4" width="16" height="16" rx="2" ry="2" />
            <rect x="9" y="9" width="6" height="6" />
            <line x1="9" y1="1" x2="9" y2="4" />
            <line x1="15" y1="1" x2="15" y2="4" />
            <line x1="9" y1="20" x2="9" y2="23" />
            <line x1="15" y1="20" x2="15" y2="23" />
            <line x1="20" y1="9" x2="23" y2="9" />
            <line x1="20" y1="14" x2="23" y2="14" />
            <line x1="1" y1="9" x2="4" y2="9" />
            <line x1="1" y1="14" x2="4" y2="14" />
        </svg>
    );
}
