export interface LayerInfo {
  layer_id: string;
  type: string;
  in_features: number;
  out_features: number;
  weights_shape: number[];
  layer_sparsity: number;
  param_count: number;
  sampled_edges: number[];
  // Conv2d-specific (optional)
  kernel_size?: number[];
  stride?: number[];
  padding?: number[];
}

export interface ModelTopology {
  model_id: string;
  original_size_kb: number;
  current_size_kb: number;
  global_sparsity: number;
  max_weight: number;
  layers: LayerInfo[];
  model_format?: string;
}

export interface HardwareDevice {
  name: string;
  category: "MCU" | "SoC" | "Accelerator" | "SBC";
  sram_kb: number;
  flash_kb: number;
  clock_mhz: number;
  macs_per_cycle: number;
  power_mw: number;
}

export const HARDWARE_DEVICE_CATEGORIES = ["MCU", "SoC", "Accelerator", "SBC"] as const;

export const HARDWARE_DEVICES: HardwareDevice[] = [
  // ── Microcontrollers (MCU) ─────────────────────────────────────
  { name: "Arduino Nano 33 BLE",    category: "MCU", sram_kb: 256,    flash_kb: 1024,    clock_mhz: 64,   macs_per_cycle: 1,    power_mw: 15 },
  { name: "Nordic nRF5340",          category: "MCU", sram_kb: 512,    flash_kb: 1024,    clock_mhz: 128,  macs_per_cycle: 1,    power_mw: 5 },
  { name: "STM32F411 (Black Pill)",  category: "MCU", sram_kb: 128,    flash_kb: 512,     clock_mhz: 100,  macs_per_cycle: 1,    power_mw: 100 },
  { name: "STM32F746",               category: "MCU", sram_kb: 320,    flash_kb: 1024,    clock_mhz: 216,  macs_per_cycle: 1,    power_mw: 300 },
  { name: "STM32H743",               category: "MCU", sram_kb: 1024,   flash_kb: 2048,    clock_mhz: 480,  macs_per_cycle: 1,    power_mw: 500 },
  { name: "STM32L476",               category: "MCU", sram_kb: 128,    flash_kb: 1024,    clock_mhz: 80,   macs_per_cycle: 1,    power_mw: 36 },
  { name: "Raspberry Pi Pico (RP2040)", category: "MCU", sram_kb: 264, flash_kb: 2048,    clock_mhz: 133,  macs_per_cycle: 1,    power_mw: 25 },
  { name: "Raspberry Pi Pico 2 (RP2350)", category: "MCU", sram_kb: 520, flash_kb: 4096, clock_mhz: 150,  macs_per_cycle: 1,    power_mw: 30 },
  { name: "Teensy 4.1",              category: "MCU", sram_kb: 1024,   flash_kb: 8192,    clock_mhz: 600,  macs_per_cycle: 1,    power_mw: 500 },
  { name: "Ambiq Apollo4 Blue Plus", category: "MCU", sram_kb: 2048,   flash_kb: 2048,    clock_mhz: 192,  macs_per_cycle: 1,    power_mw: 3 },

  // ── System-on-Chip (SoC) ──────────────────────────────────────
  { name: "ESP32-S3",                category: "SoC", sram_kb: 512,    flash_kb: 8192,    clock_mhz: 240,  macs_per_cycle: 1,    power_mw: 240 },
  { name: "ESP32-C3",                category: "SoC", sram_kb: 400,    flash_kb: 4096,    clock_mhz: 160,  macs_per_cycle: 1,    power_mw: 130 },
  { name: "ESP32-C6",                category: "SoC", sram_kb: 512,    flash_kb: 4096,    clock_mhz: 160,  macs_per_cycle: 1,    power_mw: 140 },
  { name: "Kendryte K210",           category: "SoC", sram_kb: 8192,   flash_kb: 16384,   clock_mhz: 400,  macs_per_cycle: 64,   power_mw: 300 },
  { name: "Realtek AMB82-Mini",      category: "SoC", sram_kb: 4096,   flash_kb: 16384,   clock_mhz: 500,  macs_per_cycle: 1,    power_mw: 400 },
  { name: "NXP i.MX RT1060",         category: "SoC", sram_kb: 1024,   flash_kb: 8192,    clock_mhz: 600,  macs_per_cycle: 1,    power_mw: 450 },
  { name: "NXP i.MX RT1170",         category: "SoC", sram_kb: 2048,   flash_kb: 16384,   clock_mhz: 1000, macs_per_cycle: 1,    power_mw: 700 },

  // ── AI Accelerators ───────────────────────────────────────────
  { name: "Google Edge TPU (Coral)", category: "Accelerator", sram_kb: 8192,  flash_kb: 8192,   clock_mhz: 500,  macs_per_cycle: 4000,  power_mw: 2000 },
  { name: "Intel Movidius Myriad X", category: "Accelerator", sram_kb: 4096,  flash_kb: 8192,   clock_mhz: 700,  macs_per_cycle: 1000,  power_mw: 1500 },
  { name: "Hailo-8L",                category: "Accelerator", sram_kb: 16384, flash_kb: 32768,  clock_mhz: 400,  macs_per_cycle: 6500,  power_mw: 2500 },
  { name: "NVIDIA Jetson Nano",      category: "Accelerator", sram_kb: 4194304, flash_kb: 16777216, clock_mhz: 921, macs_per_cycle: 256, power_mw: 5000 },
  { name: "MAX78000 (Maxim)",        category: "Accelerator", sram_kb: 512,   flash_kb: 512,    clock_mhz: 100,  macs_per_cycle: 64,    power_mw: 1 },
  { name: "GreenWaves GAP9",         category: "Accelerator", sram_kb: 1536,  flash_kb: 2048,   clock_mhz: 400,  macs_per_cycle: 8,     power_mw: 50 },

  // ── Single-Board Computers (SBC) ──────────────────────────────
  { name: "Raspberry Pi Zero 2 W",   category: "SBC", sram_kb: 524288,   flash_kb: 16777216, clock_mhz: 1000, macs_per_cycle: 1,   power_mw: 1200 },
  { name: "Raspberry Pi 4B (2GB)",   category: "SBC", sram_kb: 2097152,  flash_kb: 33554432, clock_mhz: 1500, macs_per_cycle: 1,   power_mw: 4000 },
  { name: "BeagleBone AI-64",        category: "SBC", sram_kb: 4194304,  flash_kb: 16777216, clock_mhz: 2000, macs_per_cycle: 8,   power_mw: 5000 },
];

/* ── Client-side computed metrics ─────────────────────── */

export function computeModelMetrics(topology: ModelTopology, bitwidth: number = 32) {
  const totalParams = topology.layers.reduce((s, l) => s + l.param_count, 0);

  const layerFlops = topology.layers.map(l => {
    let flops = 2 * l.in_features * l.out_features;
    // For Conv2d, multiply by kernel spatial dimensions
    if (l.kernel_size) {
      flops *= l.kernel_size.reduce((a, b) => a * b, 1);
    }
    return flops;
  });
  const totalFlops = layerFlops.reduce((s, f) => s + f, 0);

  // Memory footprint: weights + peak activation
  const peakActivationKb = Math.max(
    ...topology.layers.map(l => (l.out_features * 4) / 1024),
    0
  );
  const memoryFootprintKb = topology.current_size_kb + peakActivationKb;

  // Accuracy drop estimation (heuristic)
  const sparsity = topology.global_sparsity;
  let accuracyDrop = 0;
  if (sparsity > 0) {
    if (sparsity <= 50) {
      accuracyDrop = (sparsity / 10) * 0.5;
    } else {
      accuracyDrop = (50 / 10) * 0.5 + ((sparsity - 50) / 10) * 2.0;
    }
  }

  // Quantization accuracy drop based on bitwidth
  let quantAccDrop = 0;
  if (bitwidth <= 4) quantAccDrop = 3.5;
  else if (bitwidth <= 8) quantAccDrop = 0.8;
  else if (bitwidth <= 16) quantAccDrop = 0.1;
  const totalAccDrop = Math.round((accuracyDrop + quantAccDrop) * 100) / 100;

  // Latency factor
  const latencyFactor = Math.max(0.05, 1.0 - (sparsity / 100) * 0.6);

  return {
    totalParams,
    totalFlops,
    layerFlops,
    peakActivationKb: Math.round(peakActivationKb * 100) / 100,
    memoryFootprintKb: Math.round(memoryFootprintKb * 100) / 100,
    estimatedAccuracyDrop: totalAccDrop,
    latencyFactor: Math.round(latencyFactor * 10000) / 10000,
  };
}

export function computeHardwareAnalysis(
  topology: ModelTopology,
  device: HardwareDevice,
  bitwidthOrFlops: number = 32,
  peakActivationKbArg?: number,
) {
  // Support both calling conventions:
  // 1. computeHardwareAnalysis(topology, device, bitwidth) — auto-compute
  // 2. computeHardwareAnalysis(topology, device, totalFlops, peakActivationKb) — explicit
  let totalFlops: number;
  let peakActivationKb: number;

  if (peakActivationKbArg !== undefined) {
    // Explicit calling convention
    totalFlops = bitwidthOrFlops;
    peakActivationKb = peakActivationKbArg;
  } else {
    // Simplified: compute from topology
    const metrics = computeModelMetrics(topology, bitwidthOrFlops);
    totalFlops = metrics.totalFlops;
    peakActivationKb = metrics.peakActivationKb;
  }

  const flashFit = topology.current_size_kb <= device.flash_kb;
  const ramFit = peakActivationKb <= device.sram_kb;
  const flashUtil = (topology.current_size_kb / device.flash_kb) * 100;
  const ramUtil = (peakActivationKb / device.sram_kb) * 100;

  // Inference time: FLOPs / (clock * MACs/cycle * 1e6) * 1000 → ms
  const opsPerSec = device.clock_mhz * 1e6 * device.macs_per_cycle;
  const inferenceMs = (totalFlops / opsPerSec) * 1000;
  const throughput = inferenceMs > 0 ? 1000 / inferenceMs : 0;

  // Energy: power_mw * time_s → mJ
  const energyMj = device.power_mw * (inferenceMs / 1000);

  // Battery life: 1000 mAh LiPo @ 3.7V = 3700 mWh
  const batteryMwh = 3700;
  const batteryHrs = energyMj > 0 ? (batteryMwh * 3.6) / energyMj : 999999;

  return {
    flashFit,
    ramFit,
    flashUtil: Math.round(flashUtil * 10) / 10,
    ramUtil: Math.round(ramUtil * 10) / 10,
    inferenceMs: Math.round(inferenceMs * 1000) / 1000,
    throughput: Math.round(throughput * 10) / 10,
    energyMj: Math.round(energyMj * 10000) / 10000,
    batteryHrs: Math.round(batteryHrs * 10) / 10,
  };
}

export function computeReadinessScore(
  flashFit: boolean,
  ramFit: boolean,
  flashUtil: number,
  ramUtil: number,
  inferenceMs: number,
  energyMj: number,
  accuracyRetention: number,
) {
  // Memory (30 pts)
  let memScore = 0;
  if (flashFit && ramFit) {
    const avgUtil = (flashUtil + ramUtil) / 2;
    memScore = Math.round(30 * Math.max(0, 1 - avgUtil / 100));
  }

  // Latency (30 pts) — target < 100ms  
  const latScore = inferenceMs < 100
    ? Math.round(30 * (1 - inferenceMs / 100))
    : 0;

  // Energy (20 pts) — target < 1mJ
  const engScore = energyMj < 1
    ? Math.round(20 * (1 - energyMj))
    : 0;

  // Accuracy (20 pts)
  const accScore = Math.round(20 * Math.min(accuracyRetention / 100, 1));

  return {
    total: Math.max(0, Math.min(100, memScore + latScore + engScore + accScore)),
    memory: { score: memScore, max: 30 },
    latency: { score: latScore, max: 30 },
    energy: { score: engScore, max: 20 },
    accuracy: { score: accScore, max: 20 },
  };
}
