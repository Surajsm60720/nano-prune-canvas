# NanoPrune Canvas

A web-based neural network compression toolkit for visualizing and simulating magnitude-based pruning and post-training quantization. Upload PyTorch or ONNX models, interactively adjust compression parameters, and evaluate edge-device deployment readiness — all from the browser.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Running the Application](#running-the-application)
8. [Running Tests](#running-tests)
9. [API Reference](#api-reference)
10. [Supported Hardware Targets](#supported-hardware-targets)
11. [Tech Stack](#tech-stack)
12. [License](#license)

---

## Overview

NanoPrune Canvas provides an interactive dashboard for exploring how pruning and quantization affect neural network models before deploying them to resource-constrained edge devices. The tool parses model architectures, renders a node-graph visualization of the network topology, and computes real-time metrics including estimated accuracy drop, inference latency, energy consumption, and hardware compatibility across 26 embedded devices.

The application is split into a Python backend (FastAPI) that handles model parsing, weight manipulation, and export, and a React frontend (Next.js) that renders the interactive visualization and analytics.

---

## Features

**Model Support**
- Upload and parse PyTorch models (.pt, .pth) with arbitrary `nn.Module` architectures
- Upload and parse ONNX models (.onnx) with Gemm/MatMul layers
- Support for Linear and Conv2d layer types

**Pruning**
- Magnitude-based weight pruning with adjustable threshold
- Real-time global and per-layer sparsity calculation
- Visual sparsity bars on each layer node in the graph

**Quantization**
- Post-training affine fake quantization (FP32, FP16, INT8, INT4)
- Per-tensor scale/zero-point computation
- Effective size estimation combining pruning and quantization

**Visualization**
- Interactive node-graph rendering of the full network topology using React Flow
- Color-coded edges representing weight magnitudes (positive/negative)
- Animated edges for high-magnitude weights
- Node Inspector panel — click any node to view detailed layer statistics
- Per-layer sparsity bar chart (Recharts)
- Compression timeline sparkline tracking size reduction over successive adjustments

**Hardware Analysis**
- 26 embedded devices across 4 categories (MCU, SoC, Accelerator, SBC)
- Flash/SRAM fit analysis with utilization percentages
- Inference time, throughput, energy per inference, and battery life estimation
- Edge Readiness Score (0-100) with breakdown across memory, latency, energy, and accuracy

**Export**
- Download compressed model files (PyTorch .pt or ONNX .onnx)
- Generate deployment reports as PDF
- Full hardware compatibility table in the report

**Learning Resources**
- Built-in educational panel explaining pruning, quantization, accuracy-compression tradeoffs, and the TinyML inference pipeline
- Interactive weight matrix visualization
- Accuracy vs. compression curve

---

## Architecture

```
+---------------------+         HTTP/JSON         +---------------------+
|                     |  <--------------------->  |                     |
|   Frontend (3000)   |      POST /api/upload     |   Backend (8000)    |
|                     |      POST /api/prune      |                     |
|   Next.js 16        |      POST /api/quantize   |   FastAPI            |
|   React 19          |      POST /api/export     |   PyTorch (CPU)     |
|   React Flow v11    |                           |   ONNX Runtime      |
|   Recharts          |                           |   NumPy             |
|   Tailwind CSS v4   |                           |                     |
+---------------------+                           +---------------------+
```

The frontend performs all metric computation (FLOPs, memory footprint, hardware analysis, readiness scoring) client-side using the topology data returned by the backend. The backend handles model loading, weight manipulation (pruning/quantization), and binary export.

---

## Project Structure

```
nano-prune-canvas/
|
|-- backend/
|   |-- main.py                  # FastAPI app entry point, CORS configuration
|   |-- requirements.txt         # Python dependencies
|   |-- test_core.py             # 18 test functions (parser, pruner, quantizer, ONNX)
|   |-- generate_models.py       # Script to generate sample .pt/.onnx test models
|   |-- api/
|   |   |-- __init__.py
|   |   |-- routes.py            # API endpoints: upload, prune, quantize, export
|   |-- core/
|   |   |-- __init__.py
|   |   |-- parser.py            # PyTorch model loader and topology extractor
|   |   |-- pruner.py            # Magnitude-based pruning (Linear, Conv2d)
|   |   |-- quantizer.py         # Post-training fake quantization
|   |   |-- onnx_parser.py       # ONNX model parsing, pruning, and quantization
|   |-- models/                  # Pre-generated sample models (.pt and .onnx)
|       |-- tiny_classifier.*
|       |-- deep_narrow.*
|       |-- wide_shallow.*
|       |-- large_recommender.*
|       |-- mnist_mlp.*
|       |-- embedding_reducer.*
|
|-- frontend/
|   |-- package.json
|   |-- next.config.ts
|   |-- tsconfig.json
|   |-- postcss.config.mjs
|   |-- eslint.config.mjs
|   |-- src/
|   |   |-- app/
|   |   |   |-- globals.css      # Design tokens, animations, utility classes
|   |   |   |-- layout.tsx       # Root layout (Space Grotesk + JetBrains Mono)
|   |   |   |-- page.tsx         # Main page: header, 3-column layout, overlays
|   |   |-- components/
|   |   |   |-- CanvasArea.tsx    # React Flow graph with node inspector
|   |   |   |-- ControlPanel.tsx  # Upload, pruning slider, quantization buttons
|   |   |   |-- MetricsDashboard.tsx  # Model/Hardware/Readiness analytics tabs
|   |   |   |-- LearningDashboard.tsx # Educational content panel
|   |   |   |-- DeploymentReport.tsx  # PDF-exportable deployment report modal
|   |   |   |-- Icons.tsx        # SVG icon components
|   |   |-- lib/
|   |       |-- api.ts           # Backend API client functions
|   |       |-- types.ts         # TypeScript types, hardware catalog, metrics
|   |-- public/
```

---

## Prerequisites

- Python 3.11 or later
- Node.js 18 or later
- npm (included with Node.js)

---

## Installation

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch is installed from the CPU-only index by default. If you need GPU support, modify the `--index-url` in `requirements.txt` accordingly.

### Frontend

```bash
cd frontend
npm install
```

---

## Running the Application

### Start the Backend

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. A health check endpoint is at `GET /health`.

### Start the Frontend

```bash
cd frontend
npm run dev
```

The application will be available at `http://localhost:3000`.

### Using the Application

1. Open `http://localhost:3000` in your browser.
2. Upload a `.pt`, `.pth`, or `.onnx` model file using the control panel on the left (or use one of the sample models from `backend/models/`).
3. Adjust the pruning threshold slider to remove low-magnitude weights.
4. Select a quantization level (FP32, FP16, INT8, INT4).
5. View the network graph in the center panel. Click any node to open the Node Inspector.
6. Switch between Model, Hardware, and Readiness tabs in the right panel to analyze metrics.
7. Select different hardware targets from the dropdown to evaluate deployment feasibility.
8. Click "Export" in the header to generate a deployment report or download the compressed model.

### Generating Sample Models

To regenerate the sample `.pt` and `.onnx` models in `backend/models/`:

```bash
cd backend
source venv/bin/activate
python generate_models.py
```

---

## Running Tests

The backend includes 18 test functions covering parsing, pruning, quantization, Conv2d support, composability, and ONNX operations:

```bash
cd backend
source venv/bin/activate
python -m pytest test_core.py -v
```

Test coverage includes:

| Area | Tests |
|------|-------|
| PyTorch parser | `test_parser`, `test_conv2d_parser`, `test_arbitrary_module` |
| Pruning | `test_pruner_zero`, `test_pruner_high`, `test_conv2d_prune`, `test_prune_generated_models` |
| Quantization | `test_quantizer`, `test_quantizer_fp32`, `test_conv2d_quantize`, `test_real_quantization`, `test_quantize_generated_models` |
| Composability | `test_composability` |
| Performance | `test_large_model_performance` |
| ONNX | `test_onnx_parse`, `test_onnx_prune`, `test_onnx_quantize` |
| Generated models | `test_parse_generated_models` |

---

## API Reference

All endpoints are prefixed with `/api`.

### POST /api/upload

Upload a model file for parsing.

- **Content-Type**: `multipart/form-data`
- **Body**: `file` — a `.pt`, `.pth`, or `.onnx` file
- **Response**: JSON topology object

```json
{
  "model_id": "uuid",
  "original_size_kb": 48.0,
  "current_size_kb": 48.0,
  "global_sparsity": 0.0,
  "max_weight": 0.543210,
  "layers": [
    {
      "layer_id": "layer_0",
      "type": "Linear",
      "in_features": 784,
      "out_features": 128,
      "weights_shape": [128, 784],
      "layer_sparsity": 0.0,
      "param_count": 100480,
      "sampled_edges": [0.123, -0.456, ...]
    }
  ]
}
```

### POST /api/prune

Apply magnitude-based pruning.

- **Content-Type**: `application/json`
- **Body**: `{ "model_id": "uuid", "threshold": 0.1 }`
- **Response**: Updated topology object (same schema as upload)

### POST /api/quantize

Apply post-training quantization.

- **Content-Type**: `application/json`
- **Body**: `{ "model_id": "uuid", "target_bitwidth": 8 }`
- **Response**: Updated topology object

### POST /api/export

Download the compressed model file.

- **Content-Type**: `application/json`
- **Body**: `{ "model_id": "uuid" }`
- **Response**: Binary file download (`.pt` or `.onnx`)

### GET /health

Health check endpoint.

- **Response**: `{ "status": "ok" }`

---

## Supported Hardware Targets

The application includes 26 hardware devices organized into 4 categories:

### Microcontrollers (MCU)

| Device | SRAM | Flash | Clock | Power |
|--------|------|-------|-------|-------|
| Arduino Nano 33 BLE | 256 KB | 1 MB | 64 MHz | 15 mW |
| Nordic nRF5340 | 512 KB | 1 MB | 128 MHz | 5 mW |
| STM32F411 (Black Pill) | 128 KB | 512 KB | 100 MHz | 100 mW |
| STM32F746 | 320 KB | 1 MB | 216 MHz | 300 mW |
| STM32H743 | 1 MB | 2 MB | 480 MHz | 500 mW |
| STM32L476 | 128 KB | 1 MB | 80 MHz | 36 mW |
| Raspberry Pi Pico (RP2040) | 264 KB | 2 MB | 133 MHz | 25 mW |
| Raspberry Pi Pico 2 (RP2350) | 520 KB | 4 MB | 150 MHz | 30 mW |
| Teensy 4.1 | 1 MB | 8 MB | 600 MHz | 500 mW |
| Ambiq Apollo4 Blue Plus | 2 MB | 2 MB | 192 MHz | 3 mW |

### System-on-Chip (SoC)

| Device | SRAM | Flash | Clock | Power |
|--------|------|-------|-------|-------|
| ESP32-S3 | 512 KB | 8 MB | 240 MHz | 240 mW |
| ESP32-C3 | 400 KB | 4 MB | 160 MHz | 130 mW |
| ESP32-C6 | 512 KB | 4 MB | 160 MHz | 140 mW |
| Kendryte K210 | 8 MB | 16 MB | 400 MHz | 300 mW |
| Realtek AMB82-Mini | 4 MB | 16 MB | 500 MHz | 400 mW |
| NXP i.MX RT1060 | 1 MB | 8 MB | 600 MHz | 450 mW |
| NXP i.MX RT1170 | 2 MB | 16 MB | 1 GHz | 700 mW |

### AI Accelerators

| Device | SRAM | Flash | Clock | MACs/cycle | Power |
|--------|------|-------|-------|------------|-------|
| Google Edge TPU (Coral) | 8 MB | 8 MB | 500 MHz | 4000 | 2 W |
| Intel Movidius Myriad X | 4 MB | 8 MB | 700 MHz | 1000 | 1.5 W |
| Hailo-8L | 16 MB | 32 MB | 400 MHz | 6500 | 2.5 W |
| NVIDIA Jetson Nano | 4 GB | 16 GB | 921 MHz | 256 | 5 W |
| MAX78000 (Maxim) | 512 KB | 512 KB | 100 MHz | 64 | 1 mW |
| GreenWaves GAP9 | 1.5 MB | 2 MB | 400 MHz | 8 | 50 mW |

### Single-Board Computers (SBC)

| Device | RAM | Storage | Clock | Power |
|--------|-----|---------|-------|-------|
| Raspberry Pi Zero 2 W | 512 MB | 16 GB | 1 GHz | 1.2 W |
| Raspberry Pi 4B (2GB) | 2 GB | 32 GB | 1.5 GHz | 4 W |
| BeagleBone AI-64 | 4 GB | 16 GB | 2 GHz | 5 W |

---

## Tech Stack

### Backend

| Component | Version |
|-----------|---------|
| Python | 3.11+ |
| FastAPI | 0.115.0 |
| Uvicorn | 0.30.0 |
| PyTorch | CPU build |
| ONNX | 1.15.0+ |
| NumPy | 1.26.4 |

### Frontend

| Component | Version |
|-----------|---------|
| Next.js | 16.1.6 |
| React | 19.2.3 |
| React Flow | 11.11.4 |
| Recharts | 3.7.0 |
| Tailwind CSS | 4.x |
| TypeScript | 5.x |
| html2canvas | 1.4.1 |
| jsPDF | 4.2.0 |

### Design

- Font: Space Grotesk (headings/body), JetBrains Mono (numeric/code)
- Color scheme: Dark navy palette with green (#00E68A) accent and purple (#7B61FF) secondary
- Design tokens defined in CSS custom properties

---

## License

This project is provided as-is for educational and research purposes.
