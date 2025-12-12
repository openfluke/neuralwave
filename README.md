# 🌊 NeuralWave

**Real-time Neural Network Model Analysis & 3D Visualization**

NeuralWave is a web-based tool for benchmarking, analyzing, and visualizing neural network models. It provides deep insights into model architecture through interactive 3D visualizations, allowing you to inspect individual layers, neurons, and connections.

![NeuralWave Dashboard](https://img.shields.io/badge/Go-1.21+-00ADD8?style=flat&logo=go)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

### 📊 Model Management
- **Auto-Discovery**: Scans configured directories for `.gguf` model files
- **Verification**: Tests if models can be loaded successfully
- **Benchmarking**: Measures tokens-per-second (TPS) performance
- **Batch Operations**: Verify, benchmark, or analyze multiple models at once

### 🔬 Weight Analysis
- **Layer Statistics**: Mean, StdDev, Min, Max for each layer's weights
- **Parameter Counts**: Total params per layer and model-wide
- **Architecture Detection**: Identifies MLP, Attention, Norm, Embedding layers

### 🎮 3D Visualization
- **Interactive Scene**: Orbit, pan, zoom through the neural network
- **Layer Selection**: Click layers or use dropdown to focus
- **Deep Dive**: Expand any layer to see its internal structure:
  - Neurons with value-based coloring
  - Weight connections
  - Biases visualization

### 🔬 Feature Extraction
- **Attention Layers**: Visualize all 12 heads with Q/K/V neurons
- **MLP Layers**: See Gate, Up, Down projections separately
- **Norm Layers**: Scale (γ) and Shift (β) parameters
- **One-Click**: Press "Extract Features" button to decompose

---

## 🚀 Quick Start

### Prerequisites
- Go 1.21+
- [LOOM](https://github.com/openfluke/loom) - OpenFluke's neural network framework
- Hugging Face account (for downloading models)

### Installation

```bash
git clone https://github.com/openfluke/neuralwave.git
cd neuralwave
go mod tidy
go build -o neuralwave .
```

### Download Models

Use the included script to download models from Hugging Face:

```bash
./download_models.sh
```

Or manually download SafeTensors models to your `~/.cache/huggingface/hub/` directory.

### Run

```bash
./neuralwave
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 📁 Project Structure

```
neuralwave/
├── main.go              # CLI entry point & llama.cpp integration
├── web.go               # HTTP server, WebSocket, API handlers
├── views/
│   └── index.html       # Single-page app with Three.js visualization
├── static/              # Static assets
├── download_models.sh   # Model downloader script
├── go.mod
└── README.md
```

---

## 🎯 Usage

### Dashboard Tab
- View all discovered models
- Select models for batch operations
- Run Verify / Benchmark / Analyze

### Analysis Tab
- See detailed layer-by-layer statistics
- Compare model architectures

### Compare Tab
- Side-by-side layer comparison across models
- Filter by layer name or type

### Visualize 3D Tab
1. **Select a model** from the dropdown
2. Click **Generate** to render the architecture
3. **Click any layer** to zoom in and see details
4. Press **🔬 Extract Features** for deep decomposition
5. **Rotate/Zoom** freely to explore
6. Click **✕** to return to overview

---

## ⚙️ Configuration

Edit `main.go` to configure:

```go
modelRoots := []string{
    "/path/to/your/models",
}
```

---

## 🛠️ Tech Stack

- **Backend**: Go, Gorilla WebSocket
- **AI Framework**: [LOOM](https://github.com/openfluke/loom) (OpenFluke)
- **Frontend**: HTML, CSS (Tailwind), JavaScript
- **3D Engine**: Three.js (r128)
- **Model Format**: SafeTensors (Hugging Face)

---

## 📝 License

APACHE2 License - See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

**Built with 🧠 by [OpenFluke](https://openfluke.com)**
