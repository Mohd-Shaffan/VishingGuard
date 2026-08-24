---
title: ScamGuard Live
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

<div align="center">

# 🛡️ ScamGuard v3.0 — Neural Defense Grid

### AI-Powered Real-Time Vishing Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)

<p align="center">
  <strong>A cinematic, real-time vishing (voice phishing) detection engine</strong><br>
  <em>Combining DistilBERT embeddings + Logistic Regression + keyword heuristics<br>with a next-gen holographic security dashboard</em>
</p>

---

</div>

## ✨ Features

### 🧠 Hybrid AI Engine
- **DistilBERT Sentence Embeddings** (768-dim) for semantic understanding
- **Logistic Regression** classifier with 93%+ accuracy
- **Keyword Heuristic Engine** with India-specific scam detection (OTP, Aadhaar, KYC, etc.)
- **Temporal Threat Scorer** — cumulative threat tracking with time-decay
- **Real-time chunk-by-chunk analysis** via WebSocket

### 🎬 Cinematic Dashboard ("Neural Defense Grid")
- **3D Holographic Environment** — Three.js neural network with dual rotating icosahedrons
- **DNA Helix Particles** — threat-reactive color shifting (cyan → amber → red)
- **Liquid Gel Threat Meter** — GSAP physics-based animations
- **Chromatic Aberration** — RGB split overlay intensifies with threat level
- **Scanline Effect** — CRT-inspired retro overlay
- **Typewriter Log Effect** — AI "types out" analysis results
- **Spatial Audio** — dynamic ambient hum, stereo panning, heartbeat LFO on DROP
- **Jarvis Voice Mode** — "Hey ScamGuard" wake word + voice commands
- **KONAMI Code Easter Egg** — ↑↑↓↓←→←→BA unlocks confusion matrix dev mode
- **5 Color Themes** — Cyan, Emerald, Sentinel, Amber, Phantom
- **Command Palette** — Ctrl+K searchable command launcher
- **Mobile Ready** — device orientation parallax, shake-to-reset

### 🔐 Security Architecture
- Zero data persistence — all analysis in-memory only
- Session-isolated WebSocket channels
- Auto-disconnect on critical threat (DROP protocol)

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Frontend (index.html)                   │
│  Three.js · GSAP · Web Speech API · Web Audio API         │
│  ──────────────────────────────────────────────────────── │
│  WebSocket Client ←→ Real-time analysis stream             │
└──────────────────────┬───────────────────────────────────┘
                       │ WebSocket (ws://host/ws/{session})
┌──────────────────────▼───────────────────────────────────┐
│                  Backend (FastAPI + Uvicorn)               │
│  main.py  ─────→  scamguard_enhanced.py                   │
│  ├─ Session Manager      ├─ classify_intent()             │
│  ├─ WebSocket Handler     ├─ TemporalThreatScorer         │
│  └─ Threshold Control     ├─ DistilBERT Embeddings        │
│                           └─ Logistic Regression Model    │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ScamGuard.git
cd ScamGuard

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain the model
python train_vishing_model.py

# Start the server
python -m uvicorn main:app --host 0.0.0.0 --port 7860
```

Open your browser and navigate to `http://localhost:7860`

### Docker

```bash
docker build -t scamguard .
docker run -p 7860:7860 scamguard
```

### Hugging Face Spaces

1. Create a new Space (Docker SDK)
2. Upload all project files
3. The `Dockerfile` handles everything automatically

---

## 📁 Project Structure

```
ScamGuard/
├── main.py                    # FastAPI backend + WebSocket server
├── scamguard_enhanced.py      # Hybrid AI engine (NLP + heuristics)
├── train_vishing_model.py     # Dataset builder + model trainer
├── vishing_data.csv           # Cleaned & balanced training dataset
├── logistic_vishing_model.pkl # Trained model (768-dim LogReg)
├── index.html                 # Cinematic Neural Defense Grid UI
├── Dockerfile                 # Docker deployment config
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # You are here
```

---

## 🎮 Controls & Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Analyze current input |
| `Ctrl+K` | Open command palette |
| `Ctrl+M` | Toggle microphone |
| `Ctrl+V` | Jarvis voice mode |
| `Ctrl+R` | Reset session |
| `Ctrl+E` | Export report |
| `Ctrl+S` | Settings panel |
| `?` | Show all shortcuts |
| `1-4` | Load test scenarios |
| `↑↑↓↓←→←→BA` | Developer mode |

### Voice Commands (Jarvis Mode)
- **"Hey ScamGuard"** — wake word
- **"Analyze [text]"** — analyze a sentence
- **"Reset session"** — clear and restart
- **"Show report"** — export analysis report
- **"Emergency protocol"** — immediate threat response

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 93.13% |
| 5-Fold CV | 92.87% ±1.54% |
| Sanity Check | 15/16 (93.8%) |
| Safe Precision | 96% |
| Scam Recall | 97% |
| Embedding Model | DistilBERT (768-dim) |
| Classifier | Logistic Regression |

---

## 👥 Team

- **Mohd Shaffan** — Lead Developer
- **Ayush Kumar Agarwal** — AI/ML
- **Sarika** — Research
- **Sayantan Ghosh** — Research

**Manipal University Jaipur**

---

## 📄 License

This project is for educational purposes. Built at Manipal University Jaipur.

---

<div align="center">
  <sub>Built with ❤️ and lots of ☕ | ScamGuard v3.0</sub>
</div>
