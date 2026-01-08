# 🌐 Hybrid Transformer-Based Framework for Seismic Response Prediction of RC Chimneys

[![Paper](https://img.shields.io/badge/Paper-View_on_Journal-blue)](INSERT_LINK_HERE)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official implementation of the **Chimney Transformer**, a deep learning framework for rapid seismic response prediction of reinforced concrete chimneys. This model achieves high-fidelity results with a **3,000–18,000× speedup** over traditional OpenSees Nonlinear Time-History Analysis (NLTHA).

---

## 📌 Overview
Structural health monitoring and large-scale fragility assessments require rapid analysis. This framework replaces expensive numerical integration with a **Hybrid Transformer** that:
* Predicts full **Acceleration** and **Displacement** time histories.
* Estimates the first five **Modal Frequencies**.
* Generalizes across different chimney geometries and ground motions.



---

## 🏗️ Model Architecture
The `Chimney_Transformer` architecture is engineered for long-sequence seismic data:

* **Adaptive Sequence Compression:** Uses `AdaptiveAvgPool1d` and `Conv1d` to compress sequences (default ratio 8:1), enabling the Transformer to process long-duration records efficiently.
* **Rotary Positional Embeddings (RoPE):** Integrated via `precompute_rotary_embeddings` to provide relative temporal awareness.
* **Multi-Task Heads:** Specialized output heads for Acceleration and Displacement, utilizing unique `task_embeddings` and `height_embeddings`.
* **RMSNorm & GELU:** Employs Root Mean Square Layer Normalization and GELU activations for stable, high-performance training.



---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* PyTorch 2.0+
* `torchinfo`, `tqdm`, `numpy`, `matplotlib`

### Installation
```bash
git clone [https://github.com/YourUsername/Seismic-Chimney-Transformer.git](https://github.com/YourUsername/Seismic-Chimney-Transformer.git)
cd Seismic-Chimney-Transformer
pip install -r requirements.txt
