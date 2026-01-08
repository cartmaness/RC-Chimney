# 🌐 Hybrid Transformer-Based Framework for Seismic Response Prediction of RC Chimneys
> [!IMPORTANT]
> **Status: 🚧 Building in Progress** > This repository is currently undergoing active development. Code, documentation, and pre-trained weights are being updated frequently.

[![Paper](https://img.shields.io/badge/Paper-View_on_Journal-blue)](INSERT_LINK_HERE)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)

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

## 🚀 Usage Example: Inference

You can use the model to predict the structural response by passing chimney geometry, modal frequencies, and the ground motion acceleration (GMA) record.

```python
import torch
from model import Chimney_Transformer

# 1. Initialize the model
model = Chimney_Transformer(
    parameters_features=10, 
    freq_features=5, 
    response_features=11, 
    sequence_length=4000,
    compression_ratio=8
).to("cuda")

# 2. Prepare inputs (Example shapes)
params = torch.randn(1, 15).to("cuda")   # Geometry parameters
freqs = torch.randn(1, 5).to("cuda")     # Modal frequencies
gma = torch.randn(1, 5000).to("cuda")    # Ground motion record

# 3. Predict both Displacement and Acceleration
model.eval()
with torch.no_grad():
    acc_out, disp_out = model(params, freqs, gma, return_both=True)

print(f"Acceleration Prediction Shape: {acc_out.shape}") # [Batch, Time, Heights]
```
---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* PyTorch 2.0+
* `torchinfo`, `tqdm`, `numpy`, `matplotlib`
