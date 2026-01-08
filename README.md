# 🌐 Hybrid Transformer-Based Framework for Seismic Response Prediction of RC Chimneys

[![Paper](https://img.shields.io/badge/Paper-View_on_Journal-blue)](INSERT_LINK_HERE)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository contains the official implementation of the deep learning framework proposed in the paper:  
> **"A Hybrid Transformer-Based Multi-Task Framework for Rapid Seismic Response Prediction of Reinforced Concrete Chimneys"**

---

## 📌 Overview
Reinforced concrete (RC) chimneys are critical industrial structures, yet their seismic assessment via traditional **Nonlinear Time-History Analysis (NLTHA)** is computationally prohibitive for real-time applications or large-scale fragility studies.

This framework introduces a hybrid Transformer-based model that delivers high-fidelity structural response predictions in milliseconds, achieving a **3,000–18,000× speedup** over OpenSees NLTHA.

### Key Capabilities
* **Full Time-History Prediction:** Predicts displacement and acceleration sequences at multiple heights.
* **Modal Analysis:** Simultaneously estimates the first five modal frequencies.
* **High Generalization:** Validated on unseen chimney geometries and diverse ground motion records.
* **Advanced Architecture:** Utilizes Rotary Positional Embeddings (**RoPE**) and Grouped-Query Attention (**GQA**) for superior temporal modeling.

---

## 🏗️ Model Features

| Feature | Description |
| :--- | :--- |
| **Multi-Task Learning** | Shared encoder backbone for frequency and time-series prediction. |
| **Sequence Compression** | Adaptive methods to handle long-duration seismic records efficiently. |
| **Transfer Learning** | Knowledge distillation from displacement to acceleration tasks. |
| **Computational Efficiency** | Optimized attention mechanisms for rapid inference. |

---

## 🚀 Getting Started

### Installation
```bash
# Clone the repository
git clone [https://github.com/YourUsername/RepositoryName.git](https://github.com/YourUsername/RepositoryName.git)
cd RepositoryName

# Install required packages
pip install -r requirements.txt
