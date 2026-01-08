🌐 Hybrid Transformer-Based Framework for Seismic Response Prediction of RC Chimneys
This repository contains the official implementation of the deep learning framework proposed in the paper:

"A Hybrid Transformer-Based Multi-Task Framework for Rapid Seismic Response Prediction of Reinforced Concrete Chimneys"

📌 Overview
Reinforced concrete (RC) chimneys are critical yet seismically vulnerable industrial structures. Traditional nonlinear time-history analysis (NLTHA) is highly accurate but computationally prohibitive for real-time applications or large-scale studies.

To address this, we introduce a hybrid Transformer-based deep learning model that:

- Predicts full dynamic response time histories (displacement & acceleration) at multiple heights.
- Simultaneously estimates the first five modal frequencies.
- Generalizes well to unseen geometries and ground motions.
- Delivers predictions in milliseconds, achieving 3,000–18,000× speedup over OpenSees NLTHA.

Our model leverages:

- A shared encoder backbone with multi-task learning.
- Adaptive sequence compression for efficient long-sequence modeling.
- Transfer learning from displacement to acceleration prediction.
- Rotary Positional Embeddings (RoPE) and Grouped-Query Attention (GQA) for robust temporal modeling.
