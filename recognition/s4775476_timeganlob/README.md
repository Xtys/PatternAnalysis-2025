# Synthetic Limit Order Book (LOB) Sequence Generation using TimeGAN

Abstract
---
This project focuses on generating synthetic Limit Order Book (LOB) sequences using TimeGAN, a model that combines recurrent and adversarial learning to capture both temporal and statistical properties of sequential data. The objective is to produce realistic synthetic LOB data that resembles actual market dynamics, particularly in mid-price returns, spread, and order imbalance.

The dataset used is the LOBSTER feed for AMZN (2012), processed into 20-step sequences with 43 engineered features including relative prices, log volumes, and derived market indicators. The training follows a two-phase structure: Phase 1 for reconstruction and supervised consistency, and Phase 2 for adversarial fine-tuning.

Evaluation is based on the task specification metrics: KL divergence (≤ 0.1), SSIM (> 0.6), and discriminator accuracy (~0.5). The baseline model was able to learn temporal relationships but struggled to replicate the true distributional shape, leading to high discriminator accuracy and poor SSIM scores.

These findings highlight the need for further tuning. In the next stage, the model will integrate a kurtosis-based proxy loss to better represent tail behaviour and employ Optuna for hyperparameter optimization to improve stability and generalization.

Model Architecture
---
The architecture of this project follows the TimeGAN framework [1], combining autoencoding, supervised prediction, and adversarial generation to model temporal dependencies in LOB sequences. The network consists of five key modules: Embedder (E), Recovery (R), Supervisor (S), Generator (G), and Discriminator (D), all implemented using GRU layers for efficiency and temporal stability.

1. Embedder (E)
The Embedder transforms the input sequence $$X \in \mathbb{R}^{B \times T \times 43}$$ into a latent representation

Dataset loading and Preprocessing
---
The Limit Order Book (LOB) represents the full state of market supply and demand through continuously updated bid and ask prices with their corresponding volumes. Modeling such data is challenging due to its high frequency, noise, and nonlinear temporal dependencies. Traditional models like ARIMA or simple RNNs often fail to capture these complex microstructure patterns.

In this project, data is sourced from the LOBSTER dataset for Amazon (AMZN) on June 21, 2012, covering the trading interval from 09:30:00 to 16:00:00. The raw message and order-book files were processed into synchronized snapshots using a custom dataset pipeline (dataset.py).

Each sample is a sequence of 20 timesteps containing 43 standardized features, grouped into four categories:

- Price levels: 20 bid and 20 ask relative prices
- Volume levels: 20 bid and 20 ask log-transformed sizes
- Derived indicators: spread, mid-price return, and imbalance

All features are normalized using a StandardScaler and stored for reuse across training and inference. The resulting dataset contains approximately 26,973 total sequences, split into train (70%), validation (10%), and test (20%) sets.

This structured preprocessing enables stable training of TimeGAN while preserving meaningful short-term order flow dynamics within each sequence window.





References
---
[1] Yoon, J., Cho, J., & Lee, J. (2019). TimeGAN: A Time-series Generative Adversarial Network. arXiv preprint arXiv:1904.04442.

[2] https://personal.ntu.edu.sg/boan/papers/AAAI24_MarketGAN.pdf
