# Synthetic LOB Sequence Generation using TimeGAN

Abstract
---
This report presents a baseline implementation and evaluation of TimeGAN for synthetic Limit Order Book (LOB) sequence generation.
The goal was to reproduce temporal and structural patterns of high-frequency market data using adversarial training on latent time-series features.
The baseline model was trained and evaluated according to the specification metrics — KL divergence ≤ 0.1 and SSIM > 0.6 — using held-out test data.

While the generator captured overall temporal correlations, it failed to generalize across the full distribution of market states. The results showed high KL divergence, low SSIM, and discriminator overfitting (accuracy = 1.0), indicating mode collapse and loss of depth variability.
These findings highlight the limitations of the base model and motivate the next phase of improvement through kurtosis-aware loss functions and Optuna-driven hyperparameter optimization, aimed at improving distribution fidelity and model stability.

Introduction
---
This work investigates the use of TimeGAN for generating realistic Limit Order Book (LOB) sequences.
The objective is to reproduce temporal and structural properties of real market data. This includes spread, mid-price returns, and depth imbalance — while maintaining distribution similarity.

The project focuses on the evaluation phase of the TimeGAN pipeline, where generated sequences are compared to real data using metrics defined in the specification:

- KL divergence ≤ 0.1 between real and synthetic spread and mid-return distributions.
- SSIM > 0.6 on LOB depth heatmaps.
- Discriminator accuracy ≈ 0.5 for realism balance.
