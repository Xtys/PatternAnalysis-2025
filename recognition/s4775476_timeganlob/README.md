# Synthetic Limit Order Book (LOB) Sequence Generation using TimeGAN

Abstract
---
This project explores the application of TimeGAN, a generative adversarial network for sequential data, to the synthesis of Limit Order Book (LOB) time-series. The goal is to generate synthetic market microstructure data that preserves the temporal dependencies and statistical characteristics of real financial order flows.

Financial time-series data such as LOB messages and order depths are inherently non-stationary, noisy, and highly dynamic. Traditional GANs struggle to capture such temporal dependencies, leading to unstable outputs. TimeGAN combines recurrent dynamics with adversarial learning, allowing both reconstruction-based and generation-based consistency within latent space.

The dataset used in this project comes from the LOBSTER feed for AMZN (June 21, 2012), consisting of message and order-book pairs. Each input sequence has 20 time steps and 43 engineered features, including bid/ask relative prices, log-volumes, and derived features such as spread, mid-price return, and order imbalance.

This report presents the baseline implementation and evaluation of the TimeGAN model, trained using a two-phase approach: (1) reconstruction and supervised pretraining, and (2) adversarial refinement. The generated synthetic sequences are assessed against held-out test data using the evaluation metrics defined in the task specification — KL divergence (≤ 0.1), Structural Similarity Index (SSIM > 0.6), and discriminator accuracy (~0.5).

Initial results show that the model effectively reproduces broad temporal structures but fails to generalize over distributional diversity. This outcome motivates the next stage of development: integrating a kurtosis-based proxy to better capture heavy-tail behaviour, and applying Optuna for systematic hyperparameter tuning to improve model stability and realism.

Introduction
---
This work investigates the use of TimeGAN for generating realistic Limit Order Book (LOB) sequences.
The objective is to reproduce temporal and structural properties of real market data. This includes spread, mid-price returns, and depth imbalance — while maintaining distribution similarity.

The project focuses on the evaluation phase of the TimeGAN pipeline, where generated sequences are compared to real data using metrics defined in the specification:

- KL divergence ≤ 0.1 between real and synthetic spread and mid-return distributions.
- SSIM > 0.6 on LOB depth heatmaps.
- Discriminator accuracy ≈ 0.5 for realism balance.
