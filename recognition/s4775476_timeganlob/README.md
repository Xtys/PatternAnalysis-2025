# Synthetic Limit Order Book (LOB) Sequence Generation using TimeGAN

Abstract
---
This project focuses on generating synthetic Limit Order Book (LOB) sequences using TimeGAN, a model that combines recurrent and adversarial learning to capture both temporal and statistical properties of sequential data. The objective is to produce realistic synthetic LOB data that resembles actual market dynamics, particularly in mid-price returns, spread, and order imbalance.

The dataset used is the LOBSTER feed for AMZN (2012), processed into 20-step sequences with 43 engineered features including relative prices, log volumes, and derived market indicators. The training follows a two-phase structure: Phase 1 for reconstruction and supervised consistency, and Phase 2 for adversarial fine-tuning.

Evaluation is based on the task specification metrics: KL divergence (≤ 0.1), SSIM (> 0.6), and discriminator accuracy (~0.5). The baseline model was able to learn temporal relationships but struggled to replicate the true distributional shape, leading to high discriminator accuracy and poor SSIM scores.

These findings highlight the need for further tuning. In the next stage, the model integrates a kurtosis-based proxy loss to better represent tail behavior and employs Optuna for hyperparameter optimization to improve stability and generalization. The enhanced model achieves full specification compliance, with all metrics passing thresholds, demonstrating improved fidelity for downstream market simulations.

Model Architecture
---
The architecture of this project follows the TimeGAN framework [1], combining autoencoding, supervised prediction, and adversarial generation to model temporal dependencies in LOB sequences. The network consists of five key modules: Embedder (E), Recovery (R), Supervisor (S), Generator (G), and Discriminator (D), all implemented using GRU layers for efficiency and temporal stability. The latent dimension is set to $d_h = 64$, resulting in approximately 136,000 trainable parameters across all components.

1. **Embedder (E)**: The Embedder transforms the input sequence $X \in \mathbb{R}^{B \times T \times d_x}$ (where $d_x = 43$) into a latent representation $H \in \mathbb{R}^{B \times T \times d_h}$ using a single-layer GRU followed by a fully connected layer with tanh activation:

    - $H_t = \tanh(W_h \cdot \text{GRU}(X_t; \theta_E) + b_h), \quad t = 1, \dots, T$

    where $\theta_E$ denotes the GRU parameters, and $W_h, b_h$ are the projection weights and bias. This encodes the high-dimensional LOB features into a compact, temporally aware space.

2. **Recovery (R)**: The Recovery module decodes the latent $H$ back to the feature space via a GRU and linear projection, producing reconstructed sequences $\hat{X} \in \mathbb{R}^{B \times T \times d_x}$:

    - $\hat{X}_t = W_x \cdot \text{GRU}(H_t; \theta_R) + b_x, \quad t = 1, \dots, T$

    where $\theta_R$ are the GRU parameters, and $W_x, b_x$ project to the output dimension. It enforces reconstruction fidelity during pretraining via $\mathcal{L}_{recon} = \| X - \hat{X} \|^2_2$.

3. **Supervisor (S)**: An autoregressive GRU-based predictor that forecasts the next latent state $\hat{H}_{t+1}$ from $H_t$, outputting $\hat{H} \in \mathbb{R}^{B \times (T-1) \times d_h}$:

    - $\hat{H}_t = \tanh(W_s \cdot \text{GRU}(H_t; \theta_S) + b_s), \quad t = 1, \dots, T-1$

    where $\theta_S$ are the GRU parameters, and $W_s, b_s$ are the projection. This promotes temporal consistency via  $\mathcal{L}_{sup} = \| \hat{H} - H_{1:T-1} \|^2_2$.

4. **Generator (G)**: Starting from random noise $Z \in \mathbb{R}^{B \times T \times d_h} \sim \mathcal{N}(0, I)$, the Generator produces synthetic latents $\hat{H} \in \mathbb{R}^{B \times T \times d_h}$ via GRU and tanh projection:

    - $\hat{H}_t = \tanh(W_g \cdot \text{GRU}(Z_t; \theta_G) + b_g), \quad t = 1, \dots, T$

    where $\theta_G$ are the GRU parameters, and $W_g, b_g$ project. Composed with S and R, it yields full synthetic sequences $\tilde{X} = R(S(G(Z)))$.

5. **Discriminator (D)**: A GRU classifier that distinguishes real latents $H = E(X)$ from synthetic $\hat{H}$, outputting a scalar probability $p \in [0,1]$ via sigmoid:

    - $p = \sigma(W_d \cdot \text{GRU}(H_T; \theta_D) + b_d)$

    where $\theta_D$ are the GRU parameters, $H_T$ is the final hidden state, and $\sigma$ is the sigmoid. It provides adversarial signals to refine G via binary cross-entropy $\mathcal{L}_{adv} = -\mathbb{E}[\log p(H)] - \mathbb{E}[\log(1 - p(\hat{H}))]$.

Dataset loading and Preprocessing
---
The Limit Order Book (LOB) represents the full state of market supply and demand through continuously updated bid and ask prices with their corresponding volumes. Modeling such data is challenging due to its high frequency, noise, and nonlinear temporal dependencies. Traditional models like ARIMA or simple RNNs often fail to capture these complex microstructure patterns.

In this project, data is sourced from the LOBSTER dataset for Amazon (AMZN) on June 21, 2012, covering the trading interval from 09:30:00 to 16:00:00. The raw message and order-book files were processed into synchronized snapshots using a custom dataset pipeline (dataset.py).

Each sample is a sequence of 20 timesteps containing 43 standardized features, grouped into four categories:

- Price levels: 20 bid and 20 ask relative prices (ticks from mid-price, scaled by 100).
- Volume levels: 20 bid and 20 ask log-transformed sizes $(log(1 + size$ for stability).
- Derived indicators: Spread (ask1 - bid1, scaled), mid-price return $(log(mid_t / mid_{t-1}))$, and imbalance (($bid1_s - ask1_s$) / ($bid1_s + ask1_s$)).

All features are normalized using a StandardScaler and stored for reuse across training and inference. The resulting dataset contains approximately 26,973 total sequences, split into train (70%), validation (10%), and test (20%) sets.

This structured preprocessing enables stable training of TimeGAN while preserving meaningful short-term order flow dynamics within each sequence window.

Training Procedure
---
Training proceeds in two phases, implemented in train.py with configurable sweeps over hyperparameters (hidden_dim, lr, λ_sup, batch_size).
Phase 1 (Supervised Pretraining, 20 epochs): Optimize E and R for reconstruction loss (MSE: $\mathcal{L}_{recon} = \| X - R(E(X)) \|^2$) and S for latent supervision loss (MSE: $\mathcal{L}_{sup} = \| S(E(X)) - E(X)_{1:} \|^2$). This initializes robust embeddings and temporal predictors.
Phase 2 (Adversarial Training, 50 epochs): Alternate updates for D (BCE on real/fake latents) and G (adversarial BCE + λ_sup * self-sup + moment matching). Moment loss enforces mean/variance alignment: $\mathcal{L}_{mom} = \| \mu_{\tilde{X}} - \mu_X \|^2 + \| \sigma_{\tilde{X}}^2 - \sigma_X^2 \|^2$. Joint E-R-S updates maintain reconstruction on supervised paths.

Optimizers use Adam (lr=1e-3 baseline), with validation monitoring (recon/sup <0.005 target). Checkpoints and loss plots are saved per experiment.

References
---
[1] Yoon, J., Cho, J., & Lee, J. (2019). TimeGAN: A Time-series Generative Adversarial Network. arXiv preprint arXiv:1904.04442.

[2] Boan, L., et al. (2024). MarketGAN: Controllable Financial Time Series Generation with Semantic Context. Proceedings of AAAI 2024. Available at: https://personal.ntu.edu.sg/boan/papers/AAAI24_MarketGAN.pdf.
