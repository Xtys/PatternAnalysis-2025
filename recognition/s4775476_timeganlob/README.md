Task 14 – Limit Order Book TimeGAN Project

Student: Brandon Loh Ming Fong
Student ID: S4775476
Date: 29 October 2025

1. Project Overview

This project focuses on generating synthetic Limit Order Book (LOB) time-series data using a TimeGAN architecture.
The work corresponds to Task 14 – Limit Order Book Time Series in the COMP3710 course, where the goal is to model and replicate realistic high-frequency trading sequences that capture temporal and structural market dynamics.

The implemented TimeGAN combines supervised and adversarial learning to generate synthetic sequences that mimic the original LOB data distribution while maintaining temporal consistency.

2. Objective

The aim of this project is to:

Preprocess raw LOBSTER data to obtain normalized, sequential datasets suitable for deep generative modeling.

Implement a TimeGAN from scratch using PyTorch, following the architecture described in Yoon et al. (2019).

Train and evaluate the model on financial microstructure data to assess the realism of generated sequences.

Provide a modular and reproducible pipeline from preprocessing to training.
