"""
Preprocesses LOBSTER Limit Order Book (Level 10) data for TimeGAN training.

Input  : message_10.csv and orderbook_10.csv (aligned by event index)
Output : NumPy arrays (train, val, test) of shape [num_seq, seq_len, num_features]

Features used:
    midprice  = (ask1 + bid1) / 2
    spread    = ask1 - bid1
    imbalance = (bid_size1 - ask_size1) / (bid_size1 + ask_size1)

Compatible with local Linux dev and Google Colab (T4 GPU).
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Configurable parameters
DATA_DIR = "data"  # change if using Google Drive on Colab
SEQ_LEN = 100  # window length
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
