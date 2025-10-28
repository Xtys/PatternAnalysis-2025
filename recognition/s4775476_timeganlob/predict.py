"""
Predict / generation script for trained TimeGAN model.
"""

import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from modules import Embedder, Recovery, Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
