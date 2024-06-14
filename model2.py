from calendar import c
import numpy as np
import math
import matplotlib.pyplot as plt  # Visualization
import matplotlib.dates as mdates  # Formatting dates
import seaborn as sns  # Visualization
from sklearn.preprocessing import MinMaxScaler
import torch  # Library for implementing Deep Neural Network
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 64)  # Adding another linear layer
        self.linear2 = nn.Linear(64, 13)  # Final linear layer with output size 15

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear1(out)  # Apply the first linear layer
        out = torch.relu(out)  # Apply ReLU activation function
        out = self.linear2(out)  # Apply the final linear layer
        return out

