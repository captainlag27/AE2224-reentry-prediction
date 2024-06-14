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

torch.cuda.empty_cache()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
DATA_FILE = Path(__file__).parent / "validation2.csv"
MIN_GROUP_SIZE = 600
TRAIN_SPLIT_RATIO = 0.7

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

def load_data(filepath):
    """Load data from CSV, handle errors, and log the process."""
    try:
        data = pd.read_csv(filepath, delimiter=';')
        logging.info(f"Data loaded from {filepath}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        exit()
    except pd.errors.EmptyDataError:
        logging.error("No data: file is empty")
        exit()

def preprocess_data(data):
    """Clean and group data."""
    data.dropna(inplace=True)
    grouped = data.groupby("NORAD_CAT_ID").filter(lambda x: len(x) > MIN_GROUP_SIZE)
    logging.info("Data cleaned and grouped by NORAD_CAT_ID")
    return grouped

def process_grouped_data(grouped):
    """Process grouped data into a list of dataframes, sorted by 'EPOCH'."""
    dataset = [
        group.sort_values(by="EPOCH").reset_index(drop=True)
        for _, group in grouped.groupby("NORAD_CAT_ID")
    ]
    return dataset

def split_data(dataset):
    """Randomize and split the dataset into training and testing subsets."""
    np.random.seed(42)
    np.random.shuffle(dataset)
    split_idx = int(TRAIN_SPLIT_RATIO * len(dataset))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    logging.info(
        f"Train data size: {len(train_data)}, Test data size: {len(test_data)}"
    )
    return train_data, test_data

# Example function to move tensors to GPU and perform operations
def example_gpu_operations():
    a = torch.randn(1000, 1000).to(device)
    b = torch.randn(1000, 1000).to(device)
    c = torch.matmul(a, b)
    logging.info(f"Performed matrix multiplication on GPU. Result shape: {c.shape}")

def main():
    data = load_data(DATA_FILE)
    processed_data = preprocess_data(data)
    dataset = process_grouped_data(processed_data)
    train_data, test_data = split_data(dataset)

    # Example operation on GPU
    example_gpu_operations()

    return train_data, test_data, data

if __name__ == "__main__":
    train_data, test_data, data = main()