
import pandas as pd
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
import os
from model import main
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
from model import main
import time
import datetime
from model2 import LSTMModel





time_started = datetime.datetime.now()
# Has to be expanded but this section will adjust the training and test data lengths and get the dataa

# train_data = df.iloc[:training_data_len, 2:]
# test_data = df.iloc[training_data_len:, 2:]
# print(train_data.shape, test_data.shape)
print('33')
# This section will reshape the data into the required format (such as 2D or 3D)
test_data, train_data, data = main()



print('37')
scaler = MinMaxScaler(feature_range=(0, 1))

X_train, y_train = [], []
sequence_length = 150
for i in range(len(train_data)):
    sorted_train_data = train_data[i].sort_values(by="AvA")
    dataset_train = np.reshape(sorted_train_data[['DryMass','DryFlag','Length','LFlag','Diameter','DFlag','Span','SpanFlag','Inc','Ecc','AvA','F107','Ap']], (-1, 13))
    scaled_train_dataset = scaler.fit_transform(dataset_train)

    for j in range(len(scaled_train_dataset) - sequence_length):
        X_train.append(scaled_train_dataset[j : j + sequence_length])
        y_train.append(scaled_train_dataset[j + 1 : j + sequence_length + 1])

print('51')

# Convert to numpy arrays after the loop
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = [torch.tensor(x, dtype=torch.float32) for x in X_train]
y_train = [torch.tensor(y, dtype=torch.float32) for y in y_train]



X_test, y_test = [], []

sequence_length = 150
for i in range(len(test_data)):
    sorted_test_data = test_data[i].sort_values(by="AvA")
    dataset_train = np.reshape(sorted_test_data[['DryMass','DryFlag','Length','LFlag','Diameter','DFlag','Span','SpanFlag','Inc','Ecc','AvA','F107','Ap']], (-1, 13))
    scaled_test_dataset = scaler.fit_transform(dataset_train)

    for j in range(len(scaled_test_dataset) - sequence_length):
        X_test.append(scaled_test_dataset[j : j + sequence_length])
        y_test.append(scaled_test_dataset[j + 1 : j + sequence_length + 1])
print("Checkpoint 77")
# Convert to numpy arrays after the loop
X_test, y_test = np.array(X_train), np.array(y_train)

X_test = [torch.tensor(x, dtype=torch.float32) for x in X_test]
y_test = [torch.tensor(y, dtype=torch.float32) for y in y_test]

torch.save(X_train, 'X_train.pt')
torch.save(y_train, 'y_train.pt')
torch.save(X_test, 'X_test.pt')
torch.save(y_test, 'y_test.pt')
# Convert to numpy arrays after the loop
