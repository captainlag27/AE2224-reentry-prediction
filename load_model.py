import time
import datetime
from model import main

time_started = datetime.datetime.now()

import pandas as pd
import math
import numpy as np
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
from model import LSTMModel

print("Start load model")

PATH = 'model_ep80_nl1_nn16_sl300_trained_for_32789.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 14
num_layers = 1
hidden_size = 16
output_size = 1
num_epochs = 40
# Define the model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, num_layers).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scaler = MinMaxScaler(feature_range=(0, 1))

model.load_state_dict(torch.load(PATH))

print("Model loaded")

norad = 32789
test_data, train_data, data = main()
norad_id_to_predict = data[data["NORAD_CAT_ID"].astype(str).str.fullmatch(str(norad))]
print(norad)
print(norad_id_to_predict)
norad_id_to_predict_len = math.ceil(len(norad_id_to_predict))
norad_id_to_predict_not_edited = norad_id_to_predict.iloc[:norad_id_to_predict_len, 2:]
norad_id_to_predict_mid_step = norad_id_to_predict.iloc[:norad_id_to_predict_len, 3:]
print(norad_id_to_predict_not_edited)
print("MID STEP")
print(norad_id_to_predict_mid_step.head())
test_norad_id_non_edited = np.reshape(norad_id_to_predict_not_edited, (-1, 15))
test_norad_id = np.reshape(norad_id_to_predict_mid_step, (-1, 14))
sclaed_test_norad_id = scaler.fit_transform(test_norad_id)
test_data_23471 = torch.tensor(sclaed_test_norad_id, dtype=torch.float32).to(device)

print("Checkpoint 1")


with torch.no_grad():
    norad_predictions = model(test_data_23471)

# Inverse normalization of the predictions
norad_predicted_values = scaler.inverse_transform(
    norad_predictions.squeeze().cpu().numpy()
)
norad_predicted_values_reshaped = np.reshape(norad_predicted_values, (-1, 14))


print("Checkpoint 2")


# Extract the third and 10th columns from the predicted values
norad_predicted_third_column = test_norad_id_non_edited[:, 0]
norad_predicted_tenth_column = norad_predicted_values_reshaped[:, 6]
print("LAST HEAD")
print(norad_id_to_predict_mid_step.head())
norad_actual_third_column = test_norad_id_non_edited[:, 0]
norad_actual_tenth_column = test_norad_id[:, 6]

# print(norad_actual_tenth_column)
print(norad_predicted_third_column)
print(norad_predicted_tenth_column)

norad_float_third = []

for th in norad_predicted_third_column:
    norad_float_third.append(float(str(time.mktime(datetime.datetime.fromisoformat(str(th)).timetuple()))))


# Plot the third and 10th columns of the predicted values

begin = 1.675e9
end = 1.805e9
points = 659
norad_float_third = np.linspace(begin, end, points)

print("Checkpoint 3")


plt.figure(figsize=(10, 5))
plt.plot(
    norad_float_third,
    norad_predicted_tenth_column,
    label="Predicted",
    color="blue",
)
plt.plot(
    norad_float_third, norad_actual_tenth_column, label="Actual", color="red"
)
plt.title(f"Predicted and Actual Average Altitude vs Epoch for Norad {norad}, {num_epochs} Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Altitude")
plt.legend()
plt.grid(True)

time_finished = datetime.datetime.now()
difference = time_finished - time_started
seconds_in_day = 24 * 60 * 60
print(divmod(difference.days * seconds_in_day + difference.seconds, 60), "minutes, seconds")

plt.show()

print("Done")