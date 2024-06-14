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
from new_model import main_2
import statistics

time_started = datetime.datetime.now()
# Has to be expanded but this section will adjust the training and test data lengths and get the dataa

# train_data = df.iloc[:training_data_len, 2:]
# test_data = df.iloc[training_data_len:, 2:]
# print(train_data.shape, test_data.shape)
print('33')
# This section will reshape the data into the required format (such as 2D or 3D)
test_data, train_data, data = main()
test_data_2, train_data_2, data_2 = main_2()
print('85')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Checkpoint 101")


input_size = 13
num_layers = 1
hidden_size = 80
output_size = 1
num_epochs = 30
learning_rate = 5e-4
 # Define the model, loss function, and optimizer

PATH = Path(__file__).parent / "model_ep30_nl1_nn80_sl128_trained_for_25018_bs48_lr0.0005.pt"
model = LSTMModel(input_size, hidden_size, num_layers).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scaler = MinMaxScaler(feature_range=(0, 1))
model.load_state_dict(torch.load(PATH))


norad = 28362  
print(data_2)
print(data_2["EPOCH"])
time_diff = []

norad_id_to_predict = data[data["NORAD_CAT_ID"].astype(str).str.fullmatch(str(norad))]
print(norad)
print(norad_id_to_predict)
norad_id_to_predict_len = math.ceil(len(norad_id_to_predict))
df1 = pd.DataFrame(norad_id_to_predict[['DryMass','DryFlag','Length','LFlag','Diameter','DFlag','Span','SpanFlag','Inc','Ecc','AvA','F107','Ap', 'EPOCH']])
df2 = pd.DataFrame(norad_id_to_predict[['DryMass','DryFlag','Length','LFlag','Diameter','DFlag','Span','SpanFlag','Inc','Ecc','AvA','F107','Ap']])
norad_id_to_predict_not_edited = df1
norad_id_to_predict_mid_step = df2
print(norad_id_to_predict_not_edited)
print("MID STEP")
print(norad_id_to_predict_mid_step.head())
test_norad_id_non_edited = np.reshape(norad_id_to_predict_not_edited, (-1, 14))
print(test_norad_id_non_edited)

test_norad_id = np.reshape(norad_id_to_predict_mid_step, (-1, 13))
sclaed_test_norad_id = scaler.fit_transform(test_norad_id)
test_data_23471 = torch.tensor(sclaed_test_norad_id, dtype=torch.float32).to(device)


with torch.no_grad():
    norad_predictions = model(test_data_23471)

# Inverse normalization of the predictions
norad_predicted_values = scaler.inverse_transform(
    norad_predictions.squeeze().cpu().numpy()
)
norad_predicted_values_reshaped = np.reshape(norad_predicted_values, (-1, 13))

# Extract the third and 10th columns from the predicted values
norad_predicted_third_column = test_norad_id_non_edited[:,-1]
norad_predicted_tenth_column = norad_predicted_values_reshaped[:,10]
print("LAST HEAD")
print(norad_id_to_predict_mid_step.head())
norad_actual_third_column = test_norad_id_non_edited[:,-1]
norad_actual_tenth_column = test_norad_id[:,10]

# print(norad_actual_tenth_column)
print(norad_predicted_third_column)
print(norad_predicted_tenth_column)
error_rmse_total = math.sqrt(np.sum((norad_predicted_values_reshaped[:,10] - test_norad_id[:,10])**2))
print(f'Final summed AvA error for all data points is {error_rmse_total}')
error_final_point = math.sqrt(abs(np.sum(norad_predicted_values_reshaped[:,10][-1] - test_norad_id[:,10][-1])))
print(f'Final epoch AvA error is {error_final_point}')

# Plot the third and 10th columns of the predicted values
epoch_number_data = 10
historical_data = data[data["NORAD_CAT_ID"].astype(str).str.fullmatch(str(norad))][:epoch_number_data]
historical_data_cut_columns = np.reshape(historical_data[['DryMass','DryFlag','Length','LFlag','Diameter','DFlag','Span','SpanFlag','Inc','Ecc','AvA','F107','Ap']], (-1,13))
print(historical_data)
forecasted_values = historical_data_cut_columns
print(forecasted_values)


def max_value(inputlist):
    return max([sublist[6] for sublist in inputlist])
#reshaped_predicted_value_2d = forecasted_values
trial = 0
# Use the trained model to forecast future values
time_column = np.reshape(historical_data[['EPOCH']], (-1,1))

for i in range(len(time_column)-1):
    time_diff.append(abs(time_column[i + 1] - time_column[i]))


time_diff_array = np.array(time_diff)
filtered_array_time = time_diff_array[time_diff_array != 0]
filtered_list_time = filtered_array_time.tolist()

average_time_diff = statistics.mean(filtered_list_time)

print(average_time_diff)


with torch.no_grad():
    while forecasted_values[:,10][-1] > 150 and trial < 1000:
        forecasted_values_transformed = scaler.fit_transform(forecasted_values)
        forecasted_values_tensor = torch.tensor(forecasted_values_transformed, dtype=torch.float32).to(device)
        predicted_value = model(forecasted_values_tensor).cpu().numpy()
        inverse_transformed_predicted = scaler.inverse_transform(predicted_value[-1].reshape(-1,13))
        print(inverse_transformed_predicted)
        forecasted_values = np.append(forecasted_values, inverse_transformed_predicted, axis=0)
        print('printing forecasted_values')
        print(forecasted_values[:,10])
        
        trial = trial + 1
        
#forecasted_values = np.array(forecasted_values)
print('proceeding to forecasted values now')
print(forecasted_values[:,10])
print(len(forecasted_values[:,10]))

combined_list = []



ava = forecasted_values[:,10]

dates = [time_column] + [time_column[-1] + average_time_diff * i/7.7 for i in range(len(forecasted_values) - len(time_column))]

for arr in dates:
    if arr.ndim > 1:
        combined_list.extend(arr.flatten())
    else:
        combined_list.extend(arr)

dates = combined_list


plt.figure(figsize=(10, 5))
plt.plot(dates, ava, label = f'Model forecasting results after first {epoch_number_data} epochs', color='green')
re_rentry_time_diff = dates[-1] - norad_actual_third_column[-1]

print(f'Final epoch AvA error is {error_final_point}')
print(f'Final summed AvA error for all data points is {error_rmse_total}')
print(f'Final error in re-entry time at 150km is {re_rentry_time_diff/86400}')

plt.plot(
    norad_predicted_third_column,
    norad_predicted_tenth_column,
    label="Model prediction for every point on the dataset",
    color="blue",
)
plt.plot(
    norad_actual_third_column, norad_actual_tenth_column, label="Actual re-entry", color="red"
)
plt.xlabel("Unix Time Stamp")
plt.ylabel("Average Altitude (Km)")
plt.legend()
plt.grid(True)

plt.show()


