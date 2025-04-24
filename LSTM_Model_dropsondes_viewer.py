# Using the same headers and seeing how model works

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import logging


import matplotlib.pyplot as plt
import random

# Basic setup for use across notebook

attempt_number = 2
logging.basicConfig(filename=f"training_log.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

def print_and_log(text):
    print(text)
    logger.info(text)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Detect device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_and_log(f"Using device: {device}")

# Load data
df = pd.read_csv(f"data_try_3.csv")
print_and_log(f"Dataset loaded with {len(df)} rows.")
print_and_log(f"Unique header_id values: {df['header_id'].nunique()}")

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Normalize features
feature_columns = ["p_mb", "t_c", "rh_percent", "z_m_10_meter_bin", "ws_m_s"]
scalers = {col: MinMaxScaler() for col in feature_columns}
for col in feature_columns:
    df[col] = scalers[col].fit_transform(df[[col]])

print_and_log("Feature normalization complete.")

# Prepare sequences
sequence_length = 5
sequences = []
for header_id, group in df.groupby("header_id"):
    group = group.sort_values("z_m_10_meter_bin")
    values = group[feature_columns].values

    if len(values) < sequence_length:
        continue

    for i in range(len(values) - sequence_length):
        X_seq = values[i:i + sequence_length, :-1]
        y_target = values[i + sequence_length, -1]
        sequences.append((X_seq, y_target, header_id))

if not sequences:
    print_and_log(f"No sequences generated for sequence length {sequence_length}. Skipping.")


# Group-based split
all_header_ids = list(set([seq[2] for seq in sequences]))
train_ids, val_ids = train_test_split(all_header_ids, test_size=0.2, random_state=42)
train_seqs = [seq for seq in sequences if seq[2] in train_ids]
val_seqs = [seq for seq in sequences if seq[2] in val_ids]
X_train, y_train = zip(*[(x, y) for x, y, _ in train_seqs])
X_val, y_val = zip(*[(x, y) for x, y, _ in val_seqs])

# Convert to tensors
X_train = torch.tensor(np.array(X_train, dtype=np.float32)).to(device)
y_train = torch.tensor(np.array(y_train, dtype=np.float32).reshape(-1, 1)).to(device)
X_val = torch.tensor(np.array(X_val, dtype=np.float32)).to(device)
y_val = torch.tensor(np.array(y_val, dtype=np.float32).reshape(-1, 1)).to(device)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)

# Initialize model
input_size = X_train.shape[2]
model = LSTMModel(input_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print_and_log("Model initialized. Ready for programming")
# summary(model,(1, 28, 28))

groups_of_generated_w_s = []


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

model = LSTMModel(input_size).to(device)
# Load the trained model
attempt_number = 2
sequence_lengths = [5, 10, 20, 25]
max_epochs = 50
# Same header_id prediction for all runs

# Select a random header_id from validation set
df_original = pd.read_csv(f"data_try_3.csv")

random_header_id = random.choice(val_ids)
# print(random_header_id)
sample_group_for_plot = df_original[df_original["header_id"] == random_header_id].sort_values("z_m_10_meter_bin")
for col in feature_columns:
    df_original[col] = scalers[col].fit_transform(df_original[[col]])
sample_group = df_original[df_original["header_id"] == random_header_id].sort_values("z_m_10_meter_bin")
# print(sample_group)

generated_sequences_dict = []
actual_ws = 0
actual_z_m = 0
generated_z_m = 0

for sequence_length in sequence_lengths:
  model_path = f"MultipleRuns/attempt_{attempt_number}/seq_len_{sequence_length}/lstm_model_epoch_{max_epochs}.pth"
  model = LSTMModel(input_size).to(device)
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.eval()
  print_and_log(f"Model loaded from {model_path}")



  # Extract an initial seed sequence
  initial_sequence = sample_group[feature_columns].values[:sequence_length, :-1]  # Exclude target (ws_m_s)
  initial_sequence_windspeed = sample_group["ws_m_s"].values
  # print(initial_sequence)
  if len(initial_sequence) < sequence_length:
      print_and_log("Selected sequence is too short for prediction.")
  else:
      # Convert to tensor
      input_sequence = torch.tensor(initial_sequence, dtype=torch.float32).unsqueeze(0).to(device)

  generated_ws = []

  with torch.no_grad():
      # print(initial_sequence_windspeed)
      for i in range(5):
        generated_ws.append(initial_sequence_windspeed[i])
      # generated_ws.append(initial_sequence_windspeed[:sequence_length])

      for i in range(sequence_length, len(sample_group)-sequence_length):
          current_window = sample_group[feature_columns].values[i-sequence_length:i, :-1]  # exclude target
          input_sequence = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(device)

          predicted_ws = model(input_sequence).cpu().numpy().flatten()[0]
          generated_ws.append(predicted_ws)
          # print(predicted_ws)
          # print(generated_ws)

      generated_z_m = sample_group["z_m_10_meter_bin"].values[sequence_length:]

      # print(generated_ws)
      # Plot observed vs generated
      # fig, ax = plt.subplots(figsize=(10, 6))

      # Convert generated values back to original scale
      generated_ws_array = np.array(generated_ws).reshape(-1, 1)
      # print(generated_ws_array)
      original_scale_ws = scalers['ws_m_s'].inverse_transform(generated_ws_array).flatten()
      # print(original_scale_ws)
      generated_ws = original_scale_ws

      # Now original_scale_ws contains wind speed values in the original range
      print_and_log("Inverse scaling complete. Ready for plotting.")


      # Plot actual values
      actual_ws = sample_group_for_plot["ws_m_s"].values
      actual_z_m = sample_group_for_plot["z_m_10_meter_bin"].values


      # Plot generated values (future sequence)
      # generated_z_m = np.linspace(actual_z_m[-1], actual_z_m[-1] + 10 * len(generated_ws), len(generated_ws))
      generated_z_m = np.linspace(0, actual_z_m[-1], len(generated_ws))
      # print(generated_z_m)

      generated_sequences_dict.append({"generated_ws": generated_ws,"generated_z_m":generated_z_m, "sequence_length":sequence_length})

# print(generated_sequences_dict)
plt.plot(actual_ws, actual_z_m , 'bo-', label="Actual Data")
for item in generated_sequences_dict:
  seq_len = item["sequence_length"]
  gen = item["generated_ws"]
  z = item["generated_z_m"]
  plt.plot(gen, z, label=f"Generated Data sequence length {seq_len}")

plt.xlabel("ws_m_s")
plt.ylabel("z_m_10_meter_bin")
plt.legend()
plt.title(f"Generated Sequence vs. Actual Data for Header ID {random_header_id} for sequence length {sequence_length}")

print_and_log("Generated sequence plotted.")
plt.show()
plt.savefig(f"sequence_prediction_graphs/sequence_prediction_header_{random_header_id}.jpg")

