import json
import os
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os


# ----- Define Model ----- #
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ----- Setup ----- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load JSON configurations
with open("best_model_per_config.json", "r") as f:
    model_configs = json.load(f)

# Load dataset
df_original = pd.read_csv("data_try_3.csv")

# Feature columns and scalers
feature_columns = ["p_mb", "t_c", "rh_percent", "z_m_10_meter_bin", "ws_m_s"]
scalers = {col: StandardScaler() for col in feature_columns}


# Create val_ids by extracting header_ids and doing a split
header_ids = df_original["header_id"].unique()
random.shuffle(header_ids)
split_idx = int(0.8 * len(header_ids))
val_ids = header_ids[split_idx:]

# Select a header_id
random.seed(2)
random_header_id = random.choice(val_ids)
# random_header_id = 16458
sample_group_for_plot = df_original[df_original["header_id"] == random_header_id].sort_values("z_m_10_meter_bin", ascending = False)
for col in feature_columns:
    df_original[col] = scalers[col].fit_transform(df_original[[col]])
sample_group = df_original[df_original["header_id"] == random_header_id].sort_values("z_m_10_meter_bin", ascending = False)

print("sample_group ", sample_group.head(20))


# ----- Evaluate Each Model ----- #

generated_sequences_dict = []
debug_on = True
for config_name, config in model_configs.items():
    sequence_length = config["sequence_length"]
    hidden_size = config["hidden_size"]
    model_path = os.path.join("MultipleRuns/attempt_6/best_models", config["model_path"])

    # Instantiate and load model
    model = LSTMModel(input_size=len(feature_columns) - 1, hidden_size=hidden_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Extract input sequence
    initial_sequence = sample_group[feature_columns].values[:sequence_length, :-1]
    if sequence_length == 5 and hidden_size == 64 and debug_on:
        print("initial_sequence ", initial_sequence)
    initial_sequence_windspeed = sample_group["ws_m_s"].values
    if sequence_length == 5 and hidden_size == 64 and debug_on:
        print("initial_sequence_windspeed ", initial_sequence_windspeed)
    if len(initial_sequence) < sequence_length:
        continue

    generated_ws = []
    with torch.no_grad():
        generated_ws.extend(initial_sequence_windspeed[:sequence_length])
        print("after extend ")
        print(  f'{len(generated_ws) = }')
        if sequence_length == 5 and hidden_size == 64 and debug_on:
            print("sequence_length \n", sequence_length)
            print("extended \n", generated_ws)
        for i in range(sequence_length, len(sample_group) - sequence_length):
            if sequence_length == 5 and hidden_size == 64 and debug_on and i < 20:
                print("i ", i)
            current_window = sample_group[feature_columns].values[i-sequence_length:i, :-1]
            current_window_ws = sample_group[feature_columns].values[i-sequence_length:i, -1]
            input_tensor = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(device)

            predicted_ws = model(input_tensor).cpu().numpy().flatten()[0]
            if sequence_length == 5 and hidden_size == 64 and debug_on and i < 20:
                print("current_window \n", current_window)
                print("predicted_ws",predicted_ws)
                print("current_window_ws", current_window_ws[-1])

            generated_ws.append(predicted_ws)

        # Inverse transform predictions
        generated_ws_array = np.array(generated_ws).reshape(-1, 1)
        original_scale_ws = scalers['ws_m_s'].inverse_transform(generated_ws_array).flatten()
        generated_ws = original_scale_ws
        print(f'{len(generated_ws) = }')
        generated_z_m = np.linspace( sample_group_for_plot["z_m_10_meter_bin"].values[0],0, len(generated_ws))
        # generated_z_m =
        # print(sample_group_for_plot)
        generated_sequences_dict.append({
            "generated_ws": generated_ws,
            "generated_z_m": generated_z_m,
            "sequence_length": sequence_length,
            "hidden_size": hidden_size
        })


from collections import defaultdict

# ----- Plotting the predicted sequences for each model type ----- #

# Group sequences by sequence_length
grouped_by_sequence_length = defaultdict(list)
for item in generated_sequences_dict:
    grouped_by_sequence_length[item['sequence_length']].append(item)

num_groups = len(grouped_by_sequence_length)
fig, axes = plt.subplots(num_groups, 1, figsize=(10, 6 * num_groups), sharex=True)

if num_groups == 1:
    axes = [axes]

for ax, (seq_len, group_items) in zip(axes, grouped_by_sequence_length.items()):
    # Plot actual data once per subplot
    print(f'{len(actual_ws) = }')
    print(f'{len(actual_z_m) = }')
    ax.plot(actual_ws, actual_z_m, 'bo-', label="Actual Data")

    for item in group_items:
        print(f'{len(item["generated_ws"]) = }')
        print(f'{len(item["generated_ws"]) = }')

        ax.scatter(item["generated_ws"], actual_z_m[:len(item["generated_ws"])],
                   label=f"Hid Size {item['hidden_size']}")

    ax.set_ylabel("z_m_10_meter_bin")
    ax.set_title(f"Header ID {random_header_id} | Sequence Length {seq_len}")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("ws_m_s")
plt.tight_layout()
plt.show()
