import json
import os
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# ----- Define Model -----
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ----- Setup -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(2)

# Load model configs
with open("best_model_per_config.json", "r") as f:
    model_configs = json.load(f)

# Load dataset
df_original = pd.read_csv("/scratch365/abrown17/NeuralNetworks/Project/data_try_3.csv")
feature_columns = ["p_mb", "t_c", "rh_percent", "z_m_10_meter_bin", "ws_m_s"]
scalers = {col: StandardScaler() for col in feature_columns}

# Train-val split by header_id
header_ids = df_original["header_id"].unique()
random.shuffle(header_ids)
split_idx = int(0.8 * len(header_ids))
train_ids = header_ids[:split_idx]
val_ids = header_ids[split_idx:]

# Fit scalers on training data only
train_df = df_original[df_original["header_id"].isin(train_ids)]
for col in feature_columns:
    scalers[col].fit(train_df[[col]])
for col in feature_columns:
    df_original[col] = scalers[col].transform(df_original[[col]])

# Select random sample from validation set
random_header_id = 16458  # Can change this to random.choice(val_ids)
sample_group = df_original[df_original["header_id"] == random_header_id].sort_values("z_m_10_meter_bin", ascending=False)
sample_group_for_plot = sample_group.copy()
actual_ws = sample_group_for_plot["ws_m_s"].values
actual_z_m = sample_group_for_plot["z_m_10_meter_bin"].values

# ----- Evaluate Each Model -----
generated_sequences_dict = []
for config_name, config in model_configs.items():
    sequence_length = config["sequence_length"]
    hidden_size = config["hidden_size"]
    model_path = os.path.join("best_models", config["model_path"])

    # Instantiate model
    model = LSTMModel(input_size=len(feature_columns) - 1, hidden_size=hidden_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if len(sample_group) < sequence_length:
        continue

    # Generate predictions
    generated_ws = []
    with torch.no_grad():
        generated_ws.extend(sample_group["ws_m_s"].values[:sequence_length])
        for i in range(sequence_length, len(sample_group)):
            window = sample_group[feature_columns].values[i-sequence_length:i, :-1]
            input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            predicted_ws = model(input_tensor).cpu().numpy().flatten()[0]
            generated_ws.append(predicted_ws)

    # Inverse transform predictions
    generated_ws_array = np.array(generated_ws).reshape(-1, 1)
    original_scale_ws = scalers['ws_m_s'].inverse_transform(generated_ws_array).flatten()
    generated_z_m = actual_z_m[:len(original_scale_ws)]

    generated_sequences_dict.append({
        "generated_ws": original_scale_ws,
        "generated_z_m": generated_z_m,
        "sequence_length": sequence_length,
        "hidden_size": hidden_size
    })

# ----- Plot -----
grouped_by_sequence_length = defaultdict(list)
for item in generated_sequences_dict:
    grouped_by_sequence_length[item['sequence_length']].append(item)

num_groups = len(grouped_by_sequence_length)
fig, axes = plt.subplots(num_groups, 1, figsize=(10, 6 * num_groups), sharex=True)

if num_groups == 1:
    axes = [axes]

for ax, (seq_len, group_items) in zip(axes, grouped_by_sequence_length.items()):
    ax.plot(actual_ws, actual_z_m, 'bo-', label="Actual Data")
    for item in group_items:
        ax.plot(item["generated_ws"], item["generated_z_m"], label=f"Hid Size {item['hidden_size']}")
    ax.set_ylabel("z_m_10_meter_bin")
    ax.set_title(f"Header ID {random_header_id} | Sequence Length {seq_len}")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("ws_m_s")
plt.tight_layout()
plt.savefig("header_and_predictions_by_model.jpg")
plt.show()
