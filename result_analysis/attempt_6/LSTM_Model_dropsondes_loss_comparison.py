import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Define model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
print("Loading dataset...")
df = pd.read_csv("data_try_3.csv")
df.fillna(df.mean(), inplace=True)

feature_columns = ["p_mb", "t_c", "rh_percent", "z_m_10_meter_bin", "ws_m_s"]
scalers = {col: MinMaxScaler() for col in feature_columns}
print("Normalizing features...")
for col in feature_columns:
    df[col] = scalers[col].fit_transform(df[[col]])

# Parameters
sequence_lengths = [5, 10, 20, 25]
attempt_number = 2
batch_size = 16
final_train_losses = []
final_val_losses = []

for seq_len in sequence_lengths:
    print(f"\n--- Processing sequence length: {seq_len} ---")

    sequences = []
    print("Creating sequences...")
    for header_id, group in df.groupby("header_id"):
        group = group.sort_values("z_m_10_meter_bin")
        values = group[feature_columns].values
        if len(values) < seq_len:
            continue
        for i in range(len(values) - seq_len):
            X_seq = values[i:i + seq_len, :-1]
            y_target = values[i + seq_len, -1]
            sequences.append((X_seq, y_target, header_id))

    print(f"Total sequences created: {len(sequences)}")

    all_header_ids = list(set([seq[2] for seq in sequences]))
    train_ids, val_ids = train_test_split(all_header_ids, test_size=0.2, random_state=42)
    train_seqs = [seq for seq in sequences if seq[2] in train_ids]
    val_seqs = [seq for seq in sequences if seq[2] in val_ids]

    X_train, y_train = zip(*[(x, y) for x, y, _ in train_seqs])
    X_val, y_val = zip(*[(x, y) for x, y, _ in val_seqs])

    # Convert to tensors
    X_train_tensor = torch.tensor(np.array(X_train, dtype=np.float32)).to(device)
    y_train_tensor = torch.tensor(np.array(y_train, dtype=np.float32).reshape(-1, 1)).to(device)
    X_val_tensor = torch.tensor(np.array(X_val, dtype=np.float32)).to(device)
    y_val_tensor = torch.tensor(np.array(y_val, dtype=np.float32).reshape(-1, 1)).to(device)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

    # Load model
    model_path = f"MultipleRuns/attempt_{attempt_number}/seq_len_{seq_len}/lstm_model_epoch_50.pth"
    print(f"Loading model from: {model_path}")
    input_size = X_train_tensor.shape[2]
    model = LSTMModel(input_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Loss function
    criterion = nn.MSELoss()

    # Evaluate training loss
    print("Evaluating training loss...")
    train_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            train_loss += loss.item()
    train_loss_avg = train_loss / len(train_loader)
    final_train_losses.append(train_loss_avg)
    print(f"Train loss (epoch 50): {train_loss_avg:.6f}")

    # Evaluate validation loss
    print("Evaluating validation loss...")
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()
    val_loss_avg = val_loss / len(val_loader)
    final_val_losses.append(val_loss_avg)
    print(f"Validation loss (epoch 50): {val_loss_avg:.6f}")

# Plotting
print("\nPlotting results...")
bar_width = 0.35
x = np.arange(len(sequence_lengths))

plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, final_train_losses, width=bar_width, label="Train Loss (Epoch 50)")
plt.bar(x + bar_width/2, final_val_losses, width=bar_width, label="Validation Loss (Epoch 50)")

plt.xticks(x, [f"Seq {s}" for s in sequence_lengths])
plt.ylabel("MSE Loss")
plt.title("Epoch 50 Loss for Different Sequence Lengths")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("epoch_50_loss_comparison.jpg")
plt.show()
