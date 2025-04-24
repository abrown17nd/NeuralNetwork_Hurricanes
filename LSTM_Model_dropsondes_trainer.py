# LSTM_Model_dropsondes_v1.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import logging

# Set up logging
logging.basicConfig(filename="training_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

def log_message(msg):
    print(msg)
    logging.info(msg)

# Load data
df = pd.read_csv("data_try_1.csv")

# Fill missing values
df.fillna(df.mean(), inplace=True)

# Normalize features (excluding header_id and time)
feature_columns = ["p_mb", "t_c", "rh_percent", "z_m", "ws_m_s"]
scalers = {col: MinMaxScaler() for col in feature_columns}
for col in feature_columns:
    df[col] = scalers[col].fit_transform(df[[col]])

# Prepare sequences per header_id
sequence_length = 5
sequences = []

for _, group in df.groupby("header_id"):
    group = group.sort_values("t_s")  # Sort by time
    values = group[feature_columns].values  # Extract relevant columns

    for i in range(len(values) - sequence_length):
        sequences.append((values[i:i+sequence_length], values[i+sequence_length][0]))  # Target is next p_mb

# Convert sequences to tensors
X, y = zip(*sequences)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).reshape(-1, 1)

# Split into training (80%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Convert to tensors
X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Predicts a single value (p_mb)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Use last time step output

# Initialize model
input_size = len(feature_columns)
model = LSTMModel(input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop with Validation
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation Step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()

    # Log training and validation loss every epoch
    log_message(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader):.6f}, "
                f"Val Loss: {val_loss / len(val_loader):.6f}")

# Make a sample prediction
with torch.no_grad():
    sample_input = X_val[0].unsqueeze(0)  # Take first validation sequence for testing
    prediction = model(sample_input).numpy()
    predicted_p_mb = scalers["p_mb"].inverse_transform(prediction)  # Convert back to original scale
    log_message(f"Predicted next p_mb: {predicted_p_mb[0][0]:.2f}")
