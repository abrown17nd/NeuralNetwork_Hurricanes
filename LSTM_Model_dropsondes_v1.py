# LSTM_Model_dropsondes_v1.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(filename="training_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")
def log_message(message):
    print(message)
    logging.info(message)

# Load data
log_message("Loading data from data_try_1.csv...")
df = pd.read_csv("data_try_1.csv")

# Fill missing values
log_message("Filling missing values...")
df.fillna(df.mean(), inplace=True)

# Normalize features (excluding header_id and time)
feature_columns = ["p_mb", "t_c", "rh_percent", "z_m", "ws_m_s"]
scalers = {col: MinMaxScaler() for col in feature_columns}
for col in feature_columns:
    df[col] = scalers[col].fit_transform(df[[col]])

# Prepare sequences per header_id
sequence_length = 5
sequences = []

log_message("Preparing sequences for LSTM...")
for _, group in df.groupby("header_id"):
    group = group.sort_values("t_s")  # Sort by time
    values = group[feature_columns].values  # Extract relevant columns
    
    for i in range(len(values) - sequence_length):
        sequences.append((values[i:i+sequence_length], values[i+sequence_length][0]))  # Target is next p_mb

# Convert sequences to tensors
X, y = zip(*sequences)
X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)  # Make y a column vector

# Create DataLoader
batch_size = 16
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

log_message(f"Total sequences prepared: {len(sequences)}")

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

log_message("Starting training...")

# Training Loop
epochs = 50
for epoch in range(epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        log_message(f"Epoch {epoch}, Loss: {loss.item():.6f}")

log_message("Training complete.")

# Make predictions
with torch.no_grad():
    sample_input = X[0].unsqueeze(0)  # Take first sequence for testing
    prediction = model(sample_input).numpy()
    predicted_p_mb = scalers["p_mb"].inverse_transform(prediction)  # Convert back to original scale
    log_message(f"Predicted next p_mb: {predicted_p_mb[0][0]:.2f}")

log_message("Script execution complete. Check training_log.txt for details.")
