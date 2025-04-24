# Sensitivity analysis for different sequence lengths

import os
import logging
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import json



# Constants
attempt_number = 4
sequence_lengths = [5, 10, 20, 25]
hidden_sizes = [8, 16, 32, 64]
learning_rates = [0.001, 0.005, 0.01, 0.005]

# Setup logging
log_dir = f"logs_attempt_{attempt_number}"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "training.log"), level=logging.INFO,
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

for hidden_size in hidden_sizes:
    for learning_rate in learning_rates:
        for sequence_length in sequence_lengths:
            print_and_log("=" * 60)
            print_and_log(f"Starting training with sequence length: {sequence_length}")

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
                continue

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
            model = LSTMModel(input_size,hidden_size=hidden_size).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            print_and_log("Model initialized. Beginning training...")

            # Training loop
            epochs = 50
            epsilon = 1e6
            best_loss = 1e6;
            for epoch in range(epochs):
                print_and_log(f"Starting epoch {epoch}")
                model.train()
                train_loss = 0
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)

                    if torch.isnan(loss) or loss.item() > epsilon:
                        print_and_log(f"Abnormal loss at Epoch {epoch + 1}, Batch {batch_idx}. Stopping.")
                        exit()

                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        y_pred = model(X_batch)
                        loss = criterion(y_pred, y_batch)
                        if torch.isnan(loss) or loss.item() > epsilon:
                            print_and_log(f"Abnormal validation loss at Epoch {epoch + 1}. Stopping.")
                            exit()
                        val_loss += loss.item()

                # Log performance
                print_and_log(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.3e}, "
                              f"Val Loss: {val_loss / len(val_loader):.3e}")
                true_train_loss = train_loss / len(train_loader)
                data = {
                  "sequence_length": sequence_length,
                  "hidden_size": hidden_size,
                  "learning_rate": learning_rate,
                  "epoch": epoch + 1,
                  "train_loss": train_loss / len(train_loader),
                  "val_loss": val_loss / len(val_loader)
                  }

                with open(f"run_data_attempt_{attempt_number}.json", "a") as f:
                  json.dump(data, f, indent=2)


                # Define file path
                
                base_dir = os.path.join(os.getcwd(),"MultipleRuns")


                # Define model directory inside Google Drive
                model_dir = os.path.join(base_dir, f"attempt_{attempt_number}/hidden_size_{hidden_size}/learning_rate_{learning_rate}/seq_len_{sequence_length}")
                os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist

                model_path = os.path.join(model_dir, f"lstm_model_hidden_{hidden_size}_lr_{learning_rate}_seq_len_{sequence_length}_epoch_{epoch + 1}.pth")

                # Save model checkpoint
                torch.save(model.state_dict(), model_path)
                print_and_log(f"Checkpoint saved to {model_path}")

                if true_train_loss < best_loss:
                  best_loss = true_train_loss
                  best_model_dir = os.path.join(base_dir, f"attempt_{attempt_number}/best_models")
                  os.makedirs(best_model_dir, exist_ok=True)  # Create directory if it doesn't exist
                  
                  best_model_name =  f"lstm_model_hidden_{hidden_size}_lr_{learning_rate}_seq_len_{sequence_length}_epoch_{epoch + 1}.pth"
                  best_model_path = os.path.join(best_model_dir,best_model_name)
                  torch.save(model.state_dict(), best_model_path)
                  
                  # Save best model metadata to a JSON summary file
                  best_models_json_path = os.path.join(base_dir, f"attempt_{attempt_number}", "best_model_per_config.json")
                  
                  # Load or initialize best model summary
                  if os.path.exists(best_models_json_path):
                      with open(best_models_json_path, "r") as f:
                          best_models_data = json.load(f)
                  else:
                      best_models_data = {}

                  # Create a unique key per config
                  config_key = f"seq_{sequence_length}_hidden_{hidden_size}_lr_{learning_rate}"

                  # Check if this is the best so far for the config
                  if (config_key not in best_models_data) or (true_train_loss < best_models_data[config_key]["best_train_loss"]):
                      best_models_data[config_key] = {
                          "sequence_length": sequence_length,
                          "hidden_size": hidden_size,
                          "learning_rate": learning_rate,
                          "epoch": epoch + 1,
                          "best_train_loss": true_train_loss,
                          "model_path": best_model_name
                      }

                      # Save updated summary
                      with open(best_models_json_path, "w") as f:
                          json.dump(best_models_data, f, indent=2)


