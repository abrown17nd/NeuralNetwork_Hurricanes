import json

import pandas as pd
import matplotlib.pyplot as plt

# Read the improperly formatted JSON file - fix from file creation
file_name = "run_data_attempt_6.json"
with open(file_name, "r") as f:
    raw = f.read()

# Split into individual JSON objects (assuming one per line or separated cleanly)
# Ensure braces are used correctly to split
import re
objects = re.findall(r'\{.*?\}', raw, re.DOTALL)

# Parse each object
parsed = [json.loads(obj) for obj in objects]

# Save as a proper JSON array
with open(file_name, "w") as f:
    json.dump(parsed, f, indent=2)


df = pd.read_json("run_data_attempt_6.json")

# Validation or training
loss_type = "val_loss"
fig, ax = plt.subplots(figsize=(8,6))
for (seq_len, hid_size), group in df[df["learning_rate"] == 0.001].groupby(['sequence_length', 'hidden_size']):
    group.plot("epoch", loss_type, ax=ax, label=f"Seq {seq_len}, Hidden {hid_size}")

ax.legend(title="Config")
plt.title(f"All {loss_type} ")
plt.show()