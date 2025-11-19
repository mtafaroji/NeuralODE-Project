import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


from models.f_theta8 import FTheta

# ---------- Load trained model ----------
model_path = "models/neural_ode_model_windowed_split.pth"
dataset_path = "data/processed/dataset.pt"

# Load dataset
dataset = torch.load(dataset_path)
data = dataset['data']
time = dataset['time']

num_runs, num_steps, num_features = data.shape


h_true = data[4]
h0 = h_true[0]

# Load model
f_theta = FTheta(input_dim=num_features)
f_theta.load_state_dict(torch.load(model_path))
f_theta.eval()




# ---------- Predict the trajectory ----------
with torch.no_grad():
    h_pred = odeint(f_theta, h0, time)  # shape: (num_steps, num_features)

# ---------- Plot comparison ----------
plt.figure(figsize=(10, 6))
for i in range(num_features):
    plt.subplot(num_features, 1, i+1)
    plt.plot(time.numpy(), h_true[:, i].numpy(), 'b-', label=f"True Stock {i+1}")
    plt.plot(time.numpy(), h_pred[:, i].numpy(), 'r--', label=f"Predicted Stock {i+1}")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")

plt.tight_layout()
plt.show()