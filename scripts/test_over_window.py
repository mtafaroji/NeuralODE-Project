import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.f_theta8 import FTheta


# ------------------- Device -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- Load trained model ----------
model_path = "models/neural_ode_model_windowed_split.pth"
dataset_path = "data/processed/dataset.pt"

# Load dataset
dataset = torch.load(dataset_path)
data = dataset['data'].float().to(device)
time_full = dataset['time'].float().to(device)

num_runs, num_steps, num_features = data.shape

########################################
# Select trajectory to test
h_true_full = data[7]
h0 = h_true_full[0].unsqueeze(0)
########################################

# ----------- Limit to first 10 seconds -------------
# time_full assumed sorted like: 0, 1, 2, ..., 60 or small dt
mask = time_full <= 10.0           # keep only t <= 10
time_short = time_full[mask]       # time window (<= 10 sec)

h_true = h_true_full[mask]         # true states for <= 10 sec
# ----------------------------------------------------

# Load model
f_theta = FTheta(input_dim=num_features).to(device)
f_theta.load_state_dict(torch.load(model_path))
f_theta.eval()

# ---------- Predict only the 0 → 10 sec trajectory ----------
with torch.no_grad():
    h_pred = odeint(f_theta, h0, time_short, method='rk4')
    h_pred = h_pred.squeeze(1)


# Convert to CPU
h_true = h_true.cpu()
h_pred = h_pred.cpu()
time_plot = time_short.cpu()


# ---------- Plot comparison ----------
plt.figure(figsize=(10, 6))
for i in range(num_features):
    plt.subplot(num_features, 1, i+1)
    plt.plot(time_plot.numpy(), h_true[:, i].numpy(), 'b-', label=f"True Stock {i+1}")
    plt.plot(time_plot.numpy(), h_pred[:, i].numpy(), 'r--', label=f"Predicted Stock {i+1}")
    plt.legend()
    plt.xlabel("Time (0–10s)")
    plt.ylabel("Value")

plt.tight_layout()
plt.show()