#
# Read real trajectory from run1.csv,
# normalize with mean/std, compute dh/dt from the model,
# denormalize to dx/dt, and plot standard arrows along the trajectory
# using matplotlib.quiver().

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# =====================================================================
# Set paths
# =====================================================================
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[2]  # C:\...\NeuralODE-Project
sys.path.append(str(ROOT))

from models.f_theta import FTheta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Files
dataset_path = ROOT / "data" / "processed" / "TwoBasinWith48PointsDataSet.pt"
model_path   = ROOT / "models" / "TwoBasinTrainedOver72Points.pth"
csv_path     = ROOT / "data" / "raw" / "TwoBasins" / "run1.csv"

print("Loading processed dataset from:", dataset_path)
dataset = torch.load(dataset_path, map_location=device)

data_norm = dataset["data"].float()   # just to check dimensions
mean      = dataset["mean"].float()
std       = dataset["std"].float()

_, _, num_features = data_norm.shape
if num_features != 2:
    raise ValueError("This script assumes 2D states (stock1, stock2).")

mean_np = mean.cpu().numpy().reshape(-1)  # (2,)
std_np  = std.cpu().numpy().reshape(-1)   # (2,)

print("mean (real):", mean_np)
print("std  (real):", std_np)

# =====================================================================
# Load model
# =====================================================================
print("Initializing model FTheta with input_dim = 2")
f_theta = FTheta(input_dim=2).to(device)

print("Loading model weights from:", model_path)
state_dict = torch.load(model_path, map_location=device)
f_theta.load_state_dict(state_dict)
f_theta.eval()
print("Model loaded and set to eval mode.")

# =====================================================================
# Load real trajectory from CSV
# =====================================================================
print("Loading trajectory from CSV:", csv_path)
df = pd.read_csv(csv_path)

required_cols = ["stock1", "stock2"]
if not all(c in df.columns for c in required_cols):
    raise ValueError(f"CSV must contain columns: {required_cols}")

# Real trajectory: (T, 2)
X_real = df[required_cols].values.astype(np.float32)
T = X_real.shape[0]
print("Trajectory length (time steps):", T)

step = 3      
X_real_ds = X_real[::step]
print("Number of points used for quiver:", X_real_ds.shape[0])

# =====================================================================
# Normalize trajectory points for model input: h = (x - mean) / std
# =====================================================================
H_norm = (X_real_ds - mean_np) / std_np         # (N, 2)
H_norm_torch = torch.from_numpy(H_norm).to(device)

# =====================================================================
# Compute dh/dt and then dx/dt in real scale
# =====================================================================
with torch.no_grad():
    dHdt = f_theta(torch.tensor(0.0, device=device), H_norm_torch).cpu().numpy()

# dx/dt = std * dh/dt
dXdt_real = std_np * dHdt                      # (N, 2)

U_real = dXdt_real[:, 0]
V_real = dXdt_real[:, 1]

# =====================================================================
mag = np.sqrt(U_real**2 + V_real**2) + 1e-12
U_dir = U_real * 5#/ mag
V_dir = V_real * 5#/ mag

scale_value = 15  # quiver scale factor

# =====================================================================
# Plot with matplotlib and quiver
# =====================================================================
plt.figure(figsize=(8, 8))

# Full trajectory
plt.plot(X_real[:, 0], X_real[:, 1], "k-", lw=1.5, label="Trajectory (real)")

# Arrows on subsampled points
plt.quiver(
    X_real_ds[:, 0], X_real_ds[:, 1],   # starting points
    U_dir, V_dir, 
    color="blue",                       # arrow directions
    angles="xy",
    scale_units="xy",
    scale=scale_value,                  
    width=0.003,                        # arrow line thickness
)

plt.xlabel("state 1 (real scale)")
plt.ylabel("state 2 (real scale)")
plt.title("Vector Field Along Real Trajectory (run1.csv) - Matplotlib Quiver")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
