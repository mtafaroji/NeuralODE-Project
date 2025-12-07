# scripts/vectorField/vectorField_normalized_2D.py

import sys
from pathlib import Path
import numpy as np
import torch
import plotly.graph_objects as go  
import matplotlib.pyplot as plt     # for quiver

# =====================================================================
# Set paths
# =====================================================================
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[2]
sys.path.append(str(ROOT))

from models.f_theta import FTheta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset_path = ROOT / "data" / "processed" / "3BasinBoth72.pt"

#model_path   = ROOT / "models" / "2basin1FL0Der2.pth"
#model_path   = ROOT / "models" / "2Basin1_then_basin2FL0Der2.pth"
model_path   = ROOT / "models" / "3BasinBothDisL0LD20Round7.pth"

dataset = torch.load(dataset_path, map_location=device)
data = dataset["data"].float()

_, _, num_features = data.shape
if num_features != 2:
    raise ValueError("Your dataset must have 2 state variables.")

f_theta = FTheta(input_dim=2).to(device)
f_theta.load_state_dict(torch.load(model_path, map_location=device))
f_theta.eval()

# =====================================================================
# Grid
# =====================================================================
grid_minX, grid_maxX = -3.0, 3.0
grid_minY, grid_maxY = -3.0, 3.0
num_pointsX = 30
num_pointsY = 30

x_vals = np.linspace(grid_minX, grid_maxX, num_pointsX, dtype=np.float32)
y_vals = np.linspace(grid_minY, grid_maxY, num_pointsY, dtype=np.float32)
X, Y = np.meshgrid(x_vals, y_vals)

XY = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
XY_torch = torch.from_numpy(XY).to(device)

# =====================================================================
# Derivative dh/dt = f_theta(h)
# =====================================================================
with torch.no_grad():
    dH = f_theta(torch.tensor(0.0, device=device), XY_torch).cpu().numpy()

U = dH[:, 0]
V = dH[:, 1]

# =====================================================================
# Normalize vectors (direction only)
# =====================================================================
mag = np.sqrt(U**2 + V**2) + 1e-12
U_norm = 4*U / mag
V_norm = 4*V / mag

# =====================================================================
# Plot normalized vector field
# =====================================================================
plt.figure(figsize=(7.5, 7.5))

# Arrows on the normalized grid
# The larger the scale, the shorter the arrows become.
scale_value = 15  

plt.quiver(
    XY[:, 0], XY[:, 1],   # starting points of arrows (normalized grid)
    U_norm, V_norm,       # arrow directions (normalized)
    angles="xy",
    scale_units="xy",
    scale=scale_value,
    width=0.003,
    color="blue",
)

# Grid points
plt.scatter(XY[:, 0], XY[:, 1], s=10, c="red", alpha=0.4)

plt.title("Normalized Vector Field (Neural ODE, 2D)")
plt.xlabel("state 1 (normalized)")
plt.ylabel("state 2 (normalized)")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()
