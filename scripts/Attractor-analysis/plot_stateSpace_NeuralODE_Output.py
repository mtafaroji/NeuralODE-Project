import os
import glob
import pandas as pd
import torch
from torchdiffeq import odeint
import plotly.graph_objects as go

import sys
from pathlib import Path
# --- add project root to path ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from models.f_theta import FTheta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# -------------------------------------------------------------
# 1) Load dataset (for normalization stats + time vector)
# -------------------------------------------------------------
dataset = torch.load("data/processed/dataset.pt", map_location=device)

time = dataset["time"].float().to(device)          # (T,)
mean = dataset["mean"].squeeze().float().to(device)  # (3,)
std  = dataset["std"].squeeze().float().to(device)   # (3,)

# -------------------------------------------------------------
# 2) Load trained Neural ODE
# -------------------------------------------------------------
model = FTheta(input_dim=3).to(device)
state_dict = torch.load("models/neural_ode_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# -------------------------------------------------------------
# 3) Load ALL initial conditions (REAL scale) from CSV files
# -------------------------------------------------------------
RAW_DIR = "data/raw"
files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))

if len(files) == 0:
    raise SystemExit("No CSV files in data/raw")

print("CSV files found:", len(files))

x0_list = []
dfs = []   

for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

    x0 = df.loc[0, ["stock1", "stock2", "stock3"]].values.astype("float32")
    x0 = torch.tensor(x0, device=device)
    x0_list.append(x0)

# -------------------------------------------------------------
# 4) Solve Neural ODE for each initial condition
# -------------------------------------------------------------
all_traj_real = []

with torch.no_grad():
    for x0_real in x0_list:
        x0_norm = (x0_real - mean) / std  # normalize (3,)

        sol_norm = odeint(model, x0_norm, time, method="dopri5")

        # remove batch dimension if exists
        if sol_norm.dim() == 3:
            sol_norm = sol_norm[:, 0, :]   # (T,3)

        # denormalize
        sol_real = sol_norm * std + mean   # (T,3)
        all_traj_real.append(sol_real.cpu().numpy())

# -------------------------------------------------------------
# 5) Plot exactly like your AnyLogic plotting code
# -------------------------------------------------------------
cutoff_time = 30   # For Transient State: cutoff_time = None

fig = go.Figure()

for i, (traj, df_ref) in enumerate(zip(all_traj_real, dfs), start=1):

    if cutoff_time is not None:
        t_ref = df_ref["time"].values
        cut_idx = (t_ref >= cutoff_time).nonzero()[0][0]
        traj_cut = traj[cut_idx:]
    else:
        traj_cut = traj

    fig.add_trace(go.Scatter3d(
        x=traj_cut[:, 0],
        y=traj_cut[:, 1],
        z=traj_cut[:, 2],
        mode="lines",
        name=f"run {i}",
        line=dict(width=4)
    ))

fig.update_layout(
    title="Neural ODE - 3D State-Space Trajectories (Denormalized, All Runs)",
    scene=dict(
        xaxis_title="stock1",
        yaxis_title="stock2",
        zaxis_title="stock3",
    )
)

fig.show()


# fig.write_html("neuralODE_like_anylogic.html")
