# 2 Dimention Attractor: Original vs Evaluation
#


import sys
from pathlib import Path

import torch
import numpy as np
from torchdiffeq import odeint
import plotly.graph_objects as go
from pathlib import Path
# --- add project root to path ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# =====================================================================
# Project root 
# =====================================================================
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[2]  # If the file is in scripts/Attractor_Analysis, parents[2] becomes the project root.

sys.path.append(str(ROOT))

# =====================================================================
# root paths and imports
# =====================================================================
f_theta_path      = ROOT / "models" / "f_theta.py"  # For information only; import is from models.f_theta
model_path        = ROOT / "models" / "TwoBasinTrainedOver72Points.pth"
dataset_orig_path = ROOT / "data" / "processed" / "TwoBasinWith48PointsDataSet.pt"
dataset_att_path  = ROOT / "data" / "processed" / "TwoBasinWith48PointsDataSet.pt"

from models.f_theta import FTheta  # Neural network definition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================================
# denormalization function
# =====================================================================
def denormalize(data, mean, std):
    """
    data: (N, T, D) or (T, D)
    mean/std: (1, 1, D) or (D,)
    """
    if mean is None or std is None:
        return data  # Assumption: data is already real-valued
    return data * std + mean

# =====================================================================
# 1) load original dataset for real attractor
# =====================================================================
print(f"Loading original dataset from: {dataset_orig_path}")
ds_orig = torch.load(dataset_orig_path, map_location=device)

data_orig = ds_orig["data"].float().to(device)  # : (num_runs, T, D)
mean_orig = ds_orig.get("mean", None)
std_orig  = ds_orig.get("std", None)

if mean_orig is not None:
    mean_orig = mean_orig.to(device).float()
if std_orig is not None:
    std_orig = std_orig.to(device).float()

num_runs_orig, num_steps_orig, num_features_orig = data_orig.shape
print(f"datasetOrig: runs={num_runs_orig}, steps={num_steps_orig}, features={num_features_orig}")

if num_features_orig < 2:
    raise ValueError(f"datasetOrig must have at least 2 state variables for 2D plotting, but found {num_features_orig}")

# 
data_orig_real = denormalize(data_orig, mean_orig, std_orig)  # (N, T, D)

# =====================================================================
# 2) load trained model 
# =====================================================================
print(f"Loading trained model from: {model_path}")
f_theta = FTheta(input_dim=num_features_orig).to(device)
state_dict = torch.load(model_path, map_location=device)
f_theta.load_state_dict(state_dict)
f_theta.eval()
print("Model loaded and set to eval mode.")

# =====================================================================
# 3) load evaluation dataset for model attractor
# =====================================================================
print(f"Loading attacked/second dataset from: {dataset_att_path}")
ds_att = torch.load(dataset_att_path, map_location=device)

data_att = ds_att["data"].float().to(device)  # (num_runs_att, T_att, D_att)
time_att = ds_att.get("time", None)

mean_att = ds_att.get("mean", None)
std_att  = ds_att.get("std", None)

if mean_att is not None:
    mean_att = mean_att.to(device).float()
if std_att is not None:
    std_att = std_att.to(device).float()

num_runs_att, num_steps_att, num_features_att = data_att.shape
print(f"datasetAtt: runs={num_runs_att}, steps={num_steps_att}, features={num_features_att}")

if num_features_att < 2:
    raise ValueError(f"datasetAtt must also have at least 2 state variables, but found {num_features_att}")

# 
if time_att is None:
    print("No 'time' in datasetAtt, creating a uniform time grid [0, 1].")
    time_att = torch.linspace(0.0, 1.0, steps=num_steps_att, device=device)
else:
    time_att = time_att.to(device).float()

# 
h0_att = data_att[:, 0, :]  # : (num_runs_att, D)

# Solve ODE for each run separately
print("Integrating Neural ODE trajectories from initial conditions of datasetAtt...")
traj_list = []
with torch.no_grad():
    for i in range(num_runs_att):
        h0 = h0_att[i]  # (D,)
        # odeint output: (T, D)
        h_traj = odeint(f_theta, h0, time_att, method="rk4")
        traj_list.append(h_traj.unsqueeze(0))  # (1, T, D)

# h_pred: (num_runs_att, num_steps_att, D)
h_pred = torch.cat(traj_list, dim=0)

# Denormalize predicted trajectories
h_pred_real = denormalize(h_pred, mean_att, std_att)  # (num_runs_att, T, D)

# =====================================================================
# 4) 2D plotting with Plotly: real attractor + model attractor on datasetAtt
#    (phase plot: state1 vs state2)
# =====================================================================
fig = go.Figure()

# 4-1) Real attractor (datasetOrig) - blue
for i in range(num_runs_orig):
    run_real = data_orig_real[i].detach().cpu().numpy()
    x = run_real[:, 0]  # state 1
    y = run_real[:, 1]  # state 2

    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(width=2, color="blue"),
            opacity=0.7,
            showlegend=False
        )
    )

# 4-2) Model attractor (datasetAtt) - red
for i in range(num_runs_att):
    run_model = h_pred_real[i].detach().cpu().numpy()
    x = run_model[:, 0]  # state 1
    y = run_model[:, 1]  # state 2

    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(width=2, color="red"),
            opacity=0.7,
            showlegend=False
        )
    )

fig.update_layout(
    title="2D Attractors: Original vs Neural ODE (state1 vs state2, No Legend)",
    xaxis_title="state 1",
    yaxis_title="state 2",
    width=1400,
    height=1000,
    showlegend=False
)

# Save
output_html = ROOT / "attractors_orig_vs_eval_Basin2_2D_3.html"
fig.write_html(str(output_html))
print(f"Interactive 2D figure saved to: {output_html}")

fig.show()
