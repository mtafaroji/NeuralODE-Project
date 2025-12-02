# scripts/vectorField/attractor_original_vs_neural.py
#
# مقایسه‌ی اتراکتور:
#   - تراژکتوری‌های اصلی از CSV
#   - تراژکتوری‌های تولید شده توسط Neural ODE آموزش‌دیده
#
# اگر D=2 → رسم روی صفحه‌ی x-y
# اگر D=3 → رسم در فضای سه‌بعدی x-y-z
# شکل با Plotly قابل چرخاندن و ذخیره‌کردن است.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchdiffeq import odeint
import plotly.graph_objects as go

# =====================================================================
# تنظیم مسیرها مطابق ساختار پروژه
# =====================================================================
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[2]  # C:\...\NeuralODE-Project
sys.path.append(str(ROOT))

# مسیرها (مطابق چیزی که دادی)
model_def_path = ROOT / "models" / "f_theta.py"  # فقط برای اطلاعات؛ ایمپورت از models.f_theta
model_path     = ROOT / "models" / "3D_SIR2.pth"
dataset_path   = ROOT / "data" / "processed" / "3D_SIR.pt"
csv_dir        = ROOT / "data" / "raw" / "3D_SIR"  # مسیر فایل‌های run*.csv 

print("Project root:", ROOT)
print("Model file:", model_path)
print("Dataset file:", dataset_path)
print("CSV directory:", csv_dir)

from models.f_theta import FTheta  # تعریف f_theta

# =====================================================================
# Device
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================================
# Load processed dataset (normalized data + time + mean/std + file list)
# =====================================================================
dataset = torch.load(dataset_path, map_location=device)

data_norm = dataset["data"].float()   # (num_runs, T, D) - normalized
time      = dataset["time"].float()   # (T,)
mean      = dataset["mean"].float()   # (1, 1, D)
std       = dataset["std"].float()    # (1, 1, D)
files_in_dataset = dataset.get("files", None)  # لیست فایل‌ها (اگر ذخیره شده باشد)

num_runs, num_steps, num_features = data_norm.shape
print("num_runs:", num_runs, "num_steps:", num_steps, "num_features:", num_features)

if num_features not in (2, 3):
    raise ValueError(f"This script only handles 2D or 3D states, but got D={num_features}")

mean_np = mean.cpu().numpy().reshape(-1)  # (D,)
std_np  = std.cpu().numpy().reshape(-1)   # (D,)

# =====================================================================
# Load trained model
# =====================================================================
f_theta = FTheta(input_dim=num_features).to(device)
state_dict = torch.load(model_path, map_location=device)
f_theta.load_state_dict(state_dict)
f_theta.eval()
print("Model loaded and set to eval mode.")

# =====================================================================
# CSV files (original trajectories)
# =====================================================================
if files_in_dataset is not None and len(files_in_dataset) == num_runs:
    # ترجیحاً از لیست ذخیره شده در دیتاست استفاده می‌کنیم تا ترتیب‌ها دقیقاً یکی باشد
    csv_files = [Path(p) for p in files_in_dataset]
    print("Using file list from dataset['files'].")
else:
    # در غیر این صورت، از مسیر داده شده glob می‌کنیم
    csv_files = sorted(csv_dir.glob("run*.csv"))
    if len(csv_files) != num_runs:
        print(
            f"WARNING: number of CSV files ({len(csv_files)}) "
            f"differs from num_runs in dataset ({num_runs}). "
            f"Proceeding with min of the two."
        )
    # تطبیق طول
    min_len = min(len(csv_files), num_runs)
    csv_files = csv_files[:min_len]
    data_norm = data_norm[:min_len]
    num_runs = min_len

print(f"Number of trajectories to plot: {num_runs}")

# =====================================================================
# Helper: generate neural ODE trajectory from normalized initial state
# =====================================================================
def generate_neural_traj(h0_norm: torch.Tensor, t: torch.Tensor) -> np.ndarray:
    """
    h0_norm: (D,) normalized initial state
    t:      (T,) time tensor
    returns: (T, D) trajectory in REAL scale (denormalized)
    """
    h0_norm = h0_norm.to(device)
    t = t.to(device)

    with torch.no_grad():
        h_t = odeint(f_theta, h0_norm, t, method="rk4")  # (T, D)
    h_t = h_t.cpu().numpy()  # normalized

    # denormalize: x = mean + std * h
    h_real = mean_np + std_np * h_t  # broadcasting (T, D)
    return h_real

# =====================================================================
# Collect trajectories (original and neural)
# =====================================================================
orig_trajs = []   # list of np.ndarray, each (T_orig, D)
neural_trajs = [] # list of np.ndarray, each (T_model, D) using same time grid as dataset

# تعیین ستون‌ها متناسب با D
if num_features == 2:
    stock_cols = ["stock1", "stock2"]
elif num_features == 3:
    stock_cols = ["stock1", "stock2", "stock3"]

for run_idx in range(num_runs):
    csv_path = csv_files[run_idx]
    print(f"[{run_idx+1}/{num_runs}] Loading CSV:", csv_path)

    df = pd.read_csv(csv_path)
    if not all(c in df.columns for c in stock_cols):
        raise ValueError(f"CSV {csv_path} must contain columns: {stock_cols}")

    X_orig = df[stock_cols].values.astype(np.float32)  # (T_orig, D)
    orig_trajs.append(X_orig)

    # initial condition from normalized dataset (t=0)
    h0_norm = data_norm[run_idx, 0, :]  # (D,)
    X_neural = generate_neural_traj(h0_norm, time)     # (T_model, D) real scale
    neural_trajs.append(X_neural)

# =====================================================================
# Plotting with Plotly
# =====================================================================
if num_features == 2:
    # ------------------ 2D phase portrait (x-y) ----------------------
    fig = go.Figure()

    # 1) Original trajectories (black)
    for i, X_orig in enumerate(orig_trajs):
        show_legend = (i == 0)
        fig.add_trace(
            go.Scatter(
                x=X_orig[:, 0],
                y=X_orig[:, 1],
                mode="lines",
                line=dict(color="black", width=1.5),
                name="Original trajectories" if show_legend else None,
                showlegend=show_legend,
            )
        )

    # 2) Neural ODE trajectories (red, dashed)
    for i, X_neural in enumerate(neural_trajs):
        show_legend = (i == 0)
        fig.add_trace(
            go.Scatter(
                x=X_neural[:, 0],
                y=X_neural[:, 1],
                mode="lines",
                line=dict(color="red", width=1.5, dash="dot"),
                name="Neural ODE trajectories" if show_legend else None,
                showlegend=show_legend,
            )
        )

    fig.update_layout(
        title="Attractor: Original vs Neural ODE Trajectories (2D)",
        xaxis_title="state 1",
        yaxis_title="state 2",
        width=800,
        height=800,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

else:
    # ------------------ 3D phase portrait (x-y-z) --------------------
    fig = go.Figure()

    # 1) Original trajectories (black)
    for i, X_orig in enumerate(orig_trajs):
        show_legend = (i == 0)
        fig.add_trace(
            go.Scatter3d(
                x=X_orig[:, 0],
                y=X_orig[:, 1],
                z=X_orig[:, 2],
                mode="lines",
                line=dict(color="black", width=3),
                name="Original trajectories" if show_legend else None,
                showlegend=show_legend,
            )
        )

    # 2) Neural ODE trajectories (red, dashed)
    for i, X_neural in enumerate(neural_trajs):
        show_legend = (i == 0)
        fig.add_trace(
            go.Scatter3d(
                x=X_neural[:, 0],
                y=X_neural[:, 1],
                z=X_neural[:, 2],
                mode="lines",
                line=dict(color="red", width=3),
                name="Neural ODE trajectories" if show_legend else None,
                showlegend=show_legend,
            )
        )

    fig.update_layout(
        title="Attractor: Original vs Neural ODE Trajectories (3D)",
        scene=dict(
            xaxis_title="state 1",
            yaxis_title="state 2",
            zaxis_title="state 3",
            aspectmode="cube",
        ),
        width=900,
        height=800,
    )

# حالت Plotly به صورت تعاملی قابل چرخاندن و زوم است، و از نوار ابزار می‌توان شکل را ذخیره کرد.
fig.show()
