# scripts/attractors_orig_vs_att.py
#
# 1) بارگذاری datasetOrig.pt → دنرمال → رسم اترکتور واقعی (3D)
# 2) بارگذاری datasetAtt.pt → گرفتن نقاط اولیه → حل Neural ODE با FTheta
#    → دنرمال → رسم اترکتور مدل برای دسته‌ی دوم
# 3) نمایش هر دو سری در یک شکل سه‌بعدی Plotly با دو رنگ متفاوت
# 4) ذخیره‌ی شکل به صورت HTML تعاملی

import sys
from pathlib import Path

import torch
import numpy as np
from torchdiffeq import odeint
import plotly.graph_objects as go

# =====================================================================
# تنظیم مسیر ریشه‌ی پروژه
# =====================================================================
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]  # اگر فایل را در scripts/ قرار دهی، parents[1] می‌شود ریشه‌ی پروژه

sys.path.append(str(ROOT))

# =====================================================================
# مسیرها (مطابق آنچه خودت دادی)
# =====================================================================
ROOT = Path(__file__).resolve().parents[2]    # رفتن به ریشه پروژه
sys.path.append(str(ROOT))

f_theta_path   = ROOT / "models" / "f_theta.py"  # فقط برای اطلاع؛ ایمپورت از models.f_theta است
model_path     = ROOT / "models" / "3D_SIR3.pth"
dataset_orig_path = ROOT / "data" / "processed" / "3D_SIR_Orig.pt"
dataset_att_path  = ROOT / "data" / "processed" / "3D_SIR_Attr.pt"


from models.f_theta import FTheta  # تعریف شبکه

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================================
# تابع دنرمال‌سازی کمکی
# =====================================================================
def denormalize(data, mean, std):
    """
    data: (N, T, D) یا (T, D)
    mean/std: (1, 1, D) یا (D,)
    """
    if mean is None or std is None:
        return data  # فرض: داده‌ها از قبل real هستند
    # broadcast خود PyTorch کار را انجام می‌دهد
    return data * std + mean

# =====================================================================
# 1) بارگذاری datasetOrig.pt و رسم اترکتور واقعی
# =====================================================================
print(f"Loading original dataset from: {dataset_orig_path}")
ds_orig = torch.load(dataset_orig_path, map_location=device)

data_orig = ds_orig["data"].float().to(device)  # انتظار: (num_runs, T, D)
mean_orig = ds_orig.get("mean", None)
std_orig  = ds_orig.get("std", None)

if mean_orig is not None:
    mean_orig = mean_orig.to(device).float()
if std_orig is not None:
    std_orig = std_orig.to(device).float()

num_runs_orig, num_steps_orig, num_features_orig = data_orig.shape
print(f"datasetOrig: runs={num_runs_orig}, steps={num_steps_orig}, features={num_features_orig}")

if num_features_orig != 3:
    raise ValueError(f"datasetOrig must have 3 state variables for 3D plotting, but found {num_features_orig}")

# دنرمالیزه کردن داده‌ها (اگر mean/std موجود باشد)
data_orig_real = denormalize(data_orig, mean_orig, std_orig)  # (N, T, 3)

# =====================================================================
# 2) بارگذاری مدل آموزش‌دیده
# =====================================================================
print(f"Loading trained model from: {model_path}")
f_theta = FTheta(input_dim=num_features_orig).to(device)
state_dict = torch.load(model_path, map_location=device)
f_theta.load_state_dict(state_dict)
f_theta.eval()
print("Model loaded and set to eval mode.")

# =====================================================================
# 3) بارگذاری datasetAtt.pt و حل ODE از روی نقاط اولیه
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

if num_features_att != 3:
    raise ValueError(f"datasetAtt must also have 3 state variables, but found {num_features_att}")

# اگر time در ds_att نبود، یک time یکنواخت می‌سازیم
if time_att is None:
    print("No 'time' in datasetAtt, creating a uniform time grid [0, 1].")
    time_att = torch.linspace(0.0, 1.0, steps=num_steps_att, device=device)
else:
    time_att = time_att.to(device).float()

# نقاط اولیه‌ی نرمال‌شده برای هر تراژکتوری در datasetAtt
h0_att = data_att[:, 0, :]  # شکل: (num_runs_att, D)

# حل ODE برای هر run جداگانه
print("Integrating Neural ODE trajectories from initial conditions of datasetAtt...")
traj_list = []
with torch.no_grad():
    for i in range(num_runs_att):
        h0 = h0_att[i]  # (D,)
        # odeint خروجی: (T, D)
        h_traj = odeint(f_theta, h0, time_att, method="rk4")
        traj_list.append(h_traj.unsqueeze(0))  # (1, T, D)

# h_pred: (num_runs_att, num_steps_att, D)
h_pred = torch.cat(traj_list, dim=0)

# دنرمال‌سازی تراژکتوری‌های پیش‌بینی‌شده
h_pred_real = denormalize(h_pred, mean_att, std_att)  # (num_runs_att, T, 3)


# =====================================================================
# 4) رسم سه‌بعدی با Plotly: اترکتور واقعی + اترکتور مدل روی datasetAtt
# =====================================================================
fig = go.Figure()

# 4-1) اترکتور واقعی (datasetOrig)
for i in range(num_runs_orig):
    run_real = data_orig_real[i].detach().cpu().numpy()
    x = run_real[:, 0]
    y = run_real[:, 1]
    z = run_real[:, 2]

    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(width=5, color="blue"),
            opacity=0.7,
            showlegend=False
        )
    )

# 4-2) اترکتور مدل (datasetAtt)
for i in range(num_runs_att):
    run_model = h_pred_real[i].detach().cpu().numpy()
    x = run_model[:, 0]
    y = run_model[:, 1]
    z = run_model[:, 2]

    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(width=5, color="red"),
            opacity=0.7,
            showlegend=False
        )
    )

fig.update_layout(
    title="Attractors: Original vs Neural ODE (No Legend)",
    scene=dict(
        xaxis_title="state 1",
        yaxis_title="state 2",
        zaxis_title="state 3",
    ),
    width=1600,
    height=1300,
    showlegend=False   # ← لجند کامل حذف شد
)

# ذخیره
output_html = ROOT / "attractors_orig_vs_att_SIR.html"
fig.write_html(str(output_html))
print(f"Interactive 3D figure saved to: {output_html}")

fig.show()
