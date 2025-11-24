# --------------------------------------------
# scripts/vectorField/vectorField_planes_neuralODE_arrows.py
# Real 2D arrows on XY, XZ, YZ (Neural ODE)
# --------------------------------------------
import sys
from pathlib import Path
import numpy as np
import torch
import plotly.graph_objects as go

# --- add project root ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from models.f_theta import FTheta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----- load model -----
model = FTheta(input_dim=3).to(device)
state = torch.load(ROOT / "models" / "neural_ode_model.pth", map_location=device)
if isinstance(state, dict) and "state_dict" in state:
    model.load_state_dict(state["state_dict"])
else:
    model.load_state_dict(state)
model.eval()

# -------- settings --------
grid_n = 10

x_min, x_max = 70, 80
y_min, y_max =  3, 9
z_min, z_max =  15, 20

z_plane = 0.0   # for XY
y_plane = 0.0   # for XZ
x_plane = 0.0   # for YZ

arrow_len = 1.0  # طول فلش‌ها در واحدهای داده (هرچه کمتر، فلش کوتاه‌تر)
head_size = 1    # اندازه نوک فلش در Plotly (integer 1..5)

def neural_f(pts_np):
    with torch.no_grad():
        h = torch.tensor(pts_np, dtype=torch.float32, device=device)
        t0 = torch.zeros((h.shape[0],), dtype=torch.float32, device=device)
        d = model(t0, h).cpu().numpy()
    return d

def plot_plane_arrows(xg, yg, ug, vg, title, xlabel, ylabel):
    # نرمال‌سازی جهت برای اینکه فقط direction نمایش داده شود
    mag = np.sqrt(ug**2 + vg**2) + 1e-9
    ug_n = ug / mag
    vg_n = vg / mag

    fig = go.Figure()

    # برای داشتن محورهای خالی (بدون خط)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers"))

    # فلش‌ها به صورت annotation
    annotations = []
    for x0, y0, u0, v0 in zip(xg.ravel(), yg.ravel(), ug_n.ravel(), vg_n.ravel()):
        x1 = x0 + arrow_len * u0
        y1 = y0 + arrow_len * v0
        annotations.append(
            dict(
                x=x1, y=y1, ax=x0, ay=y0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=head_size,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor="royalblue"
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        annotations=annotations,
        width=750,
        height=650
    )
    fig.show()

# =========================
# XY plane (Z fixed)
# =========================
xs = np.linspace(x_min, x_max, grid_n)
ys = np.linspace(y_min, y_max, grid_n)
X, Y = np.meshgrid(xs, ys, indexing="ij")
Z = np.full_like(X, z_plane)

pts_xy = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
d_xy = neural_f(pts_xy)

Uxy = d_xy[:,0].reshape(X.shape)
Vxy = d_xy[:,1].reshape(X.shape)

plot_plane_arrows(X, Y, Uxy, Vxy,
                  title=f"Neural ODE Vector Field on XY (Z={z_plane})",
                  xlabel="x (stockX)", ylabel="y (stockY)")


# =========================
# XZ plane (Y fixed)
# =========================
xs = np.linspace(x_min, x_max, grid_n)
zs = np.linspace(z_min, z_max, grid_n)
X, Z = np.meshgrid(xs, zs, indexing="ij")
Y = np.full_like(X, y_plane)

pts_xz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
d_xz = neural_f(pts_xz)

Uxz = d_xz[:,0].reshape(X.shape)
Wxz = d_xz[:,2].reshape(X.shape)

plot_plane_arrows(X, Z, Uxz, Wxz,
                  title=f"Neural ODE Vector Field on XZ (Y={y_plane})",
                  xlabel="x (stockX)", ylabel="z (stockZ)")


# =========================
# YZ plane (X fixed)
# =========================
ys = np.linspace(y_min, y_max, grid_n)
zs = np.linspace(z_min, z_max, grid_n)
Y, Z = np.meshgrid(ys, zs, indexing="ij")
X = np.full_like(Y, x_plane)

pts_yz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
d_yz = neural_f(pts_yz)

Vyz = d_yz[:,1].reshape(Y.shape)
Wyz = d_yz[:,2].reshape(Y.shape)

plot_plane_arrows(Y, Z, Vyz, Wyz,
                  title=f"Neural ODE Vector Field on YZ (X={x_plane})",
                  xlabel="y (stockY)", ylabel="z (stockZ)")
