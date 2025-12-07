# --------------------------------------------
# CLEAN VECTOR FIELD FOR NEURAL ODE (ARROWS)
# --------------------------------------------
import sys
from pathlib import Path
import numpy as np
import torch
import plotly.graph_objects as go

# --- add project root to path ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from models.f_theta1_256 import FTheta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- load model -----
model = FTheta(input_dim=3).to(device)
state = torch.load(ROOT / "models" / "autonomous_DrivativeOnEval.pth",
                   map_location=device)
if isinstance(state, dict) and "state_dict" in state:
    model.load_state_dict(state["state_dict"])
else:
    model.load_state_dict(state)
model.eval()

# ----- vector field sampling -----
grd_size = 10
xs = np.linspace(-10, 10, grd_size)
ys = np.linspace(-10, 10, grd_size)
zs = np.linspace(-10, 10, grd_size)

X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N, 3)

# evaluate neural ODE
with torch.no_grad():
    h = torch.tensor(pts, dtype=torch.float32, device=device)
    t0 = torch.zeros((h.shape[0],), dtype=torch.float32, device=device)
    d = model(t0, h).cpu().numpy()

U, V, W = d[:, 0], d[:, 1], d[:, 2]

# ---- scale down the arrows ----
scale = 0.1
U = U * scale
V = V * scale
W = W * scale

# ---- start and end of shafts ----
X0 = pts[:, 0]
Y0 = pts[:, 1]
Z0 = pts[:, 2]

X1 = X0 + U
Y1 = Y0 + V
Z1 = Z0 + W

# ---- build line segments with NaN breaks ----
xs_lines, ys_lines, zs_lines = [], [], []
for x0, y0, z0, x1, y1, z1 in zip(X0, Y0, Z0, X1, Y1, Z1):
    xs_lines += [x0, x1, None]
    ys_lines += [y0, y1, None]
    zs_lines += [z0, z1, None]

# ---- direction and length for arrow heads ----
lengths = np.sqrt(U**2 + V**2 + W**2) + 1e-8
dir_x = U / lengths
dir_y = V / lengths
dir_z = W / lengths

head_frac = 0.12            
head_len = lengths * head_frac

cone_u = dir_x * head_len
cone_v = dir_y * head_len
cone_w = dir_z * head_len

fig = go.Figure()

# بدنه‌ی فلش‌ها (shaft)
fig.add_trace(
    go.Scatter3d(
        x=xs_lines,
        y=ys_lines,
        z=zs_lines,
        mode="lines",
        line=dict(width=3, color="blue"),
        showlegend=False,
    )
)


fig.add_trace(
    go.Cone(
        x=X1,
        y=Y1,
        z=Z1,
        u=cone_u,
        v=cone_v,
        w=cone_w,
        anchor="tip",          
        sizemode="absolute",
        sizeref=scale * 1.5,    
        showscale=False,
        colorscale=[[0, "blue"], [1, "blue"]],
        opacity=1.0,
    )
)

fig.update_layout(
    title="Neural ODE Vector Field (3D Arrows)",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z",
        aspectmode="data",
    )
)

fig.show()
