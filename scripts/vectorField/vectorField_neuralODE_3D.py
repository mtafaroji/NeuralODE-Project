# --------------------------------------------
# CLEAN VECTOR FIELD FOR NEURAL ODE
# --------------------------------------------
import sys
from pathlib import Path
import numpy as np
import torch
import plotly.graph_objects as go

# --- add project root to path ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from models.f_theta import FTheta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- load model -----
model = FTheta(input_dim=3).to(device)
state = torch.load(ROOT / "models" / "neural_ode_model.pth", map_location=device)
if isinstance(state, dict) and "state_dict" in state:
    model.load_state_dict(state["state_dict"])
else:
    model.load_state_dict(state)
model.eval()

# ----- vector field sampling -----
# coarse grid (keep small!)
xs = np.linspace(60, 80, 6)
ys = np.linspace(1,  20, 6)
zs = np.linspace(10,  24, 6)

X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

# evaluate neural ODE
with torch.no_grad():
    h = torch.tensor(pts, dtype=torch.float32, device=device)
    t0 = torch.zeros((h.shape[0],), dtype=torch.float32, device=device)
    d = model(t0, h).cpu().numpy()

U, V, W = d[:, 0], d[:, 1], d[:, 2]

# ---- scale down the arrows dramatically ----
scale = 0.05   # reduce vector size
U = U * scale
V = V * scale
W = W * scale

# ---- PLOTLY CONE PLOT ----
fig = go.Figure(
    go.Cone(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        u=U, v=V, w=W,
        sizemode="absolute",
        sizeref=0.1,    # smaller arrows
        anchor="center", # center arrows
        colorscale="Viridis",
        showscale=False
    )
)

fig.update_layout(
    title="Neural ODE Vector Field (Clean & Readable)",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="z"
    )
)

fig.show()
