# scripts/vectorField/vectorField_denormalized_2D.py

import sys
from pathlib import Path
import numpy as np
import torch
import plotly.graph_objects as go  # دیگر استفاده نمی‌شود ولی حذفش نمی‌کنم
import matplotlib.pyplot as plt     # برای quiver

# =====================================================================
# تنظیم مسیرها (مطابق ساختار پروژه‌ی شما)
# =====================================================================
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[2]
sys.path.append(str(ROOT))

from models.f_theta import FTheta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_path   = ROOT / "models" / "3BasinBothDisL0LD20Round7.pth"
dataset_path = ROOT / "data" / "processed" / "3BasinBoth72.pt"

print("Loading dataset (for mean/std) from:", dataset_path)
dataset = torch.load(dataset_path, map_location=device)

data_norm = dataset["data"].float()
mean      = dataset["mean"].float()
std       = dataset["std"].float()

_, _, num_features = data_norm.shape
if num_features != 2:
    raise ValueError("Your dataset must have 2 state variables (features=2).")

mean_np = mean.cpu().numpy().reshape(-1)
std_np  = std.cpu().numpy().reshape(-1)

print("mean (real scale):", mean_np)
print("std  (real scale):", std_np)

# =====================================================================
# بازه‌ی واقعی انتخاب‌شده توسط کاربر
# =====================================================================
x1_min_plot = -5.0
x1_max_plot = 5.0
x2_min_plot = -5.0
x2_max_plot = 5.0

num_points = 30

# =====================================================================
# بارگذاری مدل
# =====================================================================
print("Initializing model FTheta with input_dim = 2")
f_theta = FTheta(input_dim=2).to(device)

print("Loading model state from:", model_path)
state_dict = torch.load(model_path, map_location=device)
f_theta.load_state_dict(state_dict)
f_theta.eval()
print("Model loaded and set to eval mode.")

# =====================================================================
# ساخت گرید واقعی
# =====================================================================
x1_vals = np.linspace(x1_min_plot, x1_max_plot, num_points, dtype=np.float32)
x2_vals = np.linspace(x2_min_plot, x2_max_plot, num_points, dtype=np.float32)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

X_real = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float32)

# =====================================================================
# تبدیل گرید واقعی به نرمال‌شده
# =====================================================================
H_norm = (X_real - mean_np) / std_np
H_norm_torch = torch.from_numpy(H_norm).to(device)

# =====================================================================
# dh/dt و سپس dx/dt
# =====================================================================
with torch.no_grad():
    dHdt = f_theta(torch.tensor(0.0, device=device), H_norm_torch).cpu().numpy()

dXdt_real = std_np * dHdt

U_real = dXdt_real[:, 0]
V_real = dXdt_real[:, 1]

# =====================================================================
# نرمال‌سازی جهت فلش‌ها برای visualization
# =====================================================================
mag = np.sqrt(U_real**2 + V_real**2) + 1e-12
U_dir = 4*U_real / mag
V_dir = 4*V_real / mag

# =====================================================================
# رسم با matplotlib + quiver
# =====================================================================
plt.figure(figsize=(7.5, 7.5))

# فلش‌ها
scale_value = 10   # اگر کوتاه/بلند بودند این را تغییر بده

plt.quiver(
    X_real[:, 0], X_real[:, 1],   # نقاط شروع فلش‌ها
    U_dir, V_dir,                 # جهت‌ها
    angles="xy",
    scale_units="xy",
    scale=scale_value,
    width=0.003,
    color="blue",
)

# گرید واقعی (اختیاری)
plt.scatter(X_real[:, 0], X_real[:, 1], s=10, c="red", alpha=0.4)

plt.title("Denormalized Vector Field (user-defined real range)")
plt.xlabel("state 1 (real scale)")
plt.ylabel("state 2 (real scale)")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()
