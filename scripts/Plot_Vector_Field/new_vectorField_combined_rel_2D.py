# scripts/vectorField/vectorField_combined_real_2D.py
#
# 1) میدان برداری روی گرید در مقیاس واقعی (denormalized) با quiver
# 2) تمام ترژکتوری‌های واقعی از فایل‌های run*.csv در data/raw/2Basin1
# 3) فلش‌های dx/dt روی هر ترژکتوری
# سه رنگ:
#   - میدان روی گرید: آبی روشن
#   - ترژکتوری‌ها: مشکی
#   - فلش‌های روی ترژکتوری‌ها: قرمز

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# =====================================================================
# تنظیم مسیرها
# =====================================================================
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[2]  # ...\NeuralODE-Project
sys.path.append(str(ROOT))

from models.f_theta import FTheta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)





############################################################################
# مدل و دیتاست نرمال‌شده برای mean/std

#dataset_path = ROOT / "data" / "processed" / "2Basin1.pt"
#model_path   = ROOT / "models" / "2basin1FL0Der2.pth"
# ترژکتوری واقعی
#csv_dir     = ROOT / "data" / "raw" / "2Basin1"  


model_path   = ROOT / "models" / "2basin1FL0Der2.pth"
dataset_path = ROOT / "data" / "processed" / "2Basin1.pt"
# ترژکتوری واقعی
csv_dir     = ROOT / "data" / "raw" / "2Basin1" 


#model_path   = ROOT / "models" / "2Basin1_then_basin2FL0Der2.pth"
#dataset_path = ROOT / "data" / "processed" / "2Basin1.pt"
# ترژکتوری واقعی
#csv_dir     = ROOT / "data" / "raw" / "2Basin1" 

# =====================================================================
# بازه‌ی واقعی برای گرید (دستی تنظیم کن)
# =====================================================================
x1_min_plot = -6.0
x1_max_plot = 0.0
x2_min_plot = -6.0
x2_max_plot = 6.0

num_points = 50  # تعداد نقاط گرید در هر محور
#######################################################################






print("Loading processed dataset (for mean/std) from:", dataset_path)
dataset = torch.load(dataset_path, map_location=device)

data_norm = dataset["data"].float()
mean      = dataset["mean"].float()
std       = dataset["std"].float()

_, _, num_features = data_norm.shape
if num_features != 2:
    raise ValueError("Your dataset must have 2 state variables (features=2).")

mean_np = mean.cpu().numpy().reshape(-1)  # (2,)
std_np  = std.cpu().numpy().reshape(-1)   # (2,)

print("mean (real scale):", mean_np)
print("std  (real scale):", std_np)



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
# 1) میدان برداری روی گرید واقعی
# =====================================================================
x1_vals = np.linspace(x1_min_plot, x1_max_plot, num_points, dtype=np.float32)
x2_vals = np.linspace(x2_min_plot, x2_max_plot, num_points, dtype=np.float32)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# نقاط واقعی گرید: (N, 2)
X_grid_real = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float32)

# تبدیل گرید به فضای نرمال برای مدل
H_grid_norm = (X_grid_real - mean_np) / std_np
H_grid_norm_torch = torch.from_numpy(H_grid_norm).to(device)

with torch.no_grad():
    dHdt_grid = f_theta(torch.tensor(0.0, device=device), H_grid_norm_torch).cpu().numpy()

# dx/dt روی گرید واقعی
dXdt_grid_real = std_np * dHdt_grid
U_grid_real = dXdt_grid_real[:, 0]
V_grid_real = dXdt_grid_real[:, 1]

# جهت نرمال‌شده برای فلش‌ها (طول ثابت، جهت واقعی)
mag_grid = np.sqrt(U_grid_real**2 + V_grid_real**2) + 1e-12
U_grid_dir = 3 * U_grid_real / mag_grid
V_grid_dir = 3 * V_grid_real / mag_grid

# =====================================================================
# 2) خواندن تمام ترژکتوری‌های واقعی و محاسبه بردارهای dx/dt روی آن‌ها
# =====================================================================
csv_files = sorted(csv_dir.glob("run*.csv"))
if not csv_files:
    raise SystemExit(f"No CSV files matching 'run*.csv' found in {csv_dir}")

print(f"Found {len(csv_files)} trajectory files in {csv_dir}")

# تنظیمات مشترک برای رسم
step = 3          # هر چند نقطه یک‌بار فلش روی هر ترژکتوری
traj_color = "black"
traj_arrow_color = "red"

# برای legend فقط یک‌بار label می‌گذاریم
first_traj_line = True
first_traj_arrows = True

# =====================================================================
# رسم ترکیبی
# =====================================================================
plt.figure(figsize=(8, 8))

# 1) میدان برداری روی گرید (آبی روشن)
scale_grid = 10   # اگر فلش‌ها خیلی کوتاه/بلند شدند، این را تغییر بده

plt.quiver(
    X_grid_real[:, 0], X_grid_real[:, 1],
    U_grid_dir, V_grid_dir,
    angles="xy",
    scale_units="xy",
    scale=scale_grid,
    width=0.003,
    color="cornflowerblue",
    alpha=0.9,
    label="Vector field (grid)",
)

# 2) حلقه روی تمام ترژکتوری‌ها
for csv_path in csv_files:
    print("Loading real trajectory from CSV:", csv_path)
    df = pd.read_csv(csv_path)

    required_cols = ["stock1", "stock2"]
    if not all(c in df.columns for c in required_cols):
        raise ValueError(f"CSV {csv_path} must contain columns: {required_cols}")

    X_traj_real = df[required_cols].values.astype(np.float32)
    T = X_traj_real.shape[0]
    print("  Trajectory length (time steps):", T)

    # ترژکتوری کامل برای رسم خط
    if first_traj_line:
        plt.plot(
            X_traj_real[:, 0],
            X_traj_real[:, 1],
            color=traj_color,
            linewidth=2.5,
            alpha=0.9,
            label="Trajectories (real)",
        )
        first_traj_line = False
    else:
        plt.plot(
            X_traj_real[:, 0],
            X_traj_real[:, 1],
            color=traj_color,
            linewidth=1.0,
            alpha=0.6,
        )

    # نقاط زیرنمونه‌شده برای فلش‌ها
    X_traj_real_ds = X_traj_real[::step]

    # نرمال‌سازی نقاط ترژکتوری برای مدل
    H_traj_norm = (X_traj_real_ds - mean_np) / std_np
    H_traj_norm_torch = torch.from_numpy(H_traj_norm).to(device)

    with torch.no_grad():
        dHdt_traj = f_theta(torch.tensor(0.0, device=device), H_traj_norm_torch).cpu().numpy()

    dXdt_traj_real = std_np * dHdt_traj
    U_traj_real = dXdt_traj_real[:, 0]
    V_traj_real = dXdt_traj_real[:, 1]

    mag_traj = np.sqrt(U_traj_real**2 + V_traj_real**2) + 1e-12
    U_traj_dir = U_traj_real / mag_traj
    V_traj_dir = V_traj_real / mag_traj

    # مقیاس طول فلش روی ترژکتوری خاص
    range_x_traj = X_traj_real[:, 0].max() - X_traj_real[:, 0].min()
    range_y_traj = X_traj_real[:, 1].max() - X_traj_real[:, 1].min()
    base_scale_traj = max(range_x_traj, range_y_traj) if max(range_x_traj, range_y_traj) > 0 else 1.0

    arrow_length_traj = 0.05 * base_scale_traj
    Ux_traj = U_traj_dir * arrow_length_traj
    Vy_traj = V_traj_dir * arrow_length_traj

    # فلش‌ها روی ترژکتوری
    if first_traj_arrows:
        plt.quiver(
            X_traj_real_ds[:, 0],
            X_traj_real_ds[:, 1],
            Ux_traj,
            Vy_traj,
            angles="xy",
            scale_units="xy",
            scale=1.0,        # چون خودمان طول Ux/Vy را تنظیم کرده‌ایم
            width=0.004,
            color=traj_arrow_color,
            alpha=0.9,
            label="dx/dt on trajectories",
        )
        first_traj_arrows = False
    else:
        plt.quiver(
            X_traj_real_ds[:, 0],
            X_traj_real_ds[:, 1],
            Ux_traj,
            Vy_traj,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
            color=traj_arrow_color,
            alpha=0.7,
        )

# =====================================================================
# تنظیمات نهایی شکل
# =====================================================================
plt.xlabel("state 1 (real scale)")
plt.ylabel("state 2 (real scale)")
plt.title("Vector Field (grid) + Real Trajectories + Arrows Along Trajectories")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
