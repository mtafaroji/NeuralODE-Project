# scripts/vectorField/vectorField_trajectory_quiver_2D.py
#
# خواندن ترژکتوری واقعی از run1.csv،
# نرمال‌سازی با mean/std، گرفتن dh/dt از مدل،
# دنرمال کردن به dx/dt، و رسم پیکان‌های استاندارد روی ترژکتوری
# با استفاده از matplotlib.quiver (نوک پیکان استاندارد و قابل زوم).

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
ROOT = THIS_FILE.parents[2]  # C:\...\NeuralODE-Project
sys.path.append(str(ROOT))

from models.f_theta import FTheta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# فایل‌ها
dataset_path = ROOT / "data" / "processed" / "2Basin1.pt"
model_path   = ROOT / "models" / "2basin1FL0Der2.pth"
csv_path     = ROOT / "data" / "raw" / "2Basin1" / "run1.csv"

print("Loading processed dataset from:", dataset_path)
dataset = torch.load(dataset_path, map_location=device)

data_norm = dataset["data"].float()   # فقط برای چک ابعاد
mean      = dataset["mean"].float()
std       = dataset["std"].float()

_, _, num_features = data_norm.shape
if num_features != 2:
    raise ValueError("This script assumes 2D states (stock1, stock2).")

mean_np = mean.cpu().numpy().reshape(-1)  # (2,)
std_np  = std.cpu().numpy().reshape(-1)   # (2,)

print("mean (real):", mean_np)
print("std  (real):", std_np)

# =====================================================================
# بارگذاری مدل
# =====================================================================
print("Initializing model FTheta with input_dim = 2")
f_theta = FTheta(input_dim=2).to(device)

print("Loading model weights from:", model_path)
state_dict = torch.load(model_path, map_location=device)
f_theta.load_state_dict(state_dict)
f_theta.eval()
print("Model loaded and set to eval mode.")

# =====================================================================
# خواندن ترژکتوری واقعی از CSV
# =====================================================================
print("Loading trajectory from CSV:", csv_path)
df = pd.read_csv(csv_path)

required_cols = ["stock1", "stock2"]
if not all(c in df.columns for c in required_cols):
    raise ValueError(f"CSV must contain columns: {required_cols}")

# ترژکتوری واقعی: (T, 2)
X_real = df[required_cols].values.astype(np.float32)
T = X_real.shape[0]
print("Trajectory length (time steps):", T)

# برای جلوگیری از شلوغی، هر چند نقطه یک‌بار فلش رسم کنیم
step = 3      # اگر می‌خواهی همه نقاط فلش داشته باشند، این را بگذار 1
X_real_ds = X_real[::step]
print("Number of points used for quiver:", X_real_ds.shape[0])

# =====================================================================
# نرمال‌سازی نقاط ترژکتوری برای ورودی مدل: h = (x - mean) / std
# =====================================================================
H_norm = (X_real_ds - mean_np) / std_np         # (N, 2)
H_norm_torch = torch.from_numpy(H_norm).to(device)

# =====================================================================
# محاسبه‌ی dh/dt و سپس dx/dt در مقیاس واقعی
# =====================================================================
with torch.no_grad():
    dHdt = f_theta(torch.tensor(0.0, device=device), H_norm_torch).cpu().numpy()

# dx/dt = std * dh/dt
dXdt_real = std_np * dHdt                      # (N, 2)

U_real = dXdt_real[:, 0]
V_real = dXdt_real[:, 1]

# =====================================================================
# می‌توانیم یا:
#  - از magnitude واقعی استفاده کنیم تا طول فلش‌ها متناسب با اندازه مشتق باشد
#  - یا جهت را نرمال کنیم تا همه طول‌ها تقریباً یکسان شوند
# در این‌جا جهت را نرمال می‌کنیم تا شکل خواناتر شود.
# =====================================================================
mag = np.sqrt(U_real**2 + V_real**2) + 1e-12
U_dir = U_real * 5#/ mag
V_dir = V_real * 5#/ mag

# quiver خودش scale را مدیریت می‌کند؛ با scale و scale_units می‌توانی طول فلش‌ها را تنظیم کنی.
# هرچه scale بزرگ‌تر باشد فلش‌ها کوتاه‌تر می‌شوند.
scale_value = 15  # اگر فلش‌ها خیلی کوتاه/بلند بودند، این عدد را تغییر بده.

# =====================================================================
# رسم با matplotlib و quiver
# =====================================================================
plt.figure(figsize=(8, 8))

# ترژکتوری کامل
plt.plot(X_real[:, 0], X_real[:, 1], "k-", lw=1.5, label="Trajectory (real)")

# پیکان‌ها روی زیرمجموعه‌ی نقاط
plt.quiver(
    X_real_ds[:, 0], X_real_ds[:, 1],   # نقاط شروع
    U_dir, V_dir, 
    color="blue",                      # جهت فلش‌ها
    angles="xy",
    scale_units="xy",
    scale=scale_value,                  # با این عدد می‌توانی طول نسبی فلش را تنظیم کنی
    width=0.003,                        # ضخامت خط فلش
)

plt.xlabel("state 1 (real scale)")
plt.ylabel("state 2 (real scale)")
plt.title("Vector Field Along Real Trajectory (run1.csv) - Matplotlib Quiver")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
