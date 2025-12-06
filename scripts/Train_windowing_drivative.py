import torch
import torch.nn as nn
from torchdiffeq import odeint
from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.f_theta8 import FTheta   # Model that takes time as input


# ------------------- Device -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------- Load dataset -----------------
dataset_path = "data/processed/datasetEval.pt"
dataset = torch.load(dataset_path)

# data shape: (num_runs, num_steps, num_features)
data = dataset["data"].float().to(device)
time_full = dataset["time"].float().to(device)

num_runs, num_steps, num_features = data.shape
print(f"num_runs = {num_runs}, num_steps = {num_steps}, num_features = {num_features}")

if num_runs <= 3:
    raise ValueError("We need at least 4 runs to hold out the last 3 for testing.")

# ------------------- Train / Test split -----------
# Last three runs are held out for testing
num_train_runs = num_runs - 4
train_run_indices = list(range(num_train_runs))
test_run_indices = list(range(num_train_runs, num_runs))

print("Train runs:", train_run_indices)
print("Test runs (held-out):", test_run_indices)

# ------------------- Windowing parameters ----------
# Windowing is used to create smaller segments of the time series for training
WINDOW = 10  # Length of each window

start_indices = list(range(0, num_steps - WINDOW + 1, WINDOW))
num_windows_per_run = len(start_indices)
print(f"WINDOW = {WINDOW}, num_windows_per_run = {num_windows_per_run}")
print("Train runs:", train_run_indices)
print("Test runs (held-out):", test_run_indices)

# ------------------- Model / Optimizer ------------
f_theta = FTheta(input_dim=num_features).to(device)
optimizer = torch.optim.Adam(f_theta.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# وزنِ لاس مشتق (مثل کد اول)
lambda_deriv = 0.0

num_epochs = 1000

# ------------------- Training loop ----------------
for epoch in range(1, num_epochs + 1):
    f_theta.train()
    total_loss = 0.0
    total_batches = 0

    # A simple shuffle of start indices each epoch to decrease correlation
    s_perm = torch.randperm(len(start_indices))

    for s_idx in s_perm:
        s = start_indices[s_idx.item()]
        e = s + WINDOW

        # ----------------------------
        # batch of all training runs for this window
        # ----------------------------
        #  (num_train_runs, WINDOW, D)
        batch = data[train_run_indices, s:e, :]         

        # Get initial state for trajectories (num_train_runs, D)
        h0_batch = batch[:, 0, :]

        # time window : (WINDOW,)
        t_window = time_full[s:e]

        # odeint over batch: (WINDOW, batch, D)
        h_pred = odeint(f_theta, h0_batch, t_window, method='rk4')

        #   (batch, WINDOW, D)
        h_pred = h_pred.permute(1, 0, 2)

        # ----------------------------
        # 1) Loss روی تراژکتوری
        # ----------------------------
        loss_traj = loss_fn(h_pred, batch)

        # ----------------------------
        # 2) Loss روی مشتق (central difference)
        #    مشابه کد اول، ولی روی همان پنجره
        # ----------------------------
        # batch: (B, T, D)  با T = WINDOW
        B, T, D = batch.shape

        # اگر طول پنجره کمتر از 3 باشد، مشتق مرکزی ممکن نیست
        if T > 2:
            # x_{k-1} و x_{k+1}
            x_prev = batch[:, :-2, :]   # (B, T-2, D)
            x_next = batch[:,  2:, :]   # (B, T-2, D)

            # dt در وسط‌ها
            dt_center = (t_window[2:] - t_window[:-2]).view(1, -1, 1)   # (1, T-2, 1)
            dxdt_true = (x_next - x_prev) / dt_center                   # (B, T-2, D)

            # نقاط وسط برای ورودی به f_theta
            x_mid = batch[:, 1:-1, :]   # (B, T-2, D)
            t_mid = t_window[1:-1]      # (T-2,)

            dxdt_pred_list = []
            for k in range(T - 2):
                t_k = t_mid[k]                 # اسکالر
                x_k = x_mid[:, k, :]           # (B, D)
                dxdt_k = f_theta(t_k, x_k)     # (B, D)
                dxdt_pred_list.append(dxdt_k.unsqueeze(1))  # (B, 1, D)

            dxdt_pred = torch.cat(dxdt_pred_list, dim=1)    # (B, T-2, D)

            loss_deriv = loss_fn(dxdt_pred, dxdt_true)
        else:
            # اگر T <= 2، مشتق مرکزی معنی ندارد؛ اینجا صفر می‌گیریم
            loss_deriv = torch.tensor(0.0, device=device)

        # ----------------------------
        # 3) Total loss
        # ----------------------------
        loss = loss_traj + lambda_deriv * loss_deriv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)

    if epoch % 20 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] - "
              f"Avg window loss (traj + {lambda_deriv}*deriv): {avg_loss:.6f}")

# ------------------- Save trained model -----------
model_path = "models/NonAutonomous_DrivativeOff00.pth"
torch.save(f_theta.state_dict(), model_path)
print("Saved trained model to:", model_path)

