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
dataset_path = "data/processed/dataset1.pt"
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
num_train_runs = num_runs - 3
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

num_epochs = 300

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

        # odeint over batch:
        # خروجی: (WINDOW, batch, D)
        h_pred = odeint(f_theta, h0_batch, t_window, method='rk4')

        #   (batch, WINDOW, D)
        h_pred = h_pred.permute(1, 0, 2)

        # loss  over batch
        loss = loss_fn(h_pred, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)


    if epoch % 20 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] - Avg window loss: {avg_loss:.6f}")

# ------------------- Save trained model -----------
model_path = "models/neural_ode_model_windowed_split1.pth"
torch.save(f_theta.state_dict(), model_path)
print("Saved trained model to:", model_path)



'''

###############################################
# quick eval loss on held-out runs -------------
#calculation of acuracy of model over test data
###############################################
f_theta.eval()
with torch.no_grad():
    eval_loss = 0.0
    eval_windows = 0
    for r in test_run_indices:
        for s in start_indices:
            e = s + WINDOW
            h_true = data[r, s:e, :]
            h0 = h_true[0].unsqueeze(0)
            t_window = time_full[s:e]

            h_pred = odeint(f_theta, h0, t_window, method='rk4')
            if h_pred.dim() == 3:
                h_pred = h_pred.squeeze(1)

            loss = loss_fn(h_pred, h_true)
            eval_loss += loss.item()
            eval_windows += 1

    if eval_windows > 0:
        print(f"Held-out runs avg window loss: {eval_loss / eval_windows:.6f}")
    else:
        print("No held-out windows to evaluate.")
'''