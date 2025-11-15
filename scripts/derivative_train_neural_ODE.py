import torch
import torch.nn as nn
from torchdiffeq import odeint
from pathlib import Path
import sys

#from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import our model
from models.f_theta1 import FTheta



# ------------------- Device -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------- Load processed data ---------------------
dataset = torch.load("data/processed/dataset.pt")
data = dataset['data'].float().to(device)       # shape: (num_runs, time_steps, features)     
time = dataset['time'].float().to(device)       # shape: (time_steps,)  




num_runs, num_steps, num_features = data.shape

# ---------- Initialize model, optimizer, and loss ----------
f_theta = FTheta(input_dim=num_features).to(device)
optimizer = torch.optim.Adam(f_theta.parameters(), lr=1e-3)
mse = nn.MSELoss()

lambda_deriv = 0.1  # وزن لا‌س مشتق؛ می‌توانی بعداً باهاش بازی کنی

# ----------Training loop ------------------------------------
batch_size = 8
num_epochs = 1000
for epoch in range(num_epochs):
    total_loss = 0.0
    total_batches = 0

        # شافل ساده‌ی اندیس‌های ران‌ها (اختیاری ولی بهتر)
    perm = torch.randperm(num_runs, device=device)
   
    for start in range(0, num_runs, batch_size):
        #end = start + batch_size
        end = min(start + batch_size, num_runs)
        #batch = data[start:end]
        idx = perm[start:end]                # اندیس‌های این بچ
        batch = data[idx]  
                          # شکل: (B, T, D)
        h0_batch = batch[:, 0, :]
        h_true = data[start]                   
        #h0 = h_true[0]       
        B = batch.shape[0]

        # شرایط اولیه برای هر ران در این بچ
        h0_batch = batch[:, 0, :]            # شکل: (B, D)
        # Predict trajectory

        h_pred = odeint(f_theta, h0_batch, time, method='rk4')
        h_pred = h_pred.permute(1, 0, 2) 


        # Compute loss OLD
        #loss = loss_fn(h_pred, batch)


        # ---- 2. لا‌س ترازکتوری (مثل قبل) ----
        loss_traj = mse(h_pred, batch)


        # ---- 3. محاسبه‌ی مشتق واقعی با central difference ----
        # batch: (B, T, D)
        x_prev = batch[:, :-2, :]   # x_{k-1}, شکل: (B, T-2, D)
        x_next = batch[:,  2:, :]   # x_{k+1}, شکل: (B, T-2, D)

        dt_center = (time[2:] - time[:-2]).view(1, -1, 1)   # (1, T-2, 1)
        dxdt_true = (x_next - x_prev) / dt_center           # (B, T-2, D)

        # ---- 4. محاسبه‌ی مشتق پیش‌بینی‌شده از f_theta ----
        # h_pred: (B, T, D) → نقاط داخلی برای central difference:
        #h_mid = h_pred[:, 1:-1, :]                          # (B, T-2, D)
        x_mid = batch[:, 1:-1, :]                      # (B, T-2, D)
        t_mid = time[1:-1]                                  # (T-2,)

        # dxdt_pred[k] = f_theta(t_mid[k], h_mid[:, k, :])
        dxdt_pred_list = []
        for k in range(num_steps - 2):
            t_k = t_mid[k]
            #h_k = h_mid[:, k, :]                            # (B, D)
            x_k = x_mid[:, k, :]                            # (B, D)
            dxdt_k = f_theta(t_k, x_k)                      # (B, D)
            dxdt_pred_list.append(dxdt_k.unsqueeze(1))      # (B, 1, D)

        dxdt_pred = torch.cat(dxdt_pred_list, dim=1)        # (B, T-2, D)

        # ---- 5. لا‌س مشتق ----
        loss_deriv = mse(dxdt_pred, dxdt_true)

        # ---- 6. لا‌س نهایی ----
        loss = loss_traj + lambda_deriv * loss_deriv



        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    #avg_loss = total_loss / num_runs
    avg_loss = total_loss / max(total_batches, 1)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch}] - Loss: {avg_loss:.6f}")

# ---------- 4. Save trained model ----------

torch.save(f_theta.state_dict(), "models/neural_ode_model.pth")
print("Model saved to models/neural_ode_model.pth")