import torch
import torch.nn as nn
from torchdiffeq import odeint
from pathlib import Path
import sys

#from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import our model
from models.f_theta import FTheta



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
loss_fn = nn.MSELoss()

# ----------Training loop ------------------------------------

num_epochs = 400
for epoch in range(num_epochs):
    total_loss = 0.0

    batch_size = 8
    for start in range(0, num_runs, batch_size):
        end = start + batch_size
        batch = data[start:end]
        h0_batch = batch[:, 0, :]
   

        # Predict trajectory

        h_pred = odeint(f_theta, h0_batch, time, method='rk4')
        h_pred = h_pred.permute(1, 0, 2) 
      

        # Compute loss
        loss = loss_fn(h_pred, batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_runs
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch}] - Loss: {avg_loss:.6f}")

# ---------- 4. Save trained model ----------

torch.save(f_theta.state_dict(), "models/neural_ode_model.pth")
