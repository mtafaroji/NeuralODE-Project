import torch
import torch.nn as nn
from torchdiffeq import odeint
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


# Import our model
from models.f_theta1 import FTheta

# ---------- Load processed data ----------
dataset = torch.load("data/processed/dataset.pt")
data = dataset['data']      
time = dataset['time']      

num_runs, num_steps, num_features = data.shape

# ---------- Initialize model, optimizer, and loss ----------
f_theta = FTheta(input_dim=num_features)
optimizer = torch.optim.Adam(f_theta.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ----------  Training loop ----------
num_epochs = 40
for epoch in range(num_epochs):
    total_loss = 0.0

    for i in range(num_runs):
        h_true = data[i]                   
        h0 = h_true[0]       

        # Predict trajectory

        #def func(t, h):
        #    return f_theta(h, t)

        h_pred = odeint(f_theta, h0, time)

      

        # Compute loss
        loss = loss_fn(h_pred, h_true)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_runs
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.6f}")

# ---------- 4. Save trained model ----------
Path("models").mkdir(exist_ok=True)
torch.save(f_theta.state_dict(), "models/neural_ode_model.pth")
print(" Training complete.")