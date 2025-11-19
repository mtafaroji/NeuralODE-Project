import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


from models.f_theta1 import FTheta


# ------------------- Device -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# ---------- Load trained model ----------
model_path = "models/neural_ode_model.pth"
#model_path = "models/neural_ode_model_windowed.pth"

#model_path = "models/neural_ode_model_windowed_split.pth"
dataset_path = "data/processed/dataset.pt"

# Load dataset
dataset = torch.load(dataset_path)
data = dataset['data'].float().to(device)
time = dataset['time'].float().to(device)




num_runs, num_steps, num_features = data.shape

########################################
########################################
h_true = data[9]
h0 = h_true[0].unsqueeze(0)  # shape: (1, num_features)
########################################
########################################

# Load model
f_theta = FTheta(input_dim=num_features).to(device)
f_theta.load_state_dict(torch.load(model_path))
f_theta.eval()




# ---------- Predict the trajectory ----------
with torch.no_grad():
    #h_pred = odeint(f_theta, h0, time)  # shape: (num_steps, num_features)

    h_pred = odeint(f_theta, h0, time, method='rk4')
    h_pred = h_pred.squeeze(1)



h_true = h_true.cpu()
h_pred = h_pred.cpu()
time_cpu = time.cpu()


# ---------- Plot comparison ----------
plt.figure(figsize=(10, 6))
for i in range(num_features):
    plt.subplot(num_features, 1, i+1)
    plt.plot(time_cpu.numpy(), h_true[:, i].numpy(), 'b-', label=f"True Stock {i+1}")
    plt.plot(time_cpu.numpy(), h_pred[:, i].numpy(), 'r--', label=f"Predicted Stock {i+1}")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")

plt.tight_layout()
plt.show()