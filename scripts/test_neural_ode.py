import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


from models.f_theta import FTheta


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

mean = dataset['mean']   # شکل: (1, 1, D)
std  = dataset['std']    # شکل: (1, 1, D)


num_runs, num_steps, num_features = data.shape

########################################
########################################
h_true = data[2]
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



mean_cpu = mean.squeeze(0).squeeze(0).cpu()  # شکل: (D,)
std_cpu  = std.squeeze(0).squeeze(0).cpu()   # شکل: (D,)

# denormalization:
# x_real = x_norm * std + mean

h_true_denorm = h_true * std_cpu + mean_cpu      # شکل: (T, D)
h_pred_denorm = h_pred * std_cpu + mean_cpu      # شکل: (T, D)




plt.figure(figsize=(10, 6))
for i in range(num_features):
    plt.subplot(num_features, 1, i+1)
    plt.plot(time_cpu.numpy(),
             h_true_denorm[:, i].numpy(),
             'b-', label=f"True Stock {i+1}")
    plt.plot(time_cpu.numpy(),
             h_pred_denorm[:, i].numpy(),
             'r--', label=f"Predicted Stock {i+1}")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")



plt.tight_layout()
plt.show()