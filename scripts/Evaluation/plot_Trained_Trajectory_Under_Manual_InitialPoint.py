import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from models.f_theta import FTheta
# ------------------- Device -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# ----------- Load Tensor dataset -------------
dataset_path = "data/processed/TwoBasinWith48PointsDataSet.pt"     ###########################@@@@@@ Load DataSet Path
dataset = torch.load(dataset_path)

data = dataset['data'].float().to(device)
time = dataset['time'].float().to(device)
mean = dataset['mean']                       # shape: (1,1,D)
std  = dataset['std']                        # shape: (1,1,D)
num_runs, num_steps, num_features = data.shape

# ----------- Load trained model ------------
model_path = "models/TwoBasinTrainedOver72Points.pth" ###########################@@@@@@ Load Model Path


# ---- Prepare mean/std on CPU for manual normalization ----
mean_cpu = mean.squeeze(0).squeeze(0).cpu()   # shape: (D,)
std_cpu  = std.squeeze(0).squeeze(0).cpu()    # shape: (D,)




# =====================================================
#          MANUAL INITIAL CONDITION (REAL SPACE)
# =====================================================
# The system has 2 state variables
x0_real = -2.5
y0_real = -1.3

h0_real = torch.tensor([[x0_real, y0_real]], dtype=torch.float32)   # shape: (1, D)

# ---------- Normalize BEFORE sending to network ----------
h0 = (h0_real - mean_cpu) / std_cpu
h0 = h0.to(device)

print("Manual initial condition (real):", h0_real)
print("Manual initial condition (normalized):", h0)


# ---------- Load trained model ----------
model = FTheta(input_dim=num_features).to(device)
state = torch.load(model_path, map_location=device)
if isinstance(state, dict) and "state_dict" in state:
    model.load_state_dict(state["state_dict"])
else:
    model.load_state_dict(state)
model.eval()


# ---------- Predict trajectory from the manual h0 ----------
with torch.no_grad():
    h_pred = odeint(model, h0, time, method='rk4')
    h_pred = h_pred.squeeze(1)

h_pred = h_pred.cpu()

# ---------- Denormalize ----------
h_pred_denorm = h_pred * std_cpu + mean_cpu


# ---------- Plot ----------
import matplotlib.pyplot as plt
time_cpu = time.cpu()

plt.figure(figsize=(10, 6))
for i in range(num_features):
    plt.subplot(num_features, 1, i+1)
    plt.plot(time_cpu.numpy(), h_pred_denorm[:, i].numpy(),
             'r-', label=f"Predicted Stock {i+1}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

plt.tight_layout()
plt.show()
