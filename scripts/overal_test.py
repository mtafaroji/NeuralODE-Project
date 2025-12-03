import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


from models.f_theta import FTheta


# ------------------- Device -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# ---------- Load trained model ---------- #######################@@@@@@@@ Load Model
model_path = "models/drivativeOnly2.pth"

#----------- Load Tensor dataSet -------------####################@@@@@@ Load DataSet
dataset_path = "data/processed/dataset2.pt"

# Load dataset
dataset = torch.load(dataset_path)
data = dataset['data'].float().to(device)
time = dataset['time'].float().to(device)

mean = dataset['mean']   # شکل: (1, 1, D)
std  = dataset['std']    # شکل: (1, 1, D)


num_runs, num_steps, num_features = data.shape



# Load model
f_theta = FTheta(input_dim=num_features).to(device)
f_theta.load_state_dict(torch.load(model_path))
f_theta.eval()



# ----------------- تنظیم شکل Plotly -----------------
n_rows = num_runs * num_features   # برای هر run و هر feature یک ردیف

# فاصله عمودی صفر (مشکل محدودیت را حل می‌کند)
fig = make_subplots(
    rows=n_rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.0
)

row = 1
for run_idx, run in enumerate(data):
    h_true = run
    h0 = h_true[0].unsqueeze(0)

    with torch.no_grad():
        h_pred = odeint(f_theta, h0, time, method='rk4').squeeze(1)

    h_true = h_true.cpu()
    h_pred = h_pred.cpu()
    time_cpu = time.cpu()

    mean_cpu = mean.squeeze(0).squeeze(0).cpu()
    std_cpu  = std.squeeze(0).squeeze(0).cpu()

    h_true_denorm = h_true * std_cpu + mean_cpu
    h_pred_denorm = h_pred * std_cpu + mean_cpu

    for i in range(num_features):
        fig.add_trace(
            go.Scatter(
                x=time_cpu.numpy(),
                y=h_true_denorm[:, i].numpy(),
                mode="lines",
                name=f"Run {run_idx+1} - True Stock {i+1}",
                line=dict(color="blue")
            ),
            row=row, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=time_cpu.numpy(),
                y=h_pred_denorm[:, i].numpy(),
                mode="lines",
                name=f"Run {run_idx+1} - Predicted Stock {i+1}",
                line=dict(color="red", dash="dash")
            ),
            row=row, col=1
        )

        fig.update_yaxes(
            title_text=f"Run {run_idx+1} - Stock {i+1}",
            row=row, col=1
        )

        row += 1

fig.update_xaxes(title_text="Time", row=n_rows, col=1)

fig.update_layout(
    height=250 * n_rows,   # این عدد ارتفاع هر پلات را کنترل می‌کند
    width=900,
    showlegend=False
)

# ذخیره‌ی خروجی HTML
out_path = "Draivative_NonAutonomous2.html" ####################@@@@@@@@ Change the name of output file
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Saved HTML file to {out_path}")