import torch
import torch.nn as nn
from torchdiffeq import odeint
from pathlib import Path
import sys

#from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import our model
from models.f_theta3_256 import FTheta



# ------------------- Device -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------- Load processed data ---------------------
dataset = torch.load("data/processed/TwoBasinWith72PointsDataSet.pt")                     #### Changed name of DataSet file ############################
data = dataset['data'].float().to(device)       # shape: (num_runs, time_steps, features)     
time = dataset['time'].float().to(device)       # shape: (time_steps,)  




num_runs, num_steps, num_features = data.shape
num_train_runs = num_runs - 5 # reserve last 3 for testing

# ---------- Initialize model, optimizer, and loss ----------
f_theta = FTheta(input_dim=num_features).to(device)
#f_theta.load_state_dict(torch.load("models/3BasinBothDisL0LD20Round6.pth"))   #### Changed name of Loaded model ##########################

############# Layer freezing ########

#linear1 = f_theta.net[0] # First Linear
#linear2 = f_theta.net[2] # Second Linear
#linear3 = f_theta.net[4] # 3rd Linear

'''
for param in list(linear1.parameters()) + list(linear2.parameters()): # + list(linear3.parameters()):
    param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, f_theta.parameters()),
    5e-5
)
'''

optimizer = torch.optim.Adam(f_theta.parameters(), lr=5e-5) # No freezing


#####################################################################################

mse = nn.MSELoss()

lambda_deriv = 2.0  # We can tune this hyperparameter to balance the two loss terms

# ----------Training loop ------------------------------------
batch_size = 20
num_epochs = 800
for epoch in range(num_epochs):
    total_loss = 0.0
    total_batches = 0

        # A simple way to shuffle training runs each epoch
    perm = torch.randperm(num_train_runs, device=device)
   
    for start in range(0, num_train_runs, batch_size):
        #end = start + batch_size
        end = min(start + batch_size, num_train_runs)
        #batch = data[start:end]
        idx = perm[start:end]                # Indices of runs in this batch
        batch = data[idx]  
                          #  (B, T, D)
        h0_batch = batch[:, 0, :]
        h_true = data[start]                   
        #h0 = h_true[0]       
        B = batch.shape[0]

        # Initial Values for trajectories
        h0_batch = batch[:, 0, :]            # (B, D)
        # Predict trajectory

        h_pred = odeint(f_theta, h0_batch, time, method='rk4')
        h_pred = h_pred.permute(1, 0, 2) 


        # Compute loss OLD
      
        # Loss on trajectory
        loss_traj = mse(h_pred, batch)


        # Calculation of derivative with central difference method
        # batch: (B, T, D)
        x_prev = batch[:, :-2, :]   # x_{k-1},  (B, T-2, D)
        x_next = batch[:,  2:, :]   # x_{k+1},  (B, T-2, D)

        dt_center = (time[2:] - time[:-2]).view(1, -1, 1)   # (1, T-2, 1)
        dxdt_true = (x_next - x_prev) / dt_center           # (B, T-2, D)

        # Calculation of predicted derivative for middle points between t_1 to t_{T-2}
                
        x_mid = batch[:, 1:-1, :]                      # (B, T-2, D)
        t_mid = time[1:-1]                                  # (T-2,)

        dxdt_pred_list = []
        for k in range(num_steps - 2):
            t_k = t_mid[k]
            #h_k = h_mid[:, k, :]                            # (B, D)
            x_k = x_mid[:, k, :]                            # (B, D)
            dxdt_k = f_theta(t_k, x_k)                      # (B, D)
            dxdt_pred_list.append(dxdt_k.unsqueeze(1))      # (B, 1, D)

        dxdt_pred = torch.cat(dxdt_pred_list, dim=1)        # (B, T-2, D)

        # Loss on derivatives
        loss_deriv = mse(dxdt_pred, dxdt_true)

        # Total loss
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

torch.save(f_theta.state_dict(), "models/2D_2BasinOver72PointTraining.pth")  #### Changed name of saved model ############################
print("Model saved to models/2D_2BasinOver72PointTraining.pth")