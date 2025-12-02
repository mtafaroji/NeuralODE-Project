# scripts/load_anylogic_runs.py
import os
import glob
import pandas as pd
import numpy as np
import torch

# ----------------------------
# files paths and directories
# ----------------------------
#RAW_DIR = "data/raw/basins_Composed"
RAW_DIR = "data/raw/3D_SIR"     ################################@@@@@@@@@@@@@@ Change the path for raw data
PROCESSED_DIR = "data/processed" #############################@@@@@@@@@@@@@@ Change the path for processed data
os.makedirs(PROCESSED_DIR, exist_ok=True)

# finding CSV files
files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))) #######@@@@@@@@ Define the pattern of your files
if len(files) == 0:
    raise SystemExit("No *.csv files found in data/raw. Put your CSV files there.")

runs = []

# ----------------------------
# read Data and convert to tensor
# ----------------------------
for f in files:
    df = pd.read_csv(f)
    
    
    required_cols = ['time','stock1','stock2','stock3']  #3D
    #required_cols = ['time','stock1','stock2'] #2D
    if not all(c in df.columns for c in required_cols):
        raise ValueError(f"File {f} must have columns: {required_cols}")
    
    stocks = df[['stock1','stock2','stock3']].values  # shape: (time_steps, features) # 3D
    #stocks = df[['stock1','stock2']].values  # shape: (time_steps, features)        # 2D
    
    stocks_tensor = torch.from_numpy(stocks.astype('float32')).unsqueeze(0)  # shape: (1, time_steps, features)
    
    runs.append(stocks_tensor)

data_tensor = torch.cat(runs, dim=0)  # shape: (num_runs, time_steps, features)

time_tensor = torch.from_numpy(pd.read_csv(files[0])['time'].values.astype('float32'))

# ----------------------------
# normalization
# ----------------------------
mean = data_tensor.mean(dim=(0,1), keepdim=True)   
std  = data_tensor.std(dim=(0,1), keepdim=True) + 1e-8
data_norm = (data_tensor - mean) / std

# ----------------------------
# saving proceed data
# ----------------------------
out_path = os.path.join(PROCESSED_DIR, "3D_SIR.pt")  #########################@@@@@@@@ Change the name of processed data
torch.save({
    "data": data_norm,       
    "time": time_tensor,     
    "mean": mean,
    "std": std,
    "files": files           
}, out_path)

print("Loaded", len(files), "runs.")
print("Data tensor shape:", data_norm.shape)
print("Saved processed dataset to:", out_path)