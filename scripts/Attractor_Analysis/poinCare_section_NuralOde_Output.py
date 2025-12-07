import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Trajectory files From AnyLogic
RAW_DIR = "data/raw"
files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))

# Poincare section at stock3 = Z_plane
Z_plane = 20.0   # Section at stock3 = 20.0

points = []  

for f in files:
    df = pd.read_csv(f)

    Z = df["stock3"].values
    X = df["stock1"].values
    Y = df["stock2"].values

    for i in range(len(Z)-1):
        # Check for crossing
        if (Z[i] - Z_plane) * (Z[i+1] - Z_plane) < 0:

            # Calculate intersection point using linear interpolation
            alpha = (Z_plane - Z[i]) / (Z[i+1] - Z[i])

            x_int = X[i] + alpha * (X[i+1] - X[i])
            y_int = Y[i] + alpha * (Y[i+1] - Y[i])

            points.append((x_int, y_int))


points = np.array(points)

plt.figure(figsize=(8,6))
plt.scatter(points[:,0], points[:,1], s=20)
plt.xlabel("stockX")
plt.ylabel("stockY")
plt.title("Poincaré Section (Z = %.2f)" % Z_plane)
plt.grid(True)
plt.show()
