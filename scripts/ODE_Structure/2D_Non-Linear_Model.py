# run_ode.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import torch


# -------------------------------------------------
# 1) Define ODE
# -------------------------------------------------
# ODEs of the 2D Non-Linear System (2 Basin Model)
a = 3.0
b = 6.0
def rhs(t, state):
    x, y= state
    dx = y
    dy = b*x - x*x*x -a*y
    return [dx, dy]




# -------------------------------------------------
# 2) ODE solver function
# -------------------------------------------------
def simulate(y0, t_start, t_end, n_steps=1000, method="RK45"):
    t_eval = np.linspace(t_start, t_end, n_steps)
    sol = solve_ivp(rhs, (t_start, t_end), y0, t_eval=t_eval, method=method)

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.t, sol.y   # y shape: (3, N)


# -------------------------------------------------
# 3) Save to CSV
# -------------------------------------------------
def save_csv(t, Y, filename):
    os.makedirs("data/raw/madeByPython", exist_ok=True)  ### Make sure directory exists ###
    path = f"data/raw/2Basin2/{filename}"  ###########################################@@@@@ save path

    df = pd.DataFrame({
        "time": t,
        "stock1": Y[0, :],
        "stock2": Y[1, :],
        #"stock3": Y[2, :],
    })

    df.to_csv(path, index=False)
    print(f"Saved to {path}")


# -------------------------------------------------
# 4) Plot
# -------------------------------------------------
def plot_results(t, Y):
    plt.plot(t, Y[0], label="stock1")
    plt.plot(t, Y[1], label="stock2")
    #plt.plot(t, Y[2], label="stock3")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------
#
# -------------------------------------------------
if __name__ == "__main__":
    
    all_StartPoints = []
    for i in list(range(1,5)):# + list(range(1,4)):
        for j in range(-2,3):
            all_StartPoints.append((i,j))

    
    perm = torch.randperm(len(all_StartPoints))
    # Initial value
    k= 0
    for indx in perm:
        i , j = all_StartPoints[indx]

        k += 1
        y0 = [i, j]  

        # Output file name
      
     
        filename = "run" +str(k) + ".csv"

        # Solve ODE
        t, Y = simulate(y0, t_start=0, t_end=20, n_steps=100)

        # Plot
        plot_results(t, Y)

        # Save
        save_csv(t, Y, filename)
