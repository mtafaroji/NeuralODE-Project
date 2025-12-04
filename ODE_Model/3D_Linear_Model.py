# run_ode.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import torch


# -------------------------------------------------
# 1) تعریف ODE
# -------------------------------------------------
# این قسمت را فقط با معادلاتت عوض کن
'''
a = -0.5
b = 0.8
c = -0.6
d = 0.7
e = -0.8
I = -3.0
def rhs(t, state):
    x, y, z= state
    dx = a * x + b * y + I
    dy = c * x
    dz = d * y + e * z
    return [dx, dy, dz]

'''
I = 15.21
def rhs(t, state):
    x, y, z= state
    dx = -4 * x - y + 18.0
    dy = -1.2 * y -4 *z + 6.4
    dz = 4 * y -1.2 * z - 6.8
    return [dx, dy, dz]


# -------------------------------------------------
# 2) تابع حل ODE
# -------------------------------------------------
def simulate(y0, t_start, t_end, n_steps=1000, method="RK45"):
    t_eval = np.linspace(t_start, t_end, n_steps)
    sol = solve_ivp(rhs, (t_start, t_end), y0, t_eval=t_eval, method=method)

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.t, sol.y   # y shape: (3, N)


# -------------------------------------------------
# 3) ذخیره در CSV
# -------------------------------------------------
def save_csv(t, Y, filename):
    os.makedirs("data/raw/evaluation", exist_ok=True)  ### Make sure directory exists ###
    path = f"data/raw/evaluation/{filename}"  ###########################################@@@@@ save path

    df = pd.DataFrame({
        "time": t,
        "stock1": Y[0, :],
        "stock2": Y[1, :],
        "stock3": Y[2, :],
    })

    df.to_csv(path, index=False)
    print(f"Saved to {path}")


# -------------------------------------------------
# 4) رسم
# -------------------------------------------------
def plot_results(t, Y):
    plt.plot(t, Y[0], label="stock1")
    plt.plot(t, Y[1], label="stock2")
    plt.plot(t, Y[2], label="stock3")
    
    plt.xlabel("time")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# 5) MAIN – اینجا را تغییر بده
# -------------------------------------------------
if __name__ == "__main__":
    
    all_StartPoints = []
    for i in list([1, 4 , 6]):# + list(range(1,4)):
        for j in list([0,4]):
            for l in list([3,4,5]):
                all_StartPoints.append((i,j,l))


    perm = torch.randperm(len(all_StartPoints))
    # مقدار اولیه
    k= 0
    for indx in perm:
        i , j , l= all_StartPoints[indx]

        k += 1
        y0 = [i, j, l]  # تو خودت عوض کن

        # نام فایل خروجی
      
     
        filename = "3D_Linear_run" +str(k) + ".csv"

        # حل ODE
        t, Y = simulate(y0, t_start=0, t_end= 7, n_steps=40)

        # رسم
        plot_results(t, Y)

        # ذخیره
        save_csv(t, Y, filename)
