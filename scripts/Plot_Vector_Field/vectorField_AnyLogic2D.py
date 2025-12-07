import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Parameters of the differential equation
# --------------------------
a = 3.0
b = 6.0

def f(x, y):
    dx = y
    dy = b * x - x**3 - a * y
    return dx, dy

# --------------------------
# Build grid over [-4, 4]
# --------------------------
x_min, x_max = -4.0, 4.0
y_min, y_max = -4.0, 4.0

num_points = 45  

x_vals = np.linspace(x_min, x_max, num_points)
y_vals = np.linspace(y_min, y_max, num_points)

X, Y = np.meshgrid(x_vals, y_vals)  # both (num_points, num_points)

# --------------------------
# Compute vectors on the grid
# --------------------------
DX, DY = f(X, Y)  # both have shape (num_points, num_points)

# --------------------------
# Normalize vector directions for visualization
# (fixed length, true direction)
# --------------------------
mag = np.sqrt(DX**2 + DY**2) + 1e-12
DX_dir = 3*DX / mag
DY_dir = 3*DY / mag

# --------------------------
# Plot vector field with quiver
# --------------------------
plt.figure(figsize=(8, 8))

# The larger the scale, the shorter the arrows become
scale_value = 10  

plt.quiver(
    X, Y,              # arrow starting points
    DX_dir, DY_dir,    # arrow directions (normalized)
    angles="xy",
    scale_units="xy",
    scale=scale_value,
    width=0.003,
    color="blue",
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Vector Field of the Real ODE: dx/dt = y, dy/dt = 6x - x^3 - 3y")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()
