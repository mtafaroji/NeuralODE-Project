import numpy as np
import plotly.graph_objects as go

# ======================
#   AnyLogic ODEs
# ======================
'''
def f(x, y, z):
    denom = (x + y + z) + 1e-9
    dx = 0.1*z - 0.4*((x*y)/denom)
    dy = 0.4*((x*y)/denom) - 0.3*y
    dz = 0.3*y - 0.1*z
    return dx, dy, dz
'''

def f(x, y, z):
    denom = (x + y + z) + 1e-9
    dx = y
    dy = 3*x-x*x*x-0.2*y
    dz = 0
    return dx, dy, dz





# ============= Settings =============
grid_n = 30
arrow_len = 0.5
head_size = 1

x_min, x_max = -5, 5
y_min, y_max =  -5, 5
z_min, z_max =  -1, 1

z_plane = 20.0
y_plane = 15.0
x_plane = 70.0
# ====================================

def plot_plane(X, Y, U, V, title, xlabel, ylabel, xr, yr):
    # normalize directions
    mag = np.sqrt(U**2 + V**2) + 1e-9



    mag_norm = mag / mag.max()

    # normalize direction
    U_dir = U / mag
    V_dir = V / mag




    U = U / mag
    V = V / mag

    # invisible points to force axis ranges
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X.ravel(), y=Y.ravel(),
        mode="markers",
        marker=dict(size=1, opacity=0),
        showlegend=False
    ))

    annotations = []
    for x0, y0, u0, v0 in zip(X.ravel(), Y.ravel(), U.ravel(), V.ravel()):
        x1 = x0 + arrow_len * u0
        y1 = y0 + arrow_len * v0
        annotations.append(dict(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=head_size,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="royalblue"
        ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_range=xr,
        yaxis_range=yr,
        annotations=annotations,
        width=750,
        height=650
    )
    fig.show()

# ===========================
# 1) XY plane (Z = constant)
# ===========================
xs = np.linspace(x_min, x_max, grid_n)
ys = np.linspace(y_min, y_max, grid_n)
X, Y = np.meshgrid(xs, ys, indexing="ij")
Z = np.full_like(X, z_plane)

dx, dy, dz = f(X, Y, Z)
plot_plane(X, Y, dx, dy,
           title=f"Vector Field (XY plane, Z={z_plane})",
           xlabel="x", ylabel="y",
           xr=[x_min, x_max], yr=[y_min, y_max])

# ===========================
# 2) XZ plane (Y = constant)
# ===========================
xs = np.linspace(x_min, x_max, grid_n)
zs = np.linspace(z_min, z_max, grid_n)
X, Z = np.meshgrid(xs, zs, indexing="ij")
Y = np.full_like(X, y_plane)

dx, dy, dz = f(X, Y, Z)
plot_plane(X, Z, dx, dz,
           title=f"Vector Field (XZ plane, Y={y_plane})",
           xlabel="x", ylabel="z",
           xr=[x_min, x_max], yr=[z_min, z_max])

# ===========================
# 3) YZ plane (X = constant)
# ===========================
ys = np.linspace(y_min, y_max, grid_n)
zs = np.linspace(z_min, z_max, grid_n)
Y, Z = np.meshgrid(ys, zs, indexing="ij")
X = np.full_like(Y, x_plane)

dx, dy, dz = f(X, Y, Z)
plot_plane(Y, Z, dy, dz,
           title=f"Vector Field (YZ plane, X={x_plane})",
           xlabel="y", ylabel="z",
           xr=[y_min, y_max], yr=[z_min, z_max])
