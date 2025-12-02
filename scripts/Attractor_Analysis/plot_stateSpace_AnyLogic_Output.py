import os
import glob
import pandas as pd
import plotly.graph_objects as go


RAW_DIR = "data/raw/2Basin1"   # 
files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))

if len(files) == 0:
    raise SystemExit(f"No CSV files found in {RAW_DIR}")

# ----------------------------
#-- Adjust cutoff_time to remove transients
#-- If you want to see the full trajectories, set cutoff_time = None
# ---------------------------
cutoff_time = 20      

fig = go.Figure()

for i, f in enumerate(files, start=1):
    df = pd.read_csv(f)

    
    if cutoff_time is not None:
        df = df[df["time"] >= cutoff_time].copy()

    fig.add_trace(
        go.Scatter3d(
            x=df["stock1"],
            y=df["stock2"],
            #z=df["stock3"],
            mode="lines",
            name=f"run {i}",
            line=dict(width=4)
        )
    )

fig.update_layout(
    title="3D State-Space Trajectories (Interactive)",
    scene=dict(
        xaxis_title="stockX",
        yaxis_title="stockY",
        #zaxis_title="stockZ",
    ),
    legend=dict(itemsizing="constant")
)

fig.show()

# ----------------------------
#Save as HTML
# ----------------------------
# fig.write_html("state_space_3D.html")
# 
