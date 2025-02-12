import numpy as np
import plotly.graph_objects as go

# For demonstration, we create a synthetic 3D difference tensor.
# In practice, replace this with your actual difference tensor data.
nx, ny, nz = 30, 30, 30
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
z = np.linspace(0, 1, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Create some synthetic volumetric data that has interesting variation.
# For example, a combination of sine and cosine functions:
diff_data = np.sin(np.pi * X) * np.cos(np.pi * Y) * np.sin(np.pi * Z)

# Compute the overall data range for proper scaling.
data_min = np.min(diff_data)
data_max = np.max(diff_data)

# Create the Plotly volume figure.
fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=diff_data.flatten(),
    isomin=data_min,
    isomax=data_max,
    opacity=0.1,          # Adjust for desired transparency
    surface_count=25,     # Increase for smoother surfaces
    colorscale='Viridis'  # You can change this to any Plotly colorscale
))

fig.update_layout(
    title="3D Volumetric Visualization of Difference Tensor",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    )
)

fig.show()
