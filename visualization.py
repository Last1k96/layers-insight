import numpy as np
import plotly.graph_objects as go

# For demonstration, we create a synthetic 3D difference tensor.
# In practice, replace this with your actual difference tensor data.
def convert_to_volume(tensor):
    """
    Convert an input tensor to a canonical 3D volume with shape (D, H, W).

    Acceptable input shapes:
      • HW   : (H, W)         → returns (1, H, W)
      • CHW  : (C, H, W)      → returns (C, H, W)
      • NCHW : (1, C, H, W)   → returns (C, H, W)

    If the tensor has ndim == 3, it is assumed to be in CHW format.
    """
    if tensor.ndim == 2:
        # Assume a single-slice volume; add a depth channel.
        return tensor[np.newaxis, ...]
    elif tensor.ndim == 3:
        # Already in CHW format.
        return tensor
    elif tensor.ndim == 4:
        N, C, H, W = tensor.shape
        if N != 1:
            raise ValueError("Only NCHW tensors with N==1 are supported for volumetric visualization.")
        return tensor[0]  # Remove the batch dimension.
    else:
        raise ValueError("Unsupported tensor dimensions. Expected HW, CHW, or NCHW with N==1.")

def plot_volume_tensor(tensor):
    """
    Visualize a volumetric tensor using Plotly's volume rendering.

    The input tensor can be in one of these layouts:
      • HW   (shape: (H, W))
      • CHW  (shape: (C, H, W))
      • NCHW (shape: (1, C, H, W))

    The tensor is converted to a volume with shape (D, H, W), where D is interpreted as the depth.
    A Plotly volume figure is created using synthetic coordinate grids.
    """
    # Convert the input to a volume of shape (D, H, W)
    volume = convert_to_volume(tensor)
    D, H, W = volume.shape

    # Create coordinate grids for the volume.
    # We use linspace over [0, 1] for each axis.
    x = np.linspace(0, 1, D)
    y = np.linspace(0, 1, H)
    z = np.linspace(0, 1, W)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Compute the overall data range for proper scaling.
    data_min = np.min(volume)
    data_max = np.max(volume)

    # Create the Plotly volume figure.
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volume.flatten(),
        isomin=data_min,
        isomax=data_max,
        opacity=0.1,          # Adjust opacity as needed
        surface_count=25,     # Increase for smoother surfaces
        colorscale='Viridis'  # Change to any desired Plotly colorscale
    ))

    fig.update_layout(
        title="3D Volumetric Visualization",
        scene=dict(
            xaxis_title="C",
            yaxis_title="H",
            zaxis_title="W"
        ),
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig

