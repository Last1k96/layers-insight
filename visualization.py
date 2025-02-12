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
    # Convert the input tensor to a volume with shape (C, H, W)
    volume = convert_to_volume(tensor)

    # Rearrange dimensions from (C, H, W) to (C, W, H)
    volume_swapped = volume.transpose(0, 2, 1)  # Now shape is (C, W, H)
    C, W, H = volume_swapped.shape

    # Create coordinate grids using pixel indices for the new layout.
    # Now x corresponds to channels, y corresponds to width, and z corresponds to height.
    c = np.arange(C)
    w = np.arange(W)
    h = np.arange(H)
    X, Y, Z = np.meshgrid(c, w, h, indexing='ij')

    # Compute the overall data range for proper scaling.
    data_min = np.min(volume_swapped)
    data_max = np.max(volume_swapped)

    # Create the Plotly volume figure.
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volume_swapped.flatten(),
        isomin=data_min,
        isomax=data_max,
        opacity=0.1,          # Adjust opacity as needed
        surface_count=20,     # Increase for smoother surfaces
        colorscale='Viridis'  # Change to any desired Plotly colorscale
    ))

    # Update the layout with the new axis titles.
    fig.update_layout(
        title="3D Volumetric Visualization (C W H Layout)",
        scene=dict(
            dragmode='turntable',
            xaxis=dict(title="x (Channel)", autorange='reversed'),
            yaxis=dict(title="y (Width)"),
            zaxis=dict(title="z (Height)", autorange='reversed'),
            camera=dict(
                projection=dict(type='orthographic'),
                eye=dict(x=3, y=0, z=0),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig

