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
        raise ValueError(f"Unsupported tensor dimensions = {tensor.ndim}. Expected HW, CHW, or NCHW with N==1.")
    # TODO disable visualization for 1D tensors


def pool_dimension(volume: np.ndarray, axis: int, max_size: int) -> np.ndarray:
    """
    Performs an adaptive max-pool along a single axis of 'volume',
    clamping that axis's size to 'max_size' if it's larger.
    If the size along 'axis' <= max_size, no pooling is performed.

    volume: 3D array (C, H, W) or potentially ND
    axis: which dimension to pool
    max_size: maximum allowed size along that axis

    Returns a new volume (possibly the same if no pooling was needed).
    """
    current_size = volume.shape[axis]
    if current_size <= max_size:
        # No pooling needed, dimension is already small
        return volume

    # Build cut points: e.g., if current_size=1000, max_size=50,
    # then cut points ~ [0, 20, 40, ..., 980, 1000] (51 points).
    cut_points = np.linspace(0, current_size, max_size + 1, dtype=int)

    # Use np.maximum.reduceat to apply chunk-wise max along the given axis
    pooled = np.maximum.reduceat(volume, cut_points[:-1], axis=axis)

    return pooled


def pool_each_dim_individually(volume: np.ndarray, max_size: int) -> np.ndarray:
    """
    Pools each dimension of a 3D volume (C,H,W) individually so that
    each dimension is at most 'max_size'. That is:
       new_shape[i] = min(old_shape[i], max_size)
    and we do an adaptive max over each axis that gets reduced.

    volume: np.ndarray of shape (C, H, W)
    max_size: int

    Returns: pooled_volume, shape = (C_out, H_out, W_out),
             where C_out <= max_size, H_out <= max_size, W_out <= max_size.
    """
    assert volume.ndim == 3, "Expected a 3D tensor (C, H, W)."

    # Pool dimension 0 (channel) if needed
    out = pool_dimension(volume, axis=0, max_size=max_size)
    # Pool dimension 1 (height) if needed
    out = pool_dimension(out, axis=1, max_size=max_size)
    # Pool dimension 2 (width) if needed
    out = pool_dimension(out, axis=2, max_size=max_size)

    return out

def plot_volume_tensor(tensor):
    # Convert the input tensor to a volume with shape (C, H, W)
    volume = convert_to_volume(tensor)
    # Optionally you could reduce the number of points by "MaxPooling" the data.
    # volume = pool_each_dim_individually(volume, 40)

    # Rearrange dimensions from (C, H, W) to (C, W, H)
    volume_swapped = volume.transpose(0, 2, 1)  # Now shape is (C, W, H)
    C, W, H = volume_swapped.shape

    # Create coordinate grids using pixel indices for the new layout.
    # Now x corresponds to channels, y corresponds to width, and z corresponds to height.
    c = np.arange(C)
    w = np.arange(W)
    h = np.arange(H)
    X, Y, Z = np.meshgrid(c, w, h, indexing='ij')

    # Create the Plotly volume figure.
    x_coords = X.flatten()
    y_coords = Y.flatten()
    z_coords = Z.flatten()
    vals = volume_swapped.flatten()

    vals_abs = np.abs(vals)

    val_min, val_max = vals_abs.min(), vals_abs.max()
    point_sizes = 30 * (vals_abs - val_min) / (val_max - val_min)

    fig = go.Figure(
        data=go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=point_sizes,  # <-- array for per-point sizes
                color=vals,  # color by vals
                colorscale='Viridis',
                opacity=0.3,
                showscale=True
            )
        )
    )

    # Update the layout with the new axis titles.
    fig.update_layout(
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

