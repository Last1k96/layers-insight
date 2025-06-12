import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from skimage import measure
import io
import math


def reshape_to_3d(arr):
    if arr.ndim == 1:
        return arr.reshape(1, 1, 1)
    elif arr.ndim == 2:
        H, W = arr.shape
        return arr.reshape(1, H, W)
    elif arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        N, C, H, W = arr.shape
        return arr.reshape(N * C, H, W)
    elif arr.ndim == 5:
        N, D, C, H, W = arr.shape
        return arr.reshape(N * D * C, H, W)
    else:
        raise ValueError("Unsupported tensor dimensions. Supported up to 5D tensors.")


def plot_diagnostics(cpu, xpu, ref_plugin_name="CPU", main_plugin_name="XPU"):
    cpu = reshape_to_3d(cpu)
    xpu = reshape_to_3d(xpu)

    np.nan_to_num(cpu, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(xpu, nan=0.0, posinf=0.0, neginf=0.0)

    diff = cpu - xpu
    C, H, W = cpu.shape

    n_blocks_per_row = max(1, min(8, math.ceil(C / int(math.sqrt(C))) // 2 * 2))
    n_block_rows = math.ceil(C / n_blocks_per_row)

    fig = plt.figure(figsize=(8 * n_blocks_per_row, 8 * n_block_rows), facecolor='#404345')

    outer = gridspec.GridSpec(n_block_rows, n_blocks_per_row, wspace=0.08, hspace=0.08)

    max_abs_diff = np.abs(diff).max()
    global_vmin = -max_abs_diff
    global_vmax = max_abs_diff

    for i in range(C):
        block_row = i // n_blocks_per_row
        block_col = i % n_blocks_per_row

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[block_row, block_col], wspace=0.08, hspace=0.1
        )

        inner_axes = []

        # Top Left: CPU image with individual title.
        ax_cpu = fig.add_subplot(inner[0, 0])
        ax_cpu.imshow(cpu[i], cmap='gray')
        ax_cpu.set_title(f"{ref_plugin_name}", fontsize=12, color='white')
        ax_cpu.axis('off')
        inner_axes.append(ax_cpu)

        # Top Right: XPU image with individual title.
        ax_xpu = fig.add_subplot(inner[0, 1])
        ax_xpu.imshow(xpu[i], cmap='gray')
        ax_xpu.set_title(f"{main_plugin_name}", fontsize=12, color='white')
        ax_xpu.axis('off')
        inner_axes.append(ax_xpu)

        # Bottom Left: Difference image with individual title.
        ax_diff = fig.add_subplot(inner[1, 0])
        ax_diff.imshow(diff[i], cmap='bwr', vmin=global_vmin, vmax=global_vmax)
        ax_diff.set_title(f"Diff ({ref_plugin_name} - {main_plugin_name})", fontsize=12, color='black')
        ax_diff.axis('off')
        inner_axes.append(ax_diff)

        # Bottom Right: Density Map with individual title.
        ch_cpu = cpu[i]
        ch_diff = diff[i]
        bins = 64
        density, _, _ = np.histogram2d(ch_cpu.flatten(), ch_diff.flatten(), bins=bins)
        density = np.power(density, 0.4) # NOTE original scale 0.25
        ax_density = fig.add_subplot(inner[1, 1])
        ax_density.imshow(density, cmap='gray', aspect='auto', origin='lower')
        ax_density.set_title("Density Map", fontsize=12, color='black')
        ax_density.axis('off')
        inner_axes.append(ax_density)

        # Compute the union of the inner axes positions in figure coordinates.
        positions = [ax.get_position() for ax in inner_axes]
        left = min(pos.x0 for pos in positions)
        bottom = min(pos.y0 for pos in positions)
        right = max(pos.x1 for pos in positions)
        top = max(pos.y1 for pos in positions)
        width = right - left
        height = top - bottom

        # Draw a patch covering the union so that the inner 2x2 area is white.
        inner_patch = patches.Rectangle(
            (left, bottom), width, height,
            linewidth=0, edgecolor='none', facecolor='#eeffee',
            transform=fig.transFigure, zorder=0
        )
        fig.add_artist(inner_patch)

        # Draw a border around the union.
        rect = patches.Rectangle(
            (left, bottom), width, height,
            linewidth=2, edgecolor='black', facecolor='none',
            transform=fig.transFigure, zorder=10
        )
        fig.add_artist(rect)

        # Add a single channel label above the top-left corner of the border.
        overall_label_offset = 0.000  # Adjust vertical offset as needed.
        fig.text(left, top + overall_label_offset, f"Channel {i}",
                 va="bottom", ha="left", fontsize=13, fontweight='bold', color='#66ff66')

    return fig


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
    volume = reshape_to_3d(tensor)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

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
    range_val = val_max - val_min

    if range_val == 0:
        normalized = np.zeros_like(vals_abs)
    else:
        normalized = (vals_abs - val_min) / range_val

    threshold = 0.2  # Define your threshold
    clamped = np.where(normalized < threshold, 0, normalized)
    point_sizes = 30 * clamped ** 1.5

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


responsive_css = """
        <style>
          html, body {
            width: 100% !important;
            height: 100% !important;
            margin: 0;
            padding: 0;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
          }
          /* Override canvas size; inline attributes are overridden with !important */
          canvas {
            width: 100% !important;
            height: auto !important;
            max-width: 100%;
            display: block;
            margin: 0 auto !important;
          }
        </style>
        """


def animated_slices(tensor1, tensor2, axis=0, fps=10):
    tensor1 = reshape_to_3d(tensor1)
    tensor2 = reshape_to_3d(tensor2)
    diff = tensor1 - tensor2
    abs_max = max(abs(diff.min()), abs(diff.max()))

    fig, ax = plt.subplots(figsize=(14, 10))
    slices = diff.shape[axis]

    # Initial slice
    slice_indices = [slice(None)] * 3
    slice_indices[axis] = 0
    im = ax.imshow(diff[tuple(slice_indices)], cmap='bwr',
                   vmin=-abs_max, vmax=abs_max)

    plt.colorbar(im, ax=ax, label='Difference')
    ax.set_title(f'Channel {slices - 1}')

    def update(i):
        slice_indices[axis] = i
        im.set_array(diff[tuple(slice_indices)])
        ax.set_title(f'Channel {i} / {slices - 1}')
        return [im]

    anim = FuncAnimation(fig, update, frames=slices, blit=True, interval=1000 / fps)
    plt.close()

    return responsive_css + anim.to_jshtml()


def isosurface_diff(tensor1, tensor2):
    diff = np.abs(tensor1 - tensor2)
    diff = diff.transpose(0, 2, 1)

    # Default thresholds based on percentiles of the difference
    thresholds = [
        np.percentile(diff, 75),
        np.percentile(diff, 85),
        np.percentile(diff, 95)
    ]

    fig = go.Figure()

    shape = tensor1.shape
    if shape[0] < 2 or shape[1] < 2 or shape[2] < 2:
        return fig

    # Create a colormap for the thresholds
    colors = px.colors.sequential.Plasma
    colors = [colors[int(i * (len(colors) - 1) / (len(thresholds) - 1))]
              for i in range(len(thresholds))]

    data_min, data_max = diff.min(), diff.max()

    for i, threshold in enumerate(thresholds):
        if threshold <= data_min or threshold >= data_max:
            continue

        verts, faces, _, _ = measure.marching_cubes(diff, threshold)

        # Create the isosurface mesh
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

        i_faces = np.stack([faces[:, 0], faces[:, 1], faces[:, 2]], axis=1)

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i_faces[:, 0], j=i_faces[:, 1], k=i_faces[:, 2],
            opacity=0.7 - i * 0.25,
            color=colors[i],
            name=f'Threshold: {threshold:.3f}'
        ))

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
        margin=dict(l=0, r=0, t=0, b=0),
        title=f"Isosurfaces of Tensor Difference at {len(thresholds)} thresholds"
    )

    return fig


def hierarchical_diff_visualization(tensor1, tensor2, max_depth=3):
    """
    Create a hierarchical visualization of tensor differences.

    Args:
        tensor1, tensor2: Input tensors of the same shape (with CHW layout).
        max_depth: Maximum depth for hierarchical segmentation.

    Returns:
        Plotly figure.
    """
    # Compute the absolute difference between tensors.
    diff = np.abs(tensor1 - tensor2)

    # Define axis names for CHW layout.
    axis_names = ['C', 'H', 'W']

    # Recursive function to build hierarchy with improved labels.
    def build_hierarchy(tensor, path=(), depth=0):
        current_mean = np.mean(tensor)
        # Create an ID and label from the current path.
        id_str = " > ".join(path) if path else "Root"
        label_str = f"{id_str}\nMean diff: {current_mean:.4f}"

        # Stop recursion at max_depth or if tensor is empty.
        if depth >= max_depth or tensor.size == 0:
            return {
                'id': id_str,
                'value': current_mean,
                'name': label_str
            }

        result = {
            'id': id_str,
            'value': current_mean,
            'name': label_str,
            'children': []
        }

        # Use the current tensor's shape for segmentation.
        current_shape = tensor.shape
        axis = depth % len(current_shape)
        n_segments = min(4, current_shape[axis])  # Limit to 4 segments per axis.
        segment_size = current_shape[axis] // n_segments if n_segments > 0 else current_shape[axis]

        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size if i < n_segments - 1 else current_shape[axis]

            # Create slice indices for the current axis.
            idx = [slice(None)] * len(current_shape)
            idx[axis] = slice(start, end)

            subtensor = tensor[tuple(idx)]
            new_path = path + (f"{axis_names[axis]}: {start}-{end}",)

            child = build_hierarchy(subtensor, new_path, depth + 1)
            result['children'].append(child)

        return result

    # Build the hierarchy using the difference tensor.
    hierarchy = build_hierarchy(diff)

    # Flatten the hierarchy to build lists for the treemap.
    treemap_data = []

    def flatten_hierarchy(node, parent=""):
        treemap_data.append({
            'ids': node['id'],
            'labels': node['name'],
            'parents': parent,
            'value': node.get('value', 0)
        })
        if 'children' in node:
            for child in node['children']:
                flatten_hierarchy(child, node['id'])

    flatten_hierarchy(hierarchy)

    ids = [item['ids'] for item in treemap_data]
    labels = [item['labels'] for item in treemap_data]
    parents = [item['parents'] for item in treemap_data]
    values = [item['value'] for item in treemap_data]

    # Normalize values for color mapping.
    norm_values = np.array(values)
    if norm_values.max() > 0:
        norm_values = norm_values / norm_values.max()

    colors = [px.colors.sequential.Blues[min(int(v * 8), 8)] for v in norm_values]

    # Create the treemap figure.
    fig = go.Figure(go.Treemap(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors),
        textinfo="label+value",
        hoverinfo="label+value+percent parent+percent root"
    ))

    fig.update_layout(
        title='Hierarchical View of Tensor Differences',
        margin=dict(t=50, l=25, r=25, b=25)
    )

    return fig


def tensor_network_visualization(tensor1, tensor2):
    """
    Visualize tensor structure with differences as a network.

    Args:
        tensor1, tensor2: Input tensors of same shape

    Returns:
        Plotly figure
    """
    diff = np.abs(tensor1 - tensor2)
    shape = diff.shape

    # Create a simplified tensor network representation
    # Each dimension becomes a "node" and the connections represent the tensor structure

    # Create node positions
    positions = {
        'dim0': {'x': 0, 'y': 0},
        'dim1': {'x': 1, 'y': 1},
        'dim2': {'x': 2, 'y': 0}
    }

    # Calculate mean difference along each slice
    slices = []
    for dim in range(len(shape)):
        for i in range(shape[dim]):
            idx = [slice(None)] * len(shape)
            idx[dim] = i

            slice_diff = diff[tuple(idx)]
            slices.append({
                'dim': dim,
                'index': i,
                'mean_diff': np.mean(slice_diff),
                'max_diff': np.max(slice_diff)
            })

    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []

    dim_names = ['Channel', 'Height', 'Width']

    # Add dimension nodes
    for dim, pos in positions.items():
        node_x.append(pos['x'])
        node_y.append(pos['y'])
        node_text.append(f"{dim_names[int(dim[-1])]}")
        node_size.append(30)
        node_color.append('rgba(100, 100, 255, 0.8)')

    max_diff = np.max(diff)
    if max_diff == 0:
        max_diff = 1  # Or handle it as needed, e.g., skip processing

    # Add slice nodes
    for i, slice_info in enumerate(slices):
        dim = slice_info['dim']
        idx = slice_info['index']

        # Ensure that shape[dim] is not zero to avoid division by zero in the angle calculation
        if shape[dim] == 0:
            continue  # or handle appropriately

        # Position slice nodes in circles around dimension nodes
        radius = 0.3
        angle = 2 * np.pi * idx / shape[dim]
        x = positions[f'dim{dim}']['x'] + radius * np.cos(angle)
        y = positions[f'dim{dim}']['y'] + radius * np.sin(angle)

        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{dim_names[dim][0]}[{idx}]: {slice_info['mean_diff']:.4f}")

        # Size node by max difference using safe max_diff value
        node_size.append(10 + 40 * slice_info['max_diff'] / max_diff)

        # Color node by mean difference using safe max_diff value
        intensity = int(255 * slice_info['mean_diff'] / max_diff)
        node_color.append(f'rgba({255 - intensity}, {255 - intensity}, 255, 0.7)')

    # Create the figure
    fig = go.Figure()

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color='rgb(50, 50, 50)')
        ),
        text=node_text,
        hoverinfo='text'
    ))

    # Add title and styling
    fig.update_layout(
        title='Tensor Network Visualization of Differences',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40)
    )

    return fig


def channel_correlation_matrices(tensor1, tensor2):
    """
    Generate correlation matrices between channels of both tensors
    and visualize the difference in correlations.

    Args:
        tensor1: First tensor (3D numpy array)
        tensor2: Second tensor (3D numpy array)

    Returns:
        Plotly figure object
    """
    # Check tensor shapes
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")

    channels = tensor1.shape[0]

    # Create correlation matrices
    corr1 = np.zeros((channels, channels))
    corr2 = np.zeros((channels, channels))

    # Flatten each channel for correlation calculation
    tensor1_flat = [tensor1[i].flatten() for i in range(channels)]
    tensor2_flat = [tensor2[i].flatten() for i in range(channels)]

    # Calculate correlation matrices
    for i in range(channels):
        for j in range(channels):
            # Calculate Pearson correlation between channels
            corr1[i, j] = np.corrcoef(tensor1_flat[i], tensor1_flat[j])[0, 1]
            corr2[i, j] = np.corrcoef(tensor2_flat[i], tensor2_flat[j])[0, 1]

    # Calculate difference in correlation matrices
    corr_diff = corr2 - corr1

    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Tensor 1 Channel Correlations',
                        'Tensor 2 Channel Correlations',
                        'Correlation Differences'),
        horizontal_spacing=0.1
    )

    # Add heatmaps
    fig.add_trace(
        go.Heatmap(
            z=corr1,
            x=[f'Ch{i}' for i in range(channels)],
            y=[f'Ch{i}' for i in range(channels)],
            colorscale='Blues',
            zmin=-1, zmax=1
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=corr2,
            x=[f'Ch{i}' for i in range(channels)],
            y=[f'Ch{i}' for i in range(channels)],
            colorscale='Blues',
            zmin=-1, zmax=1
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Heatmap(
            z=corr_diff,
            x=[f'Ch{i}' for i in range(channels)],
            y=[f'Ch{i}' for i in range(channels)],
            colorscale='RdBu',
            zmin=-1, zmax=1
        ),
        row=1, col=3
    )

    # Update layout
    fig.update_layout(
        title='Channel Correlation Analysis',
        margin=dict(t=50, l=25, r=25, b=25)
    )

    return fig


def gradient_flow_visualization(tensor1, tensor2, channel_idx=0):
    """
    Visualize the gradient of the difference tensor as a vector field,
    showing directionality of changes.

    Args:
        tensor1: First tensor (3D numpy array)
        tensor2: Second tensor (3D numpy array)
        channel_idx: Index of channel to visualize (default: 0)

    Returns:
        Plotly figure object
    """
    # Check tensor shapes
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")

    if channel_idx >= tensor1.shape[0]:
        raise ValueError(f"Channel index {channel_idx} is out of bounds")

    # Extract the specified channel
    tensor1_channel = tensor1[channel_idx]
    tensor2_channel = tensor2[channel_idx]

    # Calculate difference
    diff = tensor2_channel - tensor1_channel

    # Calculate gradients (approximation)
    gy, gx = np.gradient(diff)

    # Create a grid of coordinates
    y, x = np.mgrid[0:diff.shape[0], 0:diff.shape[1]]

    # Subsample for better visualization (adjust as needed)
    skip = max(1, min(diff.shape) // 20)

    # Create figure
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Difference Magnitude', 'Gradient Flow'),
                        column_widths=[0.5, 0.5])

    # Add heatmap of difference magnitude
    fig.add_trace(
        go.Heatmap(
            z=diff,
            colorscale='RdBu',
            zmin=-np.max(np.abs(diff)),
            zmax=np.max(np.abs(diff))
        ),
        row=1, col=1
    )

    # Add vector field
    fig.add_trace(
        go.Quiver(
            x=x[::skip, ::skip].flatten(),
            y=y[::skip, ::skip].flatten(),
            u=gx[::skip, ::skip].flatten(),
            v=gy[::skip, ::skip].flatten(),
            scale=0.05,
            name='Gradient',
            line=dict(width=1),
            sizemode='absolute'
        ),
        row=1, col=2
    )

    # Add background heatmap for vector field
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    fig.add_trace(
        go.Heatmap(
            z=magnitude,
            colorscale='Viridis',
            opacity=0.5,
            showscale=False
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title=f'Gradient Flow Visualization (Channel {channel_idx})',
        margin=dict(t=50, l=25, r=25, b=25)
    )

    # Update axes
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig