import numpy as np
import matplotlib.pyplot as plt
from dash import html
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import fftpack
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from skimage import measure
import seaborn as sns
from scipy.stats import entropy
import io
from IPython.display import HTML
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

from visualizations.viz_bin_diff import reshape_to_3d

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


# 1. Animated Slices
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


# 2. Isosurface Rendering
def isosurface_diff(tensor1, tensor2, thresholds=None):
    diff = np.abs(tensor1 - tensor2)

    # Default thresholds based on percentiles of the difference
    if thresholds is None:
        thresholds = [
            np.percentile(diff, 75),
            np.percentile(diff, 85),
            np.percentile(diff, 95)
        ]

    fig = go.Figure()

    # Create a colormap for the thresholds
    colors = px.colors.sequential.Plasma
    colors = [colors[int(i * (len(colors) - 1) / (len(thresholds) - 1))]
              for i in range(len(thresholds))]

    for i, threshold in enumerate(thresholds):
        # Generate isosurface vertices and faces
        verts, faces, _, _ = measure.marching_cubes(diff, threshold)

        # Create the isosurface mesh
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

        i_faces = np.stack([faces[:, 0], faces[:, 1], faces[:, 2]], axis=1)

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i_faces[:, 0], j=i_faces[:, 1], k=i_faces[:, 2],
            opacity=0.7 - i * 0.2,
            color=colors[i],
            name=f'Threshold: {threshold:.3f}'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        title=f"Isosurfaces of Tensor Difference at {len(thresholds)} thresholds"
    )

    return fig


# 3. Parallel Coordinates Plot
def parallel_coordinates_diff(tensor1, tensor2, n_samples=500):
    # Flatten tensors
    tensor1_flat = tensor1.reshape(-1)
    tensor2_flat = tensor2.reshape(-1)
    diff_flat = np.abs(tensor1_flat - tensor2_flat)

    # Sample points with higher probability for larger differences
    weights = diff_flat / diff_flat.sum()
    indices = np.random.choice(
        len(tensor1_flat),
        size=min(n_samples, len(tensor1_flat)),
        p=weights,
        replace=False
    )

    # Create dataframe for parallel coordinates
    data = {
        'Index': indices,
        'Tensor1': tensor1_flat[indices],
        'Tensor2': tensor2_flat[indices],
        'Difference': diff_flat[indices]
    }

    # Create the parallel coordinates plot
    fig = px.parallel_coordinates(
        data,
        dimensions=['Tensor1', 'Tensor2', 'Difference'],
        color='Difference',
        color_continuous_scale='Viridis',
        labels={'Tensor1': 'Tensor 1', 'Tensor2': 'Tensor 2', 'Difference': 'Abs Difference'}
    )

    fig.update_layout(
        title='Parallel Coordinates View of Tensor Differences',
        coloraxis_colorbar=dict(title='Abs Difference')
    )

    return fig


# 6. Multiple-View Tensor Unfolding
def tensor_unfolding_diff(tensor1, tensor2):
    """
    Visualize tensor differences by unfolding along each mode.

    Args:
        tensor1, tensor2: Input tensors of same shape

    Returns:
        Plotly figure
    """
    # Calculate difference tensor
    diff = tensor1 - tensor2

    # Get tensor shape
    shape = tensor1.shape

    # Create unfoldings for each mode
    unfoldings = []
    for mode in range(len(shape)):
        # Rearrange axes to put the current mode first
        axes = [mode] + [i for i in range(len(shape)) if i != mode]
        unfolded = np.transpose(diff, axes).reshape(shape[mode], -1)
        unfoldings.append(unfolded)

    # Create subplots for each unfolding
    fig = make_subplots(
        rows=len(shape), cols=1,
        subplot_titles=[f"Mode-{i + 1} Unfolding" for i in range(len(shape))]
    )

    # Get global min/max for consistent color scaling
    vmin = min(np.min(u) for u in unfoldings)
    vmax = max(np.max(u) for u in unfoldings)
    abs_max = max(abs(vmin), abs(vmax))

    # Add heatmaps for each unfolding
    for i, unfolded in enumerate(unfoldings):
        fig.add_trace(
            go.Heatmap(
                z=unfolded,
                colorscale='RdBu_r',
                zmid=0,
                zmin=-abs_max,
                zmax=abs_max
            ),
            row=i + 1, col=1
        )

    fig.update_layout(
        title="Tensor Difference Unfoldings",
        height=300 * len(shape),
        width=800
    )

    return fig


# 7. Probabilistic Visualization (KL Divergence)
def probabilistic_diff(tensor1, tensor2):
    """
    Visualize KL divergence between tensors after normalizing.

    Args:
        tensor1, tensor2: Input tensors of same shape

    Returns:
        Plotly figure
    """
    # Ensure tensors are positive for treating as probabilities
    tensor1_pos = np.maximum(tensor1 - np.min(tensor1), 1e-10)
    tensor2_pos = np.maximum(tensor2 - np.min(tensor2), 1e-10)

    # Normalize each channel/slice to sum to 1
    kl_divs = []
    js_divs = []

    for c in range(tensor1.shape[0]):
        # Normalize slices
        t1_slice = tensor1_pos[c] / np.sum(tensor1_pos[c])
        t2_slice = tensor2_pos[c] / np.sum(tensor2_pos[c])

        # Compute KL divergence for each direction
        kl_t1_t2 = entropy(t1_slice.flatten(), t2_slice.flatten())
        kl_t2_t1 = entropy(t2_slice.flatten(), t1_slice.flatten())

        # Jensen-Shannon divergence (symmetric)
        m = 0.5 * (t1_slice + t2_slice)
        js_div = 0.5 * (entropy(t1_slice.flatten(), m.flatten()) +
                        entropy(t2_slice.flatten(), m.flatten()))

        kl_divs.append((c, kl_t1_t2, kl_t2_t1))
        js_divs.append((c, js_div))

    # Create bar chart for KL divergence
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=['KL Divergence by Channel',
                                        'Jensen-Shannon Divergence by Channel'])

    channels = [item[0] for item in kl_divs]
    kl_t1_t2_vals = [item[1] for item in kl_divs]
    kl_t2_t1_vals = [item[2] for item in kl_divs]
    js_vals = [item[1] for item in js_divs]

    # KL divergence (both directions)
    fig.add_trace(
        go.Bar(
            x=channels,
            y=kl_t1_t2_vals,
            name='KL(Tensor1 || Tensor2)'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=channels,
            y=kl_t2_t1_vals,
            name='KL(Tensor2 || Tensor1)'
        ),
        row=1, col=1
    )

    # Jensen-Shannon divergence
    fig.add_trace(
        go.Bar(
            x=channels,
            y=js_vals,
            name='JS Divergence',
            marker_color='green'
        ),
        row=2, col=1
    )

    fig.update_layout(
        title='Probabilistic Divergence Between Tensors',
        xaxis_title='Channel',
        yaxis_title='KL Divergence',
        xaxis2_title='Channel',
        yaxis2_title='JS Divergence',
        height=800
    )

    return fig


# 8. Interactive 3D Dashboard (Basic Version)
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def interactive_tensor_diff_dashboard(reference, target):
    """
    Create a Plotly figure that visualizes the difference between two 3D tensors in CHW layout.

    The returned figure contains a 2×2 grid:
      - Top left: Heatmap of the HW slice (fixed channel) with a slider to select the channel.
      - Top right: Heatmap of the CW slice (fixed height) with a slider to select the height.
      - Bottom left: Heatmap of the CH slice (fixed width) with a slider to select the width.
      - Bottom right: Histogram of all error values (difference between target and reference).

    Parameters:
        reference (np.ndarray): A 3D tensor (channels, height, width).
        target (np.ndarray): A 3D tensor (channels, height, width).

    Returns:
        go.Figure: A Plotly figure object with interactive sliders.
    """
    # Compute the difference tensor
    diff = target - reference
    C, H, W = diff.shape

    # Create a 2x2 grid of subplots with reduced spacing
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("HW Slice", "CW Slice", "CH Slice", "Error Histogram"),
        horizontal_spacing=0.05,
        vertical_spacing=0.12
    )

    # --- Add initial traces ---
    # Top left: HW slice (fix channel=0)
    fig.add_trace(
        go.Heatmap(z=diff[0, :, :], coloraxis="coloraxis"),
        row=1, col=1
    )

    # Top right: CW slice (fix height=0)
    fig.add_trace(
        go.Heatmap(z=diff[:, 0, :], coloraxis="coloraxis"),
        row=1, col=2
    )

    # Bottom left: CH slice (fix width=0)
    fig.add_trace(
        go.Heatmap(z=diff[:, :, 0], coloraxis="coloraxis"),
        row=2, col=1
    )

    # Bottom right: Histogram of error values (static)
    fig.add_trace(
        go.Histogram(x=diff.flatten()),
        row=2, col=2
    )

    # Set a shared color scale for heatmaps
    fig.update_layout(coloraxis=dict(colorscale='Viridis'))

    # Reverse y-axis for heatmap subplots so that the image is oriented correctly
    fig.update_yaxes(autorange='reversed', row=1, col=1)
    fig.update_yaxes(autorange='reversed', row=1, col=2)
    fig.update_yaxes(autorange='reversed', row=2, col=1)

    # --- Define sliders for each interactive view ---
    # Slider for HW slice (channel slider updates trace 0)
    steps_hw = []
    for i in range(C):
        steps_hw.append(dict(
            method="restyle",
            args=[{"z": [diff[i, :, :]]}, [0]],  # update trace 0
            label=str(i)
        ))
    slider_hw = dict(
        active=0,
        currentvalue={"prefix": "Channel: "},
        pad={"t": 10},
        steps=steps_hw,
        x=0.0, y=0.555, len=0.482
    )

    # Slider for CW slice (height slider updates trace 1)
    steps_cw = []
    for j in range(H):
        steps_cw.append(dict(
            method="restyle",
            args=[{"z": [diff[:, j, :]]}, [1]],  # update trace 1
            label=str(j)
        ))
    slider_cw = dict(
        active=0,
        currentvalue={"prefix": "Height: "},
        pad={"t": 10},
        steps=steps_cw,
        x=0.525, y=0.555, len=0.482
    )

    # Slider for CH slice (width slider updates trace 2)
    steps_ch = []
    for k in range(W):
        steps_ch.append(dict(
            method="restyle",
            args=[{"z": [diff[:, :, k]]}, [2]],  # update trace 2
            label=str(k)
        ))
    slider_ch = dict(
        active=0,
        currentvalue={"prefix": "Width: "},
        pad={"t": 10},
        steps=steps_ch,
        x=0.0, y=0.00, len=0.482
    )

    # Add the sliders to the layout (they will appear over the figure)
    fig.update_layout(
        sliders=[slider_hw, slider_cw, slider_ch],
        margin=dict(t=30, b=30, l=30, r=30),
        autosize=True
    )

    return fig



# 9. Hierarchical Visualization (Tree Map)
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import numpy as np
import plotly.graph_objects as go
import plotly.express as px


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


# 10. Tensor Network Visualization
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

    # Add dimension nodes
    for dim, pos in positions.items():
        node_x.append(pos['x'])
        node_y.append(pos['y'])
        node_text.append(f"Dimension {dim[-1]}")
        node_size.append(30)
        node_color.append('rgba(100, 100, 255, 0.8)')

    # Add slice nodes
    for i, slice_info in enumerate(slices):
        dim = slice_info['dim']
        idx = slice_info['index']

        # Position slice nodes in circles around dimension nodes
        radius = 0.3
        angle = 2 * np.pi * idx / shape[dim]
        x = positions[f'dim{dim}']['x'] + radius * np.cos(angle)
        y = positions[f'dim{dim}']['y'] + radius * np.sin(angle)

        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Dim{dim}[{idx}]: {slice_info['mean_diff']:.4f}")

        # Size node by max difference
        node_size.append(10 + 40 * slice_info['max_diff'] / np.max(diff))

        # Color node by mean difference
        intensity = int(255 * slice_info['mean_diff'] / np.max(diff))
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
        height=500,
        width=1200
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
        height=500,
        width=1000
    )

    # Update axes
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


# Additional creative visualization methods

def tensor_histogram_comparison(tensor1, tensor2):
    """
    Create a histogram comparison of tensor values with difference highlighting.

    Args:
        tensor1: First tensor (3D numpy array)
        tensor2: Second tensor (3D numpy array)

    Returns:
        Plotly figure object
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")

    channels = tensor1.shape[0]

    # Create subplots for each channel
    fig = make_subplots(
        rows=channels, cols=1,
        subplot_titles=[f'Channel {i} Distribution' for i in range(channels)],
        vertical_spacing=0.04
    )

    # Add histograms for each channel
    for i in range(channels):
        fig.add_trace(
            go.Histogram(
                x=tensor1[i].flatten(),
                opacity=0.7,
                name=f'Tensor 1 Ch{i}',
                marker_color='blue',
                nbinsx=50
            ),
            row=i + 1, col=1
        )

        fig.add_trace(
            go.Histogram(
                x=tensor2[i].flatten(),
                opacity=0.7,
                name=f'Tensor 2 Ch{i}',
                marker_color='red',
                nbinsx=50
            ),
            row=i + 1, col=1
        )

        # Add KDE curves
        x_range = np.linspace(
            min(tensor1[i].min(), tensor2[i].min()),
            max(tensor1[i].max(), tensor2[i].max()),
            1000
        )

        kde1 = stats.gaussian_kde(tensor1[i].flatten())
        kde2 = stats.gaussian_kde(tensor2[i].flatten())

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde1(x_range) * len(tensor1[i].flatten()) * (x_range[1] - x_range[0]),
                mode='lines',
                line=dict(color='blue', width=2),
                name=f'KDE Tensor 1 Ch{i}'
            ),
            row=i + 1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde2(x_range) * len(tensor2[i].flatten()) * (x_range[1] - x_range[0]),
                mode='lines',
                line=dict(color='red', width=2),
                name=f'KDE Tensor 2 Ch{i}'
            ),
            row=i + 1, col=1
        )

    # Update layout
    fig.update_layout(
        title='Tensor Distribution Comparison',
        barmode='overlay',
        height=300 * channels,
        width=800,
        showlegend=True
    )

    return fig


def spectral_analysis(tensor1, tensor2):
    """
    Perform spectral analysis on tensors and visualize the difference
    in frequency domain.

    Args:
        tensor1: First tensor (3D numpy array)
        tensor2: Second tensor (3D numpy array)

    Returns:
        Plotly figure object
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")

    channels = tensor1.shape[0]

    # Create the subplot titles correctly - flatten the nested list
    subplot_titles = []
    for i in range(channels):
        subplot_titles.extend([
            f'Ch{i} Tensor 1 Spectrum',
            f'Ch{i} Tensor 2 Spectrum',
            f'Ch{i} Spectrum Diff'
        ])

    # Create subplots
    fig = make_subplots(
        rows=channels,
        cols=3,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.02
    )

    for i in range(channels):
        # Compute FFT for each channel
        fft1 = np.abs(np.fft.fftshift(np.fft.fft2(tensor1[i])))
        fft2 = np.abs(np.fft.fftshift(np.fft.fft2(tensor2[i])))

        # Log scale for better visualization
        fft1_log = np.log1p(fft1)
        fft2_log = np.log1p(fft2)

        # Compute difference
        fft_diff = fft2_log - fft1_log

        # Add heatmaps
        fig.add_trace(
            go.Heatmap(z=fft1_log, colorscale='Viridis'),
            row=i + 1, col=1
        )

        fig.add_trace(
            go.Heatmap(z=fft2_log, colorscale='Viridis'),
            row=i + 1, col=2
        )

        fig.add_trace(
            go.Heatmap(z=fft_diff, colorscale='RdBu', zmin=-np.max(np.abs(fft_diff)), zmax=np.max(np.abs(fft_diff))),
            row=i + 1, col=3
        )

    # Update layout
    fig.update_layout(
        title='Spectral Analysis of Tensor Differences',
        height=300 * channels,
        width=1200
    )

    # Update axes
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


def eigenvalue_comparison(tensor1, tensor2):
    """
    Compare eigenvalues of flattened tensor channels to identify structural differences.

    Args:
        tensor1: First tensor (3D numpy array)
        tensor2: Second tensor (3D numpy array)

    Returns:
        Plotly figure object
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")

    channels = tensor1.shape[0]

    # Create figure
    fig = go.Figure()

    # Convert each channel to a correlation/covariance matrix and compute eigenvalues
    for i in range(channels):
        # Reshape to 2D matrix (flatten spatial dimensions)
        matrix1 = tensor1[i].reshape(-1, 1)
        matrix2 = tensor2[i].reshape(-1, 1)

        # If matrices are large, sample them
        max_size = 1000
        if matrix1.shape[0] > max_size:
            indices = np.random.choice(matrix1.shape[0], max_size, replace=False)
            matrix1 = matrix1[indices]
            matrix2 = matrix2[indices]

        # Compute correlation matrices
        corr1 = np.corrcoef(matrix1.T, matrix1.T)
        corr2 = np.corrcoef(matrix2.T, matrix2.T)

        # Compute eigenvalues
        eig1 = np.linalg.eigvalsh(corr1)
        eig2 = np.linalg.eigvalsh(corr2)

        # Sort eigenvalues in descending order
        eig1 = np.sort(eig1)[::-1]
        eig2 = np.sort(eig2)[::-1]

        # Take top eigenvalues
        top_n = min(20, len(eig1))
        eig1 = eig1[:top_n]
        eig2 = eig2[:top_n]

        # Pad with zeros if needed
        if len(eig1) < top_n:
            eig1 = np.pad(eig1, (0, top_n - len(eig1)))
        if len(eig2) < top_n:
            eig2 = np.pad(eig2, (0, top_n - len(eig2)))

        # Add traces
        fig.add_trace(
            go.Scatter(
                x=list(range(top_n)),
                y=eig1,
                mode='lines+markers',
                name=f'Tensor 1 Ch{i}',
                line=dict(dash='solid', color=px.colors.qualitative.Plotly[i % 10])
            )
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(top_n)),
                y=eig2,
                mode='lines+markers',
                name=f'Tensor 2 Ch{i}',
                line=dict(dash='dash', color=px.colors.qualitative.Plotly[i % 10])
            )
        )

    # Update layout
    fig.update_layout(
        title='Eigenvalue Comparison',
        xaxis_title='Eigenvalue Index',
        yaxis_title='Eigenvalue Magnitude',
        height=600,
        width=1000,
        yaxis_type='log'
    )

    return fig
