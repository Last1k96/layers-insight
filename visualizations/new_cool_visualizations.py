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
import umap
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
    plt.close()  # Close the figure to avoid duplicate displays

    return anim.to_jshtml()



# 2. Isosurface Rendering
def isosurface_diff(tensor1, tensor2, thresholds=None):
    """
    Visualize isosurfaces of the difference tensor.

    Args:
        tensor1, tensor2: Input tensors of same shape
        thresholds: List of threshold values for isosurfaces

    Returns:
        Plotly figure
    """
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
    """
    Create a parallel coordinates plot showing tensor differences.

    Args:
        tensor1, tensor2: Input tensors of same shape
        n_samples: Number of points to sample

    Returns:
        Plotly figure
    """
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


# 4. Dimension Reduction Visualization
def dim_reduction_diff(tensor1, tensor2, method='pca', n_samples=1000, n_components=2):
    """
    Visualize tensor differences using dimension reduction.

    Args:
        tensor1, tensor2: Input tensors of same shape
        method: 'pca', 'tsne', or 'umap'
        n_samples: Number of points to sample
        n_components: Number of components for reduction

    Returns:
        Plotly figure
    """
    # Reshape tensors to 2D (samples x features)
    tensor1_2d = tensor1.reshape(tensor1.shape[0], -1)
    tensor2_2d = tensor2.reshape(tensor2.shape[0], -1)

    # Sample if needed
    if n_samples < tensor1_2d.shape[0]:
        indices = np.random.choice(tensor1_2d.shape[0], n_samples, replace=False)
        tensor1_2d = tensor1_2d[indices]
        tensor2_2d = tensor2_2d[indices]

    # Combined data for dimension reduction
    combined = np.vstack([tensor1_2d, tensor2_2d])

    # Apply dimension reduction
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=min(30, n_samples // 5))
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components)
    else:
        raise ValueError("Method must be one of: 'pca', 'tsne', 'umap'")

    # Fit and transform
    reduced = reducer.fit_transform(combined)

    # Split back into the two tensors
    reduced_1 = reduced[:tensor1_2d.shape[0]]
    reduced_2 = reduced[tensor1_2d.shape[0]:]

    # Calculate differences for coloring
    diff = np.sqrt(np.sum((tensor1_2d - tensor2_2d) ** 2, axis=1))

    # Create scatter plot
    fig = go.Figure()

    # Add points for tensor1
    fig.add_trace(go.Scatter(
        x=reduced_1[:, 0],
        y=reduced_1[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=diff,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Difference Magnitude')
        ),
        name='Tensor 1'
    ))

    # Add points for tensor2
    fig.add_trace(go.Scatter(
        x=reduced_2[:, 0],
        y=reduced_2[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=diff,
            colorscale='Viridis',
            opacity=0.7
        ),
        name='Tensor 2'
    ))

    # Add lines connecting corresponding points
    for i in range(reduced_1.shape[0]):
        fig.add_trace(go.Scatter(
            x=[reduced_1[i, 0], reduced_2[i, 0]],
            y=[reduced_1[i, 1], reduced_2[i, 1]],
            mode='lines',
            line=dict(
                color=f'rgba({int(255 * diff[i] / max(diff))}, 0, {255 - int(255 * diff[i] / max(diff))}, 0.3)',
                width=1
            ),
            showlegend=False
        ))

    fig.update_layout(
        title=f"Tensor Comparison Using {method.upper()} Dimensionality Reduction",
        xaxis_title=f"{method.upper()} Component 1",
        yaxis_title=f"{method.upper()} Component 2"
    )

    return fig


# 5. Spectral Analysis
def spectral_diff(tensor1, tensor2):
    """
    Visualize differences in the frequency domain.

    Args:
        tensor1, tensor2: Input tensors of same shape

    Returns:
        Plotly figure
    """
    # Compute 3D FFT for both tensors
    fft1 = np.abs(fftpack.fftn(tensor1))
    fft2 = np.abs(fftpack.fftn(tensor2))

    # Compute difference in frequency domain
    fft_diff = np.abs(fft1 - fft2)

    # For visualization, take log of the absolute values (common in spectral analysis)
    log_fft1 = np.log1p(fft1)
    log_fft2 = np.log1p(fft2)
    log_fft_diff = np.log1p(fft_diff)

    # Create 2D visualizations by taking the central slice of each dimension
    slices = []
    for axis in range(3):
        slice_indices = [tensor1.shape[i] // 2 for i in range(3)]

        # Replace the current axis with slice(None) to get the full slice
        slice_indices[axis] = slice(None)

        # For the other axes, create slices across all possible index combinations
        other_axes = [i for i in range(3) if i != axis]

        for i in range(tensor1.shape[other_axes[0]]):
            slice_indices[other_axes[0]] = i

            # Visualize these slices
            slice_fft1 = log_fft1[tuple(slice_indices)]
            slice_fft2 = log_fft2[tuple(slice_indices)]
            slice_diff = log_fft_diff[tuple(slice_indices)]

            slices.append({
                'axis': axis,
                'index': i,
                'fft1': slice_fft1,
                'fft2': slice_fft2,
                'diff': slice_diff
            })

    # Choose a representative slice to visualize
    central_slices = []
    for axis in range(3):
        slice_indices = [tensor1.shape[i] // 2 for i in range(3)]
        slice_indices[axis] = slice(None)
        central_slices.append({
            'axis': axis,
            'fft1': log_fft1[tuple(slice_indices)],
            'fft2': log_fft2[tuple(slice_indices)],
            'diff': log_fft_diff[tuple(slice_indices)]
        })

    # Create a 3x3 subplot to visualize central slices
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            'FFT Tensor 1 (Axis 0)', 'FFT Tensor 2 (Axis 0)', 'FFT Difference (Axis 0)',
            'FFT Tensor 1 (Axis 1)', 'FFT Tensor 2 (Axis 1)', 'FFT Difference (Axis 1)',
            'FFT Tensor 1 (Axis 2)', 'FFT Tensor 2 (Axis 2)', 'FFT Difference (Axis 2)'
        ]
    )

    for i, slice_data in enumerate(central_slices):
        row = i + 1

        fig.add_trace(
            go.Heatmap(z=slice_data['fft1'], colorscale='Viridis'),
            row=row, col=1
        )

        fig.add_trace(
            go.Heatmap(z=slice_data['fft2'], colorscale='Viridis'),
            row=row, col=2
        )

        fig.add_trace(
            go.Heatmap(z=slice_data['diff'], colorscale='Viridis'),
            row=row, col=3
        )

    fig.update_layout(
        title="Spectral Analysis of Tensor Differences",
        height=900,
        width=1000
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
def interactive_tensor_diff_dashboard(tensor1, tensor2):
    """
    Create a simple interactive dashboard for exploring tensor differences.

    Args:
        tensor1, tensor2: Input tensors of same shape

    Returns:
        Plotly figure
    """
    diff = tensor1 - tensor2
    abs_diff = np.abs(diff)

    # Create central slices for each dimension
    slice_x = diff[:, diff.shape[1] // 2, :]
    slice_y = diff[:, :, diff.shape[2] // 2]
    slice_z = diff[diff.shape[0] // 2, :, :]

    # Get global min/max for consistent color scaling
    vmin = diff.min()
    vmax = diff.max()
    abs_max = max(abs(vmin), abs(vmax))

    # Create a figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'heatmap'}, {'type': 'heatmap'}],
            [{'type': 'heatmap'}, {'type': 'heatmap'}]
        ],
        subplot_titles=[
            'X-Slice (Difference)',
            'Y-Slice (Difference)',
            'Z-Slice (Difference)',
            'Difference Histogram'
        ]
    )

    # Add heatmaps for each slice
    fig.add_trace(
        go.Heatmap(
            z=slice_x,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=slice_y,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Heatmap(
            z=slice_z,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max
        ),
        row=2, col=1
    )

    # Add histogram of differences
    fig.add_trace(
        go.Histogram(
            x=diff.flatten(),
            nbinsx=50,
            marker_color='rgba(0, 0, 255, 0.5)'
        ),
        row=2, col=2
    )

    fig.update_layout(
        title='Interactive Tensor Difference Explorer',
        height=800,
        width=900
    )

    return fig


# 9. Hierarchical Visualization (Tree Map)
def hierarchical_diff_visualization(tensor1, tensor2, max_depth=3):
    """
    Create a hierarchical visualization of tensor differences.

    Args:
        tensor1, tensor2: Input tensors of same shape
        max_depth: Maximum depth for hierarchical segmentation

    Returns:
        Plotly figure
    """
    diff = np.abs(tensor1 - tensor2)
    shape = diff.shape

    # Generate hierarchical data
    def build_hierarchy(tensor, path=(), depth=0):
        if depth >= max_depth:
            return {
                'id': '_'.join(map(str, path)),
                'value': np.mean(tensor),
                'name': f"{'.'.join(map(str, path))}: {np.mean(tensor):.4f}"
            }

        result = {
            'id': '_'.join(map(str, path)) if path else 'root',
            'name': '.'.join(map(str, path)) if path else 'Root',
            'children': []
        }

        axis = depth % len(shape)
        n_segments = min(4, shape[axis])  # Limit to 4 segments per axis
        segment_size = shape[axis] // n_segments

        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size if i < n_segments - 1 else shape[axis]

            # Create slice indices
            idx = [slice(None)] * len(shape)
            idx[axis] = slice(start, end)

            subtensor = tensor[tuple(idx)]
            new_path = path + (f"{axis}:{start}-{end}",)

            child = build_hierarchy(subtensor, new_path, depth + 1)
            result['children'].append(child)

        return result

    # Build the hierarchy
    hierarchy = build_hierarchy(diff)

    # Flatten hierarchy for treemap
    treemap_data = []

    def flatten_hierarchy(node, parent=""):
        if 'children' in node:
            for child in node['children']:
                flatten_hierarchy(child, node['id'])
        else:
            treemap_data.append({
                'ids': node['id'],
                'labels': node['name'],
                'parents': parent,
                'values': node['value'],
                'marker': {'colors': [px.colors.sequential.Blues[int(node['value'] * 8)]]}
            })

    flatten_hierarchy(hierarchy)

    # Combine all data
    ids = []
    labels = []
    parents = []
    values = []
    for item in treemap_data:
        ids.append(item['ids'])
        labels.append(item['labels'])
        parents.append(item['parents'])
        values.append(item['value'])

    # Normalize values for color mapping
    norm_values = np.array(values)
    if norm_values.max() > 0:
        norm_values = norm_values / norm_values.max()

    colors = [px.colors.sequential.Blues[int(v * 8)] for v in norm_values]

    # Create treemap
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


def tensor_network_diagram(tensor1, tensor2, threshold=0.1):
    """
    Create a visual representation of tensor difference using a network diagram.

    Args:
        tensor1: First tensor (3D numpy array)
        tensor2: Second tensor (3D numpy array)
        threshold: Minimum difference magnitude to display

    Returns:
        Plotly figure object
    """
    diff = np.abs(tensor1 - tensor2)

    # Check tensor shapes
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")

    channels, height, width = diff.shape

    # Create a graph structure
    G = nx.Graph()

    # Add nodes for each channel
    for c in range(channels):
        avg_diff = np.mean(diff[c])
        if avg_diff > threshold:
            G.add_node(f"C{c}", size=avg_diff, group=1)

    # Add additional nodes for significant spatial locations
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                if diff[c, h, w] > threshold * 2:
                    node_id = f"P{c}_{h}_{w}"
                    G.add_node(node_id, size=diff[c, h, w] / 5, group=2)
                    G.add_edge(f"C{c}", node_id, weight=diff[c, h, w])

    # If graph is empty, add a placeholder node
    if len(G.nodes) == 0:
        G.add_node("No significant differences", size=1, group=0)

    # Create positions using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Create traces for nodes
    node_trace = []

    # Create groups of nodes (channels and points)
    for group in range(3):
        x, y, sizes, labels, colors = [], [], [], [], []

        for node in G.nodes():
            if G.nodes[node].get('group', 0) == group:
                x.append(pos[node][0])
                y.append(pos[node][1])
                size = G.nodes[node].get('size', 0.1)
                sizes.append(size * 30)
                labels.append(node)

                if group == 0:  # Placeholder
                    colors.append('gray')
                elif group == 1:  # Channels
                    colors.append('blue')
                else:  # Points
                    colors.append('red')

        if x:  # Only add trace if there are nodes in this group
            marker_size = [max(5, s) for s in sizes]
            scatter = go.Scatter(
                x=x, y=y,
                mode='markers+text',
                marker=dict(
                    size=marker_size,
                    color=colors,
                    line=dict(width=1, color='black')
                ),
                text=labels,
                textposition="top center",
                hoverinfo='text',
                name=f"Group {group}"
            )
            node_trace.append(scatter)

    # Create edges
    edge_x, edge_y, edge_colors = [], [], []
    edge_weights = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        weight = G.edges[edge].get('weight', 0.1)
        edge_weights.append(weight)
        edge_colors.extend([weight, weight, None])

    if edge_x:  # Only add trace if there are edges
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(
                width=1,
                color='rgba(150, 150, 150, 0.5)'
            ),
            hoverinfo='none',
            name='Connections'
        )
        node_trace.append(edge_trace)

    # Create the figure
    fig = go.Figure(
        data=node_trace,
        layout=go.Layout(
            title='Tensor Network Difference Diagram',
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='rgba(240, 240, 240, 0.8)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
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