import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# Define the dimensions and create the meshgrid.
nx, ny, nz = 30, 30, 30
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
z = np.linspace(0, 1, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Create synthetic volumetric data.
diff_data = np.sin(np.pi * X) * np.cos(np.pi * Y) * np.sin(np.pi * Z)

# Compute the overall data range for the entire tensor.
data_min = np.min(diff_data)
data_max = np.max(diff_data)


def create_slice_figure(slice_index):
    """
    Returns a Plotly heatmap figure for the given z-axis slice of diff_data,
    using a fixed color scale computed from the entire tensor.
    """
    # Extract the 2D slice along the z-axis.
    slice_data = diff_data[:, :, slice_index]

    # Build a heatmap of the slice with fixed zmin and zmax.
    fig = go.Figure(data=go.Heatmap(
        z=slice_data,
        colorscale='Viridis',
        zmin=data_min,
        zmax=data_max
    ))
    fig.update_layout(
        title=f"Slice at z-index: {slice_index}",
        xaxis_title="X",
        yaxis_title="Y"
    )
    return fig


# Create the Dash app.
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(
        id='slice-graph',
        figure=create_slice_figure(nz // 2)  # Start with the middle slice.
    ),
    dcc.Slider(
        id='slice-slider',
        min=0,
        max=nz - 1,
        step=1,
        value=nz // 2,
        marks={i: str(i) for i in range(0, nz, max(1, nz // 10))}
    )
])


@app.callback(
    Output('slice-graph', 'figure'),
    Input('slice-slider', 'value')
)
def update_slice(selected_slice):
    return create_slice_figure(selected_slice)


if __name__ == '__main__':
    app.run(debug=True)
