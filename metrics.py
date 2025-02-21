import numpy as np
from dash import html
import dash_bootstrap_components as dbc


# -----------------------
# Helper Functions & Data
# -----------------------

def format_value(val, precision=3): # 3 digits after the dot should be enough for fp16 tensors
    return f"{val:.{precision}f}"


def compute_metrics(data):
    return {
        "min": format_value(np.min(data)),
        "mean": format_value(np.mean(data)),
        "max": format_value(np.max(data)),
        "std": format_value(np.std(data))
    }


METRIC_INFO = {
    "MAE": (
        "MAE",
        "Mean Absolute Error (MAE) calculates the average of the absolute differences between the reference and main outputs. "
        "It is ideal for element-wise operations and simple arithmetic (e.g., Abs, Add, Subtract, Multiply, Divide) and works well with activation functions like ReLU or Sigmoid, "
        "where every error contributes equally."
    ),
    "MSE": (
        "MSE",
        "Mean Squared Error (MSE) computes the average of the squared differences between outputs, emphasizing larger errors. "
        "This makes it suitable for layers where occasional large deviations are critical, such as Convolution and Pooling (AvgPool, MaxPool), "
        "or in scenarios where numerical instabilities need to be detected."
    ),
    "RMSE": (
        "RMSE",
        "Root Mean Squared Error (RMSE) is the square root of MSE, expressing the error in the same units as the output. "
        "This direct measure of error magnitude is particularly useful in image-processing layers like AdaptiveAvgPool, AdaptiveMaxPool, Convolution, or ROIAlign."
    ),
    "Median": (
        "Median",
        "Median Error identifies the middle value in the error distribution, providing robustness against outliers. "
        "This metric is beneficial for layers (e.g., ReLU or pooling operations) where sporadic extreme values should not skew the overall error assessment."
    ),
    "IQR": (
        "IQR",
        "Interquartile Range (IQR) measures the spread of the middle 50% of errors (75th percentile minus 25th percentile). "
        "It is useful for assessing consistency in layers with skewed or non-Gaussian error distributions, such as BatchNormInference, GroupNormalization, or element-wise nonlinear operations (e.g., Sin, Cos)."
    ),
    "PSNR": (
        "PSNR",
        "Peak Signal-to-Noise Ratio (PSNR) evaluates the ratio between the maximum possible signal power and the power of the noise. "
        "It is especially relevant for image-like outputs, making it a common choice for Convolution, Deconvolution, and ROI-based layers (e.g., ROIAlign, ROIPooling) where visual fidelity is key."
    ),
    "Pearson": (
        "Pearson",
        "Pearson Correlation measures the linear correlation between the reference and main outputs, focusing on pattern or shape similarity rather than absolute differences. "
        "This metric is ideal for layers where maintaining activation trends is crucial, such as in Sigmoid, Tanh, Swish, or normalization layers like BatchNormInference and GroupNormalization."
    )
}


def metric_table_row(metric_key, value):
    """Create a table row for a metric with a tooltip on the label."""
    full_name, description = METRIC_INFO.get(metric_key, (metric_key, ""))
    # Use the title attribute for a simple tooltip on hover.
    return html.Tr([
        html.Td(html.Span(full_name, title=description)),
        html.Td(format_value(value) if isinstance(value, (int, float)) and not (
                isinstance(value, float) and np.isnan(value)) else value)
    ])


# -----------------------
# Advanced Metrics Card
# -----------------------

def advanced_diff_metrics_card(diff, ref_data, main_data):
    """Return a card with additional metrics computed on the difference tensor, grouped by type."""
    # Compute error metrics.
    mae = np.mean(np.abs(diff))
    mse_value = np.mean(diff ** 2)
    rmse_value = np.sqrt(mse_value)
    median_value = np.median(diff)
    iqr_value = np.percentile(diff, 75) - np.percentile(diff, 25)

    # Compute quality metrics.
    # PSNR: using the maximum absolute value in the reference data as the peak.
    max_val = np.max(np.abs(ref_data))
    psnr_value = float('inf') if mse_value == 0 else 20 * np.log10(max_val) - 10 * np.log10(mse_value)

    # Pearson correlation between reference and main outputs.
    if np.std(ref_data) > 0 and np.std(main_data) > 0:
        pearson_value = np.corrcoef(ref_data.flatten(), main_data.flatten())[0, 1]
    else:
        pearson_value = 0

    # Build rows for Error Metrics.
    error_metrics_rows = [
        metric_table_row("MAE", format_value(mae)),
        metric_table_row("MSE", format_value(mse_value)),
        metric_table_row("RMSE", format_value(rmse_value)),
        metric_table_row("Median", format_value(median_value)),
        metric_table_row("IQR", format_value(iqr_value)),
        metric_table_row("PSNR", format_value(psnr_value)),
        metric_table_row("Pearson", format_value(pearson_value))
    ]

    # Assemble the table with a grouping header for quality metrics.
    table = dbc.Table(
        [
            html.Tbody(
                error_metrics_rows
            )
        ],
        bordered=True,
        striped=True,
        hover=True,
        size="sm",
        style={"margin": "8px"}
    )

    return table


# -----------------------
# Main Comparison Card
# -----------------------
def comparison_card(ref_data, main_data):
    """Return a Div containing two cards:
       1. A main comparison card showing basic metrics for Reference, Main, and their Difference.
       2. A second card with advanced difference metrics.
       Plus a button at the bottom.

       The cards are arranged in a responsive layout filling the available width.
    """
    # Compute basic statistics.
    ref_metrics = compute_metrics(ref_data)
    main_metrics = compute_metrics(main_data)
    diff = ref_data - main_data
    diff_metrics = compute_metrics(diff)

    # Build the basic metrics table.
    table = dbc.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Metrics"),
                    html.Th("min"),
                    html.Th("mean"),
                    html.Th("max"),
                    html.Th("Std")
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td("Reference"),
                    html.Td(ref_metrics['min']),
                    html.Td(ref_metrics['mean']),
                    html.Td(ref_metrics['max']),
                    html.Td(ref_metrics["std"])
                ]),
                html.Tr([
                    html.Td("Main"),
                    html.Td(main_metrics['min']),
                    html.Td(main_metrics['mean']),
                    html.Td(main_metrics['max']),
                    html.Td(main_metrics["std"])
                ]),
                html.Tr([
                    html.Td("Difference"),
                    html.Td(diff_metrics['min']),
                    html.Td(diff_metrics['mean']),
                    html.Td(diff_metrics['max']),
                    html.Td(diff_metrics["std"])
                ])
            ])
        ],
        bordered=True,
        striped=True,
        hover=True,
        size="sm",
        style={"margin" : "8px"}
    )

    # Get the advanced metrics card.
    additional_card = advanced_diff_metrics_card(diff, ref_data, main_data)

    # Arrange cards and the button in a responsive layout.
    layout = dbc.Col(
        [
            dbc.Row(table),
            dbc.Row(additional_card)
        ],
        className="w-100"
    )

    return html.Div(layout)

