import numpy as np
from dash import html
import dash_bootstrap_components as dbc


# -----------------------
# Helper Functions & Data
# -----------------------

def format_value(val, precision=4):
    """Format a floating‐point value using general format."""
    try:
        return f"{val:.{precision}g}"
    except Exception:
        return str(val)


def compute_metrics(data):
    """Compute basic metrics (min, mean, max, std) for an array."""
    return {
        "min": format_value(np.min(data)),
        "mean": format_value(np.mean(data)),
        "max": format_value(np.max(data)),
        "std": format_value(np.std(data))
    }


# Dictionary of metric info for tooltips: key -> (Full Name, Description)
METRIC_INFO = {
    "MAE": (
        "MAE",
        "Mean Absolute Error. Computes the average absolute difference between the reference and main outputs. "
        "This metric is especially useful for layers that perform element‐wise operations (e.g., Abs, Acos, Asin, Atan) "
        "or simple arithmetic (Add, Subtract, Multiply, Divide) where the magnitude of per-element differences is important. "
        "It works well when outliers are not expected to dominate the error signal – for example, in many activation functions (ReLU, Sigmoid) "
        "or element‐wise transforms."
    ),
    "MSE": (
        "MSE",
        "Mean Squared Error. Averages the squared differences between outputs, making it more sensitive to large errors. "
        "This sensitivity makes MSE suitable for layers where occasional large deviations are critical – such as Convolution, "
        "Pooling (AvgPool, MaxPool), or layers that may suffer from numerical instabilities. "
        "It can be useful when assessing the overall energy of the error, particularly in layers that output values with a wide dynamic range."
    ),
    "RMSE": (
        "RMSE",
        "Root Mean Squared Error. The square root of MSE expresses error in the same units as the output. "
        "RMSE is recommended when you need a direct, interpretable measure of error magnitude – for instance, in image-processing layers "
        "(e.g., AdaptiveAvgPool, AdaptiveMaxPool, Convolution, ROIAlign) or any layer where the physical units of the output matter."
    ),
    "Median": (
        "Median",
        "Median error. Provides the middle value of the error distribution, offering robustness against outliers. "
        "This metric is particularly helpful for layers where sporadic extreme values can occur – for example, in activation functions like ReLU "
        "or in pooling operations where a few high or low values might not reflect the typical behavior."
    ),
    "IQR": (
        "IQR",
        "Interquartile Range. Measures the spread of the middle 50% of differences (75th percentile minus 25th percentile). "
        "IQR is valuable for layers that yield non-Gaussian or skewed error distributions. It helps assess consistency in layers such as "
        "normalization (BatchNormInference, GroupNormalization) or element-wise nonlinear operations (Sin, Cos, etc.) where central tendency matters more than extremes."
    ),
    "PSNR": (
        "PSNR",
        "Peak Signal-to-Noise Ratio. Evaluates the ratio between the maximum possible signal power and the power of corrupting noise. "
        "PSNR is particularly relevant for layers producing image-like outputs or high dynamic range data. "
        "It is commonly used in evaluating outputs from Convolution, Deconvolution, ROI-based operations (ROIAlign, ROIPooling), "
        "and any transformation where visual fidelity or signal quality is critical (e.g., layers converting between color spaces or scaling images)."
    ),
    "Pearson": (
        "Pearson",
        "Pearson Correlation. Measures the linear correlation between the reference and main outputs, focusing on the shape or pattern similarity "
        "rather than absolute error magnitude. This metric is ideal for layers where preserving the relative ordering or pattern of activations is key, "
        "such as in activation functions (Sigmoid, Tanh, Swish), normalization layers (BatchNormInference, GroupNormalization), "
        "or any operation where the trend of values (even if offset) is more important than their exact numeric differences. "
        "It works well for many element‐wise operations (e.g., Abs, Acos, etc.) as well."
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
        metric_table_row("MAE", mae),
        metric_table_row("MSE", mse_value),
        metric_table_row("RMSE", rmse_value),
        metric_table_row("Median", median_value),
        metric_table_row("IQR", iqr_value),
        metric_table_row("PSNR", psnr_value),
        metric_table_row("Pearson", pearson_value)
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
        size="sm"
    )

    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Additional Difference Metrics")),
            dbc.CardBody(table)
        ],
        className="mb-3"
    )


# -----------------------
# Main Comparison Card
# -----------------------

def comparison_card(ref_data, main_data):
    """Return a Div containing two cards:
       1. A main comparison card showing basic metrics for Reference, Main, and their Difference.
       2. A second card with advanced difference metrics.

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
                    html.Th("Dataset"),
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
        size="sm"
    )

    main_card = dbc.Card(
        [
            dbc.CardHeader(html.H5("Comparison Metrics")),
            dbc.CardBody(table)
        ],
        className="mb-3"
    )

    # Get the advanced metrics card.
    additional_card = advanced_diff_metrics_card(diff, ref_data, main_data)

    # Arrange both cards in a responsive layout (each card in its own row that fills the available width).
    layout = dbc.Col(
        [
            dbc.Row(main_card),
            dbc.Row(additional_card)
        ],
        className="w-100"
    )

    return html.Div(layout)
