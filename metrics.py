import numpy as np
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim

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
        "It is ideal for element-wise operations and simple arithmetic, and works well with activation functions where every error counts equally—for example, in layers such as Abs, Add, ReLU, and Sigmoid."
    ),
    "MSE": (
        "MSE",
        "Mean Squared Error (MSE) computes the average of the squared differences between outputs, emphasizing larger errors. "
        "It is suitable for layers where occasional large deviations are critical—such as Convolution and Pooling layers (e.g., AvgPool, MaxPool)."
    ),
    "NRMSE": (
        "NRMSE",
        "Normalized Root Mean Squared Error (NRMSE) scales the RMSE by the range of the reference data, allowing for relative error comparisons across layers with different value ranges. "
        "It is particularly useful in heterogeneous contexts, such as in Convolution, Pooling, or MatMul layers."
    ),
    "Median": (
        "Median",
        "Median Error identifies the middle value in the error distribution, providing robustness against outliers. "
        "This metric is beneficial when extreme values should not skew the overall error assessment—as seen in layers like ReLU or AvgPool."
    ),
    "IQR": (
        "IQR",
        "Interquartile Range (IQR) measures the spread of the middle 50% of errors (75th percentile minus 25th percentile), which is useful for assessing consistency in skewed or non-Gaussian distributions. "
        "It is well-suited for layers such as BatchNormInference or GroupNormalization."
    ),
    "PSNR": (
        "PSNR",
        "Peak Signal-to-Noise Ratio (PSNR) evaluates the ratio between the maximum possible signal power and the power of the noise. "
        "It is especially relevant for image-like outputs, as in Convolution or ROI-based layers (e.g., ROIAlign) where visual fidelity is key."
    ),
    "Pearson": (
        "Pearson",
        "Pearson Correlation measures the linear correlation between the reference and main outputs, focusing on pattern similarity rather than absolute differences. "
        "It is ideal for layers where maintaining activation trends is crucial, such as Sigmoid, Tanh, or normalization layers like BatchNormInference."
    ),
    "R2": (
        "R2",
        "Coefficient of Determination (R²) quantifies how well the main outputs explain the variance in the reference data. "
        "It is particularly useful in regression contexts or when assessing predictive performance, for example in layers like MatMul or LSTMCell."
    ),
    "SSIM": (
        "SSIM",
        "Structural Similarity Index (SSIM) evaluates the perceptual similarity between two images by comparing structural information, luminance, and contrast. "
        "It is beneficial in image processing tasks, such as in Convolution or ROI-based layers (e.g., ROIAlign) where visual fidelity matters."
    ),
    "JSD": (
        "JSD",
        "Jensen–Shannon Divergence (JSD) is a symmetric measure of the similarity between two probability distributions, derived from the Kullback–Leibler divergence. "
        "It is applicable when comparing distributions—commonly used in layers such as SoftMax."
    ),
    "EMD": (
        "EMD",
        "Earth Mover's Distance (EMD), also known as the Wasserstein distance, measures the minimum cost required to transform one distribution into another. "
        "This metric is useful for comparing the overall shape of distribution-based outputs, for example in SoftMax layers."
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

def compute_mape(ref, main):
    # Avoid division by zero: use only non-zero ref values.
    non_zero = ref != 0
    if np.any(non_zero):
        return np.mean(np.abs((ref[non_zero] - main[non_zero]) / ref[non_zero])) * 100
    else:
        return np.inf

def compute_r2(ref, main):
    ss_res = np.sum((ref - main) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 1.0


def compute_ssim(ref, main):
    # Ensure the shapes match.
    if ref.shape != main.shape:
        raise ValueError("ref and main must have the same shape")

    # Handle 2D case: HW (grayscale image).
    if ref.ndim == 2:
        data_range = ref.max() - ref.min()
        return ssim(ref, main, data_range=data_range)

    # Handle 3D (CHW) and 4D (NCHW) cases.
    elif ref.ndim in [3, 4]:
        # For 4D input, check the batch dimension.
        if ref.ndim == 4:
            # If batch size > 1, return NaN.
            if ref.shape[0] != 1:
                return np.nan
            # Squeeze the batch dimension.
            ref = ref.squeeze(0)
            main = main.squeeze(0)

        # Now ref is in CHW format.
        # If there's only one channel, treat it as grayscale.
        if ref.shape[0] == 1:
            ref_2d = ref.squeeze(0)
            main_2d = main.squeeze(0)
            data_range = ref_2d.max() - ref_2d.min()
            return ssim(ref_2d, main_2d, data_range=data_range)
        else:
            # For multichannel images, convert from CHW to HWC.
            ref_hwc = np.moveaxis(ref, 0, -1)
            main_hwc = np.moveaxis(main, 0, -1)
            data_range = ref_hwc.max() - ref_hwc.min()
            return ssim(ref_hwc, main_hwc, data_range=data_range, multichannel=True)

    else:
        # For any other number of dimensions, SSIM is not meaningful.
        return np.nan

def compute_jsd(ref, main, num_bins=50):
    # Compute histograms and normalize to form probability distributions.
    hist_ref, bins = np.histogram(ref, bins=num_bins, density=True)
    hist_main, _ = np.histogram(main, bins=bins, density=True)
    # Add a small constant to avoid division by zero issues.
    epsilon = 1e-10
    hist_ref = hist_ref + epsilon
    hist_main = hist_main + epsilon
    m = 0.5 * (hist_ref + hist_main)
    kl_ref_m = np.sum(hist_ref * np.log(hist_ref / m))
    kl_main_m = np.sum(hist_main * np.log(hist_main / m))
    return 0.5 * (kl_ref_m + kl_main_m)

def compute_emd(ref, main):
    # Flatten the arrays for EMD calculation.
    return wasserstein_distance(ref.flatten(), main.flatten())

def advanced_diff_metrics_card(diff, ref_data, main_data):
    """Return a card with additional metrics computed on the difference tensor, grouped by type."""
    # Standard error metrics.
    mae = np.mean(np.abs(diff))
    mse_value = np.mean(diff ** 2)
    rmse_value = np.sqrt(mse_value)
    # NRMSE: normalized by the range of the reference data.
    ref_range = np.max(ref_data) - np.min(ref_data)
    nrmse_value = rmse_value / ref_range if ref_range != 0 else 0
    median_value = np.median(diff)
    iqr_value = np.percentile(diff, 75) - np.percentile(diff, 25)

    # Quality metric: PSNR using the maximum absolute value in the reference data.
    max_val = np.max(np.abs(ref_data))
    psnr_value = float('inf') if mse_value == 0 else 20 * np.log10(max_val) - 10 * np.log10(mse_value)

    # Pearson correlation between reference and main outputs.
    if np.std(ref_data) > 0 and np.std(main_data) > 0:
        pearson_value = np.corrcoef(ref_data.flatten(), main_data.flatten())[0, 1]
    else:
        pearson_value = 0

    # Additional sophisticated metrics.
    mape_value = compute_mape(ref_data, main_data)
    r2_value = compute_r2(ref_data, main_data)
    ssim_value = compute_ssim(ref_data, main_data)
    jsd_value = compute_jsd(ref_data, main_data)
    emd_value = compute_emd(ref_data, main_data)

    # Build rows for Error Metrics.
    error_metrics_rows = [
        # Element-wise / simple arithmetic metrics
        metric_table_row("MAE", format_value(mae)),
        metric_table_row("MSE", format_value(mse_value)),
        metric_table_row("NRMSE", format_value(nrmse_value)),

        # Robust metrics (insensitive to outliers)
        metric_table_row("Median", format_value(median_value)),
        metric_table_row("IQR", format_value(iqr_value)),

        # Distribution similarity metrics
        metric_table_row("JSD", format_value(jsd_value)),
        metric_table_row("EMD", format_value(emd_value)),

        # Image quality metrics
        metric_table_row("PSNR", format_value(psnr_value)),
        metric_table_row("SSIM", format_value(ssim_value)),

        # Trend / correlation and regression metrics
        metric_table_row("Pearson", format_value(pearson_value)),
        metric_table_row("R2", format_value(r2_value)),
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

