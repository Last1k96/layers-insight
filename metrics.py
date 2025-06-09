import numpy as np
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim

from dash import html
import dash_bootstrap_components as dbc

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
}


def format_value(val, precision=3):  # 3 digits after the dot should be enough for fp16 tensors
    return f"{val:.{precision}f}"


def compute_metrics(data):
    return {
        "min": format_value(np.min(data)),
        "mean": format_value(np.mean(data)),
        "max": format_value(np.max(data)),
        "std": format_value(np.std(data))
    }


def metric_table_row(metric_key, value):
    full_name, description = METRIC_INFO.get(metric_key, (metric_key, ""))
    return html.Tr([
        html.Td(html.Span(full_name, title=description)),
        html.Td(format_value(value) if isinstance(value, (int, float)) and not (
                isinstance(value, float) and np.isnan(value)) else value)
    ])


def compute_r2(ref, main):
    ss_res = np.sum((ref - main) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 1.0


def compute_ssim(ref, main):
    if ref.shape != main.shape:
        return np.nan
    if ref.ndim not in [2, 3, 4]:
        return np.nan

    try:
        if ref.ndim == 2:
            data_range = ref.max() - ref.min()
            return ssim(ref, main, data_range=data_range)

        if ref.ndim == 4:
            if ref.shape[0] != 1:
                return np.nan

            ref = ref.squeeze(0)
            main = main.squeeze(0)

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
    except Exception:
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
    return wasserstein_distance(ref.flatten(), main.flatten())


def advanced_diff_metrics(diff, ref_data, main_data):
    # Standard error metrics.
    mae = np.mean(np.abs(diff))
    mse_value = np.mean(diff ** 2)
    rmse_value = np.sqrt(mse_value)
    ref_range = np.max(ref_data) - np.min(ref_data)
    nrmse_value = rmse_value / ref_range if ref_range != 0 else 0

    max_val = np.max(np.abs(ref_data))
    psnr_value = float('inf') if mse_value == 0 else 20 * np.log10(max_val) - 10 * np.log10(mse_value)

    if np.std(ref_data) > 0 and np.std(main_data) > 0:
        pearson_value = np.corrcoef(ref_data.flatten(), main_data.flatten())[0, 1]
    else:
        pearson_value = 0

    # Additional sophisticated metrics.
    r2_value = compute_r2(ref_data, main_data)
    ssim_value = compute_ssim(ref_data, main_data)

    error_metrics_rows = [
        # Element-wise / simple arithmetic metrics
        metric_table_row("MAE", format_value(mae)),
        metric_table_row("MSE", format_value(mse_value)),
        metric_table_row("NRMSE", format_value(nrmse_value)),

        # Image quality metrics
        metric_table_row("PSNR", format_value(psnr_value)),
        metric_table_row("SSIM", format_value(ssim_value)),

        # Trend / correlation and regression metrics
        metric_table_row("Pearson", format_value(pearson_value)),
        metric_table_row("R2", format_value(r2_value)),
    ]

    # Advanced metrics table with clean styling
    table = dbc.Table(
        [html.Tbody(error_metrics_rows)],
        bordered=True,
        striped=True,
        hover=True,
        size="sm",
        className="mb-2"
    )

    return table


def comparison_metrics_table(ref_data, main_data, idx):
    ref_metrics = compute_metrics(ref_data)
    main_metrics = compute_metrics(main_data)
    diff = ref_data - main_data
    diff_metrics = compute_metrics(diff)

    # Basic metrics table
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
        className="mb-2"
    )

    # Advanced metrics table
    advanced_metrics = advanced_diff_metrics(diff, ref_data, main_data)

    # Visualization button
    visualization_button = dbc.Button(
        "Visualization",
        id={"type": "visualization-button", "index": idx},
        color="secondary",
        className="w-100 mt-2"
    )

    # Clean, unified layout
    return html.Div(
        [table, advanced_metrics, visualization_button],
        className="metrics-container"
    )
