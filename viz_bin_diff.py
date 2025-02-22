#!/usr/bin/env python3
"""
A diagnostic script that compares two NumPy arrays (CPU and XPU outputs)
and visualizes their differences. The inputs are assumed to be in one of the following formats:
  - HW   (2D image)
  - CHW  (3D image with channels first)
  - NCHW (4D batch of images with channels first)

For visualization, the inputs are converted to HWC format (or unrolled to HWC if there are multiple batches).
For each channel (or batch–channel combination), a 2×2 grid is displayed:
  ┌─────────────┬─────────────┐
  │   CPU       │   XPU       │
  │ (reference) │ (perturbed) │
  ├─────────────┼─────────────┤
  │ Difference  │  Density    │
  │ (CPU - XPU) │   Map       │
  └─────────────┴─────────────┘
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def convert_to_CHW(arr):
    """
    Convert an input tensor to CHW format.

    - If the input is HW (2D, shape (H, W)), add a channel dimension → (1, H, W).
    - If the input is CHW (3D, shape (C, H, W)), return it as is.
    - If the input is NCHW (4D, shape (N, C, H, W)):
         • If N == 1, squeeze the batch dimension → (C, H, W).
         • If N > 1, unroll the batch and channel dimensions → (N * C, H, W).
    """
    if arr.ndim == 2:
        return arr[np.newaxis, ...]  # (1, H, W)
    elif arr.ndim == 3:
        # Assume already in CHW.
        return arr
    elif arr.ndim == 4:
        N, C, H, W = arr.shape
        if N == 1:
            return arr[0]  # (C, H, W)
        else:
            return arr.reshape(N * C, H, W)  # Unroll batches: (N * C, H, W)
    else:
        raise ValueError("Unsupported tensor dimensions. Expected HW, CHW, or NCHW.")


def plot_diagnostics(cpu, xpu):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as patches
    import math
    import numpy as np

    # Convert inputs to CHW format.
    cpu = convert_to_CHW(cpu)
    xpu = convert_to_CHW(xpu)

    diff = cpu - xpu
    C, H, W = cpu.shape

    # Determine the layout of channel blocks.
    n_blocks_per_row = max(2, min(math.ceil(math.sqrt(C)) // 2 * 2, 8))
    n_block_rows = math.ceil(C / n_blocks_per_row)

    # Create an overall figure.
    fig = plt.figure(figsize=(8 * n_blocks_per_row, 8 * n_block_rows), facecolor='white')

    # Outer GridSpec for the channel blocks.
    outer = gridspec.GridSpec(n_block_rows, n_blocks_per_row, wspace=0.08, hspace=0.08)

    # Compute a global symmetric scale for the difference images.
    max_abs_diff = np.abs(diff).max()
    global_vmin = -max_abs_diff
    global_vmax = max_abs_diff

    for i in range(C):
        block_row = i // n_blocks_per_row
        block_col = i % n_blocks_per_row

        # Create an inner GridSpec for the 2x2 layout within each outer block.
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[block_row, block_col], wspace=0.08, hspace=0.08
        )

        inner_axes = []

        # Top Left: CPU image with individual title.
        ax_cpu = fig.add_subplot(inner[0, 0])
        ax_cpu.imshow(cpu[i], cmap='gray')
        ax_cpu.set_title("CPU", fontsize=10)
        ax_cpu.axis('off')
        inner_axes.append(ax_cpu)

        # Top Right: XPU image with individual title.
        ax_xpu = fig.add_subplot(inner[0, 1])
        ax_xpu.imshow(xpu[i], cmap='gray')
        ax_xpu.set_title("XPU", fontsize=10)
        ax_xpu.axis('off')
        inner_axes.append(ax_xpu)

        # Bottom Left: Difference image with individual title.
        ax_diff = fig.add_subplot(inner[1, 0])
        ax_diff.imshow(diff[i], cmap='bwr', vmin=global_vmin, vmax=global_vmax)
        ax_diff.set_title("Diff (CPU - XPU)", fontsize=10)
        ax_diff.axis('off')
        inner_axes.append(ax_diff)

        # Bottom Right: Density Map with individual title.
        ch_cpu = cpu[i]
        ch_diff = diff[i]
        bins = 64
        density, _, _ = np.histogram2d(ch_diff.flatten(), ch_cpu.flatten(), bins=bins)
        density = np.power(density, 0.25)
        ax_density = fig.add_subplot(inner[1, 1])
        ax_density.imshow(density, cmap='gray', aspect='auto', origin='lower')
        ax_density.set_title("Density Map", fontsize=10)
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

        # Draw a tight 1px border around the union.
        rect = patches.Rectangle(
            (left, bottom), width, height,
            linewidth=2, edgecolor='black', facecolor='none',
            transform=fig.transFigure, zorder=10
        )
        fig.add_artist(rect)

        # Add a single overall label above the top-left corner of the border.
        # Here we place it a bit lower (offset=0.001) and with a larger font size.
        overall_label_offset = 0.001  # Adjust as needed
        fig.text(left, top + overall_label_offset, f"Channel {i}", va="bottom", ha="left", fontsize=12, fontweight='bold')

    # Hide any unused outer grid cells if C doesn't fill the entire grid.
    total_blocks = n_block_rows * n_blocks_per_row
    for j in range(C, total_blocks):
        ax_dummy = fig.add_subplot(outer[j // n_blocks_per_row, j % n_blocks_per_row])
        ax_dummy.axis('off')

    fig.patch.set_facecolor('lightgray')
    return fig


def main():
    # --- Demo 1: HW layout (2D) ---
    # H, W = 64, 64
    # cpu_input = np.random.rand(H, W).astype(np.float32) * 255
    # xpu_input = cpu_input + np.random.normal(scale=5, size=(H, W)).astype(np.float32)

    # --- Demo 2: CHW layout (3D) ---
    # C, H, W = 20, 64, 64  # e.g., 20 channels
    # cpu_input = np.random.rand(C, H, W).astype(np.float32) * 255
    # xpu_input = cpu_input + np.random.normal(scale=5, size=(C, H, W)).astype(np.float32)

    # --- Demo 3: NCHW layout (4D) ---
    # TODO test if the graph should be mirrored or flipped 180 degree
    N, C, H, W = 2, 20, 64, 64  # e.g., 2 batches, 20 channels each → 40 channels after unrolling
    cpu_input = np.random.rand(N, C, H, W).astype(np.float32) * 255
    xpu_input = cpu_input + np.random.normal(scale=5, size=(N, C, H, W)).astype(np.float32)

    # Set the number of blocks (diagnostic 2x2 blocks) per row. For example, 4.
    fig = plot_diagnostics(cpu_input, xpu_input)
    plt.show()


if __name__ == '__main__':
    main()
