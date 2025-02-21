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
    # Convert inputs to CHW format.
    cpu = convert_to_CHW(cpu)
    xpu = convert_to_CHW(xpu)

    diff = cpu - xpu
    C, H, W = cpu.shape

    n_blocks_per_row = max(2, min(math.ceil(math.sqrt(C)) // 2 * 2, 8))

    # Compute a global scale for the difference images.
    # Using a symmetric range so that 0 is centered in the 'bwr' colormap.
    max_abs_diff = np.abs(diff).max()
    global_vmin = -max_abs_diff
    global_vmax = max_abs_diff

    # Determine grid size.
    n_block_rows = math.ceil(C / n_blocks_per_row)
    total_rows = n_block_rows * 2  # each block is 2 rows tall
    total_cols = n_blocks_per_row * 2  # each block is 2 columns wide

    fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols, figsize=(4 * total_cols, 4 * total_rows))
    axs = np.atleast_2d(axs)

    for i in range(C):
        block_row = i // n_blocks_per_row
        block_col = i % n_blocks_per_row

        # Compute starting row and column indices for this block.
        r_top = block_row * 2
        c_left = block_col * 2

        # Top Left: CPU image
        ax = axs[r_top, c_left]
        im = ax.imshow(cpu[i], cmap='gray')
        ax.set_title(f"Channel {i}: CPU")
        plt.colorbar(im, ax=ax)
        ax.axis('off')

        # Top Right: XPU image
        ax = axs[r_top, c_left + 1]
        im = ax.imshow(xpu[i], cmap='gray')
        ax.set_title(f"Channel {i}: XPU")
        plt.colorbar(im, ax=ax)
        ax.axis('off')

        # Bottom Left: Difference image (CPU - XPU)
        ax = axs[r_top + 1, c_left]
        im = ax.imshow(diff[i], cmap='bwr', vmin=global_vmin, vmax=global_vmax)
        ax.set_title(f"Channel {i}: Diff (CPU - XPU)")
        plt.colorbar(im, ax=ax)
        ax.axis('off')

        # Bottom Right: Density Map
        ch_cpu = cpu[i]
        ch_diff = diff[i]
        bins = 64
        density, _, _ = np.histogram2d(ch_diff.flatten(), ch_cpu.flatten(), bins=bins)
        density = np.power(density, 0.25) #.T  # Apply power transform and transpose for better display.
        ax = axs[r_top + 1, c_left + 1]
        im = ax.imshow(density, cmap='gray', aspect='auto', origin='lower')
        ax.set_title(f"Channel {i}: Density Map")
        plt.colorbar(im, ax=ax)
        ax.axis('off')

    # Hide any extra subplot axes if total number of blocks is not a perfect multiple.
    total_blocks = n_block_rows * n_blocks_per_row
    for j in range(C, total_blocks):
        block_row = j // n_blocks_per_row
        block_col = j % n_blocks_per_row
        r_top = block_row * 2
        c_left = block_col * 2
        axs[r_top, c_left].axis('off')
        axs[r_top, c_left + 1].axis('off')
        axs[r_top + 1, c_left].axis('off')
        axs[r_top + 1, c_left + 1].axis('off')

    plt.close(fig) # TODO Is it useful at all?
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
