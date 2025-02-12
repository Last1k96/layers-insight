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
            return arr.reshape(N * C, H, W)  # unroll batches: (N * C, H, W)
    else:
        raise ValueError("Unsupported tensor dimensions. Expected HW, CHW, or NCHW.")


def plot_diagnostics(cpu, xpu):
    """
    Given two arrays in CHW format (shape (C, H, W)), display a diagnostic plot
    for each channel. For each channel, a 2×2 grid is shown:
      - Top Left: CPU image (reference)
      - Top Right: XPU image (perturbed)
      - Bottom Left: Difference image (CPU − XPU) using a 'bwr' colormap
      - Bottom Right: Density map (2D histogram of (difference, CPU activation) pairs)
    """
    cpu = convert_to_CHW(cpu)
    xpu = convert_to_CHW(xpu)

    diff = cpu - xpu
    C, H, W = cpu.shape

    fig, axs = plt.subplots(nrows=2 * C, ncols=2, figsize=(8, 4 * C))

    for c in range(C):
        # Top Left: CPU image
        ax = axs[2 * c, 0]
        im = ax.imshow(cpu[c], cmap='gray')
        ax.set_title(f"Channel {c}: CPU")
        plt.colorbar(im, ax=ax)
        ax.axis('off')

        # Top Right: XPU image
        ax = axs[2 * c, 1]
        im = ax.imshow(xpu[c], cmap='gray')
        ax.set_title(f"Channel {c}: XPU")
        plt.colorbar(im, ax=ax)
        ax.axis('off')

        # Bottom Left: Difference image (CPU - XPU)
        ax = axs[2 * c + 1, 0]
        im = ax.imshow(diff[c], cmap='bwr')
        ax.set_title(f"Channel {c}: Diff (CPU - XPU)")
        plt.colorbar(im, ax=ax)
        ax.axis('off')

        # Bottom Right: Density Map
        ch_cpu = cpu[c]
        ch_diff = diff[c]
        bins = 128
        density, _, _ = np.histogram2d(ch_diff.flatten(), ch_cpu.flatten(), bins=bins)
        density = np.power(density, 0.25).T  # apply power transform and transpose
        ax = axs[2 * c + 1, 1]
        im = ax.imshow(density, cmap='gray', aspect='auto', origin='lower')
        ax.set_title(f"Channel {c}: Density Map")
        plt.colorbar(im, ax=ax)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """
    For demonstration, generate two random arrays representing CPU and XPU outputs.
    The arrays can be in any of these layouts: HW, CHW, or NCHW.

    Uncomment one of the demo blocks below to test a particular layout.
    """
    # --- Demo 1: HW layout (2D) ---
    # H, W = 64, 64
    # cpu_input = np.random.rand(H, W).astype(np.float32) * 255
    # xpu_input = cpu_input + np.random.normal(scale=5, size=(H, W)).astype(np.float32)

    # --- Demo 2: CHW layout (3D) ---
    # C, H, W = 3, 64, 64
    # cpu_input = np.random.rand(C, H, W).astype(np.float32) * 255
    # xpu_input = cpu_input + np.random.normal(scale=5, size=(C, H, W)).astype(np.float32)

    # --- Demo 3: NCHW layout (4D) ---
    N, C, H, W = 2, 3, 64, 64  # e.g. 2 batches, 3 channels, 64x64 images
    cpu_input = np.random.rand(N, C, H, W).astype(np.float32) * 255
    xpu_input = cpu_input + np.random.normal(scale=5, size=(N, C, H, W)).astype(np.float32)

    # Convert the inputs (HW, CHW, or NCHW) to canonical CHW format.
    cpu_CHW = convert_to_CHW(cpu_input)
    xpu_CHW = convert_to_CHW(xpu_input)

    # Display the diagnostic plots.
    plot_diagnostics(cpu_CHW, xpu_CHW)


if __name__ == '__main__':
    main()