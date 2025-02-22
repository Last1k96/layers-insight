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
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches


def reshape_to_3d(arr):
    if arr.ndim == 2:
        H, W = arr.shape
        return arr.reshape(1, H, W)
    elif arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        N, C, H, W = arr.shape
        return arr.reshape(N * C, H, W)  # Unroll batches: (N * C, H, W)
    else:
        raise ValueError("Unsupported tensor dimensions. Expected HW, CHW, or NCHW.")


def plot_diagnostics(cpu, xpu, ref_plugin_name="CPU", main_plugin_name="XPU"):
    cpu = reshape_to_3d(cpu)
    xpu = reshape_to_3d(xpu)

    diff = cpu - xpu
    C, H, W = cpu.shape

    n_blocks_per_row = max(1, min(8, math.ceil(C / int(math.sqrt(C))) // 2 * 2))
    n_block_rows = math.ceil(C / n_blocks_per_row)

    fig = plt.figure(figsize=(8 * n_blocks_per_row, 8 * n_block_rows), facecolor='#404345')

    outer = gridspec.GridSpec(n_block_rows, n_blocks_per_row, wspace=0.08, hspace=0.08)

    max_abs_diff = np.abs(diff).max()
    global_vmin = -max_abs_diff
    global_vmax = max_abs_diff

    for i in range(C):
        block_row = i // n_blocks_per_row
        block_col = i % n_blocks_per_row

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[block_row, block_col], wspace=0.08, hspace=0.1
        )

        inner_axes = []

        # Top Left: CPU image with individual title.
        ax_cpu = fig.add_subplot(inner[0, 0])
        ax_cpu.imshow(cpu[i], cmap='gray')
        ax_cpu.set_title(f"{ref_plugin_name}", fontsize=12, color='white')
        ax_cpu.axis('off')
        inner_axes.append(ax_cpu)

        # Top Right: XPU image with individual title.
        ax_xpu = fig.add_subplot(inner[0, 1])
        ax_xpu.imshow(xpu[i], cmap='gray')
        ax_xpu.set_title(f"{main_plugin_name}", fontsize=12, color='white')
        ax_xpu.axis('off')
        inner_axes.append(ax_xpu)

        # Bottom Left: Difference image with individual title.
        ax_diff = fig.add_subplot(inner[1, 0])
        ax_diff.imshow(diff[i], cmap='bwr', vmin=global_vmin, vmax=global_vmax)
        ax_diff.set_title(f"Diff ({ref_plugin_name} - {main_plugin_name})", fontsize=12, color='black')
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
        ax_density.set_title("Density Map", fontsize=12, color='black')
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

        # Draw a patch covering the union so that the inner 2x2 area is white.
        inner_patch = patches.Rectangle(
            (left, bottom), width, height,
            linewidth=0, edgecolor='none', facecolor='#eeffee',
            transform=fig.transFigure, zorder=0
        )
        fig.add_artist(inner_patch)

        # Draw a border around the union.
        rect = patches.Rectangle(
            (left, bottom), width, height,
            linewidth=2, edgecolor='black', facecolor='none',
            transform=fig.transFigure, zorder=10
        )
        fig.add_artist(rect)

        # Add a single channel label above the top-left corner of the border.
        overall_label_offset = 0.001  # Adjust vertical offset as needed.
        fig.text(left, top + overall_label_offset, f"Channel {i}",
                 va="bottom", ha="left", fontsize=13, fontweight='bold', color='#66ff66')

    return fig


def main():
    nd = 4
    cpu_input, xpu_input = [], []

    if nd == 2:
        H, W = 64, 64
        cpu_input = np.random.rand(H, W).astype(np.float32) * 255
        xpu_input = cpu_input + np.random.normal(scale=5, size=(H, W)).astype(np.float32)
    elif nd == 3:
        C, H, W = 20, 64, 64  # e.g., 20 channels
        cpu_input = np.random.rand(C, H, W).astype(np.float32) * 255
        xpu_input = cpu_input + np.random.normal(scale=5, size=(C, H, W)).astype(np.float32)
    elif nd == 4:
        N, C, H, W = 2, 4, 64, 64  # e.g., 2 batches, 20 channels each → 40 channels after unrolling
        cpu_input = np.random.rand(N, C, H, W).astype(np.float32) * 255
        xpu_input = cpu_input + np.random.normal(scale=5, size=(N, C, H, W)).astype(np.float32)

    fig = plot_diagnostics(cpu_input, xpu_input)
    fig.show()


if __name__ == '__main__':
    main()
