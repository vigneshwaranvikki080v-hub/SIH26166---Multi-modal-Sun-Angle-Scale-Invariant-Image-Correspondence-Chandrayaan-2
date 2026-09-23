#!/usr/bin/env python3
"""
Uniformity / Spatial Coverage Metrics — addresses the explicit problem-statement
requirement: "sub-pixel accuracy of source image maintaining UNIFORM DISTRIBUTION
across the images."

Feature detectors naturally cluster matches on high-texture regions (crater rims,
boulder fields) and leave flat plains with few or no matches. A homography fit mostly
from one corner of the image can be locally accurate there and badly wrong elsewhere.
This module scores how evenly matches are spread, independent of match count or
inlier ratio (which say nothing about spatial distribution).

Usage (as a library):
    from uniformity_metrics import compute_uniformity_score
    score, grid_counts = compute_uniformity_score(inlier_points, image_shape, grid_size=8)

Usage (standalone, on a saved match-points CSV):
    python uniformity_metrics.py --points match_points.csv --image_shape 512,512
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import json
from pathlib import Path


def compute_uniformity_score(points, image_shape, grid_size=8):
    """
    Bin the image into a grid_size x grid_size grid and measure how evenly the
    given points (e.g. RANSAC inliers) are spread across cells.

    Args:
        points: (N, 2) array of (x, y) pixel coordinates
        image_shape: (H, W) of the image the points live in
        grid_size: number of cells per side (grid_size^2 total cells)

    Returns:
        uniformity_score: float in [0, 1]. 1.0 = perfectly uniform (every cell has
                           the same point density). 0.0 = maximally clustered
                           (all points in one cell).
        coverage_ratio: fraction of cells that contain at least one point
        grid_counts: (grid_size, grid_size) array of point counts per cell
    """
    h, w = image_shape[:2]
    points = np.asarray(points)

    if len(points) == 0:
        return 0.0, 0.0, np.zeros((grid_size, grid_size))

    cell_h = h / grid_size
    cell_w = w / grid_size

    grid_counts = np.zeros((grid_size, grid_size), dtype=int)
    for x, y in points:
        gx = min(int(x // cell_w), grid_size - 1)
        gy = min(int(y // cell_h), grid_size - 1)
        gx = max(0, gx)
        gy = max(0, gy)
        grid_counts[gy, gx] += 1

    # Coverage: fraction of cells with >=1 match
    coverage_ratio = float(np.count_nonzero(grid_counts) / grid_counts.size)

    # Uniformity: 1 - normalized entropy deficit, using a coefficient-of-variation-based
    # score that's intuitive: low spread in per-cell counts -> high uniformity.
    counts_flat = grid_counts.flatten().astype(float)
    mean_count = counts_flat.mean()
    if mean_count == 0:
        return 0.0, coverage_ratio, grid_counts

    std_count = counts_flat.std()
    cv = std_count / (mean_count + 1e-8)  # coefficient of variation
    # Map CV to a bounded [0,1] score: cv=0 (perfectly even) -> 1.0, cv>=2 -> ~0
    uniformity_score = float(np.clip(1.0 - cv / 2.0, 0.0, 1.0))

    return uniformity_score, coverage_ratio, grid_counts


def visualize_uniformity(grid_counts, output_path, title="Match Density Grid"):
    """Save a heatmap of the per-cell match density."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid_counts, cmap='viridis', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    for (i, j), val in np.ndenumerate(grid_counts):
        ax.text(j, i, int(val), ha='center', va='center',
                color='white' if val < grid_counts.max() / 2 else 'black', fontsize=9)
    plt.colorbar(im, ax=ax, label='Match count')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved uniformity heatmap: {output_path}")


def suggest_grid_enforced_keypoints(image_shape, existing_points, grid_size=8, target_per_cell=3):
    """
    Optional enhancement: identify which grid cells are under-populated, so a caller
    could re-run detection with a region-of-interest mask targeting those cells
    specifically (e.g. lower the SIFT/LoFTR confidence threshold only in sparse cells).

    Returns:
        sparse_cells: list of (row, col) grid cells with fewer than target_per_cell matches
    """
    h, w = image_shape[:2]
    _, _, grid_counts = compute_uniformity_score(existing_points, image_shape, grid_size)
    sparse_cells = [(int(r), int(c)) for r, c in zip(*np.where(grid_counts < target_per_cell))]
    return sparse_cells, grid_counts


def main(args):
    image_shape = tuple(int(x) for x in args.image_shape.split(','))

    points = []
    with open(args.points, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append([float(row['x']), float(row['y'])])
    points = np.array(points)

    score, coverage, grid_counts = compute_uniformity_score(points, image_shape, args.grid_size)

    print(f"\nUniformity score: {score:.3f} (1.0 = perfectly even, 0.0 = clustered)")
    print(f"Coverage ratio:   {coverage:.3f} (fraction of grid cells with >=1 match)")

    output_dir = Path(args.points).parent
    visualize_uniformity(grid_counts, output_dir / 'uniformity_heatmap.png')

    result = {"uniformity_score": score, "coverage_ratio": coverage, "grid_size": args.grid_size}
    with open(output_dir / 'uniformity_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✓ Saved: {output_dir}/uniformity_metrics.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute spatial uniformity of match points')
    parser.add_argument('--points', required=True, help='CSV with columns x,y (e.g. inlier match points)')
    parser.add_argument('--image_shape', required=True, help='H,W of the image, e.g. 512,512')
    parser.add_argument('--grid_size', type=int, default=8)
    args = parser.parse_args()
    main(args)
