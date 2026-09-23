#!/usr/bin/env python3
"""
Grid Bucketing for Uniform Match Distribution — ACTIVE enforcement version.

Difference from uniformity_metrics.py:
  - uniformity_metrics.py MEASURES how uniform your matches are (a score/report)
  - this module ENFORCES uniformity by capping how many matches survive per grid cell,
    before RANSAC ever runs

Why this matters: without capping, a crater rim with 500 easy matches can dominate the
RANSAC point set. The homography ends up locally very accurate near that crater and
poorly constrained everywhere else. Capping forces the point set (and therefore the
homography fit) to respect the whole image, not just the texture-rich regions.

Usage (as a library):
    from bucket_uniform import bucket_cap_matches
    pts1_capped, pts2_capped, conf_capped = bucket_cap_matches(
        pts1, pts2, confidences, image_shape, grid_size=10, max_per_cell=50
    )
"""

import numpy as np


def bucket_cap_matches(pts1, pts2, confidences, image_shape, grid_size=10, max_per_cell=50, reference='pts2'):
    """
    Cap the number of matches per grid cell, keeping the highest-confidence matches
    within each cell when there are more than max_per_cell candidates.

    Args:
        pts1, pts2: (N, 2) matched point arrays (source, reference)
        confidences: (N,) confidence/quality score per match. If you don't have real
                     confidence values (e.g. plain SIFT/RIFT matches), pass an array
                     of 1/distance or just np.ones(N) to keep arbitrary points per cell.
        image_shape: (H, W) of the image the bucketing grid is defined over
        grid_size: number of cells per side
        max_per_cell: max matches retained per cell
        reference: 'pts1' or 'pts2' — which point set's coordinates define the grid
                   (usually 'pts2', the reference/fixed image, since that's the
                   canonical coordinate system you're registering into)

    Returns:
        pts1_capped, pts2_capped, conf_capped: filtered arrays, same relative ordering
    """
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    confidences = np.asarray(confidences)

    if len(pts1) == 0:
        return pts1, pts2, confidences

    h, w = image_shape[:2]
    cell_h = h / grid_size
    cell_w = w / grid_size

    grid_pts = pts2 if reference == 'pts2' else pts1

    # Assign each match to a cell
    cell_ids = []
    for x, y in grid_pts:
        gx = min(max(int(x // cell_w), 0), grid_size - 1)
        gy = min(max(int(y // cell_h), 0), grid_size - 1)
        cell_ids.append((gy, gx))
    cell_ids = np.array(cell_ids)

    keep_indices = []
    unique_cells = set(map(tuple, cell_ids))
    for cell in unique_cells:
        mask = np.all(cell_ids == cell, axis=1)
        idx_in_cell = np.where(mask)[0]

        if len(idx_in_cell) <= max_per_cell:
            keep_indices.extend(idx_in_cell.tolist())
        else:
            # Keep the highest-confidence matches in this over-populated cell
            conf_in_cell = confidences[idx_in_cell]
            top_k = idx_in_cell[np.argsort(-conf_in_cell)[:max_per_cell]]
            keep_indices.extend(top_k.tolist())

    keep_indices = np.array(sorted(keep_indices))

    print(f"✓ Bucket capping: {len(pts1)} -> {len(keep_indices)} matches "
          f"(grid={grid_size}x{grid_size}, max_per_cell={max_per_cell}, "
          f"{len(unique_cells)} occupied cells)")

    return pts1[keep_indices], pts2[keep_indices], confidences[keep_indices]


def cells_needing_more_matches(pts2, image_shape, grid_size=10, min_per_cell=3):
    """
    Identify sparse cells (the opposite problem — cells with too FEW matches).
    Useful if you want to re-run detection with a lowered threshold targeted at
    just these regions, rather than blindly lowering the threshold everywhere.

    Returns:
        sparse_cells: list of (row, col) grid indices with fewer than min_per_cell matches
        grid_counts: (grid_size, grid_size) count array
    """
    h, w = image_shape[:2]
    cell_h = h / grid_size
    cell_w = w / grid_size

    grid_counts = np.zeros((grid_size, grid_size), dtype=int)
    for x, y in pts2:
        gx = min(max(int(x // cell_w), 0), grid_size - 1)
        gy = min(max(int(y // cell_h), 0), grid_size - 1)
        grid_counts[gy, gx] += 1

    sparse_cells = [(int(r), int(c)) for r, c in zip(*np.where(grid_counts < min_per_cell))]
    return sparse_cells, grid_counts
