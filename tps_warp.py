#!/usr/bin/env python3
"""
Thin Plate Spline (TPS) Non-Rigid Warping.

A single global homography assumes the imaged surface is FLAT. The lunar surface
isn't — craters, mountains, and elevation changes mean a purely planar (homography)
transform will have residual local error near rugged terrain, even after RANSAC
correctly rejects outliers.

TPS fits a smooth non-rigid warp that passes exactly through your inlier
correspondence points, letting different regions of the image deform by different
amounts. Use it as an optional REFINEMENT step after homography, not a replacement:
homography establishes the global alignment (and is what RANSAC needs to reject
outliers against), then TPS mops up local residual error.

Requires: opencv-contrib-python (NOT plain opencv-python -- the TPS transformer
lives in the contrib/extra modules).
    pip install opencv-contrib-python

Usage (as a library):
    from tps_warp import apply_tps_warp
    warped_img, tps_transformer = apply_tps_warp(source_img, pts1_inliers, pts2_inliers, target_shape)
"""

import cv2
import numpy as np


def apply_tps_warp(source_img, pts1_inliers, pts2_inliers, target_shape):
    """
    Warp source_img onto the reference coordinate system using Thin Plate Splines,
    driven by the inlier correspondence points from your homography/RANSAC step.

    Args:
        source_img: the image to warp (grayscale or color, uint8)
        pts1_inliers: (N, 2) inlier points in SOURCE image coordinates
        pts2_inliers: (N, 2) corresponding inlier points in REFERENCE image coordinates
        target_shape: (H, W) of the reference image (output size)

    Returns:
        warped_img: source_img warped via TPS into reference coordinates
        tps: the fitted cv2 TPS transformer (kept in case you want to warp other
             data, e.g. a mask, using the same fitted transform)
    """
    if not hasattr(cv2, 'createThinPlateSplineShapeTransformer'):
        raise ImportError(
            "cv2.createThinPlateSplineShapeTransformer not found. "
            "You have plain opencv-python installed; TPS needs opencv-contrib-python.\n"
            "Fix: pip uninstall opencv-python -y && pip install opencv-contrib-python"
        )

    if len(pts1_inliers) < 4:
        raise ValueError(f"TPS needs at least 4 points, got {len(pts1_inliers)}")

    tps = cv2.createThinPlateSplineShapeTransformer()

    # cv2's TPS API wants shape (1, N, 2) float32, and match indices as DMatch objects
    src_pts = np.float32(pts1_inliers).reshape(1, -1, 2)
    dst_pts = np.float32(pts2_inliers).reshape(1, -1, 2)

    matches = [cv2.DMatch(i, i, 0) for i in range(len(pts1_inliers))]

    tps.estimateTransformation(dst_pts, src_pts, matches)
    # Note: OpenCV's TPS warpImage maps FROM the "source shape" arg TO the "target
    # shape" arg of estimateTransformation, in that order -- dst_pts first here
    # means we're fitting the transform that pulls source_img into reference space.

    h, w = target_shape[:2]
    warped_img = tps.warpImage(source_img)

    # warpImage outputs into the same size as source_img; resize/crop to target if needed
    if warped_img.shape[:2] != (h, w):
        warped_img = cv2.resize(warped_img, (w, h))

    return warped_img, tps


def compare_homography_vs_tps(img1, img2, H, pts1_inliers, pts2_inliers, output_path=None):
    """
    Convenience function: warp with homography AND with TPS, compute RMSE for both,
    and optionally save a 3-panel comparison figure. Useful for a PPT slide showing
    "TPS reduces residual error on non-planar terrain."
    """
    import matplotlib.pyplot as plt

    h, w = img2.shape[:2]

    # Homography warp
    img1_warped_h = cv2.warpPerspective(img1, H, (w, h))

    # TPS warp
    img1_warped_tps, _ = apply_tps_warp(img1, pts1_inliers, pts2_inliers, img2.shape)

    # RMSE via reprojected inlier points
    pts1_h = np.hstack([pts1_inliers, np.ones((len(pts1_inliers), 1))])
    proj_h = pts1_h @ H.T
    proj_h = proj_h[:, :2] / proj_h[:, 2:3]
    rmse_homography = float(np.sqrt(np.mean(np.linalg.norm(proj_h - pts2_inliers, axis=1) ** 2)))

    print(f"Homography RMSE: {rmse_homography:.3f} px")
    print("(TPS RMSE is near-zero by construction on the fitted points themselves --"
          " its real value is smoother interpolation BETWEEN points, not a comparable"
          " RMSE number. Judge it visually / on held-out points instead.)")

    if output_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img2, cmap='gray'); axes[0].set_title("Reference"); axes[0].axis('off')
        axes[1].imshow(img1_warped_h, cmap='gray'); axes[1].set_title(f"Homography (RMSE={rmse_homography:.2f}px)"); axes[1].axis('off')
        axes[2].imshow(img1_warped_tps, cmap='gray'); axes[2].set_title("TPS (non-rigid refinement)"); axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved comparison: {output_path}")

    return img1_warped_h, img1_warped_tps, rmse_homography
