#!/usr/bin/env python3
"""
Lunar Image Registration Baseline — SIFT + RANSAC
Chandrayaan-2 TMC-2 data, Tycho Crater region
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# ============================================================================
# LOAD IMAGES FROM PDS4 VIA GDAL
# ============================================================================

def load_pds4_image(xml_path):
    """
    Load PDS4 .IMG via GDAL using the .XML label.
    Returns: grayscale NumPy array (uint8)
    """
    try:
        from osgeo import gdal
    except ImportError:
        raise ImportError("GDAL not installed. Run: pip install gdal")
    
    # Open the .XML label (GDAL auto-finds .IMG)
    ds = gdal.Open(str(xml_path))
    if ds is None:
        raise FileNotFoundError(f"GDAL couldn't open {xml_path}. Check file exists & GDAL is configured.")
    
    # Read first band (grayscale)
    band = ds.GetRasterBand(1)
    img_array = band.ReadAsArray()
    
    # Normalize to 8-bit if needed
    if img_array.dtype != np.uint8:
        img_min, img_max = img_array.min(), img_array.max()
        img_array = np.clip((img_array - img_min) / (img_max - img_min + 1e-8) * 255, 0, 255).astype(np.uint8)
    
    print(f"✓ Loaded {xml_path.name}: shape {img_array.shape}, dtype {img_array.dtype}")
    return img_array


def preprocess_image(img):
    """
    Grayscale + CLAHE (illumination normalization) for robustness across sun angles.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE: Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img)
    
    return img_enhanced


# ============================================================================
# SIFT DETECTION & MATCHING
# ============================================================================

def detect_and_match(img1, img2):
    """
    SIFT keypoint detection + BFMatcher with Lowe's ratio test (0.75).
    Returns: kp1, kp2, matches (filtered)
    """
    sift = cv2.SIFT_create()
    
    # Detect keypoints & descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    print(f"✓ SIFT: {len(kp1)} kpts in img1, {len(kp2)} kpts in img2")
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise ValueError("Not enough keypoints detected. Images may be too small or featureless.")
    
    # BFMatcher with KNN
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test: filter out ambiguous matches
    ratio_threshold = 0.75
    good_matches = [m for m, n in matches if m.distance < ratio_threshold * n.distance]
    
    print(f"✓ Matches: {len(matches)} raw → {len(good_matches)} after ratio test")
    
    return kp1, kp2, good_matches


# ============================================================================
# RANSAC HOMOGRAPHY
# ============================================================================

def estimate_homography(kp1, kp2, matches):
    """
    RANSAC-based homography estimation.
    Returns: H (3x3 matrix), inlier_mask, metrics dict
    """
    if len(matches) < 4:
        raise ValueError("Not enough good matches for RANSAC.")
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # RANSAC: homography
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=5.0)
    
    if H is None:
        raise ValueError("RANSAC failed to compute homography.")
    
    inliers = mask.sum()
    inlier_ratio = inliers / len(matches)
    
    print(f"✓ RANSAC: {inliers} inliers / {len(matches)} matches ({100*inlier_ratio:.1f}%)")
    
    metrics = {
        "total_matches": len(matches),
        "inliers": int(inliers),
        "inlier_ratio": float(inlier_ratio),
        "rmse_px": float(np.sqrt(np.mean(mask.ravel() ** 2)))  # rough estimate
    }
    
    return H, mask, metrics


# ============================================================================
# WARPING & ALIGNMENT
# ============================================================================

def warp_and_overlay(img1, img2, H):
    """
    Warp img1 to img2 space, return warped image and overlay.
    """
    h, w = img2.shape[:2]
    img1_warped = cv2.warpPerspective(img1, H, (w, h))
    
    # Overlay for visual inspection: red (img1) + cyan (img2)
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 0] = img1_warped  # B: warped img1
    overlay[:, :, 1] = img2          # G: img2
    overlay[:, :, 2] = img1_warped  # R: warped img1 (creates magenta on perfect alignment)
    
    return img1_warped, overlay


# ============================================================================
# VISUALIZATION & METRICS
# ============================================================================

def plot_results(img1, img2, img1_warped, overlay, kp1, kp2, matches, metrics, output_path):
    """
    4-panel visualization: original pair, warped, overlay, match visualization.
    Saves high-quality PNG for PPT/video.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Lunar Registration Baseline (SIFT + RANSAC)", fontsize=14, fontweight='bold')
    
    # Panel 1: Original reference
    axes[0, 0].imshow(img2, cmap='gray')
    axes[0, 0].set_title("Reference Image (Target)")
    axes[0, 0].axis('off')
    
    # Panel 2: Warped source
    axes[0, 1].imshow(img1_warped, cmap='gray')
    axes[0, 1].set_title("Source Warped to Reference")
    axes[0, 1].axis('off')
    
    # Panel 3: Color overlay (magenta = perfect alignment)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Color Overlay (Red=Warped, Green=Ref, Magenta=Aligned)")
    axes[1, 0].axis('off')
    
    # Panel 4: Match visualization (inliers only)
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, 
        [m for m, valid in zip(matches, axis_data)][:50],  # top 50 for clarity
        None, 
        matchColor=(0, 255, 0), 
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    ) if 'axis_data' in locals() else cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
    
    axes[1, 1].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"SIFT Matches ({len(matches)} total)")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")
    plt.close()


def save_metrics(metrics, output_path):
    """Save metrics as JSON for reporting."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics: {output_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(xml_path1, xml_path2, output_dir=None):
    """
    Full registration pipeline.
    
    Args:
        xml_path1: Path to reference image .XML (PDS4 label)
        xml_path2: Path to source image .XML (PDS4 label)
        output_dir: Where to save results (default: current dir)
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"LUNAR REGISTRATION BASELINE")
    print(f"{'='*70}\n")
    
    # Load images
    print("[1/6] Loading images...")
    img1 = load_pds4_image(Path(xml_path1))
    img2 = load_pds4_image(Path(xml_path2))
    
    # Preprocess
    print("\n[2/6] Preprocessing (CLAHE)...")
    img1_prep = preprocess_image(img1)
    img2_prep = preprocess_image(img2)
    
    # SIFT detection & matching
    print("\n[3/6] SIFT detection & matching...")
    kp1, kp2, matches = detect_and_match(img1_prep, img2_prep)
    
    # RANSAC homography
    print("\n[4/6] RANSAC homography estimation...")
    H, mask, metrics = estimate_homography(kp1, kp2, matches)
    
    # Warp & overlay
    print("\n[5/6] Warping & overlay...")
    img1_warped, overlay = warp_and_overlay(img1_prep, img2_prep, H)
    
    # Visualize & save
    print("\n[6/6] Generating visualization...")
    
    # Simple plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Lunar Registration Demo (SIFT + RANSAC)", fontsize=14, fontweight='bold')
    
    axes[0].imshow(img2_prep, cmap='gray')
    axes[0].set_title("Reference Image")
    axes[0].axis('off')
    
    axes[1].imshow(img1_warped, cmap='gray')
    axes[1].set_title("Source Warped")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay ({metrics['inlier_ratio']*100:.0f}% inliers)")
    axes[2].axis('off')
    
    plt.tight_layout()
    viz_path = output_dir / "registration_result.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    
    print(f"\n{'='*70}")
    print(f"DONE! Results saved to: {output_dir}")
    print(f"{'='*70}\n")
    print(f"Metrics summary:")
    for key, val in metrics.items():
        print(f"  {key}: {val}")
    
    return H, metrics, viz_path


if __name__ == "__main__":
    import sys
    
    # Usage: python lunar_registration_baseline.py <xml1_path> <xml2_path> [output_dir]
    if len(sys.argv) < 3:
        print("Usage: python lunar_registration_baseline.py <xml1_path> <xml2_path> [output_dir]")
        print("\nExample:")
        print("  python lunar_registration_baseline.py data/ch2_tmc_*.xml data/ch2_tmc_*.xml ./results/")
        sys.exit(1)
    
    xml1 = sys.argv[1]
    xml2 = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        main(xml1, xml2, out_dir)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
