#!/usr/bin/env python3
"""
Lunar Image Registration with LoFTR + RANSAC
Replaces SIFT with dense LoFTR matching. Includes bidirectional matching,
confidence filtering, and sub-pixel refinement.

Usage:
    python lunar_registration_loftr.py \
        --source source.png \
        --target target.png \
        --output results_loftr \
        --confidence_threshold 0.1 \
        --bidirectional
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import argparse
from typing import Tuple, List, Dict, Optional

# ============================================================================
# LOFTR MODEL SETUP
# ============================================================================

def init_loftr(device='cpu', checkpoint_path=None):
    """
    Initialize LoFTR model. Downloads checkpoint if not found.
    
    Args:
        device: 'cpu' or 'cuda'
        checkpoint_path: Path to pretrained weights. If None, tries default location.
    
    Returns:
        matcher: LoFTR model in eval mode
    """
    try:
        import os, sys
        _loftr_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LoFTR')
        if _loftr_dir not in sys.path:
            sys.path.insert(0, _loftr_dir)
        from src.loftr import LoFTR as LoFTR_, default_cfg
    except ImportError:
        raise ImportError(
            "LoFTR not installed correctly. Make sure you've cloned it as a 'LoFTR' subfolder\n"
            "inside your project folder, and installed its dependencies:\n"
            "  pip install einops yacs kornia loguru joblib pytorch-lightning"
        )
    
    # Initialize model using LoFTR's own ready-made default config
    matcher = LoFTR_(config=default_cfg)
    matcher = matcher.eval().to(device)
    
    # Load checkpoint
    if checkpoint_path is None:
        # Try default paths
        default_paths = [
            'checkpoints/outdoor_ds.ckpt',
            'LoFTR/checkpoints/outdoor_ds.ckpt',
            '/workspace/LoFTR/checkpoints/outdoor_ds.ckpt'
        ]
        for p in default_paths:
            if Path(p).exists():
                checkpoint_path = p
                break
    
    if checkpoint_path is None:
        raise FileNotFoundError(
            "Checkpoint not found. Download from:\n"
            "https://github.com/zju3dv/LoFTR/releases/download/v0.1/outdoor_ds.ckpt\n"
            "Place in: LoFTR/checkpoints/outdoor_ds.ckpt"
        )
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    matcher.load_state_dict(state_dict['state_dict'])
    
    print(f"✓ LoFTR initialized on {device}")
    print(f"✓ Loaded checkpoint: {checkpoint_path}")
    
    return matcher


# ============================================================================
# IMAGE LOADING & PREPROCESSING
# ============================================================================

def load_image(img_path):
    """Load image (PNG, JPG, or PDS4 .XML via GDAL)."""
    img_path = Path(img_path)
    
    if img_path.suffix.lower() in ('.xml', '.img', '.lbl'):
        # PDS4 (.xml label, Chandrayaan-2) or PDS3 (.img / .lbl, LRO NAC) via GDAL
        try:
            from osgeo import gdal
        except ImportError:
            raise ImportError("GDAL required for PDS3/PDS4. Run: pip install gdal")
        
        ds = gdal.Open(str(img_path))
        if ds is None:
            raise FileNotFoundError(f"GDAL couldn't open {img_path}")
        
        band = ds.GetRasterBand(1)
        img_array = band.ReadAsArray()
        
        if img_array.dtype != np.uint8:
            img_min, img_max = img_array.min(), img_array.max()
            img_array = np.clip(
                (img_array - img_min) / (img_max - img_min + 1e-8) * 255, 
                0, 255
            ).astype(np.uint8)
    else:
        # Standard image (PNG, JPG, etc.)
        img_array = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            raise FileNotFoundError(f"Couldn't load image: {img_path}")
    
    print(f"✓ Loaded {img_path.name}: shape {img_array.shape}, dtype {img_array.dtype}")
    return img_array


def preprocess_image(img):
    """CLAHE for illumination robustness."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img)
    
    return img_enhanced


# ============================================================================
# LOFTR MATCHING (DENSE)
# ============================================================================

def loftr_match(matcher, img1, img2, device='cpu', max_dim=800):
    """
    Run LoFTR dense matching.
    
    Args:
        matcher: LoFTR model
        img1, img2: Grayscale NumPy arrays (uint8)
        device: 'cpu' or 'cuda'
        max_dim: cap on the longer side (px) fed to LoFTR, applied to both
                 images independently. Default 800 keeps memory use reasonable
                 on 4-6GB cards. Callers that pre-scale img1 themselves (e.g.
                 a multi-scale search) should pass a matching max_dim, or this
                 will silently re-clip every pre-scaled variant back down to
                 the same 800px and make the pre-scaling a no-op.
    
    Returns:
        mkpts0, mkpts1: (N, 2) matched points in img1 and img2
        mconf: (N,) confidence scores
        runtime: inference time in seconds
    """
    import torch

    def _resize_to_fit(img, max_dim=max_dim):
        # LoFTR's positional encoding table caps the coarse (1/8-scale) feature
        # grid at 256x256 (~2048px), but on smaller GPUs the real limit is VRAM,
        # not that architectural cap. 800px keeps memory use reasonable on 4-6GB cards.
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img, scale

    def _pad_to_multiple(img, multiple=8):
        # LoFTR's backbone downsamples by 8x internally; both dimensions
        # must be exact multiples of 8 or the coarse/fine feature maps
        # end up mismatched sizes and it crashes.
        h, w = img.shape[:2]
        new_h = ((h + multiple - 1) // multiple) * multiple
        new_w = ((w + multiple - 1) // multiple) * multiple
        if new_h == h and new_w == w:
            return img, h, w
        padded = np.zeros((new_h, new_w), dtype=img.dtype)
        padded[:h, :w] = img
        return padded, h, w

    img1_small, scale1 = _resize_to_fit(img1)
    img2_small, scale2 = _resize_to_fit(img2)
    img1_padded, h1, w1 = _pad_to_multiple(img1_small)
    img2_padded, h2, w2 = _pad_to_multiple(img2_small)

    # Prepare inputs (LoFTR expects float32, normalized to [0, 1])
    img1_t = torch.from_numpy(img1_padded).float()[None, None] / 255.0  # (1, 1, H, W)
    img2_t = torch.from_numpy(img2_padded).float()[None, None] / 255.0
    
    img1_t = img1_t.to(device)
    img2_t = img2_t.to(device)
    
    # Forward pass
    start = time.time()
    try:
        with torch.no_grad():
            batch = {'image0': img1_t, 'image1': img2_t}
            matcher(batch)
    except torch.cuda.OutOfMemoryError:
        del img1_t, img2_t
        if device != 'cpu':
            torch.cuda.empty_cache()
        raise
    runtime = time.time() - start
    
    # Extract matches
    mkpts0 = batch['mkpts0_f'].cpu().numpy()  # (N, 2) in img1
    mkpts1 = batch['mkpts1_f'].cpu().numpy()  # (N, 2) in img2
    mconf = batch['mconf'].cpu().numpy()       # (N,) confidence

    # Drop any matches that fell inside the padded (fake, black) region
    valid = (mkpts0[:, 0] < w1) & (mkpts0[:, 1] < h1) & (mkpts1[:, 0] < w2) & (mkpts1[:, 1] < h2)
    mkpts0, mkpts1, mconf = mkpts0[valid], mkpts1[valid], mconf[valid]

    # Undo the internal resize so returned points are in the ORIGINAL img1/img2 coordinate frame
    mkpts0 = mkpts0 / scale1
    mkpts1 = mkpts1 / scale2

    print(f"✓ LoFTR: {len(mkpts0)} matches (runtime: {runtime:.2f}s)")
    
    return mkpts0, mkpts1, mconf, runtime


def filter_by_confidence(mkpts0, mkpts1, mconf, threshold=0.1):
    """Filter matches by confidence."""
    mask = mconf >= threshold
    mkpts0_f = mkpts0[mask]
    mkpts1_f = mkpts1[mask]
    mconf_f = mconf[mask]
    
    print(f"✓ Confidence filter (thresh={threshold}): {len(mkpts0)} → {len(mkpts0_f)} matches")
    
    return mkpts0_f, mkpts1_f, mconf_f


def bidirectional_matching(matcher, img1, img2, mkpts0, mkpts1, mconf, device='cpu'):
    """
    Bidirectional matching: keep only matches that are consistent in both directions.
    
    Args:
        mkpts0, mkpts1, mconf: Forward matches (img1 → img2)
    
    Returns:
        mkpts0_bi, mkpts1_bi: Bidirectional-consistent matches only
    """
    import torch
    
    # Reverse match: img2 → img1
    mkpts2_rev, mkpts1_rev, mconf_rev, _ = loftr_match(matcher, img2, img1, device)
    
    print(f"✓ Reverse LoFTR: {len(mkpts2_rev)} matches")
    
    # Find mutual matches: pts that appear in both forward and reverse
    # Forward: (p1 in img1) → (p2 in img2)
    # Reverse: (p2 in img2) → (p1 in img1)
    # Mutual: if forward p1→p2, then reverse p2→p1 exists
    
    mutual_indices = []
    for i, (pt1, pt2) in enumerate(zip(mkpts0, mkpts1)):
        # Check if this match appears in reverse direction
        # In reverse: we're looking for pt2 in mkpts2_rev that maps to pt1 in mkpts1_rev
        for j, (pt2_rev, pt1_rev) in enumerate(zip(mkpts2_rev, mkpts1_rev)):
            # Tolerance: ±1 pixel for floating point matching
            if (np.linalg.norm(pt2 - pt2_rev) < 1.0 and 
                np.linalg.norm(pt1 - pt1_rev) < 1.0):
                mutual_indices.append(i)
                break
    
    mkpts0_bi = mkpts0[mutual_indices]
    mkpts1_bi = mkpts1[mutual_indices]
    mconf_bi = mconf[mutual_indices]
    
    print(f"✓ Bidirectional filter: {len(mkpts0)} → {len(mkpts0_bi)} mutual matches")
    
    return mkpts0_bi, mkpts1_bi, mconf_bi


# ============================================================================
# RANSAC GEOMETRIC VERIFICATION
# ============================================================================

def ransac_homography(pts1, pts2, ransac_threshold=5.0):
    """
    RANSAC homography estimation.
    
    Args:
        pts1, pts2: (N, 2) point arrays
        ransac_threshold: Reprojection threshold in pixels
    
    Returns:
        H: (3, 3) homography matrix
        mask: Inlier mask
        metrics: Dict with inlier stats
    """
    if len(pts1) < 4:
        raise ValueError(f"Need ≥4 matches for RANSAC, got {len(pts1)}")
    
    H, mask = cv2.findHomography(
        np.float32(pts1), 
        np.float32(pts2),
        cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold
    )
    
    if H is None:
        raise ValueError("RANSAC failed to estimate homography")
    
    inliers = int(mask.sum())
    inlier_ratio = inliers / len(pts1)
    
    print(f"✓ RANSAC: {inliers}/{len(pts1)} inliers ({100*inlier_ratio:.1f}%)")
    
    metrics = {
        "total_matches": len(pts1),
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio)
    }
    
    return H, mask, metrics


def subpixel_refinement(img1, img2, pts1_inliers, H):
    """
    Refine inlier points to sub-pixel accuracy using corner detection.
    
    Args:
        pts1_inliers: Inlier points in img1
        H: Homography matrix
    
    Returns:
        pts1_refined, pts2_refined: Refined point locations
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    
    pts1_refined = cv2.cornerSubPix(
        img1, 
        np.float32(pts1_inliers).reshape(-1, 1, 2),
        (5, 5), 
        (-1, -1), 
        criteria
    ).reshape(-1, 2)
    
    # Warp refined points using homography
    pts1_h = np.hstack([pts1_refined, np.ones((len(pts1_refined), 1))])
    pts2_h = pts1_h @ H.T
    pts2_refined = (pts2_h[:, :2] / pts2_h[:, 2:3]).astype(np.float32)
    
    # Refine in img2 too
    pts2_refined = cv2.cornerSubPix(
        img2,
        pts2_refined.reshape(-1, 1, 2),
        (5, 5),
        (-1, -1),
        criteria
    ).reshape(-1, 2)
    
    print(f"✓ Sub-pixel refinement: {len(pts1_refined)} inliers refined")
    
    return pts1_refined, pts2_refined


# ============================================================================
# WARP & VISUALIZE
# ============================================================================

def warp_and_overlay(img1, img2, H):
    """Warp img1 to img2 space and create overlay."""
    h, w = img2.shape[:2]
    img1_warped = cv2.warpPerspective(img1, H, (w, h))
    
    # RGB overlay: Red=warped, Green=target, Magenta=perfect alignment
    overlay = np.stack([img1_warped, img2, img1_warped], axis=2)
    
    return img1_warped, overlay


def plot_results(img1, img2, img1_warped, overlay, mkpts0, mkpts1, H, 
                 output_dir, method_name='LoFTR'):
    """Generate and save visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Lunar Registration — {method_name} + RANSAC", fontsize=14, fontweight='bold')
    
    # Panel 1: Reference
    axes[0, 0].imshow(img2, cmap='gray')
    axes[0, 0].set_title("Reference Image")
    axes[0, 0].axis('off')
    
    # Panel 2: Warped source
    axes[0, 1].imshow(img1_warped, cmap='gray')
    axes[0, 1].set_title("Source Warped to Reference")
    axes[0, 1].axis('off')
    
    # Panel 3: Overlay
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(f"Color Overlay ({len(mkpts0)} matches)")
    axes[1, 0].axis('off')
    
    # Panel 4: Matches visualization
    h, w = img1.shape[:2]
    img1_display = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1
    
    h2, w2 = img2.shape[:2]
    img2_display = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2
    
    # Composite image for match display
    composite = np.hstack([
        cv2.resize(img1_display, (400, 300)),
        cv2.resize(img2_display, (400, 300))
    ])
    
    # Draw matches (sample 50 for clarity)
    sample_indices = np.linspace(0, len(mkpts0)-1, min(50, len(mkpts0)), dtype=int)
    for idx in sample_indices:
        pt1 = tuple((mkpts0[idx] * 400 / w).astype(int))
        pt2 = tuple((mkpts1[idx] * 400 / w2 + np.array([400, 0])).astype(int))
        cv2.line(composite, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(composite, pt1, 3, (0, 0, 255), -1)
        cv2.circle(composite, pt2, 3, (0, 0, 255), -1)
    
    axes[1, 1].imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"Match Visualization")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_path = Path(output_dir) / f"registration_result_{method_name.lower()}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualization: {output_path}")
    return output_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(args):
    """Full registration pipeline."""
    import torch
    
    print(f"\n{'='*70}")
    print(f"LUNAR REGISTRATION v2 — LoFTR + RANSAC")
    print(f"{'='*70}\n")
    
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Device: {device}\n")
    
    # Load images
    print("[1/7] Loading images...")
    img1 = load_image(args.source)
    img2 = load_image(args.target)
    
    # Preprocess
    print("\n[2/7] Preprocessing (CLAHE)...")
    img1_prep = preprocess_image(img1)
    img2_prep = preprocess_image(img2)
    
    # Initialize LoFTR
    print("\n[3/7] Initializing LoFTR...")
    matcher = init_loftr(device, args.checkpoint)
    
    # LoFTR matching
    print("\n[4/7] Running LoFTR matching...")
    mkpts0, mkpts1, mconf, runtime_loftr = loftr_match(matcher, img1_prep, img2_prep, device)
    
    # Confidence filtering
    if args.confidence_threshold > 0:
        print(f"\n[4.5/7] Applying confidence filter (threshold={args.confidence_threshold})...")
        mkpts0, mkpts1, mconf = filter_by_confidence(
            mkpts0, mkpts1, mconf, args.confidence_threshold
        )
    
    # Bidirectional matching
    if args.bidirectional:
        print(f"\n[4.7/7] Running bidirectional matching...")
        mkpts0, mkpts1, mconf = bidirectional_matching(
            matcher, img1_prep, img2_prep, mkpts0, mkpts1, mconf, device
        )
    
    # RANSAC homography
    print("\n[5/7] RANSAC homography estimation...")
    H, mask, ransac_metrics = ransac_homography(mkpts0, mkpts1, args.ransac_threshold)
    
    # Sub-pixel refinement (optional)
    if args.subpixel:
        print("\n[5.5/7] Sub-pixel refinement...")
        mkpts0_inliers = mkpts0[mask.ravel().astype(bool)]
        try:
            mkpts0, mkpts1 = subpixel_refinement(img1_prep, img2_prep, mkpts0_inliers, H)
        except Exception as e:
            print(f"⚠ Sub-pixel refinement failed: {e}. Skipping.")
    
    # Warp & overlay
    print("\n[6/7] Warping & overlay...")
    img1_warped, overlay = warp_and_overlay(img1_prep, img2_prep, H)
    
    # Visualize
    print("\n[7/7] Generating visualization...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_results(img1_prep, img2_prep, img1_warped, overlay, mkpts0, mkpts1, H, 
                 output_dir, method_name='LoFTR')
    
    # Compile metrics
    metrics = {
        "method": "LoFTR",
        "total_matches": len(mkpts0),
        "inliers": ransac_metrics["inliers"],
        "inlier_ratio": ransac_metrics["inlier_ratio"],
        "confidence_threshold": float(args.confidence_threshold),
        "bidirectional": args.bidirectional,
        "subpixel_refined": args.subpixel,
        "ransac_threshold_px": args.ransac_threshold,
        "runtime_loftr_s": float(runtime_loftr)
    }
    
    # Save metrics
    metrics_path = output_dir / "metrics_loftr.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ DONE! Results saved to: {output_dir}")
    print(f"{'='*70}\n")
    print("Metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val}")
    
    return metrics


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys
    
    parser = argparse.ArgumentParser(
        description='Lunar image registration using LoFTR + RANSAC'
    )
    parser.add_argument('--source', required=True, help='Source image path (.png, .jpg, or .xml for PDS4)')
    parser.add_argument('--target', required=True, help='Target/reference image path')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--checkpoint', default=None, help='LoFTR checkpoint path (auto-finds if not given)')
    parser.add_argument('--confidence_threshold', type=float, default=0.1, 
                        help='Filter matches by confidence (0.0-1.0)')
    parser.add_argument('--bidirectional', action='store_true', 
                        help='Enable bidirectional matching')
    parser.add_argument('--subpixel', action='store_true', 
                        help='Enable sub-pixel refinement')
    parser.add_argument('--ransac_threshold', type=float, default=5.0, 
                        help='RANSAC reprojection threshold (pixels)')
    parser.add_argument('--cpu', action='store_true', 
                        help='Force CPU mode (default: GPU if available)')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
