#!/usr/bin/env python3
"""
RIFT (Radiation-variation Insensitive Feature Transform) — simplified reference implementation.

Based on: Li et al., "RIFT: Multi-modal Image Matching Based on Radiation-Variation
Insensitive Feature Transform", IEEE TIP 2019.

Core idea: matches images using PHASE CONGRUENCY structure (frequency-domain edge/corner
strength) instead of raw pixel intensity. Phase congruency is invariant to monotonic
intensity transforms, which makes this robust to:
  - illumination changes (sun angle differences)
  - cross-sensor radiometric differences (different camera response curves)

This does NOT use deep learning — it's a classical computer vision pipeline, which is
exactly why it survives domain gaps that a natural-image-trained network (LoFTR) might not.

Usage:
    python rift_matcher.py --source source.png --target target.png --output results_rift
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import argparse
from scipy import ndimage


# ============================================================================
# PHASE CONGRUENCY (Kovesi-style, via phasepack if available, else fallback)
# ============================================================================

def compute_phase_congruency(img, nscale=4, norient=6):
    """
    Compute phase congruency map + orientation-wise responses.

    Returns:
        pc: (H, W) phase congruency magnitude map
        orientation_responses: (norient, H, W) log-Gabor filter responses per orientation
                                (used to build the Maximum Index Map)
    """
    try:
        from phasepack.phasecong import phasecong as pc_func
        result = pc_func(
            img.astype(np.float64), nscale=nscale, norient=norient,
            minWaveLength=3, mult=2.1, sigmaOnf=0.55
        )
        # phasepack returns (M, m, ori, ft, PC, EO, T) depending on version;
        # PC is a list of per-orientation phase congruency maps
        M = result[0]  # maximum moment (edge strength) -- used as PC magnitude
        PC_per_orient = result[4] if len(result) > 4 else None

        if PC_per_orient is not None and len(PC_per_orient) == norient:
            orientation_responses = np.stack(PC_per_orient, axis=0)
        else:
            orientation_responses = _fallback_log_gabor_bank(img, nscale, norient)

        pc = M
        return pc, orientation_responses

    except ImportError:
        print("⚠ phasepack not installed. Run: pip install phasepack")
        print("  Falling back to manual log-Gabor bank (slower, less accurate).")
        return _fallback_phase_congruency(img, nscale, norient)


def _fallback_log_gabor_bank(img, nscale, norient):
    """
    Manual log-Gabor filter bank fallback if phasepack isn't installed.
    Produces per-orientation response maps via frequency-domain log-Gabor filters.
    """
    h, w = img.shape
    img_f = np.fft.fft2(img.astype(np.float64))
    img_f_shift = np.fft.fftshift(img_f)

    y, x = np.meshgrid(np.linspace(-0.5, 0.5, h), np.linspace(-0.5, 0.5, w), indexing='ij')
    radius = np.sqrt(x**2 + y**2)
    radius[h // 2, w // 2] = 1.0  # avoid div by zero at DC
    theta = np.arctan2(-y, x)

    orientation_responses = np.zeros((norient, h, w))

    min_wavelength = 3.0
    mult = 2.1
    sigma_onf = 0.55
    d_theta_sigma = np.pi / norient / 1.2

    for o in range(norient):
        angle = o * np.pi / norient
        d_theta = np.abs(np.arctan2(np.sin(theta - angle), np.cos(theta - angle)))
        spread = np.exp(-(d_theta**2) / (2 * d_theta_sigma**2))

        response_sum = np.zeros((h, w))
        wavelength = min_wavelength
        for s in range(nscale):
            fo = 1.0 / wavelength
            log_gabor = np.exp(-(np.log(radius / fo))**2 / (2 * np.log(sigma_onf)**2))
            log_gabor[h // 2, w // 2] = 0
            filt = log_gabor * spread
            filtered = np.fft.ifft2(np.fft.ifftshift(img_f_shift * filt))
            response_sum += np.abs(filtered)
            wavelength *= mult

        orientation_responses[o] = response_sum

    pc = orientation_responses.max(axis=0)
    pc = (pc - pc.min()) / (pc.max() - pc.min() + 1e-8)

    return pc, orientation_responses


def _fallback_phase_congruency(img, nscale, norient):
    return _fallback_log_gabor_bank(img, nscale, norient)


def build_maximum_index_map(orientation_responses):
    """
    Maximum Index Map (MIM): at each pixel, record which orientation index gave the
    strongest log-Gabor response. This is the core RIFT descriptor substrate — it encodes
    LOCAL STRUCTURE ORIENTATION, not intensity, so it's stable across sensors/illumination.

    Returns:
        mim: (H, W) integer map, values in [0, norient-1]
    """
    mim = np.argmax(orientation_responses, axis=0).astype(np.uint8)
    return mim


# ============================================================================
# KEYPOINT DETECTION (on phase congruency map)
# ============================================================================

def detect_keypoints_on_pc(pc_map, max_keypoints=2000, quality=0.01, min_distance=8):
    """
    Detect keypoints using Shi-Tomasi corner detection on the phase congruency map
    (not on raw image intensity — this is what makes detection illumination-stable).
    """
    pc_8u = (np.clip(pc_map, 0, 1) * 255).astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(
        pc_8u, maxCorners=max_keypoints, qualityLevel=quality,
        minDistance=min_distance
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2)


# ============================================================================
# MIM-BASED DESCRIPTOR
# ============================================================================

def build_mim_descriptor(mim, keypoints, patch_size=48, n_bins=6, grid=6):
    """
    Build a rotation-aware descriptor per keypoint from local MIM patches.
    Similar in spirit to SIFT's spatial-histogram descriptor, but built on orientation
    INDEX (from log-Gabor bank) instead of intensity gradient.

    Returns:
        descriptors: (N, grid*grid*n_bins) array
        valid_keypoints: keypoints that had a full patch inside the image bounds
    """
    h, w = mim.shape
    half = patch_size // 2
    cell = patch_size // grid

    descriptors = []
    valid_kps = []

    for kp in keypoints:
        x, y = int(round(kp[0])), int(round(kp[1]))
        if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
            continue

        patch = mim[y - half:y + half, x - half:x + half]
        desc = []
        for gy in range(grid):
            for gx in range(grid):
                cell_patch = patch[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell]
                hist, _ = np.histogram(cell_patch, bins=n_bins, range=(0, n_bins))
                desc.extend(hist.tolist())

        desc = np.array(desc, dtype=np.float32)
        norm = np.linalg.norm(desc)
        if norm > 1e-6:
            desc = desc / norm

        descriptors.append(desc)
        valid_kps.append(kp)

    if len(descriptors) == 0:
        return np.empty((0, grid * grid * n_bins), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    return np.array(descriptors, dtype=np.float32), np.array(valid_kps, dtype=np.float32)


# ============================================================================
# MATCHING (nearest neighbor + ratio test, same principle as SIFT)
# ============================================================================

def match_descriptors(desc1, desc2, ratio_threshold=0.85):
    """
    Match MIM descriptors via brute-force nearest neighbor + Lowe's ratio test.
    Ratio threshold is looser than SIFT's typical 0.75 because MIM histograms are
    coarser than SIFT gradient descriptors.
    """
    if len(desc1) == 0 or len(desc2) == 0:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    print(f"  [debug] raw knn matches: {len(matches)}, desc1: {len(desc1)}, desc2: {len(desc2)}")

    good_matches = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    # De-duplicate: keep only the best (lowest-distance) match per target
    # keypoint (trainIdx). Without this, crossCheck=False lets one "generic"
    # target descriptor become the nearest neighbor for many different source
    # keypoints -- producing many "distinct" matches that all collapse onto
    # the same target coordinate (this was the bucket-capping /
    # subpixel-refine collapse bug: the root cause was upstream, here).
    best_per_train = {}
    for m in good_matches:
        if m.trainIdx not in best_per_train or m.distance < best_per_train[m.trainIdx].distance:
            best_per_train[m.trainIdx] = m

    return list(best_per_train.values())


# ============================================================================
# FULL RIFT PIPELINE
# ============================================================================

def rift_match(img1, img2, nscale=4, norient=6, max_keypoints=2000, ratio_threshold=0.85, max_dim=1200):
    """
    Full RIFT matching pipeline.

    Returns:
        pts1, pts2: matched point coordinates (N, 2) each
        runtime: seconds
    """
    start = time.time()

    # Phase congruency needs several full-size float64 arrays in memory at once
    # (one per orientation/scale), which can exceed available RAM on low-memory
    # machines even at moderate resolutions. Shrink first, then scale matched
    # points back up to the ORIGINAL image coordinate frame before returning.
    def _resize_to_fit(img, max_dim):
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img, scale

    img1_small, scale1 = _resize_to_fit(img1, max_dim)
    img2_small, scale2 = _resize_to_fit(img2, max_dim)

    print("  Computing phase congruency (image 1)...")
    pc1, ori1 = compute_phase_congruency(img1_small, nscale, norient)
    print("  Computing phase congruency (image 2)...")
    pc2, ori2 = compute_phase_congruency(img2_small, nscale, norient)

    mim1 = build_maximum_index_map(ori1)
    mim2 = build_maximum_index_map(ori2)

    print("  Detecting keypoints on phase congruency maps...")
    kp1 = detect_keypoints_on_pc(pc1, max_keypoints=max_keypoints)
    kp2 = detect_keypoints_on_pc(pc2, max_keypoints=max_keypoints)
    print(f"  Keypoints: img1={len(kp1)}, img2={len(kp2)}")

    print("  Building MIM descriptors...")
    desc1, kp1_valid = build_mim_descriptor(mim1, kp1)
    desc2, kp2_valid = build_mim_descriptor(mim2, kp2)

    print("  Matching descriptors...")
    matches = match_descriptors(desc1, desc2, ratio_threshold)

    pts1 = np.float32([kp1_valid[m.queryIdx] for m in matches])
    pts2 = np.float32([kp2_valid[m.trainIdx] for m in matches])

    # Undo the internal resize so returned points are in the ORIGINAL img1/img2 coordinate frame
    if len(pts1) > 0:
        pts1 = pts1 / scale1
        pts2 = pts2 / scale2

    runtime = time.time() - start
    print(f"✓ RIFT: {len(pts1)} matches (runtime: {runtime:.2f}s)")

    return pts1, pts2, runtime


# ============================================================================
# RANSAC + WARP + VISUALIZATION (shared logic, self-contained here)
# ============================================================================

def ransac_homography(pts1, pts2, ransac_threshold=5.0):
    if len(pts1) < 4:
        raise ValueError(f"Need >=4 matches for RANSAC, got {len(pts1)}")
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=ransac_threshold)
    if H is None:
        raise ValueError("RANSAC failed to estimate homography")
    inliers = int(mask.sum())
    inlier_ratio = inliers / len(pts1)
    print(f"✓ RANSAC: {inliers}/{len(pts1)} inliers ({100*inlier_ratio:.1f}%)")
    return H, mask, {"total_matches": len(pts1), "inliers": inliers, "inlier_ratio": float(inlier_ratio)}


def warp_and_overlay(img1, img2, H):
    h, w = img2.shape[:2]
    img1_warped = cv2.warpPerspective(img1, H, (w, h))
    overlay = np.stack([img1_warped, img2, img1_warped], axis=2)
    return img1_warped, overlay


def preprocess_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def load_image(img_path):
    img_path = Path(img_path)
    if img_path.suffix.lower() in ('.xml', '.img', '.lbl'):
        from osgeo import gdal
        ds = gdal.Open(str(img_path))
        if ds is None:
            raise FileNotFoundError(f"GDAL couldn't open {img_path}")
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        if arr.dtype != np.uint8:
            lo, hi = arr.min(), arr.max()
            arr = np.clip((arr - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)
        return arr
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Couldn't load image: {img_path}")
    return img


def main(args):
    print(f"\n{'='*70}\nLUNAR REGISTRATION — RIFT (Radiation-variation Insensitive)\n{'='*70}\n")

    img1 = load_image(args.source)
    img2 = load_image(args.target)
    img1p = preprocess_image(img1)
    img2p = preprocess_image(img2)

    pts1, pts2, runtime = rift_match(
        img1p, img2p,
        nscale=args.nscale, norient=args.norient,
        max_keypoints=args.max_keypoints, ratio_threshold=args.ratio_threshold
    )

    H, mask, ransac_metrics = ransac_homography(pts1, pts2, args.ransac_threshold)
    img1_warped, overlay = warp_and_overlay(img1p, img2p, H)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Lunar Registration — RIFT + RANSAC", fontsize=14, fontweight='bold')
    axes[0].imshow(img2p, cmap='gray'); axes[0].set_title("Reference"); axes[0].axis('off')
    axes[1].imshow(img1_warped, cmap='gray'); axes[1].set_title("Source Warped (RIFT)"); axes[1].axis('off')
    axes[2].imshow(overlay); axes[2].set_title(f"Overlay ({ransac_metrics['inlier_ratio']*100:.0f}% inliers)"); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "registration_result_rift.png", dpi=150, bbox_inches='tight')
    plt.close()

    metrics = {
        "method": "RIFT",
        **ransac_metrics,
        "runtime_s": float(runtime)
    }
    with open(output_dir / "metrics_rift.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Saved: {output_dir}/registration_result_rift.png")
    print(f"✓ Saved: {output_dir}/metrics_rift.json")
    print("\nMetrics:", json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RIFT-based lunar image registration')
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--nscale', type=int, default=4, help='Log-Gabor scales')
    parser.add_argument('--norient', type=int, default=6, help='Log-Gabor orientations')
    parser.add_argument('--max_keypoints', type=int, default=2000)
    parser.add_argument('--ratio_threshold', type=float, default=0.85)
    parser.add_argument('--ransac_threshold', type=float, default=5.0)
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
