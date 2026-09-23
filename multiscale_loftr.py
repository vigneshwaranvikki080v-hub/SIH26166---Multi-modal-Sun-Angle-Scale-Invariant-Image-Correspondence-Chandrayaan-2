#!/usr/bin/env python3
"""
Multi-Scale LoFTR wrapper — addresses the SCALE VARIATION challenge.

Chandrayaan-2 instruments and LRO NAC operate at very different spatial resolutions
(e.g. TMC-2 ~5m/px, OHRC ~25cm/px, LRO NAC ~0.5-2m/px). LoFTR alone has no explicit
scale search — it matches at whatever resolution you feed it. This wrapper tests the
source image resized at several candidate scale factors, runs LoFTR at each, and keeps
whichever scale produces the most RANSAC inliers.

Requires: lunar_registration_loftr.py's core functions (imported directly).

Usage:
    python multiscale_loftr.py --source source.png --target target.png \
        --output results_loftr_ms --scales 0.5,0.75,1.0,1.25,1.5
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import argparse
import sys

# Reuse core functions from the single-scale LoFTR script
sys.path.insert(0, str(Path(__file__).parent))
from lunar_registration_loftr import (
    init_loftr, load_image, preprocess_image, loftr_match,
    filter_by_confidence, ransac_homography, warp_and_overlay
)


def rescale_image(img, scale_factor):
    """Rescale image by a given factor (returns resized image + the factor for later point rescaling)."""
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    if new_h < 32 or new_w < 32:
        return None
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_CUBIC)
    return resized


def run_single_scale(matcher, img1_full, img2, scale_factor, device, confidence_threshold, ransac_threshold,
                      base_max_dim=800, safety_cap=1600):
    """
    Run LoFTR at one candidate scale factor. Returns match results rescaled back to
    the ORIGINAL source image coordinate system so downstream homography is consistent.

    NOTE: loftr_match() internally re-caps whatever it's given to `max_dim` px (800 by
    default) before running the model. Without passing a matching max_dim here, every
    scale_factor collapses back down to the same ~800px image once it reaches LoFTR --
    the scale sweep becomes a no-op that just re-tests the same effective resolution
    five times. We scale max_dim WITH scale_factor so each candidate genuinely reaches
    the model at a different resolution. base_max_dim=800 keeps scale_factor=1.0
    identical to the old fixed-800 behavior; safety_cap guards VRAM at the top end.
    """
    img1_scaled = rescale_image(img1_full, scale_factor)
    if img1_scaled is None:
        return None

    effective_max_dim = int(np.clip(round(base_max_dim * scale_factor), 200, safety_cap))
    mkpts0, mkpts1, mconf, runtime = loftr_match(matcher, img1_scaled, img2, device, max_dim=effective_max_dim)

    if confidence_threshold > 0 and len(mkpts0) > 0:
        mkpts0, mkpts1, mconf = filter_by_confidence(mkpts0, mkpts1, mconf, confidence_threshold)

    if len(mkpts0) < 4:
        return {
            "scale_factor": scale_factor, "matches": 0, "inliers": 0,
            "inlier_ratio": 0.0, "H": None, "mkpts0": None, "mkpts1": None, "runtime": runtime
        }

    # Rescale matched points from the resized-source coordinate system back to full-res source coords
    mkpts0_full = mkpts0 / scale_factor

    try:
        H, mask, ransac_metrics = ransac_homography(mkpts0_full, mkpts1, ransac_threshold)
    except ValueError:
        return {
            "scale_factor": scale_factor, "matches": len(mkpts0), "inliers": 0,
            "inlier_ratio": 0.0, "H": None, "mkpts0": None, "mkpts1": None, "runtime": runtime
        }

    return {
        "scale_factor": scale_factor,
        "matches": len(mkpts0),
        "inliers": ransac_metrics["inliers"],
        "inlier_ratio": ransac_metrics["inlier_ratio"],
        "H": H,
        "mkpts0": mkpts0_full,
        "mkpts1": mkpts1,
        "runtime": runtime
    }


def multiscale_loftr_match(matcher, img1_full, img2, scales, device='cpu',
                           confidence_threshold=0.1, ransac_threshold=5.0):
    """
    Try LoFTR at each candidate scale. Return the best result by inlier count.
    """
    print(f"\nTesting {len(scales)} scale factors: {scales}")
    results = []
    for sf in scales:
        print(f"\n  --- Scale factor {sf} ---")
        try:
            res = run_single_scale(matcher, img1_full, img2, sf, device, confidence_threshold, ransac_threshold)
        except Exception as e:
            print(f"  Scale {sf} failed: {e}")
            try:
                import torch
                if device != 'cpu':
                    torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        if res is not None:
            results.append(res)
            print(f"  Scale {sf}: {res['matches']} matches, {res['inliers']} inliers ({res['inlier_ratio']*100:.1f}%)")

    if not results:
        raise ValueError("No valid results at any scale")

    best = max(results, key=lambda r: r['inliers'])
    print(f"\n✓ Best scale factor: {best['scale_factor']} ({best['inliers']} inliers)")

    return best, results


def main(args):
    import torch

    scales = [float(s) for s in args.scales.split(',')]
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    print(f"\n{'='*70}\nMULTI-SCALE LoFTR — Scale Variation Robustness\n{'='*70}\n")
    print(f"Device: {device}")

    img1 = load_image(args.source)
    img2 = load_image(args.target)
    img1p = preprocess_image(img1)
    img2p = preprocess_image(img2)

    matcher = init_loftr(device, args.checkpoint)

    best, all_results = multiscale_loftr_match(
        matcher, img1p, img2p, scales, device,
        args.confidence_threshold, args.ransac_threshold
    )

    if best["H"] is None:
        raise ValueError("Best scale still produced no valid homography — try different scale range")

    img1_warped, overlay = warp_and_overlay(img1p, img2p, best["H"])

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualization: main result + scale search summary bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Multi-Scale LoFTR (best scale={best['scale_factor']})", fontsize=14, fontweight='bold')
    axes[0].imshow(img2p, cmap='gray'); axes[0].set_title("Reference"); axes[0].axis('off')
    axes[1].imshow(img1_warped, cmap='gray'); axes[1].set_title("Source Warped"); axes[1].axis('off')
    axes[2].imshow(overlay); axes[2].set_title(f"Overlay ({best['inlier_ratio']*100:.0f}% inliers)"); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "registration_result_multiscale.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Scale search bar chart (good for PPT — shows the search worked)
    fig, ax = plt.subplots(figsize=(8, 5))
    sfs = [r['scale_factor'] for r in all_results]
    inliers = [r['inliers'] for r in all_results]
    colors = ['#4ECDC4' if sf == best['scale_factor'] else '#CCCCCC' for sf in sfs]
    ax.bar([str(s) for s in sfs], inliers, color=colors, edgecolor='black')
    ax.set_xlabel('Scale Factor Tested')
    ax.set_ylabel('RANSAC Inliers')
    ax.set_title('Scale Search: Inliers per Candidate Scale')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "scale_search_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    metrics = {
        "method": "LoFTR-MultiScale",
        "best_scale_factor": best["scale_factor"],
        "total_matches": best["matches"],
        "inliers": best["inliers"],
        "inlier_ratio": best["inlier_ratio"],
        "scales_tested": scales,
        "all_scale_results": [
            {"scale_factor": r["scale_factor"], "matches": r["matches"],
             "inliers": r["inliers"], "inlier_ratio": r["inlier_ratio"]}
            for r in all_results
        ]
    }
    with open(output_dir / "metrics_multiscale.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Saved: {output_dir}/registration_result_multiscale.png")
    print(f"✓ Saved: {output_dir}/scale_search_summary.png")
    print(f"✓ Saved: {output_dir}/metrics_multiscale.json")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-scale LoFTR for scale-variant lunar image registration')
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scales', default='0.5,0.75,1.0,1.25,1.5',
                        help='Comma-separated scale factors to test')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--confidence_threshold', type=float, default=0.1)
    parser.add_argument('--ransac_threshold', type=float, default=5.0)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
