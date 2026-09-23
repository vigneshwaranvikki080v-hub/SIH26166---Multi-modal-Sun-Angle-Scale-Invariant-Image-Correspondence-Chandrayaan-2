#!/usr/bin/env python3
"""
Unified Lunar Registration Pipeline v2 — adds active uniformity ENFORCEMENT
(grid bucket capping, not just scoring) and optional TPS non-rigid refinement
for non-planar terrain.

New vs unified_pipeline.py:
    --bucket_grid_size / --max_per_cell : cap matches per grid cell before RANSAC
    --tps                                : after homography, also fit a TPS warp
                                            and save it as an additional output

Usage:
    python unified_pipeline_v2.py --source source.png --target target.png \
        --output results_v2 --bucket_grid_size 10 --max_per_cell 50 --tps
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import csv
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent))

from uniformity_metrics import compute_uniformity_score, visualize_uniformity
from bucket_uniform import bucket_cap_matches
from rift_matcher import (
    rift_match, load_image as rift_load_image, preprocess_image as rift_preprocess
)


def run_loftr_branch(source_path, target_path, args):
    try:
        import torch
        from lunar_registration_loftr import init_loftr, load_image, preprocess_image
        from multiscale_loftr import multiscale_loftr_match

        device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
        img1 = preprocess_image(load_image(source_path))
        img2 = preprocess_image(load_image(target_path))

        matcher = init_loftr(device, args.checkpoint)
        scales = [float(s) for s in args.scales.split(',')]

        best, _ = multiscale_loftr_match(
            matcher, img1, img2, scales, device,
            args.confidence_threshold, args.ransac_threshold
        )
        if best["H"] is None:
            print("⚠ LoFTR branch: no valid homography at any scale")
            return None

        pts1, pts2 = best["mkpts0"], best["mkpts1"]
        conf = np.ones(len(pts1))  # fallback confidence for bucket capping

        return {
            "method": "LoFTR-MultiScale", "img1": img1, "img2": img2,
            "pts1": pts1, "pts2": pts2, "conf": conf,
            "scale_factor": best["scale_factor"]
        }
    except Exception as e:
        print(f"⚠ LoFTR branch failed: {e}")
        return None


def run_rift_branch(source_path, target_path, args):
    try:
        img1 = rift_preprocess(rift_load_image(source_path))
        img2 = rift_preprocess(rift_load_image(target_path))

        pts1, pts2, runtime = rift_match(
            img1, img2, nscale=args.rift_nscale, norient=args.rift_norient,
            max_keypoints=args.rift_max_keypoints, ratio_threshold=args.rift_ratio_threshold
        )
        if len(pts1) < 4:
            print("⚠ RIFT branch: fewer than 4 matches")
            return None

        conf = np.ones(len(pts1))

        return {
            "method": "RIFT", "img1": img1, "img2": img2,
            "pts1": pts1, "pts2": pts2, "conf": conf
        }
    except Exception as e:
        print(f"⚠ RIFT branch failed: {e}")
        return None


def process_candidate(result, args):
    """Apply bucket capping, RANSAC, and scoring to one candidate branch's raw matches."""
    if result is None:
        return None

    img2_shape = result["img2"].shape

    pts1_capped, pts2_capped, conf_capped = bucket_cap_matches(
        result["pts1"], result["pts2"], result["conf"], img2_shape,
        grid_size=args.bucket_grid_size, max_per_cell=args.max_per_cell
    )

    if len(pts1_capped) < 4:
        print(f"⚠ {result['method']}: fewer than 4 matches after bucket capping")
        return None

    H, mask = cv2.findHomography(pts1_capped, pts2_capped, cv2.RANSAC,
                                  ransacReprojThreshold=args.ransac_threshold)
    if H is None:
        print(f"⚠ {result['method']}: RANSAC failed after bucket capping")
        return None

    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())

    uniformity, coverage, _ = compute_uniformity_score(
        pts2_capped[inlier_mask], img2_shape, grid_size=args.bucket_grid_size
    )

    score = inliers * (1.0 + 0.3 * uniformity)

    return {
        "method": result["method"],
        "img1": result["img1"], "img2": result["img2"],
        "H": H,
        "pts1_inliers": pts1_capped[inlier_mask],
        "pts2_inliers": pts2_capped[inlier_mask],
        "total_matches": len(pts1_capped),
        "inliers": inliers,
        "inlier_ratio": inliers / len(pts1_capped),
        "uniformity": uniformity,
        "coverage": coverage,
        "score": score,
        "scale_factor": result.get("scale_factor", 1.0)
    }


def subpixel_refine(img1, img2, pts1, pts2):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    try:
        pts1_r = cv2.cornerSubPix(img1, np.float32(pts1).reshape(-1, 1, 2), (5, 5), (-1, -1), criteria).reshape(-1, 2)
        pts2_r = cv2.cornerSubPix(img2, np.float32(pts2).reshape(-1, 1, 2), (5, 5), (-1, -1), criteria).reshape(-1, 2)
        return pts1_r, pts2_r
    except Exception as e:
        print(f"⚠ Sub-pixel refinement skipped: {e}")
        return pts1, pts2


def compute_rmse(H, pts1, pts2):
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    proj = pts1_h @ H.T
    proj = proj[:, :2] / proj[:, 2:3]
    return float(np.sqrt(np.mean(np.linalg.norm(proj - pts2, axis=1) ** 2)))


def main(args):
    print(f"\n{'='*70}\nUNIFIED PIPELINE v2 — LoFTR + RIFT + Bucket Uniformity + TPS\n{'='*70}\n")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = {}
    if not args.rift_only:
        print("\n### Branch 1: Multi-Scale LoFTR ###")
        raw["loftr"] = run_loftr_branch(args.source, args.target, args)
    if not args.loftr_only:
        print("\n### Branch 2: RIFT ###")
        raw["rift"] = run_rift_branch(args.source, args.target, args)

    # Branch 3: pool BOTH matchers' raw matches together before RANSAC.
    # Safe because both branches return points in the same (original image)
    # coordinate frame -- same preprocessing, and each matcher's own internal
    # resizing is already undone before it returns.
    if raw.get("loftr") is not None and raw.get("rift") is not None:
        print("\n### Branch 3: LoFTR + RIFT combined pool ###")
        combined_pts1 = np.vstack([raw["loftr"]["pts1"], raw["rift"]["pts1"]])
        combined_pts2 = np.vstack([raw["loftr"]["pts2"], raw["rift"]["pts2"]])
        combined_conf = np.concatenate([raw["loftr"]["conf"], raw["rift"]["conf"]])
        raw["combined"] = {
            "method": "LoFTR+RIFT-Combined",
            "img1": raw["loftr"]["img1"], "img2": raw["loftr"]["img2"],
            "pts1": combined_pts1, "pts2": combined_pts2, "conf": combined_conf,
            "scale_factor": raw["loftr"].get("scale_factor", 1.0)
        }
        print(f"  Pooled {len(raw['loftr']['pts1'])} LoFTR + {len(raw['rift']['pts1'])} RIFT = {len(combined_pts1)} candidate matches")

    print("\n### Bucket capping + RANSAC + scoring per candidate ###")
    processed = {}
    for name, result in raw.items():
        if result is not None:
            p = process_candidate(result, args)
            if p is not None:
                processed[name] = p
                print(f"  {p['method']}: inliers={p['inliers']}, uniformity={p['uniformity']:.3f}, "
                      f"coverage={p['coverage']:.3f}, score={p['score']:.1f}")

    if not processed:
        print("\n❌ All branches failed. Check thresholds / inputs.")
        sys.exit(1)

    winner_name = max(processed, key=lambda k: processed[k]["score"])
    winner = processed[winner_name]
    print(f"\n✓ Winner: {winner['method']} (score={winner['score']:.1f})")

    print("\n### Sub-pixel refinement ###")
    pts1_r, pts2_r = subpixel_refine(winner["img1"], winner["img2"],
                                     winner["pts1_inliers"], winner["pts2_inliers"])

    H_final, _ = cv2.findHomography(pts1_r, pts2_r, cv2.RANSAC,
                                     ransacReprojThreshold=args.ransac_threshold)
    if H_final is None:
        H_final = winner["H"]
    rmse = compute_rmse(H_final, pts1_r, pts2_r)

    h, w = winner["img2"].shape[:2]
    registered = cv2.warpPerspective(winner["img1"], H_final, (w, h))
    cv2.imwrite(str(output_dir / "registered_source.png"), registered)
    np.save(output_dir / "registered_source.npy", registered)

    with open(output_dir / "match_points.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["source_x", "source_y", "reference_x", "reference_y"])
        for (sx, sy), (rx, ry) in zip(pts1_r, pts2_r):
            writer.writerow([f"{sx:.3f}", f"{sy:.3f}", f"{rx:.3f}", f"{ry:.3f}"])

    _, _, grid_counts = compute_uniformity_score(pts2_r, winner["img2"].shape, grid_size=args.bucket_grid_size)
    visualize_uniformity(grid_counts, output_dir / "uniformity_heatmap.png",
                         title=f"Match Density ({winner['method']}, bucket-capped)")

    tps_output = None
    if args.tps:
        print("\n### TPS non-rigid refinement (non-planar terrain) ###")
        try:
            from tps_warp import compare_homography_vs_tps
            _, warped_tps, rmse_h = compare_homography_vs_tps(
                winner["img1"], winner["img2"], H_final, pts1_r, pts2_r,
                output_path=output_dir / "homography_vs_tps.png"
            )
            cv2.imwrite(str(output_dir / "registered_source_tps.png"), warped_tps)
            tps_output = "registered_source_tps.png"
            print(f"✓ Saved TPS-refined registered image: {tps_output}")
        except ImportError as e:
            print(f"⚠ TPS skipped: {e}")

    overlay = np.stack([registered, winner["img2"], registered], axis=2)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Final Registration v2 — Winner: {winner['method']}", fontsize=14, fontweight='bold')
    axes[0].imshow(winner["img2"], cmap='gray'); axes[0].set_title("Reference"); axes[0].axis('off')
    axes[1].imshow(registered, cmap='gray'); axes[1].set_title("Registered Source"); axes[1].axis('off')
    axes[2].imshow(overlay); axes[2].set_title(f"Overlay (RMSE={rmse:.2f}px)"); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_panel.png", dpi=150, bbox_inches='tight')
    plt.close()

    metrics = {
        "winning_method": winner["method"],
        "candidates_tested": list(processed.keys()),
        "total_matches": winner["total_matches"],
        "inliers": len(pts1_r),
        "inlier_ratio": winner["inlier_ratio"],
        "rmse_px": rmse,
        "uniformity_score": winner["uniformity"],
        "coverage_ratio": winner["coverage"],
        "scale_factor_used": winner["scale_factor"],
        "bucket_grid_size": args.bucket_grid_size,
        "max_per_cell": args.max_per_cell,
        "tps_applied": args.tps and tps_output is not None,
        "per_candidate_scores": {
            k: {"score": v["score"], "uniformity": v["uniformity"],
                "coverage": v["coverage"], "inliers": v["inliers"]}
            for k, v in processed.items()
        }
    }
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*70}\n✓ ALL DELIVERABLES SAVED TO: {output_dir}\n{'='*70}")
    print(f"\nFinal metrics:\n{json.dumps(metrics, indent=2)}")
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified LoFTR+RIFT pipeline v2 with bucket uniformity + TPS')
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--output', required=True)

    parser.add_argument('--loftr_only', action='store_true')
    parser.add_argument('--rift_only', action='store_true')

    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--scales', default='0.5,0.75,1.0,1.25,1.5')
    parser.add_argument('--confidence_threshold', type=float, default=0.1)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--rift_nscale', type=int, default=4)
    parser.add_argument('--rift_norient', type=int, default=6)
    parser.add_argument('--rift_max_keypoints', type=int, default=2000)
    parser.add_argument('--rift_ratio_threshold', type=float, default=0.85)

    parser.add_argument('--ransac_threshold', type=float, default=5.0)

    parser.add_argument('--bucket_grid_size', type=int, default=10, help='Grid size for uniformity enforcement')
    parser.add_argument('--max_per_cell', type=int, default=50, help='Max matches kept per grid cell')
    parser.add_argument('--tps', action='store_true', help='Also compute TPS non-rigid refinement')

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
