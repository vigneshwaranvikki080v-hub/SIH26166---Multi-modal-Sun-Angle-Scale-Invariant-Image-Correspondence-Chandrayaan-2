#!/usr/bin/env python3
"""Tiled LoFTR: splits large images into overlapping tiles, matches each, merges to global coords."""
import cv2, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from lunar_registration_loftr import init_loftr, preprocess_image, loftr_match, filter_by_confidence


def load_image(img_path):
    """Loads .xml (PDS4, needs GDAL) / .IMG (self-labeled PDS3, GDAL reads directly) / plain images."""
    img_path = Path(img_path)
    suffix = img_path.suffix.lower()
    if suffix in ('.xml', '.img'):
        from osgeo import gdal
        ds = gdal.Open(str(img_path))
        if ds is None:
            raise FileNotFoundError(f"GDAL couldn't open {img_path}")
        arr = ds.GetRasterBand(1).ReadAsArray()
        if arr.dtype != np.uint8:
            lo, hi = arr.min(), arr.max()
            arr = np.clip((arr - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)
        print(f"✓ Loaded {img_path.name} via GDAL: shape {arr.shape}")
        return arr
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Couldn't load: {img_path}")
    return img


def tiled_loftr_match(matcher, img1, img2, tile_size=800, overlap=100, device='cpu', conf_thresh=0.1):
    h1, w1 = img1.shape[:2]
    all_pts1, all_pts2, all_conf = [], [], []
    step = tile_size - overlap
    for y in range(0, h1, step):
        for x in range(0, w1, step):
            tile1 = img1[y:min(y+tile_size, h1), x:min(x+tile_size, w1)]
            if tile1.shape[0] < 100 or tile1.shape[1] < 100:
                continue
            try:
                m0, m1, conf, _ = loftr_match(matcher, tile1, img2, device)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"  ⚠ OOM at tile ({x},{y}), skipping — consider smaller --tile_size")
                    import torch; torch.cuda.empty_cache()
                    continue
                raise
            if len(m0) == 0:
                continue
            if conf_thresh > 0:
                m0, m1, conf = filter_by_confidence(m0, m1, conf, conf_thresh)
            if len(m0) == 0:
                continue
            m0_global = m0 + np.array([x, y])  # shift tile coords back to full-image coords
            all_pts1.append(m0_global); all_pts2.append(m1); all_conf.append(conf)
    if not all_pts1:
        return np.empty((0,2)), np.empty((0,2)), np.empty((0,))
    pts1 = np.vstack(all_pts1); pts2 = np.vstack(all_pts2); conf = np.concatenate(all_conf)
    print(f"✓ Tiled LoFTR total: {len(pts1)} matches across all tiles")
    return pts1, pts2, conf


if __name__ == '__main__':
    import argparse, torch, json
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True); p.add_argument('--target', required=True)
    p.add_argument('--output', required=True); p.add_argument('--tile_size', type=int, default=512)
    p.add_argument('--overlap', type=int, default=80); p.add_argument('--ransac_threshold', type=float, default=8.0)
    p.add_argument('--confidence_threshold', type=float, default=0.1); p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    img1 = preprocess_image(load_image(args.source))
    img2 = preprocess_image(load_image(args.target))
    matcher = init_loftr(device)

    pts1, pts2, conf = tiled_loftr_match(matcher, img1, img2, args.tile_size, args.overlap, device, args.confidence_threshold)
    if len(pts1) < 4:
        print("❌ Still too few matches — try larger overlap or check images actually overlap in content")
        sys.exit(1)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=args.ransac_threshold)
    inliers = int(mask.sum())
    print(f"✓ RANSAC: {inliers}/{len(pts1)} inliers ({100*inliers/len(pts1):.1f}%)")

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    warped = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
    cv2.imwrite(str(out/"registered_source_tiled.png"), warped)
    with open(out/"metrics_tiled.json", 'w') as f:
        json.dump({"total_matches": len(pts1), "inliers": inliers, "inlier_ratio": inliers/len(pts1)}, f, indent=2)
    print(f"✓ Saved to {out}")
