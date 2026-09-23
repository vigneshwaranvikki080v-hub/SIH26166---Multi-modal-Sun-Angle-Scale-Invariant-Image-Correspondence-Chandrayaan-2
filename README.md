# MEMORAX1360 — Lunar Image Registration (SIH26166)

Dual-matcher pipeline for finding sub-pixel, cross-sensor correspondence between Chandrayaan-2 optical imagery (OHRC / TMC-2 / IIRS) and lunar reference imagery (LRO NAC / SELENE), built for **SIH26166 (ISRO)**.

## 🌐 Live Demo

**Try the Interactive Registration Studio:**  
https://claude.ai/artifact/35592VEHQzen7191mQDikh

Explore real-time tie-point visualization, wipe slider comparison, and registration metrics.

## The problem

Chandrayaan-2 images need to be precisely aligned against trusted reference imagery for mission planning and science, despite:
- **Illumination variation** — different sun azimuth/elevation between missions
- **Viewpoint variation** — different orbit geometry
- **Scale variation** — different altitudes/resolutions across instruments

## Our approach

We run **two independent matchers** on every image pair instead of committing to one method:
- **LoFTR** (multi-scale wrapped) — deep-learning dense matcher
- **RIFT** — classical, illumination-invariant matcher built on phase congruency

Both are scored on `inliers × spatial uniformity`, the higher-scoring candidate is kept, refined with RANSAC homography + sub-pixel refinement, and optionally a Thin Plate Spline warp for the Moon's real (non-flat) terrain.

```
Source (Chandrayaan-2)   Target (LRO NAC)
        \                       /
         \                     /
          Preprocess (grayscale + normalize)
                    |
           ┌────────┴────────┐
        LoFTR               RIFT
   (multiscale deep)   (phase congruency)
           └────────┬────────┘
          Bucket-cap matches (uniform spread)
                    |
          Score & pick winner (inliers × uniformity)
                    |
          RANSAC homography + sub-pixel refine
                    |
             Optional TPS warp
                    |
      ┌─────────────┼─────────────┐
Registered image  Match points  Metrics
                    csv           json
```

## Setup

```bash
conda create -n lunar python=3.10
conda activate lunar
conda install -c conda-forge gdal          # PDS3/PDS4 planetary imagery support
pip install opencv-contrib-python numpy matplotlib scipy phasepack scikit-image
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or /cpu for no GPU
pip install einops yacs kornia loguru joblib pytorch-lightning

git clone https://github.com/zju3dv/LoFTR.git
```

Download the pretrained checkpoint from the [official LoFTR weights folder](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf) (`weights/outdoor_ds.ckpt`) into `LoFTR/checkpoints/`.

## Usage

```bash
python unified_pipeline_v2.py \
  --source source_crop.png \
  --target target_crop.png \
  --output results \
  --bucket_grid_size 4 \
  --max_per_cell 50 \
  --rift_ratio_threshold 0.95 \
  --tps
```

Add `--cpu` to force CPU-only. Both `.png` and PDS3/PDS4 (`.xml`, `.img`, `.lbl`) inputs are supported directly.

**Outputs** (in `--output` folder):
- `registered_source.png` / `registered_source_tps.png` — aligned image
- `match_points.csv` — `source_x, source_y, reference_x, reference_y` for every correspondence
- `metrics.json` — inliers, inlier ratio, RMSE (px), uniformity/coverage score, winning method
- `uniformity_heatmap.png`, `homography_vs_tps.png` — visual diagnostics

## Repo structure

```
lunar_registration_baseline.py   # SIFT + RANSAC baseline
lunar_registration_loftr.py      # LoFTR single-scale matcher + loader (GDAL for PDS3/PDS4)
multiscale_loftr.py              # Wraps LoFTR across scale factors
rift_matcher.py                  # RIFT from scratch: phase congruency, MIM descriptors, matching
bucket_uniform.py                # Enforces uniform match spread before RANSAC
uniformity_metrics.py            # Scores match spatial spread
tps_warp.py                      # Thin Plate Spline non-rigid refinement
unified_pipeline_v2.py           # Main orchestrator — entry point
compare_sift_vs_loftr.py         # Benchmarking utility
make_demo_pair.py                # Generates a synthetic test pair (no real data needed)
sample_output/                   # One representative run's outputs
```

## Current results

Tested end-to-end on real Chandrayaan-2 OHRC data and real LRO NAC (PDS3 EDR) data, near the lunar south pole, at two independent locations:

| Site | Winning matcher | Inliers | RMSE (px) | Notes |
|---|---|---|---|---|
| A | LoFTR & RIFT (tied) | 6 | 3.12 | Two independent methods converged on the same result |
| B | RIFT | 10 | 4.39 | Harder-lit site — RIFT's illumination-invariance won out over LoFTR |

## Known limitations / in progress

- Inlier counts and spatial coverage are still modest — this is an early validated prototype, not production-accuracy yet.
- Current demo pair has a slight edge-overlap mismatch between the Chandrayaan-2 and LRO NAC crops rather than full overlap; sourcing a more precisely co-located NAC frame is the next concrete step to improve inlier count and uniformity.
- No georeferenced GeoTIFF output yet (currently plain PNG).
- No ASIFT branch yet for severe viewpoint distortion.
- Shadow occlusion (features invisible in one image entirely) is not solved — flagged as a known open limitation, not silently ignored.
- No jury-facing UI yet (Streamlit planned).

## References

- Li, J. et al., *RIFT2: Speeding-up RIFT with a new rotation-invariance technique*, arXiv:2303.00319 (2023)
- Sun, J. et al., *LoFTR: Detector-Free Local Feature Matching with Transformers*, CVPR 2021

## Team — MEMORAX1360

Keruthiga D (Team Lead) · Namithkrishna N · Vigneshwaran J · Tamizh Maran S R · Varsshini R D · Rituparna B
